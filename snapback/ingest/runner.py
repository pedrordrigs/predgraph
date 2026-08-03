"""Discovery/linking and bar polling.

Discovery attaches a market to the graph only through an ontology anchor. A
market that matches nothing is left in quarantine rather than being wired in on
a guess — an unlinked market costs us one alert we never had, a wrongly linked
one poisons every signal computed through it.
"""

from __future__ import annotations

import logging
from datetime import timedelta

import sqlalchemy as sa

from snapback.db import get_engine, utcnow
from snapback.db import market_bars as bars_t
from snapback.db import markets as markets_t
from snapback.ingest.base import MarketRef, Quote
from snapback.ingest.kalshi import KalshiClient
from snapback.ingest.polymarket import PolymarketClient

log = logging.getLogger(__name__)

# Linking a market to the graph is cheap; polling it every minute is not, and a
# watchlist of hundreds is how you get alert fatigue and rate limits. Kalshi
# lists every strike of every ladder out to 2036, so the polling budget needs
# real gates: time to diffuse, a tradeable price band, and per-ladder caps.
MIN_HOURS_TO_CLOSE = 48.0
MAX_DAYS_TO_CLOSE = 180.0
MIN_PRICE = 0.05
MAX_PRICE = 0.95
MAX_SPREAD = 0.10
POLY_MIN_LIQUIDITY = 500.0
POLY_MIN_VOLUME_24H = 1000.0
KALSHI_MIN_OPEN_INTEREST = 100.0
KALSHI_MIN_VOLUME = 500.0
WATCH_LIMIT = 60
MAX_PER_EVENT = 3


def _has_room_to_diffuse(ref: MarketRef) -> bool:
    """Diffusion needs time to land, but a 2036 market barely moves on today's news."""
    if ref.close_time is None:
        return False
    now = utcnow()
    return (
        now + timedelta(hours=MIN_HOURS_TO_CLOSE)
        < ref.close_time
        < now + timedelta(days=MAX_DAYS_TO_CLOSE)
    )


def _is_liquid_enough(ref: MarketRef) -> bool:
    meta = ref.meta
    if ref.venue == "polymarket":
        return (meta.get("liquidity_num") or 0) >= POLY_MIN_LIQUIDITY or (
            meta.get("volume_24h") or 0
        ) >= POLY_MIN_VOLUME_24H
    return (meta.get("open_interest") or 0) >= KALSHI_MIN_OPEN_INTEREST or (
        meta.get("volume") or 0
    ) >= KALSHI_MIN_VOLUME


def _quoted_price(ref: MarketRef) -> tuple[float | None, float | None]:
    meta = ref.meta
    if ref.venue == "polymarket":
        return meta.get("best_bid"), meta.get("best_ask")
    return meta.get("yes_bid"), meta.get("yes_ask")


def _is_tradeable_band(ref: MarketRef) -> bool:
    """Tails are where spread and longshot bias eat any edge we might have."""
    bid, ask = _quoted_price(ref)
    if bid is None and ask is None:
        return True  # genuinely unknown: Kalshi discovery carries no book
    if bid is None or ask is None:
        # A book quoting one side and not the other is not unknown, it is dead:
        # nobody will take the other side of an exit. Treating this as "unknown,
        # allow" put 80 of 300 sports outcomes on the watchlist, every one of
        # them unable to produce a signal the engine could ever act on.
        return False
    mid = (bid + ask) / 2.0
    return MIN_PRICE < mid < MAX_PRICE and (ask - bid) <= MAX_SPREAD


def _liquidity_score(ref: MarketRef) -> float:
    meta = ref.meta
    if ref.venue == "polymarket":
        # `liquidity_num` is reported per event, not per outcome, and inside a
        # grouped market it runs *backwards*: the dead 0.1c drivers showed
        # $880k against $240k for the contenders. Adding it made the ranking
        # prefer exactly the markets that cannot move. Only `volume_24h` is
        # genuinely per-outcome, and it separates them cleanly.
        return meta.get("volume_24h") or 0.0
    return (meta.get("open_interest") or 0.0) + (meta.get("volume_24h") or 0.0)


def _event_key(ref: MarketRef) -> str:
    """Groups the strikes of one ladder so a single event cannot flood the list."""
    if ref.venue == "kalshi":
        event = ref.meta.get("event_ticker")
        if event:
            return str(event)
        return ref.tags[0] if ref.tags else ref.id
    return ref.event_title or ref.slug or ref.id


def select_watchlist(refs: list[MarketRef], limit: int = WATCH_LIMIT) -> set[str]:
    """Pick the markets worth a request every minute.

    Eligible markets are ranked by liquidity, capped per event so one strike
    ladder cannot crowd out every other driver, then truncated to `limit`.
    """
    eligible = [
        ref
        for ref in refs
        if _has_room_to_diffuse(ref) and _is_liquid_enough(ref) and _is_tradeable_band(ref)
    ]
    eligible.sort(key=_liquidity_score, reverse=True)

    per_event: dict[str, int] = {}
    chosen: set[str] = set()
    for ref in eligible:
        key = _event_key(ref)
        if per_event.get(key, 0) >= MAX_PER_EVENT:
            continue
        per_event[key] = per_event.get(key, 0) + 1
        chosen.add(ref.id)
        if len(chosen) >= limit:
            break
    return chosen


def _upsert_market(conn, ref: MarketRef, watch: bool) -> str:
    """Insert or refresh one market row."""
    market_values = {
        "venue": ref.venue,
        "venue_id": ref.venue_id,
        "question": ref.question,
        "slug": ref.slug,
        "event_title": ref.event_title,
        "token_id": ref.token_id,
        "status": ref.status,
        "open_time": ref.open_time,
        "close_time": ref.close_time,
        "tags": ref.tags,
        "meta": ref.meta,
        "watch": watch,
        "updated_at": utcnow(),
    }
    exists = conn.execute(
        sa.select(markets_t.c.id).where(markets_t.c.id == ref.id)
    ).first()
    if exists:
        conn.execute(markets_t.update().where(markets_t.c.id == ref.id).values(**market_values))
    else:
        conn.execute(markets_t.insert().values(id=ref.id, created_at=utcnow(), **market_values))
    return ref.id


# Polymarket category tags, verified 2026-07-31. The fade edge showed up across
# all of these, so restricting the universe to macro/geo markets — which is what
# ontology anchors do — was leaving most of the opportunity unwatched.
FADE_TAGS = {
    "sports": "1",
    "crypto": "21",
    "us-politics": "789",
    "geopolitics": "100265",
    "pop-culture": "596",
    "business": "107",
    "tech": "1401",
    "world": "101970",
    "elections": "2",
}


FADE_PER_CATEGORY = 25
# Per-outcome 24h volume. The old gate read `volume_num`, which Polymarket
# reports for the whole event, so every driver in a $12M championship cleared a
# $20k bar - including the ones quoted at a tenth of a cent.
#
# Set low on purpose. 380 markets pass the band and close-time gates, and the
# ranking plus FADE_TARGET_TOTAL already keep the best 200; a high bar here
# just discards liquid-enough candidates before that choice is made. Depth and
# spread are re-checked per trade by the engine, which is where they belong.
FADE_MIN_VOLUME_24H = 50.0
# Discovery band, deliberately a little wider than the widest trading band
# (0.15-0.95) so a market can drift into range between daily rediscoveries -
# but not so wide that it admits markets needing to triple before they qualify.
FADE_PRICE_LO, FADE_PRICE_HI = 0.10, 0.96
# Held at the size the storage budget was measured against: ~93 MB/day at
# 3-day retention. The gates below decide quality; this decides cost.
FADE_TARGET_TOTAL = 200
# Polymarket category tags, verified 2026-07-31. The fade edge showed up across
# all of these, so restricting the universe to macro/geo markets — which is what
# ontology anchors do — was leaving most of the opportunity unwatched.
FADE_TAGS = {
    "sports": "1",
    "crypto": "21",
    "us-politics": "789",
    "geopolitics": "100265",
    "pop-culture": "596",
    "business": "107",
    "tech": "1401",
    "world": "101970",
    "elections": "2",
}


def _in_fade_band(ref: MarketRef) -> bool:
    """Could this market plausibly produce a signal the engine would act on?

    The engine cannot enter outside 0.15-0.95, so a market quoted at a tenth of
    a cent is not a long shot on the watchlist - it is a market that will never
    fire, consuming a poll slot and 93 MB/day of storage to prove it. Unlike
    the generic band check this one has no unknown-is-fine escape: a fade
    candidate with no quote is not a candidate.
    """
    bid, ask = _quoted_price(ref)
    if bid is None or ask is None:
        return False
    mid = (bid + ask) / 2.0
    return FADE_PRICE_LO <= mid <= FADE_PRICE_HI


def discover_fade_universe(per_category: int = FADE_PER_CATEGORY) -> dict:
    """Select the watchlist by tradeability, then by per-outcome volume.

    Ranked on per-outcome 24h volume among markets the engine could actually
    enter, capped per category so one hot topic cannot take the whole list.
    """
    poly = PolymarketClient()
    try:
        refs: dict[str, MarketRef] = {}
        for category, tag_id in FADE_TAGS.items():
            try:
                found = poly.discover(pages=0, tag_ids=[tag_id])
            except Exception as exc:  # noqa: BLE001 - one tag must not stop discovery
                log.warning("fade discovery: tag %s failed: %s", category, exc)
                continue
            ranked = sorted(found, key=_liquidity_score, reverse=True)
            kept = 0
            for ref in ranked:
                if ref.id in refs:
                    continue
                if (ref.meta.get("volume_24h") or 0) < FADE_MIN_VOLUME_24H:
                    continue
                if not _has_room_to_diffuse(ref) or not _is_tradeable_band(ref):
                    continue
                if not _in_fade_band(ref):
                    continue
                ref.meta["fade_category"] = category
                refs[ref.id] = ref
                kept += 1
                if kept >= per_category:
                    break
            log.info("fade discovery: %s -> %d markets", category, kept)

        # The tag pages alone yield ~109 markets once the dead ones are gone.
        # A volume-ranked sweep finds plenty more that qualify, but it is not
        # tag-scoped, so it runs last and is labelled honestly rather than
        # being attributed to whichever category happened to be first.
        if len(refs) < FADE_TARGET_TOTAL:
            try:
                sweep = poly.discover(pages=3)
            except Exception as exc:  # noqa: BLE001 - backfill is best-effort
                log.warning("fade discovery: sweep failed: %s", exc)
                sweep = []
            added = 0
            for ref in sorted(sweep, key=_liquidity_score, reverse=True):
                if len(refs) >= FADE_TARGET_TOTAL:
                    break
                if ref.id in refs:
                    continue
                if (ref.meta.get("volume_24h") or 0) < FADE_MIN_VOLUME_24H:
                    continue
                if not _has_room_to_diffuse(ref) or not _is_tradeable_band(ref):
                    continue
                if not _in_fade_band(ref):
                    continue
                ref.meta["fade_category"] = "sweep"
                refs[ref.id] = ref
                added += 1
            log.info("fade discovery: sweep -> %d markets", added)
    finally:
        poly.close()

    engine = get_engine()
    stats = {"seen": len(refs), "watched": 0, "by_category": {}}
    with engine.begin() as conn:
        conn.execute(markets_t.update().values(watch=False))
        for ref in refs.values():
            _upsert_market(conn, ref, watch=True)
            stats["watched"] += 1
            category = ref.meta.get("fade_category", "?")
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
    return stats


def watched_markets() -> list[dict]:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sa.select(
                markets_t.c.id,
                markets_t.c.venue,
                markets_t.c.venue_id,
                markets_t.c.token_id,
                markets_t.c.question,
            ).where(markets_t.c.watch.is_(True))
        ).all()
    return [dict(row._mapping) for row in rows]


def _store_quotes(quotes: list[Quote]) -> int:
    if not quotes:
        return 0
    engine = get_engine()
    # Two round trips for the whole batch, not two per quote. A SELECT-then-
    # INSERT per market is free against local SQLite but dominates everything
    # against a hosted database: 200 markets meant 400 sequential trips, which
    # on its own pushed a poll past the 60-second cadence the strategy needs.
    min_ts = min(quote.ts for quote in quotes)
    with engine.begin() as conn:
        existing = {
            (row.market_id, row.ts)
            for row in conn.execute(
                sa.select(bars_t.c.market_id, bars_t.c.ts).where(
                    sa.and_(
                        bars_t.c.market_id.in_({quote.market_id for quote in quotes}),
                        bars_t.c.ts >= min_ts,
                    )
                )
            )
        }
        rows = []
        for quote in quotes:
            key = (quote.market_id, quote.ts)
            # `existing` also absorbs duplicates inside this batch, which the
            # per-row version could not hit but a bulk insert would fail on.
            if key in existing:
                continue
            existing.add(key)
            rows.append(
                {
                    "market_id": quote.market_id,
                    "ts": quote.ts,
                    "mid": quote.mid,
                    "bid": quote.bid,
                    "ask": quote.ask,
                    "spread": quote.spread,
                    "last": quote.last,
                    "volume": quote.volume,
                    "liquidity": quote.liquidity,
                    "depth_2c": quote.depth_2c,
                }
            )
        if rows:
            conn.execute(bars_t.insert(), rows)
    return len(rows)


def poll_once(markets: list[dict] | None = None) -> dict:
    markets = markets if markets is not None else watched_markets()
    if not markets:
        return {"markets": 0, "quotes": 0, "written": 0}

    poly = PolymarketClient()
    kalshi = KalshiClient()
    quotes: list[Quote] = []
    try:
        for market in markets:
            if market["venue"] == "polymarket":
                if not market["token_id"]:
                    continue
                quote = poly.quote(market["id"], market["token_id"])
            else:
                quote = kalshi.quote(market["id"], market["venue_id"])
            if quote is not None and quote.mid is not None:
                quotes.append(quote)
    finally:
        poly.close()
        kalshi.close()

    written = _store_quotes(quotes)
    return {"markets": len(markets), "quotes": len(quotes), "written": written}
