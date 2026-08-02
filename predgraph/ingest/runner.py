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

from predgraph.db import edges as edges_t
from predgraph.db import get_engine, utcnow
from predgraph.db import market_bars as bars_t
from predgraph.db import markets as markets_t
from predgraph.db import nodes as nodes_t
from predgraph.db import quarantine as quarantine_t
from predgraph.ingest.base import MarketRef, Quote
from predgraph.ingest.kalshi import KalshiClient
from predgraph.ingest.polymarket import PolymarketClient
from predgraph.ontology import AnchorSpec, Ontology, load_ontology

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
    if bid is None or ask is None:
        return True  # unknown at discovery; the liquidity gate still applies
    mid = (bid + ask) / 2.0
    return MIN_PRICE < mid < MAX_PRICE and (ask - bid) <= MAX_SPREAD


def _liquidity_score(ref: MarketRef) -> float:
    meta = ref.meta
    if ref.venue == "polymarket":
        return (meta.get("volume_24h") or 0.0) + (meta.get("liquidity_num") or 0.0)
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


def _upsert_market(conn, ref: MarketRef, anchors: list[AnchorSpec], watch: bool) -> str:
    """Insert/update the market's graph node, market row and anchor edges."""
    node_values = {
        "kind": "market",
        "label": ref.question[:250],
        "axis_def": "higher = YES outcome more likely",
        "domain": None,
        "status": "active",
        "aliases": [],
        "meta": {"venue": ref.venue},
        "updated_at": utcnow(),
    }
    exists = conn.execute(sa.select(nodes_t.c.id).where(nodes_t.c.id == ref.id)).first()
    if exists:
        conn.execute(nodes_t.update().where(nodes_t.c.id == ref.id).values(**node_values))
    else:
        conn.execute(nodes_t.insert().values(id=ref.id, created_at=utcnow(), **node_values))

    market_values = {
        "venue": ref.venue,
        "venue_id": ref.venue_id,
        "question": ref.question,
        "slug": ref.slug,
        "event_title": ref.event_title,
        "outcome": "YES",
        "token_id": ref.token_id,
        "status": ref.status,
        "open_time": ref.open_time,
        "close_time": ref.close_time,
        "tags": ref.tags,
        "meta": ref.meta,
        "watch": watch,
        "updated_at": utcnow(),
    }
    exists = conn.execute(sa.select(markets_t.c.id).where(markets_t.c.id == ref.id)).first()
    if exists:
        conn.execute(markets_t.update().where(markets_t.c.id == ref.id).values(**market_values))
    else:
        conn.execute(markets_t.insert().values(id=ref.id, created_at=utcnow(), **market_values))

    for anchor in anchors:
        mechanism = anchor.mechanism or anchor.id
        values = {
            "sign": anchor.sign,
            "weight": anchor.weight,
            "delay_h": anchor.delay_h,
            "halflife_h": anchor.halflife_h,
            "edge_class": "structural",
            "provenance": f"anchor:{anchor.id}",
        }
        existing = conn.execute(
            sa.select(edges_t.c.id).where(
                sa.and_(
                    edges_t.c.src == anchor.driver,
                    edges_t.c.dst == ref.id,
                    edges_t.c.mechanism == mechanism,
                )
            )
        ).first()
        if existing:
            conn.execute(edges_t.update().where(edges_t.c.id == existing.id).values(**values))
        else:
            conn.execute(
                edges_t.insert().values(
                    src=anchor.driver,
                    dst=ref.id,
                    mechanism=mechanism,
                    valid_from=utcnow(),
                    created_at=utcnow(),
                    **values,
                )
            )
    return ref.id


def discover_and_link(
    ontology: Ontology | None = None,
    poly_pages: int = 4,
    quarantine_unmatched: bool = False,
    watch_limit: int = WATCH_LIMIT,
) -> dict:
    ontology = ontology or load_ontology()
    poly = PolymarketClient()
    kalshi = KalshiClient()
    try:
        refs = poly.discover(pages=poly_pages, tag_ids=ontology.polymarket_tags)
        refs += kalshi.discover(ontology.kalshi_series)
    finally:
        poly.close()
        kalshi.close()

    matched: list[tuple[MarketRef, list[AnchorSpec]]] = []
    unmatched: list[MarketRef] = []
    for ref in refs:
        anchors = ontology.match_anchors(ref.venue, ref.match_text)
        if anchors:
            matched.append((ref, anchors))
        else:
            unmatched.append(ref)

    watchlist = select_watchlist([ref for ref, _ in matched], limit=watch_limit)

    stats = {
        "seen": len(refs),
        "linked": 0,
        "watched": 0,
        "unmatched": len(unmatched),
        "by_anchor": {},
        "by_venue": {},
    }
    engine = get_engine()
    with engine.begin() as conn:
        # The selected set is authoritative: clear stale flags so markets that
        # closed or fell out of the ranking stop consuming polling budget.
        conn.execute(markets_t.update().values(watch=False))
        for ref, anchors in matched:
            watch = ref.id in watchlist
            _upsert_market(conn, ref, anchors, watch)
            stats["linked"] += 1
            stats["watched"] += int(watch)
            if watch:
                stats["by_venue"][ref.venue] = stats["by_venue"].get(ref.venue, 0) + 1
            for anchor in anchors:
                stats["by_anchor"][anchor.id] = stats["by_anchor"].get(anchor.id, 0) + int(watch)
        if quarantine_unmatched:
            for ref in unmatched:
                conn.execute(
                    quarantine_t.insert().values(
                        kind="market",
                        payload={"id": ref.id, "question": ref.question, "venue": ref.venue},
                        rationale="no ontology anchor matched",
                        source="discovery",
                        created_at=utcnow(),
                    )
                )
    return stats


FADE_PER_CATEGORY = 25
FADE_MIN_VOLUME = 20_000.0
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


def discover_fade_universe(per_category: int = FADE_PER_CATEGORY) -> dict:
    """Select the watchlist by liquidity across categories, not by ontology.

    The fade strategy needs markets that are liquid and that move, wherever they
    are. The graph's anchors answer a different question — which markets a macro
    story should touch — and using them here simply hid the categories where the
    effect measured strongest.
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
                if (ref.meta.get("volume_num") or 0) < FADE_MIN_VOLUME:
                    continue
                if not _has_room_to_diffuse(ref) or not _is_tradeable_band(ref):
                    continue
                ref.meta["fade_category"] = category
                refs[ref.id] = ref
                kept += 1
                if kept >= per_category:
                    break
            log.info("fade discovery: %s -> %d markets", category, kept)
    finally:
        poly.close()

    engine = get_engine()
    stats = {"seen": len(refs), "watched": 0, "by_category": {}}
    with engine.begin() as conn:
        conn.execute(markets_t.update().values(watch=False))
        for ref in refs.values():
            _upsert_market(conn, ref, anchors=[], watch=True)
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
