"""Historical price fetching for both venues.

Two venue traps are handled here, both found by hitting them:

* Polymarket returns 400 — not an empty list — when the requested window falls
  outside a market's trading period, and closed markets routinely carry an
  `endDate` in the future because they resolved early. Every request is clamped
  to the market's own [open, close] and to now.
* Kalshi's `price.close_dollars` only exists where a trade happened (401 of 824
  candles on a liquid Fed market), while `yes_bid`/`yes_ask` are populated on
  100% of them. Mid comes from bid/ask; using last trade would silently drop
  half the timeline and bias every lag measurement toward trade arrivals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa

from predgraph.db import get_engine
from predgraph.db import history_bars as hist_t
from predgraph.db import markets as markets_t
from predgraph.ingest.base import as_float
from predgraph.ingest.kalshi import BASE as KALSHI_BASE
from predgraph.ingest.polymarket import CLOB
from predgraph.net import build_client

log = logging.getLogger(__name__)

# Kalshi rejects requests spanning more than ~5000 candles, so 1-minute pulls
# have to be chunked. 3 days keeps us clear of the cap.
KALSHI_MAX_CANDLES = 4800


@dataclass(slots=True)
class HistBar:
    ts: datetime
    mid: float | None
    bid: float | None = None
    ask: float | None = None


def _to_naive_utc(value: datetime) -> datetime:
    return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value


def _clamp_window(
    market: dict, start: datetime, end: datetime
) -> tuple[datetime, datetime] | None:
    now = datetime.now(UTC).replace(tzinfo=None)
    lo, hi = start, min(end, now)
    if market.get("open_time"):
        lo = max(lo, market["open_time"])
    if market.get("close_time"):
        hi = min(hi, market["close_time"])
    if hi <= lo:
        return None
    return lo, hi


class HistoryFetcher:
    def __init__(self) -> None:
        self._client = build_client()

    def close(self) -> None:
        self._client.close()

    def fetch(self, market: dict, start: datetime, end: datetime, resolution_min: int) -> list[HistBar]:
        window = _clamp_window(market, start, end)
        if window is None:
            return []
        lo, hi = window
        if market["venue"] == "polymarket":
            return self._polymarket(market, lo, hi, resolution_min)
        return self._kalshi(market, lo, hi, resolution_min)

    def _polymarket(self, market: dict, lo: datetime, hi: datetime, res: int) -> list[HistBar]:
        if not market.get("token_id"):
            return []
        params = {
            "market": market["token_id"],
            "startTs": int(lo.replace(tzinfo=UTC).timestamp()),
            "endTs": int(hi.replace(tzinfo=UTC).timestamp()),
            "fidelity": res,
        }
        try:
            response = self._client.get(f"{CLOB}/prices-history", params=params)
            if response.status_code == 400:
                # Window outside the market's life despite clamping (stale metadata).
                return []
            response.raise_for_status()
            history = response.json().get("history", [])
        except Exception as exc:  # noqa: BLE001 - one market must not abort a backfill
            log.warning("history %s failed: %s", market["id"], exc)
            return []
        return [
            HistBar(ts=datetime.fromtimestamp(point["t"], UTC).replace(tzinfo=None),
                    mid=as_float(point.get("p")))
            for point in history
            if point.get("t")
        ]

    def _kalshi(self, market: dict, lo: datetime, hi: datetime, res: int) -> list[HistBar]:
        series = (market.get("meta") or {}).get("series")
        if not series:
            return []
        url = f"{KALSHI_BASE}/series/{series}/markets/{market['venue_id']}/candlesticks"
        chunk = timedelta(minutes=res * KALSHI_MAX_CANDLES)
        bars: list[HistBar] = []
        cursor = lo
        while cursor < hi:
            chunk_end = min(cursor + chunk, hi)
            params = {
                "start_ts": int(cursor.replace(tzinfo=UTC).timestamp()),
                "end_ts": int(chunk_end.replace(tzinfo=UTC).timestamp()),
                "period_interval": res,
            }
            try:
                response = self._client.get(url, params=params)
                response.raise_for_status()
                candles = response.json().get("candlesticks", [])
            except Exception as exc:  # noqa: BLE001 - skip the chunk, keep the series
                log.warning("candles %s failed: %s", market["id"], exc)
                candles = []
            for candle in candles:
                bid = as_float((candle.get("yes_bid") or {}).get("close_dollars"))
                ask = as_float((candle.get("yes_ask") or {}).get("close_dollars"))
                if bid is None and ask is None:
                    continue
                mid = (bid + ask) / 2.0 if bid is not None and ask is not None else (bid or ask)
                bars.append(
                    HistBar(
                        ts=datetime.fromtimestamp(candle["end_period_ts"], UTC).replace(tzinfo=None),
                        mid=mid,
                        bid=bid,
                        ask=ask,
                    )
                )
            cursor = chunk_end
        return bars


def load_markets(market_ids: list[str]) -> dict[str, dict]:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sa.select(
                markets_t.c.id,
                markets_t.c.venue,
                markets_t.c.venue_id,
                markets_t.c.token_id,
                markets_t.c.question,
                markets_t.c.open_time,
                markets_t.c.close_time,
                markets_t.c.meta,
            ).where(markets_t.c.id.in_(market_ids))
        ).all()
    return {row.id: dict(row._mapping) for row in rows}


def store(market_id: str, bars: list[HistBar], resolution_min: int) -> int:
    if not bars:
        return 0
    engine = get_engine()
    with engine.begin() as conn:
        existing = {
            row.ts
            for row in conn.execute(
                sa.select(hist_t.c.ts).where(
                    sa.and_(
                        hist_t.c.market_id == market_id,
                        hist_t.c.resolution_min == resolution_min,
                    )
                )
            )
        }
        rows = [
            {
                "market_id": market_id,
                "ts": bar.ts,
                "resolution_min": resolution_min,
                "mid": bar.mid,
                "bid": bar.bid,
                "ask": bar.ask,
            }
            for bar in bars
            if bar.ts not in existing and bar.mid is not None
        ]
        if rows:
            conn.execute(hist_t.insert(), rows)
    return len(rows)


def load_series(market_id: str, resolution_min: int) -> list[tuple[datetime, float]]:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sa.select(hist_t.c.ts, hist_t.c.mid)
            .where(
                sa.and_(
                    hist_t.c.market_id == market_id,
                    hist_t.c.resolution_min == resolution_min,
                    hist_t.c.mid.isnot(None),
                )
            )
            .order_by(hist_t.c.ts)
        ).all()
    return [(row.ts, float(row.mid)) for row in rows]


def backfill(
    market_ids: list[str], days: int = 90, resolution_min: int = 60
) -> dict:
    end = datetime.now(UTC).replace(tzinfo=None)
    start = end - timedelta(days=days)
    markets = load_markets(market_ids)
    fetcher = HistoryFetcher()
    stats = {"markets": 0, "bars": 0, "empty": 0}
    try:
        for market_id in market_ids:
            market = markets.get(market_id)
            if market is None:
                continue
            bars = fetcher.fetch(market, start, end, resolution_min)
            written = store(market_id, bars, resolution_min)
            stats["markets"] += 1
            stats["bars"] += written
            if not bars:
                stats["empty"] += 1
    finally:
        fetcher.close()
    return stats
