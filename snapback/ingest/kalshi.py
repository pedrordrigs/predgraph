"""Kalshi ingest (trade-api v2).

Two things the older docs get wrong and that cost silent nulls if you trust them:
the market fields are now dollar-denominated strings (`yes_bid_dollars`, not
`yes_bid` in cents), and the order book comes back as `orderbook_fp` with
separate `yes_dollars` / `no_dollars` ladders. A NO bid at p is a YES offer at
1-p, which is how the ask side is derived below.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from snapback.ingest.base import MarketRef, Quote, as_float, book_metrics, parse_ts
from snapback.net import build_client

log = logging.getLogger(__name__)

BASE = "https://api.elections.kalshi.com/trade-api/v2"
VENUE = "kalshi"


class KalshiClient:
    def __init__(self) -> None:
        self._client = build_client()

    def close(self) -> None:
        self._client.close()

    def discover(self, series_tickers: list[str], page_limit: int = 1000) -> list[MarketRef]:
        refs: list[MarketRef] = []
        for series in series_tickers:
            cursor: str | None = None
            found = 0
            while True:
                params = {"series_ticker": series, "status": "open", "limit": page_limit}
                if cursor:
                    params["cursor"] = cursor
                try:
                    response = self._client.get(f"{BASE}/markets", params=params)
                    response.raise_for_status()
                    payload = response.json()
                except Exception as exc:  # noqa: BLE001 - one bad series must not stop discovery
                    log.warning("kalshi: series %s failed: %s", series, exc)
                    break
                batch = payload.get("markets") or []
                for raw in batch:
                    refs.append(self._to_ref(raw, series))
                found += len(batch)
                cursor = payload.get("cursor")
                if not cursor or not batch:
                    break
            if found:
                log.debug("kalshi: series %s -> %d markets", series, found)
        log.info("kalshi: discovered %d open markets across %d series", len(refs), len(series_tickers))
        return refs

    def _to_ref(self, raw: dict, series: str) -> MarketRef:
        ticker = raw.get("ticker")
        subtitle = raw.get("yes_sub_title") or raw.get("subtitle") or ""
        title = raw.get("title") or ""
        return MarketRef(
            id=f"kalshi:{ticker}",
            venue=VENUE,
            venue_id=str(ticker),
            # The threshold lives in the subtitle ("Above 4.25%"), so the two
            # must be joined or every strike in a series looks identical.
            question=f"{title} {subtitle}".strip(),
            slug=None,
            event_title=raw.get("event_ticker"),
            token_id=None,
            status="open" if raw.get("status") == "active" else str(raw.get("status")),
            open_time=parse_ts(raw.get("open_time")),
            close_time=parse_ts(raw.get("close_time")),
            tags=[series],
            meta={
                "series": series,
                "event_ticker": raw.get("event_ticker"),
                "subtitle": subtitle,
                "volume": as_float(raw.get("volume_fp")),
                "volume_24h": as_float(raw.get("volume_24h_fp")),
                "open_interest": as_float(raw.get("open_interest_fp")),
                "liquidity": as_float(raw.get("liquidity_dollars")),
                "yes_bid": as_float(raw.get("yes_bid_dollars")),
                "yes_ask": as_float(raw.get("yes_ask_dollars")),
                "last": as_float(raw.get("last_price_dollars")),
                "market_type": raw.get("market_type"),
            },
        )

    def quote(self, market_id: str, ticker: str) -> Quote | None:
        try:
            response = self._client.get(f"{BASE}/markets/{ticker}/orderbook")
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # noqa: BLE001 - a venue hiccup skips this tick, not the loop
            log.warning("kalshi orderbook %s failed: %s", market_id, exc)
            return None

        book = payload.get("orderbook_fp") or payload.get("orderbook") or {}
        yes_levels = book.get("yes_dollars") or book.get("yes") or []
        no_levels = book.get("no_dollars") or book.get("no") or []

        bids = [(float(p), float(s)) for p, s in yes_levels if p is not None]
        # A NO bid at p is a YES offer at 1-p.
        asks = [(round(1.0 - float(p), 6), float(s)) for p, s in no_levels if p is not None]

        metrics = book_metrics(bids, asks)
        return Quote(
            market_id=market_id,
            ts=datetime.now(UTC).replace(tzinfo=None, microsecond=0),
            **metrics,
        )
