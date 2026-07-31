"""Polymarket ingest: Gamma for discovery/metadata, CLOB for the live book."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from predgraph.ingest.base import MarketRef, Quote, as_float, book_metrics, parse_ts
from predgraph.net import build_client

log = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
VENUE = "polymarket"


def _loads(value, default):
    if isinstance(value, (list, dict)):
        return value
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


class PolymarketClient:
    def __init__(self) -> None:
        self._client = build_client()

    def close(self) -> None:
        self._client.close()

    def discover(self, pages: int = 4, page_size: int = 200) -> list[MarketRef]:
        """Most-traded open markets, newest volume first."""
        refs: list[MarketRef] = []
        for page in range(pages):
            response = self._client.get(
                f"{GAMMA}/markets",
                params={
                    "limit": page_size,
                    "offset": page * page_size,
                    "active": "true",
                    "closed": "false",
                    "order": "volume24hr",
                    "ascending": "false",
                },
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            for raw in batch:
                ref = self._to_ref(raw)
                if ref is not None:
                    refs.append(ref)
        log.info("polymarket: discovered %d tradeable markets", len(refs))
        return refs

    def _to_ref(self, raw: dict) -> MarketRef | None:
        if not raw.get("enableOrderBook"):
            return None
        outcomes = _loads(raw.get("outcomes"), [])
        token_ids = _loads(raw.get("clobTokenIds"), [])
        if not token_ids:
            return None
        # Index of the YES leg; binary markets are ["Yes", "No"].
        yes_index = 0
        for i, outcome in enumerate(outcomes):
            if str(outcome).strip().lower() == "yes":
                yes_index = i
                break
        if yes_index >= len(token_ids):
            return None

        condition_id = raw.get("conditionId") or raw.get("id")
        events = raw.get("events") or []
        event_title = events[0].get("title") if events and isinstance(events[0], dict) else None

        return MarketRef(
            id=f"poly:{condition_id}",
            venue=VENUE,
            venue_id=str(condition_id),
            question=raw.get("question") or "",
            slug=raw.get("slug"),
            event_title=event_title,
            token_id=str(token_ids[yes_index]),
            status="open" if raw.get("acceptingOrders") else "paused",
            open_time=parse_ts(raw.get("startDate")),
            close_time=parse_ts(raw.get("endDate")),
            tags=[str(o) for o in outcomes],
            meta={
                "volume_num": as_float(raw.get("volumeNum")),
                "volume_24h": as_float(raw.get("volume24hr")),
                "liquidity_num": as_float(raw.get("liquidityNum")),
                "best_bid": as_float(raw.get("bestBid")),
                "best_ask": as_float(raw.get("bestAsk")),
                "gamma_spread": as_float(raw.get("spread")),
                "neg_risk": raw.get("negRisk"),
            },
        )

    def quote(self, market_id: str, token_id: str) -> Quote | None:
        try:
            response = self._client.get(f"{CLOB}/book", params={"token_id": token_id})
            response.raise_for_status()
            book = response.json()
        except Exception as exc:  # network/venue hiccup: skip this tick, keep polling
            log.warning("polymarket book %s failed: %s", market_id, exc)
            return None

        bids = [
            (float(level["price"]), float(level["size"]))
            for level in book.get("bids", [])
            if level.get("price") is not None
        ]
        asks = [
            (float(level["price"]), float(level["size"]))
            for level in book.get("asks", [])
            if level.get("price") is not None
        ]
        metrics = book_metrics(bids, asks)
        return Quote(
            market_id=market_id,
            ts=datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0),
            last=as_float(book.get("last_trade_price")),
            **metrics,
        )
