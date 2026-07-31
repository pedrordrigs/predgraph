"""Polymarket ingest: Gamma for discovery/metadata, CLOB for the live book."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

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

    def _page(self, params: dict, pages: int, page_size: int) -> list[MarketRef]:
        refs: list[MarketRef] = []
        for page in range(pages):
            response = self._client.get(
                f"{GAMMA}/markets",
                params={**params, "limit": page_size, "offset": page * page_size},
            )
            response.raise_for_status()
            batch = response.json()
            if not batch:
                break
            for raw in batch:
                ref = self._to_ref(raw)
                if ref is not None:
                    refs.append(ref)
            if len(batch) < page_size:
                break
        return refs

    def discover(
        self,
        pages: int = 4,
        page_size: int = 200,
        tag_ids: list[str] | None = None,
    ) -> list[MarketRef]:
        """Open markets by topic tag, plus a volume-ranked sweep.

        The volume sweep alone is dominated by sports and whatever is hot
        today, so a quiet-but-well-connected oil market — exactly where a
        lagged repricing would show up — never surfaces. Tags fix the recall
        problem; the sweep stays as a safety net for untagged markets.
        """
        by_id: dict[str, MarketRef] = {}

        for tag_id in tag_ids or []:
            try:
                found = self._page(
                    {"tag_id": tag_id, "active": "true", "closed": "false"}, pages=3, page_size=100
                )
            except Exception as exc:  # noqa: BLE001 - one bad tag must not stop discovery
                log.warning("polymarket: tag %s failed: %s", tag_id, exc)
                continue
            for ref in found:
                by_id.setdefault(ref.id, ref)

        tagged = len(by_id)
        sweep = self._page(
            {
                "active": "true",
                "closed": "false",
                "order": "volume24hr",
                "ascending": "false",
            },
            pages=pages,
            page_size=page_size,
        )
        for ref in sweep:
            by_id.setdefault(ref.id, ref)

        log.info(
            "polymarket: discovered %d markets (%d from tags, %d added by volume sweep)",
            len(by_id),
            tagged,
            len(by_id) - tagged,
        )
        return list(by_id.values())

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
        except Exception as exc:  # noqa: BLE001 - a venue hiccup skips this tick, not the loop
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
            ts=datetime.now(UTC).replace(tzinfo=None, microsecond=0),
            last=as_float(book.get("last_trade_price")),
            **metrics,
        )
