from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime


def parse_ts(value: str | None) -> datetime | None:
    """Parse venue ISO timestamps into naive UTC."""
    if not value:
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(UTC).replace(tzinfo=None)
    return parsed


def as_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass(slots=True)
class MarketRef:
    id: str
    venue: str
    venue_id: str
    question: str
    slug: str | None = None
    event_title: str | None = None
    token_id: str | None = None
    status: str = "open"
    open_time: datetime | None = None
    close_time: datetime | None = None
    tags: list[str] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    @property
    def match_text(self) -> str:
        """Text an ontology anchor is matched against."""
        parts = [self.question, self.slug or "", self.event_title or "", " ".join(self.tags)]
        return " ".join(p for p in parts if p)


@dataclass(slots=True)
class Quote:
    market_id: str
    ts: datetime
    mid: float | None = None
    bid: float | None = None
    ask: float | None = None
    spread: float | None = None
    last: float | None = None
    volume: float | None = None
    liquidity: float | None = None
    depth_2c: float | None = None


def book_metrics(
    bids: list[tuple[float, float]],
    asks: list[tuple[float, float]],
    band: float = 0.02,
) -> dict[str, float | None]:
    """Best bid/ask, mid, spread and notional depth within `band` of mid.

    Depth is in USD notional (size x price) because that is what actually caps
    how much of a signal we could ever trade.
    """
    best_bid = max((p for p, _ in bids), default=None)
    best_ask = min((p for p, _ in asks), default=None)
    if best_bid is None and best_ask is None:
        return {"bid": None, "ask": None, "mid": None, "spread": None, "depth_2c": None}
    if best_bid is None:
        mid = best_ask
    elif best_ask is None:
        mid = best_bid
    else:
        mid = (best_bid + best_ask) / 2.0

    spread = None
    if best_bid is not None and best_ask is not None:
        spread = round(best_ask - best_bid, 6)

    depth = 0.0
    for price, size in bids:
        if mid is not None and price >= mid - band:
            depth += price * size
    for price, size in asks:
        if mid is not None and price <= mid + band:
            depth += price * size

    return {
        "bid": best_bid,
        "ask": best_ask,
        "mid": round(mid, 6) if mid is not None else None,
        "spread": spread,
        "depth_2c": round(depth, 2),
    }
