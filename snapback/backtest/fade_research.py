"""Deep analysis of the fade mechanism.

Builds one row per spike with the features available *before* the trade would be
placed, plus the forward price path, so every question below is answered off the
same dataset rather than from separate ad-hoc runs:

* how long to hold, and whether the optimum is a plateau (robust) or a point
  (overfit to this sample);
* whether to fade or follow, conditioned on what kind of spike it is;
* whether any signal — graph neighbours, market-wide breadth, prior trend —
  separates "real news that keeps going" from "liquidity noise that snaps back".

PnL is expressed as return on capital deployed, because a prediction-market
position posts (1 - price) to short and (price) to go long, so identical price
moves mean very different returns depending on where the market is trading.
"""

from __future__ import annotations

import logging
import statistics
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import pairwise

import sqlalchemy as sa

from snapback.db import get_engine
from snapback.db import history_bars as hist_t
from snapback.db import markets as markets_t
from snapback.signal.prices import logit

log = logging.getLogger(__name__)


def ladder_keys() -> dict[str, str]:
    """Group markets that are strikes of the same underlying event."""
    keys: dict[str, str] = {}
    with get_engine().connect() as conn:
        for row in conn.execute(
            sa.select(
                markets_t.c.id, markets_t.c.event_title, markets_t.c.slug, markets_t.c.meta
            )
        ):
            meta = row.meta or {}
            keys[row.id] = str(
                meta.get("event_ticker") or row.event_title or row.slug or row.id
            )
    return keys


def bar_counts(resolution_min: int = 60) -> dict[str, int]:
    with get_engine().connect() as conn:
        return {
            row.market_id: row.n
            for row in conn.execute(
                sa.select(hist_t.c.market_id, sa.func.count().label("n"))
                .where(hist_t.c.resolution_min == resolution_min)
                .group_by(hist_t.c.market_id)
            )
        }


def collapse_by_ladder(
    market_ids: list[str], ladder: dict[str, str], counts: dict[str, int]
) -> list[str]:
    """One representative per ladder — the strike with the most history."""
    best: dict[str, str] = {}
    for market_id in market_ids:
        key = ladder.get(market_id, market_id)
        current = best.get(key)
        if current is None or counts.get(market_id, 0) > counts.get(current, 0):
            best[key] = market_id
    return list(best.values())

FORWARD_MIN = (5, 15, 30, 60, 120, 240, 480, 720, 1440, 2880)
BREADTH_WINDOW_MIN = 15
PRE_TREND_H = 6.0
MAX_QUOTE_AGE_MIN = 20


@dataclass
class Spike:
    market_id: str
    category: str
    venue: str
    ts: datetime
    jump_logit: float
    velocity_min: float
    entry_price: float
    sigma: float | None
    pre_trend_logit: float | None
    hours_to_close: float | None
    hour_utc: int
    breadth: int = 0
    neighbours_moved: bool | None = None
    # forward fade return on capital, per horizon (positive = fade profitable)
    forward: dict[int, float] = field(default_factory=dict)
    mfe: float = 0.0  # best fade return seen within 24h
    mae: float = 0.0  # worst fade return seen within 24h

    @property
    def direction(self) -> int:
        """+1 if the price jumped up (fade = short), -1 if it jumped down."""
        return 1 if self.jump_logit > 0 else -1

    @property
    def capital_per_unit(self) -> float:
        """Posted to fade: buy NO after an up-spike, buy YES after a down-spike."""
        return 1.0 - self.entry_price if self.direction > 0 else self.entry_price


class Series:
    def __init__(self, points: list[tuple[datetime, float]]) -> None:
        self.ts = [t for t, _ in points]
        self.px = [p for _, p in points]

    def at(self, when: datetime, max_age_min: int = MAX_QUOTE_AGE_MIN) -> float | None:
        i = bisect_right(self.ts, when)
        if i == 0:
            return None
        if (when - self.ts[i - 1]) > timedelta(minutes=max_age_min):
            return None
        return self.px[i - 1]

    def between(self, start: datetime, end: datetime) -> list[tuple[datetime, float]]:
        lo = bisect_left(self.ts, start)
        hi = bisect_right(self.ts, end)
        return list(zip(self.ts[lo:hi], self.px[lo:hi], strict=True))


def fade_return_on_capital(spike: Spike, price_later: float) -> float:
    """Return on posted capital if the fade were closed at `price_later`."""
    move = price_later - spike.entry_price
    pnl_per_unit = -spike.direction * move
    return pnl_per_unit / spike.capital_per_unit if spike.capital_per_unit > 0.02 else 0.0


def extract_spikes(
    market_id: str,
    category: str,
    minute_points: list[tuple[datetime, float]],
    hourly_points: list[tuple[datetime, float]],
    close_time: datetime | None,
    min_jump: float = 0.30,
    window_min: int = 60,
    lockout_h: float = 24.0,
    price_lo: float = 0.10,
    price_hi: float = 0.90,
) -> list[Spike]:
    if len(minute_points) < 120:
        return []
    series = Series(minute_points)
    hourly = Series(hourly_points) if hourly_points else None

    sigma = None
    if hourly_points and len(hourly_points) >= 30:
        lg = [logit(p) for _, p in hourly_points]
        diffs = [b - a for a, b in pairwise(lg)]
        if len(diffs) >= 20:
            candidate = statistics.pstdev(diffs)
            sigma = candidate if candidate > 1e-6 else None

    spikes: list[Spike] = []
    last_ts: datetime | None = None
    left = 0
    for i, now in enumerate(series.ts):
        while series.ts[left] < now - timedelta(minutes=window_min):
            left += 1
        if left >= i:
            continue
        delta = logit(series.px[i]) - logit(series.px[left])
        if abs(delta) < max(min_jump, (sigma or 0.0) * 3.0):
            continue
        if last_ts is not None and now - last_ts < timedelta(hours=lockout_h):
            continue
        entry_price = series.px[i]
        if not (price_lo <= entry_price <= price_hi):
            continue

        direction = 1 if delta > 0 else -1
        base = logit(series.px[left])
        t10 = t90 = None
        for k in range(left, i + 1):
            progress = (logit(series.px[k]) - base) * direction
            if t10 is None and progress >= 0.1 * abs(delta):
                t10 = series.ts[k]
            if t90 is None and progress >= 0.9 * abs(delta):
                t90 = series.ts[k]
                break
        if t10 is None or t90 is None:
            continue

        pre_trend = None
        if hourly is not None:
            before = hourly.at(now - timedelta(hours=PRE_TREND_H), max_age_min=180)
            start_of_window = series.px[left]
            if before is not None:
                pre_trend = logit(start_of_window) - logit(before)

        spike = Spike(
            market_id=market_id,
            category=category,
            venue=market_id.split(":", 1)[0],
            ts=now,
            jump_logit=delta,
            velocity_min=(t90 - t10).total_seconds() / 60.0,
            entry_price=entry_price,
            sigma=sigma,
            pre_trend_logit=pre_trend,
            hours_to_close=(
                (close_time - now).total_seconds() / 3600.0 if close_time else None
            ),
            hour_utc=now.hour,
        )

        for minutes in FORWARD_MIN:
            later = series.at(now + timedelta(minutes=minutes), max_age_min=60)
            if later is not None:
                spike.forward[minutes] = fade_return_on_capital(spike, later)

        path = series.between(now, now + timedelta(hours=24))
        if path:
            returns = [fade_return_on_capital(spike, p) for _, p in path]
            spike.mfe = max(returns)
            spike.mae = min(returns)

        spikes.append(spike)
        last_ts = now
    return spikes


def add_breadth(spikes: list[Spike]) -> None:
    """How many other markets spiked at the same moment.

    A lone spike is idiosyncratic — thin book, one big order. A cluster is the
    market reacting to something real, which is the cheapest available proxy
    for "news happened" without any news feed at all.
    """
    ordered = sorted(spikes, key=lambda s: s.ts)
    stamps = [s.ts for s in ordered]
    for spike in ordered:
        lo = bisect_left(stamps, spike.ts - timedelta(minutes=BREADTH_WINDOW_MIN))
        hi = bisect_right(stamps, spike.ts + timedelta(minutes=BREADTH_WINDOW_MIN))
        others = {
            ordered[i].market_id
            for i in range(lo, hi)
            if ordered[i].market_id != spike.market_id
        }
        spike.breadth = len(others)


# --- exit rule simulation ---------------------------------------------------


def simulate_exit_rule(
    spike: Spike,
    series: Series,
    hold_min: int,
    target_frac: float | None,
    stop_logit: float | None,
) -> tuple[float, str]:
    """Return on capital and the reason the position closed."""
    entry_logit = logit(spike.entry_price)
    direction = spike.direction
    target_level = (
        entry_logit - direction * target_frac * abs(spike.jump_logit)
        if target_frac
        else None
    )
    stop_level = entry_logit + direction * stop_logit if stop_logit else None

    for ts, price in series.between(
        spike.ts + timedelta(minutes=1), spike.ts + timedelta(minutes=hold_min)
    ):
        value = logit(price)
        if target_level is not None and (value - target_level) * direction <= 0:
            return fade_return_on_capital(spike, price), "target"
        if stop_level is not None and (value - stop_level) * direction >= 0:
            return fade_return_on_capital(spike, price), "stop"

    final = series.at(spike.ts + timedelta(minutes=hold_min), max_age_min=120)
    if final is None:
        return 0.0, "no_data"
    return fade_return_on_capital(spike, final), "time"


def bucket_stats(values: list[float], cost_on_capital: float = 0.0) -> dict:
    if not values:
        return {"n": 0}
    adjusted = [v - cost_on_capital for v in values]
    return {
        "n": len(adjusted),
        "mean": round(statistics.mean(adjusted), 4),
        "median": round(statistics.median(adjusted), 4),
        "win": round(100.0 * sum(1 for v in adjusted if v > 0) / len(adjusted), 1),
    }
