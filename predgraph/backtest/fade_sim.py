"""Minute-resolution fade simulation with emulated executions.

Emulates, on historical minute data, exactly what the live engine would do:
detect a jump from a rolling minute window, wait out detection latency, enter
*against* the move, exit on retracement target / continuation stop / time stop.

Honesty rules baked in rather than left to discipline:

* Entry happens 2 minutes after the signal, at the first fresh quote — the live
  engine polls every 60s, and pretending we trade the signal bar is how
  backtests lie.
* Polymarket history is mid prices only, so execution costs are scenario
  parameters (round-trip 1/2/3 cents), not something we get to forget. Kalshi
  bars carry real bid/ask and use them.
* All rules were fixed before the first run: retrace target 50%, continuation
  stop 0.5 logit, time stop 24h, band 0.10-0.90, one episode per market at a
  time. No mid-run tuning; a changed rule means a rerun labeled as such.
"""

from __future__ import annotations

import logging
import statistics
from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta

from predgraph.signal.damage import logit

log = logging.getLogger(__name__)

# Signal
JUMP_WINDOW_MIN = 60
JUMP_MIN_ABS_LOGIT = 0.30
JUMP_Z = 3.0
# Execution emulation
ENTRY_DELAY_MIN = 2
MAX_QUOTE_AGE_MIN = 20
# Exits
RETRACE_TARGET = 0.5
CONTINUATION_STOP_LOGIT = 0.5
TIME_STOP_H = 24.0
# Eligibility
PRICE_LO, PRICE_HI = 0.10, 0.90
EPISODE_LOCKOUT_H = 24.0
ROUND_TRIP_COSTS = (0.01, 0.02, 0.03)


@dataclass(slots=True)
class SimTrade:
    market_id: str
    category: str
    venue: str
    signal_ts: datetime
    jump_logit: float
    jump_velocity_min: float | None
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    exit_reason: str  # target | stop | time | data_end
    direction: int  # -1 = fade an up-jump (short YES), +1 = fade a down-jump

    @property
    def gross_pnl(self) -> float:
        """Price-point PnL per unit stake, before costs."""
        return self.direction * (self.exit_price - self.entry_price)

    def net_pnl(self, round_trip_cost: float) -> float:
        return self.gross_pnl - round_trip_cost

    @property
    def holding_h(self) -> float:
        return (self.exit_ts - self.entry_ts).total_seconds() / 3600.0


class MinuteSeries:
    """Minute series with staleness-aware lookup, O(log n) per query."""

    def __init__(self, series: list[tuple[datetime, float]]) -> None:
        self.ts = [t for t, _ in series]
        self.px = [p for _, p in series]

    def at(self, when: datetime, max_age_min: int = MAX_QUOTE_AGE_MIN) -> float | None:
        i = bisect_right(self.ts, when)
        if i == 0:
            return None
        if (when - self.ts[i - 1]) > timedelta(minutes=max_age_min):
            return None
        return self.px[i - 1]

    def first_at_or_after(
        self, when: datetime, within_min: int = 30
    ) -> tuple[datetime, float] | None:
        i = bisect_left(self.ts, when)
        if i >= len(self.ts):
            return None
        if (self.ts[i] - when) > timedelta(minutes=within_min):
            return None
        return self.ts[i], self.px[i]


def hourly_sigma(hourly: list[tuple[datetime, float]]) -> float | None:
    """This market's own typical 1h logit move — the yardstick for 'jump'."""
    if len(hourly) < 30:
        return None
    from itertools import pairwise

    logits = [(t, logit(p)) for t, p in hourly]
    moves = [
        b[1] - a[1] for a, b in pairwise(logits) if (b[0] - a[0]) <= timedelta(hours=2)
    ]
    if len(moves) < 20:
        return None
    sigma = statistics.pstdev(moves)
    return sigma if sigma > 1e-6 else None


def detect_minute_jumps(
    minutes: MinuteSeries, sigma: float | None
) -> list[tuple[datetime, float, float | None]]:
    """(signal_ts, jump_delta_logit, velocity_min) for threshold crossings.

    velocity = minutes the move took from 10% to 90% of its amplitude inside
    the trailing window; a proxy for spike vs grind.
    """
    threshold = max(JUMP_MIN_ABS_LOGIT, (sigma or 0.0) * JUMP_Z)
    jumps: list[tuple[datetime, float, float | None]] = []
    last_signal: datetime | None = None
    n = len(minutes.ts)
    left = 0
    for i in range(n):
        now = minutes.ts[i]
        while minutes.ts[left] < now - timedelta(minutes=JUMP_WINDOW_MIN):
            left += 1
        if left >= i:
            continue
        base = logit(minutes.px[left])
        delta = logit(minutes.px[i]) - base
        if abs(delta) < threshold:
            continue
        if last_signal is not None and now - last_signal < timedelta(hours=EPISODE_LOCKOUT_H):
            continue
        # velocity: walk the window for 10% and 90% crossing times
        t10 = t90 = None
        direction = 1 if delta > 0 else -1
        for k in range(left, i + 1):
            progress = (logit(minutes.px[k]) - base) * direction
            if t10 is None and progress >= 0.1 * abs(delta):
                t10 = minutes.ts[k]
            if t90 is None and progress >= 0.9 * abs(delta):
                t90 = minutes.ts[k]
                break
        velocity = (
            (t90 - t10).total_seconds() / 60.0 if t10 is not None and t90 is not None else None
        )
        jumps.append((now, delta, velocity))
        last_signal = now
    return jumps


def simulate_market(
    market_id: str,
    category: str,
    minute_series: list[tuple[datetime, float]],
    hourly_series: list[tuple[datetime, float]],
    close_time: datetime | None,
) -> list[SimTrade]:
    if len(minute_series) < 120:
        return []
    minutes = MinuteSeries(minute_series)
    sigma = hourly_sigma(hourly_series)
    venue = market_id.split(":", 1)[0]
    trades: list[SimTrade] = []

    for signal_ts, jump_delta, velocity in detect_minute_jumps(minutes, sigma):
        entry = minutes.first_at_or_after(signal_ts + timedelta(minutes=ENTRY_DELAY_MIN))
        if entry is None:
            continue
        entry_ts, entry_price = entry
        if not (PRICE_LO <= entry_price <= PRICE_HI):
            continue
        if close_time is not None and close_time - entry_ts < timedelta(hours=72):
            continue

        direction = -1 if jump_delta > 0 else 1  # fade
        entry_logit = logit(entry_price)
        # Target: retrace RETRACE_TARGET of the jump from the entry level.
        target_logit = entry_logit + direction * RETRACE_TARGET * abs(jump_delta)
        stop_logit = entry_logit - direction * CONTINUATION_STOP_LOGIT

        exit_ts, exit_price, reason = None, None, "data_end"
        deadline = entry_ts + timedelta(hours=TIME_STOP_H)
        i = bisect_left(minutes.ts, entry_ts + timedelta(minutes=1))
        while i < len(minutes.ts) and minutes.ts[i] <= deadline:
            price_logit = logit(minutes.px[i])
            if (price_logit - target_logit) * direction >= 0:
                exit_ts, exit_price, reason = minutes.ts[i], minutes.px[i], "target"
                break
            if (price_logit - stop_logit) * direction <= 0:
                exit_ts, exit_price, reason = minutes.ts[i], minutes.px[i], "stop"
                break
            i += 1
        if exit_ts is None:
            last = minutes.at(deadline, max_age_min=180)
            if last is None:
                continue  # cannot price the exit; not a measurable trade
            exit_ts, exit_price, reason = deadline, last, "time"

        trades.append(
            SimTrade(
                market_id=market_id,
                category=category,
                venue=venue,
                signal_ts=signal_ts,
                jump_logit=jump_delta,
                jump_velocity_min=velocity,
                entry_ts=entry_ts,
                entry_price=entry_price,
                exit_ts=exit_ts,
                exit_price=exit_price,
                exit_reason=reason,
                direction=direction,
            )
        )
    return trades


def summarize_trades(trades: list[SimTrade], key) -> dict:
    groups: dict[str, list[SimTrade]] = {}
    for trade in trades:
        groups.setdefault(key(trade), []).append(trade)
    out: dict[str, dict] = {}
    for name, group in sorted(groups.items()):
        gross = [t.gross_pnl for t in group]
        row = {
            "n": len(group),
            "win_pct": round(100.0 * sum(1 for g in gross if g > 0) / len(gross), 1),
            "mean_gross": round(statistics.mean(gross), 4),
            "median_gross": round(statistics.median(gross), 4),
            "stop_pct": round(
                100.0 * sum(1 for t in group if t.exit_reason == "stop") / len(group), 1
            ),
            "target_pct": round(
                100.0 * sum(1 for t in group if t.exit_reason == "target") / len(group), 1
            ),
            "mean_hold_h": round(statistics.mean(t.holding_h for t in group), 1),
        }
        for cost in ROUND_TRIP_COSTS:
            net = [t.net_pnl(cost) for t in group]
            row[f"mean_net_{int(cost * 100)}c"] = round(statistics.mean(net), 4)
        out[name] = row
    return out
