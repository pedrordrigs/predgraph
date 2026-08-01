"""1-minute closure study: is there sub-hour diffusion the hourly study blurred?

The hourly M1 verdict was "graph-linked markets do not detectably co-move". One
escape hatch remains: if propagation completes within the same hourly bar, the
hourly study could score it as noise. This measures at minute resolution around
each trigger jump, with a placebo column (same response markets, same clock
time, 48h earlier) so "flat" and "signal" are distinguishable from noise.

Also measures cross-venue twin lead-lag at 1-minute — who moves first on the
same claim — which is a direct input for the twin monitor's alerting side.

Retention facts this design leans on (probed 2026-07-31): Polymarket serves
fidelity=1 for any window in a market's life (1441 points/day, no decay).
Kalshi 1-min candles exist only where there was activity, so coverage gates do
real work there.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from predgraph.backtest import history
from predgraph.backtest.lag_study import (
    Cohort,
    _min_hops,
    bar_counts,
    collapse_by_ladder,
    ladder_keys,
    load_twins,
)
from predgraph.signal.damage import (
    Jump,
    logit,
    move,
    to_logit_series,
)

log = logging.getLogger(__name__)

PAD_BEFORE = timedelta(hours=6)
PAD_AFTER = timedelta(hours=48)
MINUTE_HORIZONS_MIN = (5, 15, 30, 60, 120, 240)
# Tighter than the hourly staleness bound: at a 5-minute horizon a 2-hour-old
# quote is not an observation.
MINUTE_STALENESS_H = 1 / 3
PLACEBO_SHIFT = timedelta(hours=48)


@dataclass(slots=True)
class Window:
    start: datetime
    end: datetime


def merge_windows(windows: list[Window]) -> list[Window]:
    if not windows:
        return []
    ordered = sorted(windows, key=lambda w: w.start)
    merged = [ordered[0]]
    for window in ordered[1:]:
        if window.start <= merged[-1].end:
            merged[-1] = Window(merged[-1].start, max(merged[-1].end, window.end))
        else:
            merged.append(window)
    return merged


def plan_fetch(
    cohorts: list[Cohort],
    jumps_by_source: dict[str, list[tuple[str, Jump]]],
    max_responses: int = 8,
) -> dict[str, list[Window]]:
    """Markets and merged time windows the study needs at 1-minute resolution."""
    ladder = ladder_keys()
    counts = bar_counts(60)
    needed: dict[str, list[Window]] = {}

    def add(market_id: str, ts: datetime) -> None:
        needed.setdefault(market_id, []).append(
            # The placebo window is the same span 48h earlier; one widened
            # window covers both real and placebo measurements.
            Window(ts - PLACEBO_SHIFT - PAD_BEFORE, ts + PAD_AFTER)
        )

    for cohort in cohorts:
        jumps = jumps_by_source.get(cohort.source, [])
        responses = collapse_by_ladder(list(cohort.responses), ladder, counts)
        responses.sort(key=lambda r: -abs(cohort.responses[r].contribution))
        responses = responses[:max_responses]
        for trigger_id, jump in jumps:
            add(trigger_id, jump.ts)
            for response_id in responses:
                if ladder.get(response_id) != ladder.get(trigger_id):
                    add(response_id, jump.ts)

    return {market_id: merge_windows(ws) for market_id, ws in needed.items()}


def fetch_planned(plan: dict[str, list[Window]]) -> dict:
    markets = history.load_markets(list(plan))
    fetcher = history.HistoryFetcher()
    stats = {"markets": 0, "requests": 0, "bars": 0}
    try:
        for market_id, windows in plan.items():
            market = markets.get(market_id)
            if market is None:
                continue
            stats["markets"] += 1
            for window in windows:
                bars = fetcher.fetch(market, window.start, window.end, 1)
                stats["requests"] += 1
                stats["bars"] += history.store(market_id, bars, 1)
    finally:
        fetcher.close()
    return stats


def refine_jump_minute(trigger_series: list, jump: Jump) -> datetime | None:
    """Locate, inside the hourly jump bar, the minute the move actually happened."""
    window = [
        (ts, price)
        for ts, price in trigger_series
        if jump.ts <= ts <= jump.ts + timedelta(hours=1, minutes=10)
    ]
    if len(window) < 3:
        return None
    base = logit(window[0][1])
    target = abs(jump.delta_logit) * 0.5
    direction = 1 if jump.delta_logit > 0 else -1
    for ts, price in window[1:]:
        if (logit(price) - base) * direction >= target:
            return ts
    return None


@dataclass(slots=True)
class MinuteObservation:
    source: str
    response_id: str
    hops: int
    predicted_sign: int
    realized: dict[int, float] = field(default_factory=dict)
    placebo: dict[int, float] = field(default_factory=dict)


def measure(
    cohorts: list[Cohort],
    jumps_by_source: dict[str, list[tuple[str, Jump]]],
    max_responses: int = 8,
) -> list[MinuteObservation]:
    ladder = ladder_keys()
    counts = bar_counts(60)
    cache: dict[str, list] = {}

    def minutes(market_id: str) -> list:
        if market_id not in cache:
            cache[market_id] = history.load_series(market_id, 1)
        return cache[market_id]

    observations: list[MinuteObservation] = []
    refined = 0
    for cohort in cohorts:
        responses = collapse_by_ladder(list(cohort.responses), ladder, counts)
        responses.sort(key=lambda r: -abs(cohort.responses[r].contribution))
        responses = responses[:max_responses]

        for trigger_id, jump in jumps_by_source.get(cohort.source, []):
            t0 = refine_jump_minute(minutes(trigger_id), jump)
            if t0 is None:
                continue
            refined += 1
            trigger_sign = 1 if cohort.triggers[trigger_id].contribution >= 0 else -1
            implied = (1 if jump.delta_logit > 0 else -1) * trigger_sign

            for response_id in responses:
                if ladder.get(response_id) == ladder.get(trigger_id):
                    continue
                series = minutes(response_id)
                if not series:
                    continue
                impact = cohort.responses[response_id]
                observation = MinuteObservation(
                    source=cohort.source,
                    response_id=response_id,
                    hops=_min_hops(impact),
                    predicted_sign=implied * (1 if impact.contribution >= 0 else -1),
                )
                logits = to_logit_series(series)
                for horizon in MINUTE_HORIZONS_MIN:
                    real = move(logits, t0, horizon / 60.0, MINUTE_STALENESS_H)
                    if real is not None:
                        observation.realized[horizon] = real
                    fake = move(
                        logits, t0 - PLACEBO_SHIFT, horizon / 60.0, MINUTE_STALENESS_H
                    )
                    if fake is not None:
                        observation.placebo[horizon] = fake
                if observation.realized:
                    observations.append(observation)
    log.info("refined %d trigger jumps to the minute", refined)
    return observations


def summarize_minutes(observations: list[MinuteObservation]) -> dict:
    """Signed mean response and conditional hit rate vs the placebo, by horizon."""

    def stats_for(values: list[float], signs: list[int]) -> dict:
        signed = [v * s for v, s in zip(values, signs, strict=True)]
        material = [
            (v > 0) == (s > 0) for v, s in zip(values, signs, strict=True) if abs(v) >= 0.05
        ]
        return {
            "n": len(signed),
            "mean_signed": round(statistics.mean(signed), 4) if signed else None,
            "hit_material": (
                round(100.0 * sum(material) / len(material), 1) if material else None
            ),
            "n_material": len(material),
        }

    summary: dict = {}
    for horizon in MINUTE_HORIZONS_MIN:
        real_vals = [
            (o.realized[horizon], o.predicted_sign)
            for o in observations
            if horizon in o.realized
        ]
        placebo_vals = [
            (o.placebo[horizon], o.predicted_sign)
            for o in observations
            if horizon in o.placebo
        ]
        summary[horizon] = {
            "real": stats_for([v for v, _ in real_vals], [s for _, s in real_vals]),
            "placebo": stats_for(
                [v for v, _ in placebo_vals], [s for _, s in placebo_vals]
            ),
        }
    return summary


# --- twin lead-lag ----------------------------------------------------------


def twin_lead_lag(
    max_lag_min: int = 30,
    move_floor: float = 0.10,
    respond_floor: float = 0.05,
) -> list[dict]:
    """Who moves first when the same claim is listed on both venues.

    Two measurements per pair: cross-correlation of minute-level logit changes
    at shifted lags (sign of the peak lag says who leads), and an event study —
    after a material 10-minute move on one side, how long until the other side
    follows in the same direction.
    """
    results = []
    for pair in load_twins():
        series_a = history.load_series(pair["a"], 1)
        series_b = history.load_series(pair["b"], 1)
        if len(series_a) < 500 or len(series_b) < 500:
            continue

        grid_a = {ts.replace(second=0, microsecond=0): logit(p) for ts, p in series_a}
        grid_b = {ts.replace(second=0, microsecond=0): logit(p) for ts, p in series_b}

        def deltas(grid: dict) -> dict:
            out = {}
            for ts, value in grid.items():
                prev = grid.get(ts - timedelta(minutes=1))
                if prev is not None:
                    out[ts] = value - prev
            return out

        da, db = deltas(grid_a), deltas(grid_b)
        best_lag, best_corr = 0, 0.0
        for lag in range(-max_lag_min, max_lag_min + 1):
            paired = [
                (da[ts], db.get(ts + timedelta(minutes=lag)))
                for ts in da
                if db.get(ts + timedelta(minutes=lag)) is not None
            ]
            xs = [x for x, y in paired if abs(x) > 1e-9 or abs(y) > 1e-9]
            ys = [y for x, y in paired if abs(x) > 1e-9 or abs(y) > 1e-9]
            if len(xs) < 200:
                continue
            try:
                corr = statistics.correlation(xs, ys)
            except statistics.StatisticsError:
                continue
            if abs(corr) > abs(best_corr):
                best_corr, best_lag = corr, lag

        def response_times(lead: dict, follow: dict) -> list[float]:
            times = []
            stamps = sorted(lead)
            for ts in stamps:
                earlier = lead.get(ts - timedelta(minutes=10))
                if earlier is None:
                    continue
                jump_size = lead[ts] - earlier
                if abs(jump_size) < move_floor:
                    continue
                direction = 1 if jump_size > 0 else -1
                base = follow.get(ts)
                if base is None:
                    continue
                for minutes_ahead in range(1, 121):
                    later = follow.get(ts + timedelta(minutes=minutes_ahead))
                    if later is not None and (later - base) * direction >= respond_floor:
                        times.append(float(minutes_ahead))
                        break
            return times

        a_leads = response_times(grid_a, grid_b)
        b_leads = response_times(grid_b, grid_a)
        results.append(
            {
                "note": pair.get("note", ""),
                "xcorr_peak_lag_min": best_lag,
                "xcorr_peak": round(best_corr, 3),
                "a_to_b_median_min": statistics.median(a_leads) if a_leads else None,
                "a_to_b_n": len(a_leads),
                "b_to_a_median_min": statistics.median(b_leads) if b_leads else None,
                "b_to_a_n": len(b_leads),
            }
        )
    return results
