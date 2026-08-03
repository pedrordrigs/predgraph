"""Price maths, in logit space.

A move from 0.50 to 0.55 and one from 0.90 to 0.95 are the same five points of
probability but wildly different amounts of information, and averaging them in
raw probability quietly makes every statistic wrong. Everything downstream -
jump size, sigma, targets, stops - is expressed in logits for that reason.

The backtest and the live engine share these functions, so a threshold measured
in a study and one applied in production cannot drift apart.
"""

from __future__ import annotations

import bisect
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta

EPSILON = 1e-4
# Venues only emit a bar when something happened, so a quote older than this is
# treated as no observation at all rather than carried forward.
MAX_STALENESS_H = 2.0
Series = list[tuple[datetime, float]]


def logit(p: float, epsilon: float = EPSILON) -> float:
    clipped = min(max(p, epsilon), 1.0 - epsilon)
    return math.log(clipped / (1.0 - clipped))


def to_logit_series(series: Series) -> Series:
    return [(ts, logit(p)) for ts, p in series]


def value_at(
    series: Series, when: datetime, max_staleness_h: float | None = MAX_STALENESS_H
) -> float | None:
    """Most recent sample at or before `when`, if one is recent enough to trust.

    Markets are step functions, so carrying the last quote forward is right in
    principle — but venues only emit a bar when something happened, and these
    series have gaps of hundreds of hours. Without a staleness bound, a "4-hour
    move" silently becomes a comparison between quotes weeks apart, which is
    pure noise wearing the costume of a measurement.
    """
    if not series:
        return None
    index = bisect.bisect_right([ts for ts, _ in series], when)
    if index == 0:
        return None
    ts, value = series[index - 1]
    if max_staleness_h is not None and (when - ts) > timedelta(hours=max_staleness_h):
        return None
    return value


def move(
    series: Series,
    start: datetime,
    hours: float,
    max_staleness_h: float | None = MAX_STALENESS_H,
) -> float | None:
    """Signed change in logit over a window; None unless both ends are observed."""
    before = value_at(series, start, max_staleness_h)
    after = value_at(series, start + timedelta(hours=hours), max_staleness_h)
    if before is None or after is None:
        return None
    return after - before


def coverage(series: Series, start: datetime, hours: float, step_h: float = 1.0) -> float:
    """Fraction of hourly slots in a window that have a non-stale quote."""
    if hours <= 0:
        return 0.0
    slots = int(hours / step_h)
    if slots <= 0:
        return 0.0
    present = sum(
        1
        for i in range(slots)
        if value_at(series, start + timedelta(hours=i * step_h), step_h) is not None
    )
    return present / slots


def baseline_sigma(series: Series, window_h: float, min_samples: int = 8) -> float | None:
    """Typical size of a `window_h` move for this market, as its own yardstick.

    A market that normally drifts 0.3 logits an hour and one that never moves
    cannot share a threshold, so every move is scored against this.
    """
    if len(series) < min_samples + 1:
        return None
    step = timedelta(hours=window_h)
    changes: list[float] = []
    cursor = series[0][0]
    end = series[-1][0]
    while cursor + step <= end:
        delta = move(series, cursor, window_h)
        if delta is not None:
            changes.append(delta)
        cursor += step
    if len(changes) < min_samples:
        return None
    sigma = statistics.pstdev(changes)
    return sigma if sigma > 1e-6 else None


@dataclass(slots=True)
class Jump:
    ts: datetime
    delta_logit: float
    z: float
    price_before: float
    price_after: float


def detect_jumps(
    price_series: Series,
    window_h: float = 1.0,
    z_threshold: float = 3.0,
    min_abs_logit: float = 0.20,
    cooldown_h: float = 12.0,
) -> list[Jump]:
    """Find abrupt repricings — our proxy for "news landed here".

    Requires both a z-score and an absolute floor: a very quiet market has a
    tiny sigma, and without the floor every ripple looks like a 10-sigma event.
    A cooldown keeps one event from being counted once per sample as it unfolds.
    """
    if len(price_series) < 10:
        return []
    logits = to_logit_series(price_series)
    sigma = baseline_sigma(logits, window_h)
    if sigma is None:
        return []

    jumps: list[Jump] = []
    last_ts: datetime | None = None
    for ts, _ in logits:
        delta = move(logits, ts, window_h)
        if delta is None:
            continue
        if abs(delta) < min_abs_logit or abs(delta) < z_threshold * sigma:
            continue
        if last_ts is not None and ts - last_ts < timedelta(hours=cooldown_h):
            # Keep the larger move within the same episode.
            if abs(delta) <= abs(jumps[-1].delta_logit):
                continue
            jumps.pop()
        before = value_at(price_series, ts)
        after = value_at(price_series, ts + timedelta(hours=window_h))
        if before is None or after is None:
            continue
        jumps.append(
            Jump(ts=ts, delta_logit=delta, z=delta / sigma, price_before=before, price_after=after)
        )
        last_ts = ts
    return jumps
