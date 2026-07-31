"""M1: does an n-hop market reprice late enough to be tradeable?

Method, and why it needs no news feed. A 1-hop market repricing abruptly *is*
the observable arrival of news at that node — so we use it as the event clock.
For each such jump we ask what markets further out on the graph did next, and
whether they moved in the direction the signed path predicts.

Three ways this could flatter itself, and the guards:

* Everything drifts together on a risk-on day, so connected markets are scored
  against a control cohort with no path from the same source. A hit rate that
  does not beat control is market beta, not propagation.
* A market that has already moved before the trigger is not an opportunity, so
  responses are measured strictly forward from the jump.
* Direction is predicted from the composed path sign, fixed before the outcome
  is looked at — not fitted afterwards.
"""

from __future__ import annotations

import logging
import random
import statistics
from dataclasses import dataclass, field
from datetime import datetime

from predgraph.backtest import history
from predgraph.graph.algo import Impact, load_adjacency, propagate
from predgraph.signal.damage import (
    Jump,
    coverage,
    detect_jumps,
    move,
    time_to_fraction,
    to_logit_series,
)

log = logging.getLogger(__name__)

HORIZONS_H = (1.0, 2.0, 4.0, 8.0, 24.0, 48.0)
DEFAULT_SOURCES = (
    "military_escalation_me",
    "us_iran_tension",
    "russia_ukraine_escalation",
    "oil_supply_risk",
    "brent_price",
    "inflation_pressure_us",
    "fed_policy_dovishness",
)


def _is_market(node_id: str) -> bool:
    return node_id.startswith(("poly:", "kalshi:"))


@dataclass(slots=True)
class Observation:
    source: str
    trigger_id: str
    response_id: str
    hops: int
    jump_ts: datetime
    predicted_sign: int
    realized: dict[float, float] = field(default_factory=dict)
    time_to_half_h: float | None = None
    control: bool = False

    def agrees(self, horizon: float) -> bool | None:
        delta = self.realized.get(horizon)
        if delta is None or abs(delta) < 1e-9:
            return None
        return (delta > 0) == (self.predicted_sign > 0)


@dataclass
class Cohort:
    source: str
    triggers: dict[str, Impact]
    responses: dict[str, Impact]


def build_cohorts(sources: list[str]) -> list[Cohort]:
    """Split each source's reachable markets into 1-hop triggers and n-hop responses."""
    adjacency = load_adjacency()
    cohorts: list[Cohort] = []
    for source in sources:
        impacts = propagate(source, adjacency=adjacency)
        triggers: dict[str, Impact] = {}
        responses: dict[str, Impact] = {}
        for impact in impacts:
            if not _is_market(impact.target) or not impact.paths:
                continue
            hops = min(path.hops for path in impact.paths)
            if hops <= 1:
                triggers[impact.target] = impact
            else:
                responses[impact.target] = impact
        if triggers and responses:
            cohorts.append(Cohort(source=source, triggers=triggers, responses=responses))
            log.info(
                "%s: %d trigger / %d response markets", source, len(triggers), len(responses)
            )
    return cohorts


def _min_hops(impact: Impact) -> int:
    return min(path.hops for path in impact.paths)


def observe(
    cohorts: list[Cohort],
    resolution_min: int = 60,
    z_threshold: float = 3.0,
    max_responses: int = 40,
    control_pool: list[str] | None = None,
    seed: int = 7,
) -> list[Observation]:
    rng = random.Random(seed)
    series_cache: dict[str, list] = {}

    def series(market_id: str):
        if market_id not in series_cache:
            series_cache[market_id] = history.load_series(market_id, resolution_min)
        return series_cache[market_id]

    observations: list[Observation] = []
    for cohort in cohorts:
        response_ids = list(cohort.responses)[:max_responses]
        for trigger_id, trigger_impact in cohort.triggers.items():
            trigger_series = series(trigger_id)
            if len(trigger_series) < 24:
                continue
            jumps = detect_jumps(trigger_series, z_threshold=z_threshold)
            if not jumps:
                continue
            # Sign of the source move implied by the trigger's own move.
            trigger_sign = 1 if trigger_impact.contribution >= 0 else -1

            for jump in jumps:
                implied_source_direction = (1 if jump.delta_logit > 0 else -1) * trigger_sign
                for response_id in response_ids:
                    if response_id == trigger_id:
                        continue
                    observations.append(
                        _measure(
                            cohort.source,
                            trigger_id,
                            response_id,
                            _min_hops(cohort.responses[response_id]),
                            jump,
                            implied_source_direction
                            * (1 if cohort.responses[response_id].contribution >= 0 else -1),
                            series(response_id),
                        )
                    )
                for control_id in rng.sample(
                    control_pool or [], min(len(control_pool or []), 6)
                ):
                    if control_id in cohort.responses or control_id in cohort.triggers:
                        continue
                    observations.append(
                        _measure(
                            cohort.source,
                            trigger_id,
                            control_id,
                            -1,
                            jump,
                            rng.choice((1, -1)),
                            series(control_id),
                            control=True,
                        )
                    )
    return [o for o in observations if o is not None and o.realized]


def _measure(
    source: str,
    trigger_id: str,
    response_id: str,
    hops: int,
    jump: Jump,
    predicted_sign: int,
    response_series: list,
    control: bool = False,
) -> Observation:
    observation = Observation(
        source=source,
        trigger_id=trigger_id,
        response_id=response_id,
        hops=hops,
        jump_ts=jump.ts,
        predicted_sign=predicted_sign,
        control=control,
    )
    if not response_series:
        return observation
    logits = to_logit_series(response_series)
    for horizon in HORIZONS_H:
        delta = move(logits, jump.ts, horizon)
        if delta is not None:
            observation.realized[horizon] = delta
    observation.time_to_half_h = time_to_fraction(response_series, jump.ts, 48.0)
    return observation


def positive_control(
    cohorts: list[Cohort],
    resolution_min: int = 60,
    horizon_h: float = 4.0,
    z_threshold: float = 3.0,
    min_bars: int = 200,
    min_coverage: float = 0.75,
    limit: int = 60,
) -> dict:
    """Can the instrument detect a relationship that must exist?

    Two strikes of the same CPI ladder, or two outcomes of the same Fed meeting,
    are mechanically linked: when one reprices on news the other has to move
    too. So agreement between 1-hop markets on the same driver is a floor the
    method must clear. If this reads near 50%, the measurement is noise and no
    hop-level number from the same pipeline means anything — report this before
    reporting any verdict.
    """
    cache: dict[str, list] = {}

    def series(market_id: str):
        if market_id not in cache:
            cache[market_id] = history.load_series(market_id, resolution_min)
        return cache[market_id]

    results = {}
    for cohort in cohorts:
        triggers = [t for t in cohort.triggers if len(series(t)) >= min_bars][:limit]
        hits: list[bool] = []
        for source_market in triggers:
            for jump in detect_jumps(series(source_market), z_threshold=z_threshold):
                implied = (1 if jump.delta_logit > 0 else -1) * (
                    1 if cohort.triggers[source_market].contribution >= 0 else -1
                )
                for peer in triggers:
                    if peer == source_market:
                        continue
                    if coverage(series(peer), jump.ts, horizon_h) < min_coverage:
                        continue
                    delta = move(to_logit_series(series(peer)), jump.ts, horizon_h)
                    if delta is None or abs(delta) < 1e-9:
                        continue
                    predicted = implied * (
                        1 if cohort.triggers[peer].contribution >= 0 else -1
                    )
                    hits.append((delta > 0) == (predicted > 0))
        if hits:
            results[cohort.source] = {
                "agreement_pct": round(100.0 * sum(hits) / len(hits), 1),
                "n": len(hits),
                "markets": len(triggers),
            }
    return results


def summarize(observations: list[Observation]) -> dict:
    """Hit rate and move size by hop count, against the control cohort."""
    groups: dict[str, list[Observation]] = {}
    for observation in observations:
        key = "control" if observation.control else f"hop {observation.hops}"
        groups.setdefault(key, []).append(observation)

    summary = {}
    for key, group in sorted(groups.items()):
        row = {"n": len(group)}
        for horizon in HORIZONS_H:
            verdicts = [o.agrees(horizon) for o in group]
            hits = [v for v in verdicts if v is not None]
            row[f"hit_{horizon:g}h"] = (
                round(100.0 * sum(hits) / len(hits), 1) if hits else None
            )
            moves = [abs(o.realized[horizon]) for o in group if horizon in o.realized]
            row[f"absmove_{horizon:g}h"] = round(statistics.median(moves), 3) if moves else None
        halves = [o.time_to_half_h for o in group if o.time_to_half_h is not None]
        row["median_time_to_half_h"] = round(statistics.median(halves), 1) if halves else None
        row["n_with_half"] = len(halves)
        summary[key] = row
    return summary
