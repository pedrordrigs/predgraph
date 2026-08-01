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
from pathlib import Path

import sqlalchemy as sa
import yaml

from predgraph.backtest import history
from predgraph.config import REPO_ROOT
from predgraph.db import get_engine
from predgraph.db import history_bars as hist_t
from predgraph.db import markets as markets_t
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

# A response smaller than this is a bid tick, not a repricing, and its sign is
# a coin flip. Scoring every nonzero move drags every cohort toward 50% and was
# hiding whatever real signal exists — hit rates are reported conditional on it.
MATERIAL_LOGIT = 0.10
# One CPI print moves every strike of the ladder, so an undeduped run counts the
# same event dozens of times and calls correlated observations independent.
DEDUP_BUCKET_H = 6.0
JUMP_MIN_ABS_LOGIT = 0.30
MIN_BARS = 200
MIN_COVERAGE = 0.75
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
    predicted_magnitude: float = 0.0
    realized: dict[float, float] = field(default_factory=dict)
    time_to_half_h: float | None = None
    control: bool = False

    def agrees(self, horizon: float, material_floor: float = 0.0) -> bool | None:
        delta = self.realized.get(horizon)
        if delta is None or abs(delta) < max(material_floor, 1e-9):
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


def ladder_keys() -> dict[str, str]:
    """Group markets that are strikes of the same underlying event."""
    engine = get_engine()
    keys: dict[str, str] = {}
    with engine.connect() as conn:
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


def collapse_by_ladder(
    market_ids: list[str], ladder: dict[str, str], bar_counts: dict[str, int]
) -> list[str]:
    """One representative per ladder — the strike with the most history."""
    best: dict[str, str] = {}
    for market_id in market_ids:
        key = ladder.get(market_id, market_id)
        current = best.get(key)
        if current is None or bar_counts.get(market_id, 0) > bar_counts.get(current, 0):
            best[key] = market_id
    return list(best.values())


def bar_counts(resolution_min: int = 60) -> dict[str, int]:
    engine = get_engine()
    with engine.connect() as conn:
        return {
            row.market_id: row.n
            for row in conn.execute(
                sa.select(hist_t.c.market_id, sa.func.count().label("n"))
                .where(hist_t.c.resolution_min == resolution_min)
                .group_by(hist_t.c.market_id)
            )
        }


def _dedupe_jumps(
    jumps: list[tuple[str, Jump]], ladder: dict[str, str]
) -> list[tuple[str, Jump]]:
    """Keep the strongest jump per (ladder, time bucket) so one event counts once."""
    best: dict[tuple[str, int], tuple[str, Jump]] = {}
    for market_id, jump in jumps:
        bucket = int(jump.ts.timestamp() // (DEDUP_BUCKET_H * 3600))
        key = (ladder.get(market_id, market_id), bucket)
        if key not in best or abs(jump.z) > abs(best[key][1].z):
            best[key] = (market_id, jump)
    return list(best.values())


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

    ladder = ladder_keys()
    counts = bar_counts(resolution_min)
    observations: list[Observation] = []

    for cohort in cohorts:
        # Collapse each strike ladder to its best-covered member before doing
        # anything else, on both sides. Otherwise a single Fed meeting supplies
        # a dozen triggers and a dozen responses, and every count is inflated
        # with observations that are near-copies of each other.
        trigger_ids = collapse_by_ladder(
            [t for t in cohort.triggers if counts.get(t, 0) >= MIN_BARS], ladder, counts
        )
        response_ids = collapse_by_ladder(
            [r for r in cohort.responses if counts.get(r, 0) >= MIN_BARS], ladder, counts
        )[:max_responses]
        if not trigger_ids or not response_ids:
            continue

        raw_jumps = [
            (trigger_id, jump)
            for trigger_id in trigger_ids
            for jump in detect_jumps(
                series(trigger_id),
                z_threshold=z_threshold,
                min_abs_logit=JUMP_MIN_ABS_LOGIT,
            )
        ]

        for trigger_id, jump in _dedupe_jumps(raw_jumps, ladder):
            trigger_sign = 1 if cohort.triggers[trigger_id].contribution >= 0 else -1
            implied_source_direction = (1 if jump.delta_logit > 0 else -1) * trigger_sign

            for response_id in response_ids:
                # A market in the trigger's own ladder is the same claim, not a
                # multi-hop response.
                if ladder.get(response_id) == ladder.get(trigger_id):
                    continue
                if coverage(series(response_id), jump.ts, 4.0) < MIN_COVERAGE:
                    continue
                impact = cohort.responses[response_id]
                observations.append(
                    _measure(
                        cohort.source,
                        trigger_id,
                        response_id,
                        _min_hops(impact),
                        jump,
                        implied_source_direction * (1 if impact.contribution >= 0 else -1),
                        series(response_id),
                        magnitude=abs(impact.contribution),
                    )
                )

            for control_id in rng.sample(control_pool or [], min(len(control_pool or []), 6)):
                if control_id in cohort.responses or control_id in cohort.triggers:
                    continue
                if coverage(series(control_id), jump.ts, 4.0) < MIN_COVERAGE:
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
    magnitude: float = 0.0,
) -> Observation:
    observation = Observation(
        source=source,
        trigger_id=trigger_id,
        response_id=response_id,
        hops=hops,
        jump_ts=jump.ts,
        predicted_sign=predicted_sign,
        predicted_magnitude=magnitude,
        control=control,
    )
    if not response_series:
        return observation
    logits = to_logit_series(response_series)
    for horizon in HORIZONS_H:
        delta = move(logits, jump.ts, horizon)
        if delta is not None:
            observation.realized[horizon] = delta
    # Time-to-half is only meaningful if the market actually went somewhere;
    # on a non-move it measures how long noise took to cross a noise threshold.
    eventual = observation.realized.get(48.0)
    if eventual is not None and abs(eventual) >= MATERIAL_LOGIT:
        observation.time_to_half_h = time_to_fraction(response_series, jump.ts, 48.0)
    return observation


def load_twins(path: Path | None = None) -> list[dict]:
    """Hand-mapped cross-venue pairs that are the same real-world claim."""
    file = path or (REPO_ROOT / "twins.yaml")
    if not file.exists():
        return []
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return data.get("pairs", [])


def twin_control(
    resolution_min: int = 60,
    horizon_h: float = 4.0,
    z_threshold: float = 3.0,
    material_floor: float = MATERIAL_LOGIT,
    pairs: list[dict] | None = None,
) -> dict:
    """The honest instrument check: do two listings of the same claim agree?

    Same-ladder agreement has a structural ceiling — news that sharpens the
    distribution moves "above 3.75%" and "above 4.25%" in opposite directions,
    so those markets can legitimately disagree. Two venues listing the *same*
    question cannot. If this does not read high on material moves, the pipeline
    is broken; if it does, a weak n-hop result is a fact about the world.
    """
    pairs = pairs if pairs is not None else load_twins()
    if not pairs:
        return {}

    cache: dict[str, list] = {}

    def series(market_id: str):
        if market_id not in cache:
            cache[market_id] = history.load_series(market_id, resolution_min)
        return cache[market_id]

    hits: list[bool] = []
    used: set[str] = set()
    jump_count = 0
    for pair in pairs:
        a, b = pair.get("a"), pair.get("b")
        if not a or not b:
            continue
        # opposite: the twin states the same claim inverted (e.g. "no change" vs "cut")
        polarity = -1 if pair.get("opposite") else 1
        for trigger, responder in ((a, b), (b, a)):
            trigger_series, responder_series = series(trigger), series(responder)
            if len(trigger_series) < MIN_BARS or len(responder_series) < MIN_BARS:
                continue
            for jump in detect_jumps(
                trigger_series, z_threshold=z_threshold, min_abs_logit=JUMP_MIN_ABS_LOGIT
            ):
                if coverage(responder_series, jump.ts, horizon_h) < MIN_COVERAGE:
                    continue
                delta = move(to_logit_series(responder_series), jump.ts, horizon_h)
                if delta is None or abs(delta) < material_floor:
                    continue
                predicted = (1 if jump.delta_logit > 0 else -1) * polarity
                hits.append((delta > 0) == (predicted > 0))
                used.update((trigger, responder))
                jump_count += 1

    if not hits:
        return {"agreement_pct": None, "n": 0, "markets": 0, "jumps": 0}
    return {
        "agreement_pct": round(100.0 * sum(hits) / len(hits), 1),
        "n": len(hits),
        "markets": len(used),
        "jumps": jump_count,
    }


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

    ladder = ladder_keys()
    results = {}
    for cohort in cohorts:
        triggers = [t for t in cohort.triggers if len(series(t)) >= min_bars][:limit]
        by_tier: dict[str, list[bool]] = {"same_ladder": [], "same_driver": []}
        raw = [
            (t, j)
            for t in triggers
            for j in detect_jumps(
                series(t), z_threshold=z_threshold, min_abs_logit=JUMP_MIN_ABS_LOGIT
            )
        ]
        for source_market, jump in _dedupe_jumps(raw, ladder):
            implied = (1 if jump.delta_logit > 0 else -1) * (
                1 if cohort.triggers[source_market].contribution >= 0 else -1
            )
            for peer in triggers:
                if peer == source_market:
                    continue
                if coverage(series(peer), jump.ts, horizon_h) < min_coverage:
                    continue
                delta = move(to_logit_series(series(peer)), jump.ts, horizon_h)
                # Conditional on materiality: a one-tick move has a random sign
                # and only dilutes whatever agreement is really there.
                if delta is None or abs(delta) < MATERIAL_LOGIT:
                    continue
                predicted = implied * (1 if cohort.triggers[peer].contribution >= 0 else -1)
                tier = (
                    "same_ladder"
                    if ladder.get(peer) == ladder.get(source_market)
                    else "same_driver"
                )
                by_tier[tier].append((delta > 0) == (predicted > 0))

        for tier, hits in by_tier.items():
            if hits:
                results[f"{cohort.source} [{tier}]"] = {
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
            # Unconditional hit rate is reported for continuity, but the
            # conditional one is the number that means something.
            any_hits = [v for v in (o.agrees(horizon) for o in group) if v is not None]
            material_hits = [
                v
                for v in (o.agrees(horizon, MATERIAL_LOGIT) for o in group)
                if v is not None
            ]
            row[f"hit_{horizon:g}h"] = (
                round(100.0 * sum(any_hits) / len(any_hits), 1) if any_hits else None
            )
            row[f"mhit_{horizon:g}h"] = (
                round(100.0 * sum(material_hits) / len(material_hits), 1)
                if material_hits
                else None
            )
            row[f"mn_{horizon:g}h"] = len(material_hits)
            moves = [abs(o.realized[horizon]) for o in group if horizon in o.realized]
            row[f"absmove_{horizon:g}h"] = round(statistics.median(moves), 3) if moves else None

        halves = [o.time_to_half_h for o in group if o.time_to_half_h is not None]
        row["median_time_to_half_h"] = round(statistics.median(halves), 1) if halves else None
        row["n_with_half"] = len(halves)

        # Does a stronger predicted effect produce a bigger realized move? Sign
        # agreement can pass on noise; this cannot.
        pairs = [
            (o.predicted_sign * o.predicted_magnitude, o.realized[4.0])
            for o in group
            if 4.0 in o.realized and o.predicted_magnitude > 0
        ]
        row["magnitude_corr_4h"] = (
            round(statistics.correlation(*zip(*pairs, strict=True)), 3)
            if len(pairs) >= 30
            else None
        )
        summary[key] = row
    return summary
