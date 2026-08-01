"""Run the fade simulation across every market with minute data, then the
news-overreaction cut: do jumps confirmed by graph neighbors revert less?"""

from __future__ import annotations

import json
import logging
import pathlib
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import sqlalchemy as sa

from predgraph.backtest import history
from predgraph.backtest.fade_sim import SimTrade, simulate_market, summarize_trades
from predgraph.backtest.lag_study import DEFAULT_SOURCES, build_cohorts, ladder_keys
from predgraph.db import get_engine
from predgraph.db import history_bars as hist_t
from predgraph.db import markets as markets_t
from predgraph.signal.damage import logit

logging.basicConfig(level=logging.WARNING)

ROOT = pathlib.Path(__file__).resolve().parent.parent
UNIVERSE = ROOT / "data" / "fade_universe.json"

ANCHOR_CATEGORY = {
    "fed": "econ-macro",
    "inflation": "econ-macro",
    "unemployment": "econ-macro",
    "gdp": "econ-macro",
    "recession": "econ-macro",
    "treasury": "econ-macro",
    "tariff": "econ-macro",
    "oil": "energy",
    "opec": "energy",
    "gas": "energy",
    "iran": "geopolitics",
    "me_": "geopolitics",
    "ukraine": "geopolitics",
    "sanctions": "geopolitics",
}


def cohort_market_categories() -> dict[str, str]:
    """Category for graph-linked markets, derived from their anchor provenance."""
    engine = get_engine()
    from predgraph.db import edges as edges_t

    categories: dict[str, str] = {}
    with engine.connect() as conn:
        for row in conn.execute(
            sa.select(edges_t.c.dst, edges_t.c.provenance).where(
                edges_t.c.provenance.like("anchor:%")
            )
        ):
            anchor = row.provenance.split(":", 1)[1]
            for prefix, category in ANCHOR_CATEGORY.items():
                if anchor.startswith(prefix) or prefix in anchor:
                    categories.setdefault(row.dst, category)
                    break
    return categories


def minute_market_ids() -> list[str]:
    engine = get_engine()
    with engine.connect() as conn:
        return [
            row.market_id
            for row in conn.execute(
                sa.select(hist_t.c.market_id)
                .where(hist_t.c.resolution_min == 1)
                .group_by(hist_t.c.market_id)
                .having(sa.func.count() >= 120)
            )
        ]


def close_times(market_ids: list[str]) -> dict[str, datetime]:
    engine = get_engine()
    with engine.connect() as conn:
        return {
            row.id: row.close_time
            for row in conn.execute(
                sa.select(markets_t.c.id, markets_t.c.close_time).where(
                    markets_t.c.id.in_(market_ids)
                )
            )
            if row.close_time is not None
        }


def main() -> None:
    universe = json.loads(UNIVERSE.read_text(encoding="utf-8")) if UNIVERSE.exists() else {}
    categories = {mid: info["category"] for mid, info in universe.items()}
    categories.update(cohort_market_categories())
    closes = close_times(minute_market_ids())
    for mid, info in universe.items():
        if info.get("close_time"):
            closes[mid] = datetime.fromisoformat(info["close_time"])

    trades: list[SimTrade] = []
    markets_used = 0
    for market_id in minute_market_ids():
        category = categories.get(market_id, "uncategorized")
        minute = history.load_series(market_id, 1)
        hourly = history.load_series(market_id, 60)
        result = simulate_market(market_id, category, minute, hourly, closes.get(market_id))
        if result:
            markets_used += 1
            trades.extend(result)

    print(f"\nSIM: {len(trades)} trades across {markets_used} markets", flush=True)

    def table(title: str, summary: dict) -> None:
        print(f"\n=== {title}")
        header = f"{'group':22s} {'n':>5s} {'win%':>6s} {'gross':>8s} {'net2c':>8s} {'net3c':>8s} {'stop%':>6s} {'tgt%':>6s} {'hold_h':>7s}"
        print(header)
        for name, row in summary.items():
            print(
                f"{name[:22]:22s} {row['n']:5d} {row['win_pct']:6.1f} {row['mean_gross']:8.4f} "
                f"{row['mean_net_2c']:8.4f} {row['mean_net_3c']:8.4f} {row['stop_pct']:6.1f} "
                f"{row['target_pct']:6.1f} {row['mean_hold_h']:7.1f}"
            )

    table("ALL", summarize_trades(trades, lambda t: "all"))
    table("BY CATEGORY", summarize_trades(trades, lambda t: t.category))
    table("BY VENUE", summarize_trades(trades, lambda t: t.venue))

    def velocity_bucket(trade: SimTrade) -> str:
        v = trade.jump_velocity_min
        if v is None:
            return "unknown"
        if v <= 5:
            return "spike <=5m"
        if v <= 30:
            return "fast 5-30m"
        return "grind >30m"

    table("BY JUMP VELOCITY", summarize_trades(trades, velocity_bucket))
    table(
        "BY ENTRY PRICE",
        summarize_trades(
            trades,
            lambda t: "tail 0.1-0.3"
            if t.entry_price < 0.3
            else ("mid 0.3-0.7" if t.entry_price <= 0.7 else "tail 0.7-0.9"),
        ),
    )
    table(
        "BY JUMP SIZE",
        summarize_trades(
            trades,
            lambda t: "0.30-0.50" if abs(t.jump_logit) < 0.5 else ("0.50-1.0" if abs(t.jump_logit) < 1.0 else ">1.0"),
        ),
    )

    # --- news-overreaction cut: neighbor confirmation on cohort markets -----
    cohorts = build_cohorts(list(DEFAULT_SOURCES))
    ladder = ladder_keys()
    member_of: dict[str, list] = defaultdict(list)
    for cohort in cohorts:
        for market_id, impact in cohort.triggers.items():
            member_of[market_id].append((cohort, impact))

    cache: dict[str, list] = {}

    def minutes(mid: str) -> list:
        if mid not in cache:
            cache[mid] = history.load_series(mid, 1)
        return cache[mid]

    def confirmed_by_neighbors(trade: SimTrade) -> str:
        memberships = member_of.get(trade.market_id)
        if not memberships:
            return "no-graph"
        jump_dir = 1 if trade.jump_logit > 0 else -1
        any_checked = False
        for cohort, impact in memberships:
            implied = jump_dir * (1 if impact.contribution >= 0 else -1)
            for response_id, response_impact in cohort.responses.items():
                if ladder.get(response_id) == ladder.get(trade.market_id):
                    continue
                series = minutes(response_id)
                if not series:
                    continue
                base = after = None
                for ts, price in series:
                    if ts <= trade.signal_ts - timedelta(minutes=30):
                        base = price
                    if ts <= trade.signal_ts + timedelta(minutes=30):
                        after = price
                    else:
                        break
                if base is None or after is None:
                    continue
                any_checked = True
                delta = logit(after) - logit(base)
                expected = implied * (1 if response_impact.contribution >= 0 else -1)
                if abs(delta) >= 0.05 and (delta > 0) == (expected > 0):
                    return "confirmed"
        return "unconfirmed" if any_checked else "no-neighbor-data"

    # Concentration guard: a handful of hyperactive markets can dominate the
    # pooled mean. Per-market aggregation gives each market one vote.
    per_market: dict[str, list[float]] = defaultdict(list)
    for trade in trades:
        per_market[trade.market_id].append(trade.net_pnl(0.02))
    market_means = sorted(
        ((statistics.mean(v), len(v), k) for k, v in per_market.items()), reverse=True
    )
    positive = sum(1 for m, _, _ in market_means if m > 0)
    print(
        f"\n=== CONCENTRATION: {len(per_market)} markets; "
        f"{positive} ({100 * positive / len(per_market):.0f}%) net-positive at 2c; "
        f"equal-weight per-market mean net2c = "
        f"{statistics.mean(m for m, _, _ in market_means):+.4f}"
    )
    print("  busiest:", [(k.split(':')[1][:14], n) for _, n, k in sorted(market_means, key=lambda x: -x[1])[:5]])

    graph_trades = [t for t in trades if t.market_id in member_of]
    if graph_trades:
        table(
            "NEWS TEST: neighbor confirmation (graph-linked trades only)",
            summarize_trades(graph_trades, confirmed_by_neighbors),
        )


if __name__ == "__main__":
    sys.exit(main())
