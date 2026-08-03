"""Run the fade simulation across every market with minute data."""

from __future__ import annotations

import json
import logging
import pathlib
import sys
from datetime import datetime

import sqlalchemy as sa

from snapback.backtest import history
from snapback.backtest.fade_sim import SimTrade, simulate_market, summarize_trades
from snapback.db import get_engine
from snapback.db import history_bars as hist_t
from snapback.db import markets as markets_t

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
    from snapback.db import edges as edges_t

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

    print("\n(neighbour-confirmation test removed: it showed no separation)")


if __name__ == "__main__":
    sys.exit(main())
