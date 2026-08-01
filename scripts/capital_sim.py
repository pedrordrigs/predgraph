"""What would $100 have made, on the data we actually pooled?

Per-trade edge in cents is not a return. This converts both mechanisms into a
capital-constrained equity curve: what you must post to hold the position, how
many positions run at once, when capital frees up, and what the venue takes.

Two corrections that change the answer materially:

* Capital, not price points. Fading an up-spike means buying NO at (1 - yes),
  so a 3c gain on a share that cost 36c is an 8% return, not "3 cents". The
  headline per-trade numbers understate returns and overstate capacity.
* Kalshi charges 0.07 x P x (1-P) per contract on execution, which at a 50c
  price is 1.75c -- larger than the entire fade edge. Polymarket charges no
  trading fee. Any strategy with a Kalshi leg must clear that bar.

Twin arbitrage is modelled as the locked trade it actually is: buy YES on the
cheap venue and NO on the expensive one, pay (1 - spread), collect exactly 1 at
settlement whichever way it resolves. Profit is the spread minus costs, and the
only true risk is the two contracts resolving differently.
"""

from __future__ import annotations

import json
import logging
import pathlib
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime

logging.basicConfig(level=logging.WARNING)

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BANKROLL = 100.0
MAX_FRACTION_PER_TRADE = 0.20  # fade: at most 20% of bankroll in one episode
MAX_CONCURRENT = 5
TWIN_MAX_FRACTION = 0.50  # only a handful of verified pairs exist
# Round-trip execution cost in price points, applied to mid-based fills.
FADE_COST_SCENARIOS = (0.01, 0.02, 0.03)
# Per-leg half-spread paid when crossing on each venue.
TWIN_HALF_SPREAD = (0.005, 0.010, 0.015)
TWIN_ENTRY_SPREAD = 0.04
TWIN_EXIT_SPREAD = 0.01


def kalshi_fee(price: float, contracts: float) -> float:
    """Kalshi execution fee: 0.07 * P * (1-P) per contract."""
    return 0.07 * price * (1.0 - price) * contracts


@dataclass
class Position:
    opened: datetime
    closes: datetime
    capital: float
    payoff: float  # capital returned when the position closes


def run_equity(events: list[tuple[datetime, datetime, float, float]], bankroll: float,
               max_fraction: float, max_concurrent: int) -> dict:
    """Event-driven equity curve.

    events: (entry_ts, exit_ts, cost_per_unit, pnl_per_unit) sorted by entry.
    Capital is committed at entry and released at exit, so a trade is skipped
    when the bankroll is already deployed -- which is the real constraint a
    small account hits, and the reason per-trade means overstate what $100 does.
    """
    equity = bankroll
    free = bankroll
    open_positions: list[Position] = []
    taken = skipped = 0
    curve: list[tuple[datetime, float]] = []
    peak = equity
    max_dd = 0.0

    for entry_ts, exit_ts, cost_per_unit, pnl_per_unit in events:
        # Release anything that closed before this entry.
        still_open = []
        for position in sorted(open_positions, key=lambda p: p.closes):
            if position.closes <= entry_ts:
                free += position.capital + position.payoff
                equity += position.payoff
                curve.append((position.closes, equity))
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)
            else:
                still_open.append(position)
        open_positions = still_open

        if len(open_positions) >= max_concurrent or cost_per_unit <= 0:
            skipped += 1
            continue
        allocation = min(free, equity * max_fraction)
        if allocation < 1.0:
            skipped += 1
            continue

        units = allocation / cost_per_unit
        free -= allocation
        open_positions.append(
            Position(entry_ts, exit_ts, allocation, units * pnl_per_unit)
        )
        taken += 1

    for position in sorted(open_positions, key=lambda p: p.closes):
        free += position.capital + position.payoff
        equity += position.payoff
        curve.append((position.closes, equity))
        peak = max(peak, equity)
        max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0.0)

    return {
        "final_equity": round(equity, 2),
        "return_pct": round(100.0 * (equity / bankroll - 1.0), 1),
        "taken": taken,
        "skipped": skipped,
        "max_drawdown_pct": round(100.0 * max_dd, 1),
        "curve": curve,
    }


# --------------------------------------------------------------------------
# Mechanism 1: fade
# --------------------------------------------------------------------------

def fade_events(cost: float, spike_and_big_only: bool):
    import importlib.util

    from predgraph.backtest import history
    from predgraph.backtest.fade_sim import simulate_market

    spec = importlib.util.spec_from_file_location("rfs", ROOT / "scripts" / "run_fade_sim.py")
    rfs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rfs)

    universe_path = ROOT / "data" / "fade_universe.json"
    universe = json.loads(universe_path.read_text(encoding="utf-8")) if universe_path.exists() else {}
    categories = {m: i["category"] for m, i in universe.items()}
    categories.update(rfs.cohort_market_categories())
    closes = rfs.close_times(rfs.minute_market_ids())
    for market_id, info in universe.items():
        if info.get("close_time"):
            closes[market_id] = datetime.fromisoformat(info["close_time"])

    trades = []
    for market_id in rfs.minute_market_ids():
        trades.extend(
            simulate_market(
                market_id,
                categories.get(market_id, "uncat"),
                history.load_series(market_id, 1),
                history.load_series(market_id, 60),
                closes.get(market_id),
            )
        )

    if spike_and_big_only:
        trades = [
            t
            for t in trades
            if t.jump_velocity_min is not None
            and t.jump_velocity_min <= 5
            and abs(t.jump_logit) >= 0.5
        ]

    events = []
    for trade in sorted(trades, key=lambda t: t.entry_ts):
        # Capital posted: fading an up-move means buying NO at (1 - yes).
        cost_per_unit = (
            1.0 - trade.entry_price if trade.direction < 0 else trade.entry_price
        )
        if cost_per_unit <= 0.02:
            continue
        pnl = trade.net_pnl(cost)
        if trade.venue == "kalshi":
            # Fee charged on execution, both legs of the round trip.
            pnl -= kalshi_fee(trade.entry_price, 1.0) + kalshi_fee(trade.exit_price, 1.0)
        events.append((trade.entry_ts, trade.exit_ts, cost_per_unit, pnl))
    return events, trades


# --------------------------------------------------------------------------
# Mechanism 2: cross-venue twin arbitrage
# --------------------------------------------------------------------------

def twin_events(half_spread: float, hold_to_settlement: bool):
    import sqlalchemy as sa

    from predgraph.backtest import history
    from predgraph.backtest.lag_study import load_twins
    from predgraph.db import get_engine
    from predgraph.db import markets as markets_t

    with get_engine().connect() as conn:
        closes = {
            row.id: row.close_time
            for row in conn.execute(sa.select(markets_t.c.id, markets_t.c.close_time))
        }

    events = []
    episodes = []
    for pair in load_twins():
        a_series = history.load_series(pair["a"], 60)
        b_series = history.load_series(pair["b"], 60)
        if len(a_series) < 200 or len(b_series) < 200:
            continue
        grid_a = {t.replace(minute=0, second=0, microsecond=0): p for t, p in a_series}
        grid_b = {t.replace(minute=0, second=0, microsecond=0): p for t, p in b_series}
        common = sorted(set(grid_a) & set(grid_b))
        if len(common) < 100:
            continue

        settle = closes.get(pair["b"]) or closes.get(pair["a"])
        in_position = False
        entry_ts = entry_spread = None
        for ts in common:
            spread = grid_a[ts] - grid_b[ts]  # poly minus kalshi
            if not in_position and abs(spread) >= TWIN_ENTRY_SPREAD:
                in_position = True
                entry_ts, entry_spread = ts, abs(spread)
            elif in_position and abs(spread) <= TWIN_EXIT_SPREAD:
                exit_ts = ts
                cheap_price = min(grid_a[entry_ts], grid_b[entry_ts])
                # Cost of the locked pair: YES cheap + NO expensive, plus a
                # half-spread crossed on each leg.
                cost = 1.0 - entry_spread + 2 * half_spread
                if hold_to_settlement:
                    gross = entry_spread - 2 * half_spread
                    close_ts = settle or exit_ts
                else:
                    gross = (entry_spread - abs(spread)) - 4 * half_spread
                    close_ts = exit_ts
                # One leg is on Kalshi: fee on entry, and again on exit if we
                # unwind rather than let it settle.
                kalshi_price = grid_b[entry_ts]
                fee = kalshi_fee(kalshi_price, 1.0)
                if not hold_to_settlement:
                    fee += kalshi_fee(grid_b[exit_ts], 1.0)
                pnl = gross - fee
                events.append((entry_ts, close_ts, cost, pnl))
                episodes.append(
                    {
                        "pair": pair["note"][:40],
                        "entry_spread": round(entry_spread, 3),
                        "hours": round((close_ts - entry_ts).total_seconds() / 3600, 1),
                        "pnl_per_unit": round(pnl, 4),
                        "cheap": round(cheap_price, 3),
                    }
                )
                in_position = False
    events.sort(key=lambda e: e[0])
    return events, episodes


def main() -> None:
    print("=" * 78)
    print("$100 CAPITAL SIMULATION — pooled historical data")
    print("=" * 78)

    print("\n### MECHANISM 1: FADE (spike <=5min AND >=0.5 logit)\n")
    for cost in FADE_COST_SCENARIOS:
        events, trades = fade_events(cost, spike_and_big_only=True)
        if not events:
            continue
        result = run_equity(events, BANKROLL, MAX_FRACTION_PER_TRADE, MAX_CONCURRENT)
        span_days = (
            (max(e[1] for e in events) - min(e[0] for e in events)).days if events else 0
        )
        mean_cost = statistics.mean(e[2] for e in events)
        print(
            f"  cost {int(cost * 100)}c: ${result['final_equity']:7.2f} "
            f"({result['return_pct']:+6.1f}%) over {span_days}d | "
            f"{result['taken']} taken / {result['skipped']} skipped (capital-bound) | "
            f"maxDD {result['max_drawdown_pct']}% | avg capital/unit ${mean_cost:.2f}"
        )

    print("\n  same, WITHOUT the spike+size filter (all jumps):")
    for cost in (0.02,):
        events, _ = fade_events(cost, spike_and_big_only=False)
        result = run_equity(events, BANKROLL, MAX_FRACTION_PER_TRADE, MAX_CONCURRENT)
        print(
            f"  cost {int(cost * 100)}c: ${result['final_equity']:7.2f} "
            f"({result['return_pct']:+6.1f}%) | {result['taken']} taken"
        )

    print("\n### MECHANISM 2: CROSS-VENUE TWIN ARBITRAGE\n")
    for hold in (True, False):
        label = "hold to settlement" if hold else "unwind on convergence"
        print(f"  -- {label}")
        for half in TWIN_HALF_SPREAD:
            events, episodes = twin_events(half, hold_to_settlement=hold)
            if not events:
                print(f"     half-spread {half * 100:.1f}c: no qualifying episodes")
                continue
            result = run_equity(events, BANKROLL, TWIN_MAX_FRACTION, MAX_CONCURRENT)
            days = (max(e[1] for e in events) - min(e[0] for e in events)).days
            mean_hold = statistics.mean(ep["hours"] for ep in episodes)
            winners = sum(1 for ep in episodes if ep["pnl_per_unit"] > 0)
            print(
                f"     half-spread {half * 100:.1f}c/leg: ${result['final_equity']:7.2f} "
                f"({result['return_pct']:+6.1f}%) over {days}d | {result['taken']} taken "
                f"({winners}/{len(episodes)} episodes +) | mean hold {mean_hold:.0f}h"
            )
        if hold:
            _, episodes = twin_events(0.010, hold_to_settlement=True)
            if episodes:
                print("     episode detail (half-spread 1.0c):")
                for ep in episodes[:8]:
                    print(
                        f"       {ep['pair']:40s} spread {ep['entry_spread']:.3f} "
                        f"pnl/unit {ep['pnl_per_unit']:+.4f} hold {ep['hours']:.0f}h"
                    )


if __name__ == "__main__":
    sys.exit(main())
