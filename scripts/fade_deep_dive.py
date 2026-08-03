"""Fade mechanism: how long to hold, fade or follow, and does news help."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import statistics
import sys
from collections import defaultdict
from datetime import datetime

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from snapback.backtest import history
from snapback.backtest.fade_research import (
    FORWARD_MIN,
    Series,
    add_breadth,
    extract_spikes,
    simulate_exit_rule,
)

# Cost expressed on capital: a 2c round trip on a ~48c posted position.
COST_ON_CAPITAL = 0.042


def load_all():
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

    spikes, series_by_market = [], {}
    for market_id in rfs.minute_market_ids():
        minute = history.load_series(market_id, 1)
        hourly = history.load_series(market_id, 60)
        found = extract_spikes(
            market_id, categories.get(market_id, "uncat"), minute, hourly, closes.get(market_id)
        )
        if found:
            spikes.extend(found)
            series_by_market[market_id] = Series(minute)
    add_breadth(spikes)
    return spikes, series_by_market


def line(label: str, values: list[float], cost: float = COST_ON_CAPITAL) -> str:
    if not values:
        return f"  {label:26s}      -"
    net = [v - cost for v in values]
    mean = statistics.mean(net)
    win = 100.0 * sum(1 for v in net if v > 0) / len(net)
    # standard error, so a difference between buckets can be read as real or not
    stderr = statistics.pstdev(net) / (len(net) ** 0.5) if len(net) > 1 else 0.0
    return (
        f"  {label:26s} n={len(net):4d}  mean {mean:+7.2%}  +/-{stderr:5.2%}  "
        f"median {statistics.median(net):+7.2%}  win {win:4.1f}%"
    )


def main() -> None:
    spikes, series_by_market = load_all()
    print(f"spikes extracted: {len(spikes)} across {len(series_by_market)} markets")
    print(f"(returns below are ON CAPITAL POSTED, net of a {COST_ON_CAPITAL:.1%} cost)\n")

    print("=" * 78)
    print("1. HOW LONG TO HOLD — fade return by fixed horizon")
    print("=" * 78)
    for minutes in FORWARD_MIN:
        vals = [s.forward[minutes] for s in spikes if minutes in s.forward]
        label = f"{minutes}m" if minutes < 60 else f"{minutes // 60}h"
        print(line(f"hold {label}", vals))

    print("\n  same, split by jump speed:")
    for name, sel in (
        ("spike <=5m", lambda s: s.velocity_min <= 5),
        ("grind >5m", lambda s: s.velocity_min > 5),
    ):
        print(f"   -- {name}")
        for minutes in (15, 60, 240, 720, 1440):
            vals = [s.forward[minutes] for s in spikes if sel(s) and minutes in s.forward]
            label = f"{minutes}m" if minutes < 60 else f"{minutes // 60}h"
            print(line(f"     hold {label}", vals))

    print("\n" + "=" * 78)
    print("2. FADE OR FOLLOW — sign of the edge by spike type")
    print("=" * 78)
    print("  (positive = fading works; negative = the move continued, follow it)")
    for name, sel in (
        ("spike <=5m, big >=0.5", lambda s: s.velocity_min <= 5 and abs(s.jump_logit) >= 0.5),
        ("spike <=5m, small <0.5", lambda s: s.velocity_min <= 5 and abs(s.jump_logit) < 0.5),
        ("grind >5m, big >=0.5", lambda s: s.velocity_min > 5 and abs(s.jump_logit) >= 0.5),
        ("grind >5m, small <0.5", lambda s: s.velocity_min > 5 and abs(s.jump_logit) < 0.5),
    ):
        vals = [s.forward[240] for s in spikes if sel(s) and 240 in s.forward]
        print(line(name + " @4h", vals))

    print("\n" + "=" * 78)
    print("3. EXIT RULE GRID — is the optimum a plateau or a point?")
    print("=" * 78)
    tradeable = [
        s for s in spikes if s.velocity_min <= 5 and abs(s.jump_logit) >= 0.5
    ]
    print(f"  (on the {len(tradeable)} spike+big signals the engine would take)\n")
    print(f"  {'hold':>7s} " + "".join(f"{f'tgt {int(t * 100)}%':>12s}" for t in (0.25, 0.5, 0.75, 1.0)) + f"{'no target':>12s}")
    for hold_min in (60, 120, 240, 480, 720, 1440, 2880):
        row = f"  {hold_min // 60:5d}h  "
        for target in (0.25, 0.5, 0.75, 1.0, None):
            results = [
                simulate_exit_rule(s, series_by_market[s.market_id], hold_min, target, 0.5)[0]
                for s in tradeable
            ]
            net = [r - COST_ON_CAPITAL for r in results if r != 0.0]
            row += f"{statistics.mean(net):+11.2%} " if net else f"{'-':>11s} "
        print(row)

    print("\n  stop-loss sensitivity (target 50%, hold 4h):")
    for stop in (0.3, 0.5, 0.75, 1.0, None):
        results = [
            simulate_exit_rule(s, series_by_market[s.market_id], 240, 0.5, stop)[0]
            for s in tradeable
        ]
        net = [r - COST_ON_CAPITAL for r in results if r != 0.0]
        label = f"stop {stop} logit" if stop else "no stop"
        print(line(label, [r + COST_ON_CAPITAL for r in net]))

    print("\n" + "=" * 78)
    print("4. DOES 'NEWS' HELP? — breadth as a news proxy, and the graph")
    print("=" * 78)
    print("  breadth = other markets spiking within 15 minutes")
    for name, sel in (
        ("isolated (breadth 0)", lambda s: s.breadth == 0),
        ("small cluster (1-2)", lambda s: 1 <= s.breadth <= 2),
        ("broad cluster (3+)", lambda s: s.breadth >= 3),
    ):
        vals = [s.forward[240] for s in spikes if sel(s) and 240 in s.forward]
        print(line(name + " @4h", vals))

    print("\n  same, restricted to the tradeable spike+big signals:")
    for name, sel in (
        ("isolated (breadth 0)", lambda s: s.breadth == 0),
        ("clustered (1+)", lambda s: s.breadth >= 1),
    ):
        vals = [s.forward[240] for s in tradeable if sel(s) and 240 in s.forward]
        print(line(name + " @4h", vals))

    print("\n  prior trend (was the market already moving that way?):")
    for name, sel in (
        ("continuation of a trend", lambda s: s.pre_trend_logit is not None
            and s.pre_trend_logit * s.direction > 0.1),
        ("against prior trend", lambda s: s.pre_trend_logit is not None
            and s.pre_trend_logit * s.direction < -0.1),
        ("no prior trend", lambda s: s.pre_trend_logit is not None
            and abs(s.pre_trend_logit) <= 0.1),
    ):
        vals = [s.forward[240] for s in spikes if sel(s) and 240 in s.forward]
        print(line(name + " @4h", vals))

    print("\n" + "=" * 78)
    print("5. OTHER STRUCTURE")
    print("=" * 78)
    print("  direction of the spike:")
    for name, sel in (("up-spike (fade = short)", lambda s: s.direction > 0),
                      ("down-spike (fade = long)", lambda s: s.direction < 0)):
        vals = [s.forward[240] for s in tradeable if sel(s) and 240 in s.forward]
        print(line(name, vals))

    print("\n  entry price region:")
    for name, sel in (
        ("cheap 0.10-0.30", lambda s: s.entry_price < 0.30),
        ("mid 0.30-0.70", lambda s: 0.30 <= s.entry_price <= 0.70),
        ("rich 0.70-0.90", lambda s: s.entry_price > 0.70),
    ):
        vals = [s.forward[240] for s in tradeable if sel(s) and 240 in s.forward]
        print(line(name, vals))

    print("\n  hour of day (UTC), tradeable signals:")
    by_hour = defaultdict(list)
    for s in tradeable:
        if 240 in s.forward:
            by_hour[s.hour_utc // 6].append(s.forward[240])
    for block in sorted(by_hour):
        print(line(f"{block * 6:02d}-{block * 6 + 6:02d}h UTC", by_hour[block]))

    print("\n  excursions within 24h (how much heat before it works):")
    mfe = [s.mfe for s in tradeable]
    mae = [s.mae for s in tradeable]
    print(f"    median best-case  +{statistics.median(mfe):.1%}   median worst-case {statistics.median(mae):+.1%}")
    print(f"    share whose worst-case exceeded -25%: {100 * sum(1 for m in mae if m < -0.25) / len(mae):.0f}%")

    print("\n  category:")
    by_cat = defaultdict(list)
    for s in tradeable:
        if 240 in s.forward:
            by_cat[s.category].append(s.forward[240])
    for cat in sorted(by_cat, key=lambda c: -len(by_cat[c])):
        print(line(cat, by_cat[cat]))


if __name__ == "__main__":
    sys.exit(main())
