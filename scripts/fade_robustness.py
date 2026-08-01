"""Try to break the fade capital result before believing it.

A +1272% backtest is a claim about reality that has to survive: concentration
(is it five lucky trades?), compounding (is the number an artifact of sizing?),
cost sensitivity (where does it die?), and stability across time.
"""

from __future__ import annotations

import pathlib
import random
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.capital_sim import fade_events

FIXED_STAKE = 20.0


def main() -> None:
    events, _ = fade_events(0.02, spike_and_big_only=True)
    rets = [pnl / cost for _, _, cost, pnl in events]
    ordered = sorted(rets)
    total = sum(rets)

    print(f"trades = {len(rets)}")
    print(
        f"return on capital per trade: mean {statistics.mean(rets):+.2%}  "
        f"median {statistics.median(rets):+.2%}  win {100 * sum(1 for r in rets if r > 0) / len(rets):.1f}%"
    )
    print(f"  best 5:  {[f'{r:+.0%}' for r in ordered[-5:]]}")
    print(f"  worst 5: {[f'{r:+.0%}' for r in ordered[:5]]}")
    print(
        f"  CONCENTRATION: top 5 = {100 * sum(ordered[-5:]) / total:.0f}% of total, "
        f"top 10 = {100 * sum(ordered[-10:]) / total:.0f}%, "
        f"top 20 = {100 * sum(ordered[-20:]) / total:.0f}%"
    )

    print("\nFIXED STAKE (no compounding), $20 per trade:")
    print(f"  final ${100 + sum(FIXED_STAKE * r for r in rets):.2f}")

    print("\nCOST SENSITIVITY (fixed stake, and compounded at 20%):")
    for cost in (0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15):
        evs, _ = fade_events(cost, spike_and_big_only=True)
        rr = [p / c for _, _, c, p in evs]
        compounded = 100.0
        for r in rr:
            compounded *= 1 + 0.20 * r
        print(
            f"  round-trip {int(cost * 100):2d}c: mean/trade {statistics.mean(rr):+6.2%}  "
            f"fixed ${100 + sum(FIXED_STAKE * r for r in rr):8.2f}  compounded ${compounded:9.2f}"
        )

    print("\nSTABILITY across the sample:")
    third = len(events) // 3
    for name, sub in (
        ("first third", events[:third]),
        ("middle third", events[third : 2 * third]),
        ("last third", events[2 * third :]),
    ):
        rr = [p / c for _, _, c, p in sub]
        print(
            f"  {name:13s} n={len(rr):3d}  mean {statistics.mean(rr):+6.2%}  "
            f"fixed-stake pnl ${sum(FIXED_STAKE * r for r in rr):+8.2f}"
        )

    print("\nPER-MARKET concentration (one vote per market):")
    from collections import defaultdict

    per_market = defaultdict(list)
    for (_, _, cost, pnl), event in zip(events, events, strict=True):
        pass
    per_market.clear()
    evs, trades = fade_events(0.02, spike_and_big_only=True)
    for trade, (_, _, cost, pnl) in zip(trades, evs, strict=False):
        per_market[trade.market_id].append(pnl / cost)
    means = [statistics.mean(v) for v in per_market.values()]
    print(
        f"  {len(means)} markets, {sum(1 for m in means if m > 0)} positive "
        f"({100 * sum(1 for m in means if m > 0) / len(means):.0f}%), "
        f"equal-weight mean {statistics.mean(means):+.2%}"
    )

    print("\nBOOTSTRAP of trade order (2000 shuffles, 20% compounded):")
    random.seed(1)
    finals = []
    for _ in range(2000):
        shuffled = rets[:]
        random.shuffle(shuffled)
        equity = 100.0
        for r in shuffled:
            equity *= 1 + 0.20 * r
        finals.append(equity)
    finals.sort()
    print(
        f"  p5 ${finals[100]:.0f}   median ${finals[1000]:.0f}   p95 ${finals[1900]:.0f}"
    )

    print("\nBOOTSTRAP resampling trades with replacement (does the edge survive?):")
    random.seed(2)
    means_bs = []
    for _ in range(2000):
        sample = [random.choice(rets) for _ in rets]
        means_bs.append(statistics.mean(sample))
    means_bs.sort()
    print(
        f"  mean return/trade: p5 {means_bs[100]:+.2%}  median {means_bs[1000]:+.2%}  "
        f"p95 {means_bs[1900]:+.2%}"
    )


if __name__ == "__main__":
    sys.exit(main())
