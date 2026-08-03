# PredGraph

Temporal prediction-market intelligence graph. Connects news-bearing latent states to
open Polymarket/Kalshi markets and hunts the **R+ D−** quadrant: news pressure has
landed somewhere in a market's neighbourhood (**R**epresentation) but the market has
not repriced yet (**D**amage). The edge being tested is *lag*, not convergence.

Design doc: `prediction-market-graph-v0-plan.md` (in Downloads).
This repo is milestone **M0** of that plan.

## What works today

- **DNS-pinned HTTP client** — the local ISP resolver NXDOMAINs the venue domains; we
  resolve them against public DNS and dial the IP while keeping the real hostname in
  TLS SNI and the `Host` header, so certificates still verify. No VPN.
- **Curated ontology** — 2 domains, 46 axis-bearing nodes, 39 signed/weighted/delayed
  edges, 20 market anchors. Validated on load: a latent node without an axis is an error.
- **Market discovery + linking** — ~1,700 markets scanned, ~1,400 linked into the graph
  through anchors, top 60 selected for polling.
- **Bar polling** — live order books from both venues into 1-minute bars with spread and
  tradeable depth within 2¢ of mid.
- **Signed multi-hop propagation** — the R signal in structural form (time decay lands in M3).

Verified end to end on live data: a Middle East escalation move propagates to
ceasefire markets at 1 hop (−0.85, YES down), WTI crude at 3 hops (+0.82),
CPI markets at 4 hops (+0.44), and *"Fed Rate Hike by September 2026?"* at 5 hops
(+0.23) — each with the sign composed edge by edge.

## Quick start

```bash
py -3.12 -m venv .venv && ./.venv/Scripts/python.exe -m pip install -e .
cp .env.example .env

predgraph db init
predgraph ontology validate
predgraph markets discover          # link markets, pick the watchlist
predgraph markets poll --once       # one round of bars
predgraph graph impact military_escalation_me --direction 1
predgraph status

predgraph run                       # the collector service: poll + rediscover
```

`run` is what should be up between sessions — the D signal is worthless without an
unbroken price history. To survive reboots, register it yourself (needs your
elevation, so it is deliberately not automated):

```powershell
schtasks /create /tn PredGraphCollector /sc onstart /rl highest /f `
  /tr "F:\Projeto\PredGraph\predgraph\.venv\Scripts\python.exe -m predgraph.cli run"
```

## Deviations from the plan

**SQLite instead of Postgres + pgvector.** The plan called for Postgres in Docker, but
this machine has no Docker Desktop, no Postgres, and `sudo` inside the WSL distro needs a
password — so a Postgres install could not be completed unattended. Everything is written
in portable SQLAlchemy Core, so the move is a DSN change in `.env`. Do it when the M2 dedup
cascade needs pgvector for ANN search over article embeddings, or whenever Docker lands;
until then embeddings are float32 blobs and similarity is computed in-process. At v0
volumes (60 markets × 1 bar/min ≈ 86k rows/day) SQLite is not the constraint.

**Hop cap is 5, not 3.** Set by measurement, not taste. The cross-domain chain this system
exists to catch — escalation → oil supply → crude → inflation → Fed stance → market — lands
at exactly 5 hops. On the live graph 5 hops enumerates ~1.3k paths; 6 jumps to ~3.2k and
starts connecting everything to everything. A compressed `brent_price → inflation_pressure_us`
edge exists alongside the detailed chain so the flagship path fits under the cap.

## Things the live APIs corrected

Worth knowing before trusting any doc or LLM-generated client:

- **Kalshi fields are dollar-denominated strings now** (`yes_bid_dollars`, `volume_fp`,
  `liquidity_dollars`), not the `yes_bid`-in-cents shape most docs still show. Writing
  against the old names yields silently empty bars.
- **The Kalshi book is `orderbook_fp` with `yes_dollars` / `no_dollars` ladders.** A NO bid
  at *p* is a YES offer at *1−p*; that conversion is how the ask side is derived.
- **Most short Kalshi series are retired.** `RATECUT`, `U3`, `PAYROLLS`, `RECSSNBER`,
  `PCECORE`, `OIL`, `WTI`, `NGAS` all exist in `/series` with **zero open markets**. The
  live ones are `KX`-prefixed (`KXFED`, `KXFEDDECISION`, `KXCPIYOY`, `KXU3`, `KXGDP`, …).
- **Kalshi has no open oil/gas price markets right now** (checked across all 5,600 open
  events), so energy coverage comes from Polymarket.
- **Fed markets are phrased as thresholds** ("will the upper bound of the federal funds rate
  be above 4.25%") and holds are phrased as *"Hike rates by 0bps / Fed maintains"* — both
  need their own anchors with their own signs.

## Layout

```
ontology/        domain YAML — the curated moat; edit here, not in code
predgraph/
  net.py         DNS-pinned HTTP client
  config.py      settings (.env)
  db.py          schema (portable SQLAlchemy Core)
  ontology.py    specs, validation, DB sync
  ingest/        polymarket.py kalshi.py runner.py (discovery, watchlist, polling)
  graph/algo.py  signed bounded propagation
  cli.py         typer CLI
tests/           pure-logic tests, no network
```

## Historical data: verified feasible for M1

Checked 2026-07-31 against both venues, since the lag study lives or dies on this:

| | Polymarket | Kalshi |
|---|---|---|
| Closed/settled markets queryable | yes, back to 2023 (`closed=true`) | yes (`status=settled`) |
| Resolution | **1 minute** (17.5k points over 13 days) | **1 minute** candlesticks |
| Per-request window at 1-min | full market lifetime | ~3.5 days (max-candle cap) |

Two traps worth knowing before writing the fetcher:

- **Polymarket 400s when the requested window falls outside the market's trading
  period** — and closed markets routinely have an `endDate` in the future because they
  resolved early. Clamp every request to `[startDate, closedTime]`.
- **Kalshi's `price.close_dollars` is sparse** (401 of 824 candles on a liquid Fed
  market) because it only exists where a trade happened, but `yes_bid`/`yes_ask` are
  populated on **100%** of candles. Reconstruct mid from bid/ask, never from last trade.

## What this is now

A **fade bot**, trading paper money. It polls a liquid, category-diverse watchlist
every 60 seconds, and when a market spikes hard and fast it takes the other side,
holding for a partial retrace. Every rule traces to a measurement rather than to a
plausible-sounding idea.

**Two rule sets run side by side**, each with its own $1,000 paper account, both
trading the same tape:

| | `fade` (calibrated) | `fade_wide` |
|---|---|---|
| Minimum jump | 0.50 logit | 0.35 logit |
| Maximum velocity | 5 min | 5 min |
| Price band | 0.30–0.90 | 0.15–0.95 |
| Re-entry lockout | 24h | 6h |
| Caps | 8 open / 6 per day | 12 open / 12 per day |
| Backtested throughput | +0.759/day | **+1.110/day** |

Shared by both: up-moves only, exit at a **75% retrace**, a 1.0-logit continuation
stop, or 48 hours, and gates on depth, spread and ≥72h to resolution.

Why the shared rules, one line each:
- **Instant, not gradual** — grinds are information being priced in and lose to fade;
  only spikes snap back. Following grinds is worse still (−0.217 ROC, 26% win).
- **Up-moves only** — down-spikes lose significantly on their own
  (−0.110 ROC, CI [−0.177, −0.043]).
- **48h, not 24h** — reversion accrues slowly; the fade is *negative* for the first
  half hour and still improving at 48 hours.
- **75% retrace** — beat 50%, 100%, a tighter stop and a 72h hold.

Why a second rule set instead of just replacing the first: the sweep that produced
the wider dials tried ~40 configurations, which makes its numbers optimistic by
construction. Running both and comparing ledgers is the only measurement that
isn't. The A/B tab on the dashboard is where that settles.

Known risk: the strategy is positive-skew. The median trade is a small loss; the mean
is carried by a minority of large winners, so a short losing streak proves nothing and
the paper run needs ~40 trades before it says anything.

## Running it

```bat
setup.bat      REM once: venv, deps, database, ontology, first discovery
start.bat      REM every time: collector + dashboard at http://127.0.0.1:8765
```

The dashboard has four tabs:
- **Performance** — equity curve, graduation criteria against live counts, exit-reason
  and breadth splits.
- **A/B: wide rule** — both rule sets side by side, with the backtested expectation
  printed next to the live result so the comparison has a reference.
- **Positions** — open positions marked to market, and closed ones with exit reason
  and hold time. The `Rule` column says which rule set opened each.
- **Watchlist** — what is being polled, with live mid/spread/depth.

### Off the local machine

`docs/deploy.md` covers running it in the cloud for free. The short version: the
dashboard goes on Vercel, but the collector cannot — Hobby cron fires once a day
and its functions live for seconds, while the strategy needs a 60-second
heartbeat. So the collector runs as a long GitHub Actions job (`collect.yml`,
~55 minutes per run, chained) and both sides share a hosted Postgres. Two things
to know before committing to it: the repo has to be **public** for the Actions
minutes to be free, and quote bars must be pruned daily or a free Postgres tier
fills up in five days.

## Strategy sweep, 2026-08-02

~40 configurations over 306 Polymarket markets × 52 days of minute data, plus 1,546
markets of hourly for the slower ideas. Everything was declared before results were
seen, headline figures are equal-weighted per market with a bootstrap CI over markets,
and entry is delayed 2 minutes because entering at the peak is unobtainable.

**Adopted** — `fade_wide` above. Loosening jump, band and lockout together traded 45%
of the per-trade edge for 2.7× the volume: throughput +0.759 → **+1.110/day**. It held
in both calendar halves, survived collapsing to one unit per event-day (+0.097, CI
[+0.044, +0.163]), and was significant on **Kalshi** — a venue that took no part in
choosing it — where the calibrated rule was not. Break-even cost 9¢ against ~1–2¢ live
spreads.

**Rejected, with evidence.** These are the useful half; each is a thing not worth
trying again:

| Idea | Result |
|---|---|
| Fade down-spikes | **−0.110 ROC**, CI [−0.177, −0.043] — significantly losing |
| Momentum on slow grinds | **−0.217 ROC**, 26% win — decisively bad |
| Fade slow grinds instead | +0.051, CI spans zero — untradeable either way |
| Short cheap contracts (longshot bias) | −0.049, CI entirely negative |
| Relax the 5-min velocity limit | Throughput flat from ≤5m to ≤30m — buys nothing |
| Change the exit rule | 75% retrace / 48h beat all four alternatives |

**Breadth does not replicate.** The earlier deep dive read clustered spikes as
reverting harder (+18.8% vs +7.8%) at ~1.7σ, and flagged a possible same-event
confound. Measured properly — against qualifying spikes rather than every tick — the
buckets show no ordering at all. It stays a recorded column and was never a gate.

**One result deliberately not adopted.** Shorting 0.85–0.95 contracts showed +187% ROC
on n=442. It is not a horizon artifact (92% of exits land at 14–16 days), but three Fed
ladders contribute 183 of those observations — the same rate decision at 61 strikes
each. Collapsed to independent events it falls to +58% on 42 effective units, all
inside one 3-month window. A lead, not a strategy.

Also held back: a volatility-normalised (z ≥ 3σ) trigger had the best raw throughput
(+1.228, 11 signals/day) but keeps only **+1.33¢ per trade**, and its CI fails once
costs reach 5¢. The bot now records real spreads, so that becomes answerable with data
instead of an assumption.

The exploration scripts are not in this repo — they are research scratch, and the
findings that survived are in the rule sets above.

## M1: the instrument works, and the thesis does not show up

**Instrument: verified.** The cross-venue twin control reads **100%** (6/6 material moves at
the 0.10 logit floor, 7/7 at 0.05). When the same claim is listed on both venues and one
side reprices materially, the other moves the same way — every time. Jump detection,
timestamp alignment, logit measurement and sign composition are all doing their jobs. The n
is small, so read it as "no evidence of a broken pipeline" rather than a tight bound.

**Thesis: not supported by 90 days of hourly data.** With a working instrument, over 10,045
observations from 626 deduplicated trigger jumps:

| cohort | hit 1h | hit 4h | hit 24h | hit 48h | corr 4h |
|---|---|---|---|---|---|
| hop 2 | 45.8% | 50.5% | 50.8% | 50.7% | 0.008 |
| hop 3 | 52.6% | 53.3% | 51.2% | 49.5% | 0.020 |
| hop 4 | 54.1% | 50.7% | 51.5% | 51.8% | 0.010 |
| hop 5 | 55.2% | 55.0% | 45.0% | 48.5% | 0.055 |
| control | 53.8% | 47.6% | 50.0% | 64.3% | — |

No cohort separates from chance, and the magnitude correlation — which sign agreement
cannot fake — is essentially zero everywhere. A real lagged propagation should leave *some*
correlation at 4–24h even at hourly resolution.

**The one gradient that did appear** is informative about the ontology rather than the
thesis: peers in the same strike ladder agree ~62%, peers sharing only an ontology driver
agree ~51%, and cross-venue twins agree 100%. Mechanical linkage shows up clearly;
"linked through my graph" does not. That points at the anchor signs for threshold ladders
being too crude — every strike of `KXFED-*` gets one sign, though a deep-ITM and a
deep-OTM strike respond very differently to the same news.

**What is left before calling it.** One cheap check remains: 1-minute measurement around
each jump (now that the fetcher chunks correctly, this is a few hours of compute). Set
expectations low — hourly granularity blurs a multi-hour effect, it does not erase it.
If that comes back flat too, the honest conclusion is that this thesis does not hold for
these domains, and the fallbacks that do **not** depend on it are the R−D+ unexplained-move
alerter and the fade-the-overreaction quadrant.

## Earlier diagnosis (superseded, kept for the record)

The study is built and runs (`predgraph backtest fetch` then `predgraph backtest lag`),
over 285k backfilled bars across 879 markets. **It does not yet answer the question**,
and the reason matters more than the numbers.

Every run first reports a **positive control**: agreement between 1-hop markets on the
same driver. Different strikes of one CPI ladder, or different outcomes of one Fed
meeting, are mechanically linked — when one reprices on news the others must move too.
That control currently reads **52–64%**, where a working measurement should read 80%+.

So the instrument cannot detect a relationship that certainly exists, and no hop-level
number from the same pipeline can be trusted. The cohort table prints as diagnostics with
an explicit warning rather than as a result. Three things were ruled out along the way:
illiquidity (filtering to liquid markets: still 55%), stale quotes (a real bug — fixed
with a staleness bound, `value_at` used to carry a price forward across gaps of up to 644
hours), and wide spreads plus tail prices (filtering collapses the sample to n<100).

The diagnosis is data granularity. Hourly venue history is sparse and quote-driven: the
median market has 156 bars across a 1,300-hour span, and an event plus its response
frequently land in the same bucket. Polymarket's cohort scores highest (64.5%), which
fits — its history is continuous while Kalshi emits candles only on activity.

**Next attempt, in order:** measure at 1-minute resolution around each detected jump
(verified available on both venues, and never used yet — this is the biggest lever);
prefer live collector bars, which are continuous and carry real depth, over backfilled
venue history; and keep the positive control as the gate — no verdict until it clears 80%.
