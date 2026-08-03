# Snapback

A paper-trading bot that fades instantaneous overreactions in prediction markets.

When a Polymarket or Kalshi contract jumps hard and *fast*, it tends to snap back.
Snapback polls a liquid watchlist every 60 seconds, takes the other side of those
spikes, and holds for a partial retrace. It trades mock money and keeps a ledger,
because the point is to find out whether the edge survives contact with live
spreads — not to be right on a backtest.

Every rule below traces to a measurement. The ones that didn't survive measurement
are listed too, further down, because knowing where the edge *isn't* was most of
the work.

---

## The edge, in one paragraph

Across 306 Polymarket markets × 52 days of minute data, a jump of **≥0.35 logit
completing in ≤5 minutes** reverts often enough to pay for the spread: **73% of
faded spikes were profitable**, worth **+0.15 return on capital per trade** after a
3¢ round-trip cost. Speed is the discriminator, not size — a move that *grinds* to
the same place is information being priced in and loses money to fade
(−0.217 ROC, 26% win). A move that snaps there is a liquidity event.

## Two rules, one tape

Both run live, side by side, each with its own $1,000 book. A signal that clears
both opens a position in each.

| | `fade` (calibrated) | `fade_wide` |
|---|---|---|
| Minimum jump | 0.50 logit | **0.35 logit** |
| Maximum velocity | 5 min | 5 min |
| Price band | 0.30–0.90 | **0.15–0.95** |
| Re-entry lockout | 24h | **6h** |
| Position caps | 8 open / 6 per day | **12 / 12** |
| Backtested throughput | +0.759/day | **+1.110/day** |
| Break-even cost | 13.0¢ | 9.0¢ |

Shared by both: up-moves only, exit at a **75% retrace**, a 1.0-logit continuation
stop, or **48 hours** — whichever comes first. Entry is gated on depth ≥$300,
spread ≤5¢, and ≥72h to resolution, and fills are booked at the *executable* side
of the book, so the spread is paid in the ledger rather than assumed away.

**Why two.** The wider dials came out of a sweep that tried ~40 configurations,
which makes their backtest numbers optimistic by construction. Replacing the
calibrated rule with them would have destroyed the only clean measurement
available. Running both on the same quotes and comparing ledgers is the honest
version, and it costs nothing but a second row in a table.

Expect `fade_wide` to show a **lower win rate and smaller mean P&L per trade** —
that's the design, paid for by ~2.7× the volume. Throughput (trades/day × edge per
trade) is the column that settles it, and neither means anything before ~40 closed
trades each.

## What didn't work

Each of these was tested and rejected. They are not open questions.

| Idea | Result |
|---|---|
| Fade **down**-spikes | −0.110 ROC, CI [−0.177, −0.043] — significantly losing |
| **Momentum** on slow grinds | −0.217 ROC, 26% win — decisively bad |
| Fade slow grinds instead | +0.051, CI spans zero — untradeable either way |
| Short cheap contracts (longshot bias) | −0.049 ROC, CI entirely negative |
| Relax the 5-minute velocity cap | Throughput flat from ≤5m to ≤30m — buys nothing |
| Other exit rules | 75% retrace / 48h beat all four alternatives |

**The news graph this project started as.** The original thesis was that news
pressure propagates through an ontology of latent states and reprices connected
markets on a lag. It was built, and then measured: **no hop cohort beat chance**
(45–55% across horizons, magnitude correlation ≈0.008–0.055) over 10,045
observations, while the cross-venue control read 100%. The instrument worked; the
thesis was wrong. The graph, the ontology and the news tables have been deleted.

**Breadth.** A follow-up read clustered spikes as reverting harder (+18.8% vs
+7.8%) at ~1.7σ. Measured properly — against qualifying spikes rather than every
tick — the buckets show no ordering at all. It stays a recorded column, never a
gate.

**One result deliberately not traded.** Shorting 0.85–0.95 contracts showed +187%
ROC over 442 observations. It isn't a horizon artifact, but three Fed ladders
supply 183 of those observations — the same rate decision at 61 strikes each.
Collapsed to independent events it falls to +58% on 42 effective units, all inside
one 3-month window. A lead, not a strategy.

## Running it

```bash
py -3.12 -m venv .venv && ./.venv/Scripts/python.exe -m pip install -e .

snapback db init
snapback markets discover      # build the watchlist
snapback run                   # collector + engine, 60s cadence
snapback web                   # dashboard on http://127.0.0.1:8765
```

`snapback collect --minutes 350` is the same loop bounded in time — the shape a CI
scheduler can execute. `snapback ledger` prints results per rule; `snapback status`
prints row counts and bar coverage.

### Off this machine

`docs/deploy.md` has the full setup. The short version: **Vercel serves the
dashboard, GitHub Actions runs the collector, hosted Postgres is the shared
state.** Vercel cannot run the bot — Hobby cron fires once a day and its functions
live for seconds, while the strategy needs a 60-second heartbeat — so the collector
runs as a ~6-hour Actions job that dispatches its own successor. Four things in
that doc will each cost you an evening if you skip them: Vercel installs from
`pyproject.toml` and never from `requirements.txt`; it finds the ASGI app by static
analysis, so `app` must be a plain top-level assignment; Supabase's direct database
host is IPv6-only and unreachable from CI; and its session pooler allows 15 clients
in total, which one warm serverless pool will exhaust on its own.

## The watchlist

Rebuilt daily: ranked by **per-outcome 24h volume**, among markets quoted on both
sides and inside the entry band, with time left before resolution.

That sounds obvious and was originally wrong in three compounding ways. A missing
bid was read as "unknown, allow" rather than "dead". The volume gate read
Polymarket's *event*-level figure, so all 22 drivers in a $12M championship cleared
a $20k bar. And the ranking key summed in `liquidity_num`, which inside a grouped
market runs *backwards* — the dead outcomes carried the largest values. The result
was a watchlist where **two thirds of markets had never once been inside the entry
band** and 90% hadn't moved a cent.

## Layout

```
snapback/
  net.py            DNS-pinned HTTP client (the venue domains NXDOMAIN on some ISPs)
  config.py         settings; SNAPBACK_* env, legacy PREDGRAPH_* still honoured
  db.py             five tables, all of them read by the running bot
  ingest/           polymarket.py kalshi.py runner.py — discovery and polling
  signal/
    prices.py       logit-space price maths, shared by backtest and engine
    engine.py       rule sets, spike detection, exits, the paper account
  backtest/         historical harness the calibration came from
  web/              read-only dashboard (FastAPI + one static page)
  cli.py            typer CLI
tests/              49 tests, no network
docs/deploy.md      cloud setup and the traps in it
```

## Reading the dashboard

- **Performance** — combined balance, the equity curve per rule, graduation
  progress, exit-reason split.
- **Rules A/B** — the two rule sets side by side, with the backtested expectation
  printed next to the live result so the comparison has a reference.
- **Positions** — open positions marked to market, closed ones with exit reason and
  hold time.
- **Watchlist** — what's being polled, with live mid/spread/depth.

`collector_live` goes false after five minutes of silence; that's the first thing
to check when the numbers stop moving.

## Status

Paper only. No real orders, no venue credentials, no withdrawal path — the ledger
is a simulation and the balance is a record of a hypothesis, not money. Graduation
criteria were fixed before the run started: **n ≥ 40 closed trades, mean net P&L >
0, win rate ≥ 55%**, per rule. Changing a threshold mid-run resets the clock.
