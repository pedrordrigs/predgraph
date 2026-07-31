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
predgraph markets poll              # continuous, 60s
predgraph graph impact military_escalation_me --direction 1
predgraph status
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

## Next (M1)

The go/no-go milestone: a retrospective lag study using Polymarket `/prices-history` and
Kalshi candlesticks over known historical events, measuring whether n-hop markets actually
reprice with a lag. If they don't, the thesis dies cheaply — before news ingest,
extraction, and the judge get built.
