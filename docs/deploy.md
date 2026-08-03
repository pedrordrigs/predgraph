# Running PredGraph off the local machine

## Why it isn't just "deploy to Vercel"

The bot is two different workloads with incompatible hosting needs:

| | needs | Vercel Hobby offers |
|---|---|---|
| Dashboard | a URL that responds for ~200ms | exactly this |
| Collector | a 60-second heartbeat, 24/7, with state | cron **once per day**, functions that die in seconds, no persistent filesystem |

So Vercel hosts the dashboard, and only the dashboard. The collector runs as a
long GitHub Actions job, and both sides share a hosted Postgres.

```
GitHub Actions (collect.yml)        Neon Postgres            Vercel
  poll 200 markets every 60s  --->   market_bars     <---  dashboard (read-only)
  run the fade engine                paper_trades
  ~55 min per run, chained           alerts
```

Cadence is not a detail here. The deep dive measured that entering ~30 minutes
after a spike gives up most of the edge, so a 5-minute poll would quietly
destroy the strategy. That constraint is what forces the long-job design
instead of a normal serverless cron.

## Before you start: two decisions

**1. The repo must be public for this to be free.**
Private repos get 2,000 Actions minutes/month. Continuous collection needs
~43,000. Public repos have unlimited minutes for standard runners. Nothing
secret lives in the code — the DB URL is a GitHub secret — but the strategy
and its calibration would be publicly readable.

Note also that a 24/7 data collector is not really what Actions is sold for
(the ToS frames it around building and testing the repo's own software). It
works, but it's a gray area worth knowing about before you rely on it. If that
matters, an Oracle Cloud Always Free VM runs the existing `predgraph run`
unchanged, 24/7, with no such ambiguity.

**2. Storage has to be pruned.**
Measured on real rows: 322 bytes each including the index, 288k rows/day, so
**~93 MB/day**. Against a 500 MB free tier that is five days of runway with no
margin. `maintain.yml` runs `predgraph prune` daily keeping **3 days** (~280 MB),
which still clears both limits that matter: 72 hourly points for a baseline
that needs 20, and 72h of history against a 48h maximum hold. Trades and alerts
are the results, and are never pruned.

Re-measure if you widen the watchlist — cost scales with markets × cadence, and
both are things you may want to raise.

## Setup

### 1. Postgres

Create a free project at [neon.com](https://neon.com) and copy the connection
string. Any `postgres://` or `postgresql://` URL works as-is — the app rewrites
the scheme to the psycopg3 driver it ships.

**On Supabase, take the Session pooler string, not the direct one.** Its direct
host (`db.<ref>.supabase.co`) publishes only an AAAA record, and GitHub Actions
runners are IPv4-only, so the collector can never reach it — it fails at DNS
with `getaddrinfo failed`, which reads like a typo rather than a missing address
family. The pooler (`<prefix>-<region>.pooler.supabase.com:5432`, username
`postgres.<ref>`) is dual-stack.

**Prefer port 6543 (transaction mode) over 5432 (session mode).** Session mode
allows only **15 clients across everything you run**, and warm serverless
instances alone exhaust it — connections then fail with `EMAXCONNSESSION`,
intermittently and under load, which is the worst way to find out. Transaction
mode has no such ceiling. The usual objection does not apply here: psycopg
promotes repeated queries to prepared statements, which transaction pooling
cannot honour, so the app disables that explicitly. If the password contains `@ : / ? #`, percent-encode
it — those are URL delimiters and will otherwise truncate the DSN.

Initialise the schema once, from your machine:

```bash
PREDGRAPH_DB_URL="postgresql://...:...@...neon.tech/predgraph?sslmode=require" predgraph db init
```

### 2. GitHub

Push the repo, then add under **Settings → Secrets and variables → Actions**:

| Secret | Required | Purpose |
|---|---|---|
| `PREDGRAPH_DB_URL` | yes | Postgres connection string |
| `ACTIONS_PAT` | optional | Fine-grained PAT with **Actions: read and write** on this repo. Lets each collector run dispatch the next one, closing the gap left by GitHub's cron delays. Without it the hourly cron alone carries the schedule. |

Seed the watchlist by running the **maintain** workflow once from the Actions
tab, then start **collect** the same way. After that they self-schedule.

### 3. Vercel

Import the repo at [vercel.com/new](https://vercel.com/new). `vercel.json`
already points the build at `api/index.py`. Add two environment variables:

Two things about this build that cost real time, worth knowing before editing
either file. Vercel installs from **`pyproject.toml`**, not `requirements.txt`, which it
ignores whenever a pyproject exists — and it installs the **base dependency set
only, never an extra**. Anything needed at runtime has to be a core dependency;
`fastapi` and `psycopg` each cost a deploy cycle learning this.
And it locates the ASGI app by **static analysis**, so `app` has to be a plain
top-level assignment; binding it inside a `try` fails the build outright with
`PYTHON_ENTRYPOINT_NOT_FOUND`.

| Variable | Purpose |
|---|---|
| `PREDGRAPH_DB_URL` | same string as above |
| `PREDGRAPH_DASHBOARD_TOKEN` | optional; any random string. When set, every route needs `?k=<token>` |

The dashboard is then at `https://<project>.vercel.app/?k=<token>`.

## Checking on it

Everything needed to judge the bot is reachable over HTTP, so a review needs
nothing but the URL:

| Endpoint | Answers |
|---|---|
| `/api/health` | is the deployment up (never token-gated) |
| `/api/status` | is the collector alive — `collector_live` goes false after 5 minutes of silence |
| `/api/performance` | equity curve, win rate, exit-reason and breadth splits, graduation criteria |
| `/api/trades` | every open and closed position, with mark-to-market PnL |

`collector_live: false` with a healthy dashboard means the Actions side stalled
— check the collect workflow's most recent run.

## Local development is unchanged

With no `PREDGRAPH_DB_URL` set, everything still points at
`data/predgraph.db`, and `predgraph run` still runs collector and engine
together in one process. Nothing about the research scripts changed; they read
`history_bars`, which is local-only and never populated in the cloud.
