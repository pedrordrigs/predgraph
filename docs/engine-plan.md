# Signal Engine v1 — fade + twins, one engine

> Status: planned 2026-07-31, after M1 closed against the multi-hop thesis.
> Both strategies below are backed by measurements on our own 90-day data,
> not by the original diffusion hypothesis.

## Why these two

| Strategy | Evidence (2026-07-31, 90d hourly) |
|---|---|
| **Fade** — trade against large jumps | 1,803 deduped ≥3σ jumps: mean follow-through **−0.31 logits at +24h**, 60% revert vs 22% continue; consistent on both venues independently |
| **Twins** — same claim, two venues | Fed no-change Oct pair: **47% of hours >4¢ apart**, 27% >6¢; divergence runs of median 4h, max 92h; twin agreement on material moves is 100%, so divergence is mispricing, not disagreement about the claim |

The graph/watchlist layer stays as the *scoping* mechanism (which markets we poll,
which pairs exist, alert routing). It is not a predictive input in v1.

## Architecture: a post-poll hook in the collector

The collector (`predgraph run`) already ticks every 60s writing fresh bars. The
engine is one additional step per tick — no new process, no queue, all state in
SQLite so a restart resumes cleanly.

```
poll tick (60s)
  └─ bars written ──▶ engine.tick()
        ├─ FadeDetector      per watched market, on live 1-min bars
        ├─ TwinMonitor       per verified twin pair
        ├─ gates             band ▸ depth ▸ time-to-close ▸ rate ▸ novelty
        ├─ ledger            paper entries at the WORSE side, MTM, exits
        └─ notify            Telegram (if configured) + alerts table + dashboard
```

### FadeDetector
- Trigger: |Δlogit| over trailing 60m ≥ max(0.30, 3σ), σ from that market's own
  trailing 30d of 1h moves. Post-jump price in **0.10–0.90**; `depth_2c ≥ $300`
  on the entry side; time-to-close ≥ 72h; one open episode per event ladder.
- Action: paper-trade **against** the jump at the worse side of the book.
  Target: 50% retracement of the jump (retro median reversion ≈ 0.18 logit).
  Stops: time-stop 48h; invalidation-stop if the move *continues* ≥0.5 logit
  beyond the jump end (that is what real news looks like).

### TwinMonitor
- Registry: `twins` table seeded from `twins.yaml`; only pairs with
  `verified = true` trade. Verification = human confirmation that resolution
  criteria genuinely match (the one way a twin trade loses structurally, e.g.
  an intermeeting rate change settling the two sides differently).
- Trigger: **executable** divergence — `bid(rich) − ask(cheap) ≥ 3¢` net of
  assumed fees, persisting ≥ 3 consecutive ticks, depth on both legs.
- Action: paper long the cheap ask, short the rich bid, exit on convergence
  ≤1¢ or 7-day time-stop. Record which side moved away from the other — the
  1-minute lead-lag study determines whether v1.1 should fade only the laggard.
- Expansion: `predgraph twins discover` — embedding similarity between Kalshi
  event titles and Polymarket questions (local worker; free) → candidates into
  quarantine → `twins approve <id>` after human check.

### Shared gates (all cheap, run before any alert)
price band ▸ depth ▸ time-to-close ▸ rate limits (≤2 alerts/market/day, ≤10/day
global) ▸ novelty (no re-alert while an episode is open on that ladder).

### Ledger
`paper_trades` gains `strategy` ('fade'|'twin') and `legs` JSON (twin = two venue
legs). Entries and exits always price at the executable side; PnL is always net
of spread. Mark-to-market each tick; `predgraph ledger report` prints per-strategy
n / win% / mean net / equity curve. Dashboard gets a **Signals** tab: open
episodes, alert feed, per-strategy PnL.

## Graduation criteria (decided now, before any result exists)

| Strategy | Run for | Graduates to "real edge" if | Killed if |
|---|---|---|---|
| Fade | 3–4 weeks | n ≥ 40, mean net PnL > 0, win ≥ 55% | mean net ≤ 0 at n ≥ 40 |
| Twin | 3–4 weeks | ≥ 70% of episodes converge ≤1¢ within 7d | convergence < 50% or persistent one-sided drift |

No threshold tuning against live results mid-run; changes reset the clock.

## Minute-study input (2026-07-31, run before building)

The 1-minute closure study (586 observations, 172 markets, 1.48M minute bars,
placebo-controlled) found that graph diffusion **exists but is tiny and fast**:
material responses agree with the predicted direction 65.6% at +15m and 60.4%
at +30m (placebo: 53.8% / 57.1%), with the mean signed response peaking at
+0.009 logits (~0.2¢) at 30 minutes and fully decayed by +60m. That is why the
hourly study read null — the effect is over before the first hourly bar closes —
and also why it is not a standalone business: the whole move is a fraction of
one spread.

Two consequences for this engine:

- **v1.1 candidate — neighbor-confirmation filter for the fade.** A jump whose
  graph neighbors moved in sympathy within 15–30m is more likely *real news*
  (the kind that does not revert); a jump with silent neighbors is more likely
  the noise we want to fade. The propagation machinery becomes a risk filter,
  not an alpha source. Ship v1 without it; evaluate on ledger data.
- **Twin lead-lag was inconclusive from history** (Kalshi minute candles are
  activity-sparse in the fetched windows). The TwinMonitor should measure it
  live instead: record, per divergence episode, which side moved away and which
  converged — that is the lead/lag answer, collected as a by-product.

## Build order
1. **E1**: `signal/engine.py` + FadeDetector + ledger + alerts wiring + dashboard
   Signals tab + synthetic-bar tests. Collector calls `engine.tick()`.
2. **E2**: twins table + TwinMonitor + `twins discover/approve`; force-watch twin
   members and open-episode markets so episodes never go blind mid-trade.
3. **E3**: Telegram delivery, `ledger report`, config knobs. Then leave it
   running and let the ledger speak.

## Known risks, stated up front
- The retro fade effect may be partly phantom (stale-quote regression): the live
  paper test at executable prices is precisely the arbiter. If live fade PnL is
  flat while retro said −0.31, the retro number was quote artifact.
- Twin risk is resolution mismatch, not price risk — hence the verification gate.
- The verdicts are only as good as collector uptime (`start.bat` / Task Scheduler).
