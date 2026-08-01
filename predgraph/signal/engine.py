"""Signal Engine v1 — the fade strategy, live, on paper.

Runs as one step after each collector poll. Everything it needs is in SQLite, so
a restart mid-episode resumes cleanly rather than orphaning a position.

The trigger is what the minute simulation validated, not what seemed plausible:
a jump must be BIG (>= 0.5 logit) *and* INSTANT (10%->90% of its amplitude in
<= 5 minutes). Big-but-gradual moves were net losers to fade (-1.2c at 3c
costs) because a grind is information being incorporated, while a spike is a
liquidity event that snaps back. Velocity is the discriminator; graph-neighbour
confirmation was tested and did not separate, so it is recorded but not gated.

Fills are booked at the executable side of the book — sell into the bid, buy
back at the ask — so the spread is paid in the ledger rather than assumed away.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import pairwise

import sqlalchemy as sa

from predgraph.db import alerts as alerts_t
from predgraph.db import get_engine, utcnow
from predgraph.db import market_bars as bars_t
from predgraph.db import markets as markets_t
from predgraph.db import paper_trades as trades_t
from predgraph.signal.damage import logit

log = logging.getLogger(__name__)

# --- calibrated by the minute simulation, 2026-07-31 ------------------------
JUMP_WINDOW_MIN = 60
MIN_JUMP_LOGIT = 0.50
JUMP_Z = 3.0
MAX_VELOCITY_MIN = 5.0
RETRACE_TARGET = 0.5
CONTINUATION_STOP_LOGIT = 0.5
TIME_STOP_H = 24.0
# --- gates ------------------------------------------------------------------
PRICE_LO, PRICE_HI = 0.10, 0.90
MIN_DEPTH_2C = 300.0
MIN_HOURS_TO_CLOSE = 72.0
MAX_SPREAD = 0.05
EPISODE_LOCKOUT_H = 24.0
MAX_OPEN_TRADES = 8
MAX_TRADES_PER_DAY = 6
STAKE = 100.0


@dataclass(slots=True)
class Bar:
    ts: datetime
    mid: float
    bid: float | None
    ask: float | None
    depth: float | None


@dataclass(slots=True)
class FadeSignal:
    market_id: str
    ts: datetime
    jump_logit: float
    velocity_min: float
    sigma: float | None
    bar: Bar

    @property
    def direction(self) -> int:
        """-1 fades an up-jump (sell YES), +1 fades a down-jump (buy YES)."""
        return -1 if self.jump_logit > 0 else 1


def _load_bars(conn, market_id: str, minutes: int) -> list[Bar]:
    rows = conn.execute(
        sa.select(bars_t.c.ts, bars_t.c.mid, bars_t.c.bid, bars_t.c.ask, bars_t.c.depth_2c)
        .where(
            sa.and_(
                bars_t.c.market_id == market_id,
                bars_t.c.ts >= utcnow() - timedelta(minutes=minutes),
                bars_t.c.mid.isnot(None),
            )
        )
        .order_by(bars_t.c.ts)
    ).all()
    return [Bar(r.ts, float(r.mid), r.bid, r.ask, r.depth_2c) for r in rows]


def market_sigma(conn, market_id: str, days: int = 30) -> float | None:
    """Typical hourly logit move for this market, from its own recent history."""
    rows = conn.execute(
        sa.select(bars_t.c.ts, bars_t.c.mid)
        .where(
            sa.and_(
                bars_t.c.market_id == market_id,
                bars_t.c.ts >= utcnow() - timedelta(days=days),
                bars_t.c.mid.isnot(None),
            )
        )
        .order_by(bars_t.c.ts)
    ).all()
    if len(rows) < 60:
        return None
    hourly: dict[datetime, float] = {}
    for row in rows:
        hourly[row.ts.replace(minute=0, second=0, microsecond=0)] = float(row.mid)
    stamps = sorted(hourly)
    moves = [
        logit(hourly[b]) - logit(hourly[a])
        for a, b in pairwise(stamps)
        if (b - a) <= timedelta(hours=2)
    ]
    if len(moves) < 20:
        return None
    sigma = statistics.pstdev(moves)
    return sigma if sigma > 1e-6 else None


def detect_fade_signal(bars: list[Bar], sigma: float | None) -> FadeSignal | None:
    """A jump that is both large and near-instantaneous, ending on the last bar."""
    if len(bars) < 5:
        return None
    now = bars[-1]
    window = [b for b in bars if b.ts >= now.ts - timedelta(minutes=JUMP_WINDOW_MIN)]
    if len(window) < 5:
        return None

    base = window[0]
    delta = logit(now.mid) - logit(base.mid)
    threshold = max(MIN_JUMP_LOGIT, (sigma or 0.0) * JUMP_Z)
    if abs(delta) < threshold:
        return None

    direction = 1 if delta > 0 else -1
    t10 = t90 = None
    for bar in window:
        progress = (logit(bar.mid) - logit(base.mid)) * direction
        if t10 is None and progress >= 0.1 * abs(delta):
            t10 = bar.ts
        if t90 is None and progress >= 0.9 * abs(delta):
            t90 = bar.ts
            break
    if t10 is None or t90 is None:
        return None
    velocity = (t90 - t10).total_seconds() / 60.0
    if velocity > MAX_VELOCITY_MIN:
        return None  # a grind, not a spike: information, not overreaction

    return FadeSignal(
        market_id="",
        ts=now.ts,
        jump_logit=delta,
        velocity_min=velocity,
        sigma=sigma,
        bar=now,
    )


def _tradeable(signal: FadeSignal, close_time: datetime | None) -> str | None:
    """Returns a rejection reason, or None if the signal passes every gate."""
    bar = signal.bar
    if bar.bid is None or bar.ask is None:
        return "no quote"
    if not (PRICE_LO <= bar.mid <= PRICE_HI):
        return "outside price band"
    if (bar.ask - bar.bid) > MAX_SPREAD:
        return "spread too wide"
    if (bar.depth or 0) < MIN_DEPTH_2C:
        return "insufficient depth"
    if close_time is None or close_time - utcnow() < timedelta(hours=MIN_HOURS_TO_CLOSE):
        return "too close to resolution"
    return None


def _entry_price(signal: FadeSignal) -> float:
    """Executable side: selling YES hits the bid, buying YES lifts the ask."""
    return signal.bar.bid if signal.direction < 0 else signal.bar.ask  # type: ignore[return-value]


def _exit_price(bar: Bar, direction: int) -> float | None:
    """Closing a short YES lifts the ask; closing a long YES hits the bid."""
    return bar.ask if direction < 0 else bar.bid


def open_episodes(conn) -> list[dict]:
    rows = conn.execute(
        sa.select(trades_t).where(
            sa.and_(trades_t.c.status == "open", trades_t.c.strategy == "fade")
        )
    ).all()
    return [dict(r._mapping) for r in rows]


def _recent_episode(conn, market_id: str) -> bool:
    cutoff = utcnow() - timedelta(hours=EPISODE_LOCKOUT_H)
    return (
        conn.execute(
            sa.select(sa.func.count())
            .select_from(trades_t)
            .where(
                sa.and_(
                    trades_t.c.market_id == market_id,
                    trades_t.c.strategy == "fade",
                    trades_t.c.entry_ts >= cutoff,
                )
            )
        ).scalar()
        or 0
    ) > 0


def manage_open(conn) -> list[dict]:
    """Mark to market and close episodes that hit target, stop or time limit."""
    closed: list[dict] = []
    for trade in open_episodes(conn):
        bars = _load_bars(conn, trade["market_id"], minutes=10)
        if not bars:
            continue
        bar = bars[-1]
        meta = trade.get("meta") or {}
        direction = -1 if trade["side"] == "sell_yes" else 1
        target_logit = meta.get("target_logit")
        stop_logit = meta.get("stop_logit")
        price_logit = logit(bar.mid)

        reason = None
        if target_logit is not None and (price_logit - target_logit) * direction >= 0:
            reason = "target"
        elif stop_logit is not None and (price_logit - stop_logit) * direction <= 0:
            reason = "stop"
        elif trade["entry_ts"] is not None and utcnow() - trade["entry_ts"] >= timedelta(
            hours=trade.get("window_h") or TIME_STOP_H
        ):
            reason = "time"
        if reason is None:
            continue

        exit_price = _exit_price(bar, direction)
        if exit_price is None:
            continue
        pnl = direction * (exit_price - trade["entry_price"]) * (trade["size"] or STAKE)
        conn.execute(
            trades_t.update()
            .where(trades_t.c.id == trade["id"])
            .values(
                exit_ts=bar.ts,
                exit_mid=bar.mid,
                exit_price=exit_price,
                pnl=round(pnl, 4),
                status=f"closed_{reason}",
            )
        )
        closed.append({**trade, "exit_reason": reason, "pnl": round(pnl, 4)})
    return closed


def scan(conn) -> list[tuple[FadeSignal, dict]]:
    """Signals passing every gate, with their market row."""
    markets = conn.execute(
        sa.select(
            markets_t.c.id, markets_t.c.question, markets_t.c.venue, markets_t.c.close_time
        ).where(markets_t.c.watch.is_(True))
    ).all()

    found: list[tuple[FadeSignal, dict]] = []
    for market in markets:
        bars = _load_bars(conn, market.id, minutes=JUMP_WINDOW_MIN + 10)
        signal = detect_fade_signal(bars, market_sigma(conn, market.id))
        if signal is None:
            continue
        signal.market_id = market.id
        rejection = _tradeable(signal, market.close_time)
        if rejection is not None:
            log.info("fade signal on %s rejected: %s", market.id, rejection)
            continue
        if _recent_episode(conn, market.id):
            continue
        found.append((signal, dict(market._mapping)))
    return found


def tick(notify=None) -> dict:
    """One engine step: manage open episodes, then open new ones."""
    engine = get_engine()
    opened: list[dict] = []
    with engine.begin() as conn:
        closed = manage_open(conn)

        n_open = len(open_episodes(conn))
        today = (
            conn.execute(
                sa.select(sa.func.count())
                .select_from(trades_t)
                .where(
                    sa.and_(
                        trades_t.c.strategy == "fade",
                        trades_t.c.entry_ts >= utcnow() - timedelta(hours=24),
                    )
                )
            ).scalar()
            or 0
        )

        for signal, market in scan(conn):
            if n_open >= MAX_OPEN_TRADES or today >= MAX_TRADES_PER_DAY:
                break
            entry = _entry_price(signal)
            entry_logit = logit(signal.bar.mid)
            direction = signal.direction
            payload = {
                "jump_logit": round(signal.jump_logit, 4),
                "velocity_min": round(signal.velocity_min, 2),
                "sigma": round(signal.sigma, 4) if signal.sigma else None,
                "target_logit": entry_logit
                + direction * RETRACE_TARGET * abs(signal.jump_logit),
                "stop_logit": entry_logit - direction * CONTINUATION_STOP_LOGIT,
                "entry_bid": signal.bar.bid,
                "entry_ask": signal.bar.ask,
                "depth_2c": signal.bar.depth,
            }
            thesis = (
                f"{'up' if signal.jump_logit > 0 else 'down'}-spike of "
                f"{abs(signal.jump_logit):.2f} logit in {signal.velocity_min:.0f}min; "
                f"fading toward a {int(RETRACE_TARGET * 100)}% retrace"
            )
            alert_id = conn.execute(
                alerts_t.insert().values(
                    market_id=signal.market_id,
                    ts=signal.ts,
                    quadrant="R?D+",
                    r_signed=None,
                    d_pct=None,
                    event_ids=[],
                    judge={"strategy": "fade", "thesis": thesis, **payload},
                    delivered=False,
                )
            ).inserted_primary_key[0]

            conn.execute(
                trades_t.insert().values(
                    alert_id=alert_id,
                    market_id=signal.market_id,
                    strategy="fade",
                    meta=payload,
                    side="sell_yes" if direction < 0 else "buy_yes",
                    entry_ts=signal.ts,
                    entry_mid=signal.bar.mid,
                    entry_price=entry,
                    size=STAKE,
                    thesis=thesis,
                    invalidation=f"move continues {CONTINUATION_STOP_LOGIT} logit further",
                    window_h=TIME_STOP_H,
                    status="open",
                )
            )
            opened.append({"market": signal.market_id, "question": market["question"], **payload})
            n_open += 1
            today += 1

    if notify is not None:
        for episode in opened:
            notify(episode)
    return {"opened": len(opened), "closed": len(closed), "episodes": opened, "exits": closed}


def ledger_summary(strategy: str = "fade") -> dict:
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sa.select(trades_t).where(trades_t.c.strategy == strategy)
        ).all()
    closed = [r for r in rows if r.pnl is not None]
    wins = [r for r in closed if r.pnl > 0]
    return {
        "open": sum(1 for r in rows if r.status == "open"),
        "closed": len(closed),
        "win_pct": round(100.0 * len(wins) / len(closed), 1) if closed else None,
        "total_pnl": round(sum(r.pnl for r in closed), 2) if closed else 0.0,
        "mean_pnl": round(statistics.mean([r.pnl for r in closed]), 3) if closed else None,
        "by_exit": {
            reason: sum(1 for r in closed if r.status == f"closed_{reason}")
            for reason in ("target", "stop", "time")
        },
    }
