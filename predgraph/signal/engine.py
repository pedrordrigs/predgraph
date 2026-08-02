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

# --- calibrated by the deep dive over 808 spikes, 2026-07-31 ----------------
# Every constant here traces to a measurement, and the measurements used a
# 2-minute entry delay, so these are reachable by a 60s-poll bot rather than
# theoretical. Do not tune them against live results mid-run.
JUMP_WINDOW_MIN = 60
MIN_JUMP_LOGIT = 0.50  # small spikes are net losers to fade (-3.7%)
JUMP_Z = 3.0
MAX_VELOCITY_MIN = 5.0  # grinds are information; only instant moves revert
# 48h beat 24h at every target level; reversion accrues slowly and the old 24h
# stop was cutting trades before the edge had arrived.
TIME_STOP_H = 48.0
# 75% retrace slightly beat 50% at a 48h hold, and the whole 50-75% x 4-48h
# region is one plateau, which is what a real effect looks like.
RETRACE_TARGET = 0.75
# Tighter stops cost steadily (0.3 -> 1.0 logit improved monotonically), but a
# stop stays because the backtest under-samples the case this protects against:
# a spike that is real news and never comes back.
CONTINUATION_STOP_LOGIT = 1.0
# --- gates ------------------------------------------------------------------
# Fading below 30c lost outright (-2.7%); mid and rich both worked.
PRICE_LO, PRICE_HI = 0.30, 0.90
# Up-spikes reverted far better than down-spikes (+16.8% vs +3.5%), consistent
# with excitement overpricing YES.
UP_SPIKES_ONLY = True
MIN_DEPTH_2C = 300.0
MIN_HOURS_TO_CLOSE = 72.0
MAX_SPREAD = 0.05
EPISODE_LOCKOUT_H = 24.0
MAX_OPEN_TRADES = 8
MAX_TRADES_PER_DAY = 6
# Contracts per trade, not dollars: PnL is a price delta times this count. What
# a trade actually ties up depends on which side is taken and at what price.
STAKE = 100.0
# The paper account. Sized so the cap of 8 concurrent trades cannot exhaust it
# at normal prices, which keeps the balance a record of the edge rather than a
# second constraint that quietly changes which signals get taken.
STARTING_BALANCE = 1000.0


def trade_capital(side: str, entry_price: float, size: float = STAKE) -> float:
    """Cash a position ties up until it settles.

    Shorting YES at 0.85 posts the 0.15 it could lose, not 0.85; going long
    posts the price paid. Using the notional for both would overstate what a
    short costs by several times and make the account meaningless.
    """
    per_contract = (1.0 - entry_price) if side == "sell_yes" else entry_price
    return round(per_contract * size, 2)
# Recorded, not gated on: clustered spikes reverted more (+18.8% vs +7.8%), but
# at ~1.7 sigma with a possible same-event confound. Live data decides.
BREADTH_WINDOW_MIN = 15


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


def _load_bars_bulk(conn, market_ids, minutes: int) -> dict[str, list[Bar]]:
    """Recent bars for many markets in one round trip.

    Querying per market is free on local SQLite and ruinous against a hosted
    database - 200 markets meant 200 sequential round trips per tick.
    """
    if not market_ids:
        return {}
    rows = conn.execute(
        sa.select(
            bars_t.c.market_id, bars_t.c.ts, bars_t.c.mid,
            bars_t.c.bid, bars_t.c.ask, bars_t.c.depth_2c,
        )
        .where(
            sa.and_(
                bars_t.c.market_id.in_(list(market_ids)),
                bars_t.c.ts >= utcnow() - timedelta(minutes=minutes),
                bars_t.c.mid.isnot(None),
            )
        )
        .order_by(bars_t.c.market_id, bars_t.c.ts)
    ).all()
    out: dict[str, list[Bar]] = {}
    for r in rows:
        out.setdefault(r.market_id, []).append(
            Bar(r.ts, float(r.mid), r.bid, r.ask, r.depth_2c)
        )
    return out


# Sigma is a 30-day baseline of hourly moves; it barely shifts between ticks,
# so recomputing it every 60 seconds is pure cost. Cached per process and
# refreshed hourly, which also means a fresh CI run pays for it exactly once.
_SIGMA_CACHE: dict[str, float | None] = {}
_SIGMA_AT: datetime | None = None
SIGMA_TTL = timedelta(hours=1)


def _hour_expr(conn):
    """Truncate a timestamp to the hour, in whichever dialect is in play."""
    if conn.dialect.name == "postgresql":
        return sa.func.date_trunc("hour", bars_t.c.ts)
    return sa.func.strftime("%Y-%m-%d %H", bars_t.c.ts)


def market_sigmas(conn, market_ids, days: int = 30) -> dict[str, float | None]:
    """Typical hourly logit move per market, from each market's own history.

    Downsamples to one bar per hour inside the database. Pulling the raw bars
    would mean well over a million rows per tick at a 60-second cadence, when
    only the ~24 hourly points a day are ever used.
    """
    global _SIGMA_AT
    now = utcnow()
    if _SIGMA_AT is not None and now - _SIGMA_AT < SIGMA_TTL:
        return _SIGMA_CACHE
    if not market_ids:
        return {}

    hour = _hour_expr(conn).label("hour")
    ranked = (
        sa.select(
            bars_t.c.market_id,
            hour,
            bars_t.c.mid,
            sa.func.row_number()
            .over(
                partition_by=[bars_t.c.market_id, _hour_expr(conn)],
                order_by=bars_t.c.ts.desc(),
            )
            .label("rn"),
        )
        .where(
            sa.and_(
                bars_t.c.market_id.in_(list(market_ids)),
                bars_t.c.ts >= now - timedelta(days=days),
                bars_t.c.mid.isnot(None),
            )
        )
        .subquery()
    )
    rows = conn.execute(
        sa.select(ranked.c.market_id, ranked.c.hour, ranked.c.mid)
        .where(ranked.c.rn == 1)
        .order_by(ranked.c.market_id, ranked.c.hour)
    ).all()

    hourly: dict[str, list[tuple[datetime, float]]] = {}
    for r in rows:
        # Postgres date_trunc returns a datetime; SQLite strftime returns
        # "YYYY-MM-DD HH", which needs the minutes and seconds put back.
        stamp = r.hour if isinstance(r.hour, datetime) else datetime.fromisoformat(f"{r.hour}:00:00")
        hourly.setdefault(r.market_id, []).append((stamp, float(r.mid)))

    result: dict[str, float | None] = {}
    for market_id, points in hourly.items():
        moves = [
            logit(b[1]) - logit(a[1])
            for a, b in pairwise(points)
            if (b[0] - a[0]) <= timedelta(hours=2)
        ]
        if len(moves) < 20:
            result[market_id] = None
            continue
        sigma = statistics.pstdev(moves)
        result[market_id] = sigma if sigma > 1e-6 else None

    _SIGMA_CACHE.clear()
    _SIGMA_CACHE.update(result)
    _SIGMA_AT = now
    return result


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
    if UP_SPIKES_ONLY and signal.jump_logit < 0:
        return "down-spike (fades poorly)"
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


def market_breadth(conn, at: datetime) -> int:
    """How many watched markets moved sharply in the same window.

    A lone spike is one big order into a thin book; a cluster is the market
    reacting to something. Recorded on every episode so the live ledger can
    settle whether it predicts anything.
    """
    since = at - timedelta(minutes=BREADTH_WINDOW_MIN)
    rows = conn.execute(
        sa.select(bars_t.c.market_id, bars_t.c.ts, bars_t.c.mid)
        .where(sa.and_(bars_t.c.ts >= since, bars_t.c.mid.isnot(None)))
        .order_by(bars_t.c.market_id, bars_t.c.ts)
    ).all()
    by_market: dict[str, list[float]] = {}
    for row in rows:
        by_market.setdefault(row.market_id, []).append(float(row.mid))
    moved = 0
    for prices in by_market.values():
        if len(prices) >= 2 and abs(logit(prices[-1]) - logit(prices[0])) >= 0.20:
            moved += 1
    return moved


def scan(conn) -> list[tuple[FadeSignal, dict]]:
    """Signals passing every gate, with their market row."""
    markets = conn.execute(
        sa.select(
            markets_t.c.id, markets_t.c.question, markets_t.c.venue, markets_t.c.close_time
        ).where(markets_t.c.watch.is_(True))
    ).all()

    ids = [m.id for m in markets]
    bars_by_market = _load_bars_bulk(conn, ids, minutes=JUMP_WINDOW_MIN + 10)
    sigmas = market_sigmas(conn, ids)

    found: list[tuple[FadeSignal, dict]] = []
    for market in markets:
        bars = bars_by_market.get(market.id, [])
        signal = detect_fade_signal(bars, sigmas.get(market.id))
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


def account(conn) -> dict:
    """The paper account: starting balance plus everything realised so far."""
    realised = (
        conn.execute(
            sa.select(sa.func.coalesce(sa.func.sum(trades_t.c.pnl), 0.0)).where(
                sa.and_(trades_t.c.strategy == "fade", trades_t.c.status != "open")
            )
        ).scalar()
        or 0.0
    )
    open_rows = conn.execute(
        sa.select(trades_t.c.side, trades_t.c.entry_price, trades_t.c.size).where(
            sa.and_(trades_t.c.strategy == "fade", trades_t.c.status == "open")
        )
    ).all()
    committed = sum(
        trade_capital(r.side, r.entry_price, r.size or STAKE) for r in open_rows
    )
    balance = STARTING_BALANCE + float(realised)
    return {
        "starting_balance": STARTING_BALANCE,
        "realised_pnl": round(float(realised), 2),
        "balance": round(balance, 2),
        "committed": round(committed, 2),
        "free": round(balance - committed, 2),
        "open_positions": len(open_rows),
    }


def tick(notify=None) -> dict:
    """One engine step: manage open episodes, then open new ones."""
    engine = get_engine()
    opened: list[dict] = []
    with engine.begin() as conn:
        closed = manage_open(conn)

        free_capital = account(conn)["free"]
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
            side = "sell_yes" if direction < 0 else "buy_yes"
            capital = trade_capital(side, entry)
            if capital > free_capital:
                log.info(
                    "fade signal on %s skipped: needs $%.2f, $%.2f free",
                    signal.market_id, capital, free_capital,
                )
                continue
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
                "breadth": market_breadth(conn, signal.ts),
                "capital": capital,
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
                    side=side,
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
            # Several signals can fire in one tick, so the capital each takes
            # has to come off the running total before the next is considered.
            free_capital -= capital

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
