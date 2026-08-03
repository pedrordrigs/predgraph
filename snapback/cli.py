from __future__ import annotations

import logging
import threading
import time
import webbrowser
from datetime import UTC, datetime, timedelta

import sqlalchemy as sa
import typer
from apscheduler.schedulers.blocking import BlockingScheduler
from rich.console import Console
from rich.table import Table

from snapback.config import setup_logging
from snapback.db import get_engine, init_db
from snapback.db import market_bars as bars_t
from snapback.db import markets as markets_t
from snapback.db import paper_trades as trades_t
from snapback.ingest.runner import discover_fade_universe, poll_once, watched_markets

app = typer.Typer(help="Snapback - fading instantaneous overreactions in prediction markets", no_args_is_help=True)
db_app = typer.Typer(help="Database operations", no_args_is_help=True)
markets_app = typer.Typer(help="Market discovery and polling", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(markets_app, name="markets")

console = Console()


@db_app.command("init")
def db_init() -> None:
    """Create tables (idempotent)."""
    setup_logging()
    console.print(f"[green]schema ready[/green] {init_db()}")


@markets_app.command("discover")
def markets_discover(
    per_category: int = typer.Option(25, help="Markets kept per category"),
) -> None:
    """Build the fade watchlist: liquid, moving markets across categories."""
    setup_logging()
    init_db()
    stats = discover_fade_universe(per_category=per_category)
    console.print(
        f"seen [bold]{stats['seen']}[/bold]  watching [bold green]{stats['watched']}[/bold green]"
    )
    table = Table("category", "markets")
    for category, count in sorted(stats["by_category"].items(), key=lambda kv: -kv[1]):
        table.add_row(category, str(count))
    console.print(table)


@markets_app.command("list")
def markets_list(
    watched_only: bool = typer.Option(True, "--watched/--all"),
    limit: int = typer.Option(40),
) -> None:
    """Show linked markets and their driver edges."""
    setup_logging()
    engine = get_engine()
    query = sa.select(
        markets_t.c.id,
        markets_t.c.venue,
        markets_t.c.question,
        markets_t.c.close_time,
        markets_t.c.watch,
    )
    if watched_only:
        query = query.where(markets_t.c.watch.is_(True))
    query = query.order_by(markets_t.c.close_time).limit(limit)

    with engine.connect() as conn:
        rows = conn.execute(query).all()

    table = Table("market", "venue", "question", "closes")
    for row in rows:
        table.add_row(
            row.id[:34],
            row.venue,
            (row.question or "")[:60],
            row.close_time.strftime("%Y-%m-%d") if row.close_time else "-",
        )
    console.print(table)
    console.print(f"{len(rows)} market(s)")


@markets_app.command("poll")
def markets_poll(
    once: bool = typer.Option(False, "--once", help="Single pass then exit"),
    interval: int = typer.Option(60, help="Seconds between passes"),
) -> None:
    """Poll the live book for watched markets and write 1-minute bars."""
    setup_logging()
    init_db()
    markets = watched_markets()
    if not markets:
        console.print("[yellow]no watched markets — run 'snapback markets discover' first[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"polling {len(markets)} market(s); ctrl-c to stop")
    while True:
        started = time.monotonic()
        stats = poll_once(markets)
        console.print(
            f"quotes {stats['quotes']}/{stats['markets']}  written {stats['written']}  "
            f"({time.monotonic() - started:.1f}s)"
        )
        if once:
            break
        time.sleep(max(0.0, interval - (time.monotonic() - started)))


@app.command("run")
def run(
    poll_seconds: int = typer.Option(60, help="Seconds between bar polls"),
    discover_minutes: int = typer.Option(60, help="Minutes between rediscovery"),
) -> None:
    """Run the collector as a service: poll bars, rediscover periodically.

    Bars only accumulate while this is running, and the D side of the signal is
    worthless without an unbroken price history — so this is what should be
    running between sessions, not a hand-started poll loop.
    """
    setup_logging()
    init_db()
    logger = logging.getLogger("snapback.run")

    def poll_job() -> None:
        # Reloaded every tick so a rediscovery takes effect without a restart.
        markets = watched_markets()
        if not markets:
            logger.warning("no watched markets; waiting for discovery")
            return
        stats = poll_once(markets)
        logger.info(
            "poll: %d/%d quotes, %d bars written",
            stats["quotes"],
            stats["markets"],
            stats["written"],
        )
        # The engine runs on fresh bars, in the same tick, so a signal is acted
        # on within a minute of the move completing.
        engine_job()

    def engine_job() -> None:
        from snapback.signal import engine as fade_engine

        result = fade_engine.tick()
        for episode in result["episodes"]:
            console.print(
                f"[bold yellow]FADE[/bold yellow] {episode['question'][:60]} | "
                f"jump {episode['jump_logit']:+.2f} in {episode['velocity_min']:.0f}min"
            )
        for exit_ in result["exits"]:
            colour = "green" if (exit_["pnl"] or 0) > 0 else "red"
            console.print(
                f"[{colour}]EXIT[/{colour}] {exit_['market_id'][:34]} "
                f"{exit_['exit_reason']} pnl {exit_['pnl']:+.2f}"
            )
        if result["opened"] or result["closed"]:
            logger.info("engine: %d opened, %d closed", result["opened"], result["closed"])

    def discover_job() -> None:
        stats = discover_fade_universe()
        logger.info("discover: %d seen, %d watched", stats["seen"], stats["watched"])

    if not watched_markets():
        logger.info("no watchlist yet; running discovery first")
        discover_job()

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(
        poll_job, "interval", seconds=poll_seconds, id="poll",
        max_instances=1, coalesce=True, next_run_time=datetime.now(UTC),
    )
    scheduler.add_job(
        discover_job, "interval", minutes=discover_minutes, id="discover",
        max_instances=1, coalesce=True,
    )
    console.print(
        f"[green]collector running[/green]: bars every {poll_seconds}s, "
        f"rediscovery every {discover_minutes}min; ctrl-c to stop"
    )
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        console.print("stopped")


@app.command("engine")
def engine_cmd(
    once: bool = typer.Option(True, "--once/--loop"),
    interval: int = typer.Option(60),
) -> None:
    """Run the fade engine against current bars (the collector does this too)."""
    setup_logging()
    init_db()
    from snapback.signal import engine as fade_engine

    while True:
        result = fade_engine.tick()
        console.print(
            f"opened {result['opened']}, closed {result['closed']}"
            + ("" if result["episodes"] else "  [dim](no qualifying spikes)[/dim]")
        )
        for episode in result["episodes"]:
            console.print(
                f"  [yellow]FADE[/yellow] {episode['question'][:56]} "
                f"jump {episode['jump_logit']:+.2f} in {episode['velocity_min']:.0f}min"
            )
        if once:
            break
        time.sleep(interval)


@app.command("ledger")
def ledger(strategy: str = typer.Option("fade")) -> None:
    """Paper-trade results by strategy."""
    setup_logging()
    init_db()
    from snapback.signal.engine import ledger_summary

    summary = ledger_summary(strategy)
    table = Table("metric", "value")
    for key, value in summary.items():
        table.add_row(key, str(value))
    console.print(table)

    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sa.select(
                trades_t.c.market_id,
                trades_t.c.side,
                trades_t.c.entry_ts,
                trades_t.c.entry_price,
                trades_t.c.exit_price,
                trades_t.c.pnl,
                trades_t.c.status,
                trades_t.c.thesis,
            )
            .where(trades_t.c.strategy == strategy)
            .order_by(trades_t.c.entry_ts.desc())
            .limit(20)
        ).all()
    if rows:
        detail = Table("market", "side", "entry", "exit", "pnl", "status")
        for row in rows:
            detail.add_row(
                row.market_id[:30],
                row.side,
                f"{row.entry_price:.3f}" if row.entry_price else "-",
                f"{row.exit_price:.3f}" if row.exit_price else "-",
                f"{row.pnl:+.2f}" if row.pnl is not None else "-",
                row.status,
            )
        console.print(detail)


@app.command("prune")
def prune(
    keep_days: float = typer.Option(3.0, help="Days of quote bars to retain"),
) -> None:
    """Drop quote bars older than the engine can use.

    Measured against real rows: 322 bytes each, 288k rows a day, so ~93 MB a
    day. Three days is what fits a 500 MB free tier with room for bloat, and it
    still clears both limits that matter - 72 hourly points for a baseline that
    needs 20, and 72h of history against a 48h maximum hold. Trades and alerts
    are the actual results and are never pruned.
    """
    setup_logging()
    init_db()
    cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=keep_days)
    engine = get_engine()
    with engine.begin() as conn:
        before = conn.execute(sa.select(sa.func.count()).select_from(bars_t)).scalar() or 0
        deleted = conn.execute(bars_t.delete().where(bars_t.c.ts < cutoff)).rowcount
    console.print(
        f"[green]pruned[/green] {deleted:,} of {before:,} bars older than "
        f"{cutoff:%Y-%m-%d %H:%M}"
    )


@app.command("collect")
def collect(
    minutes: float = typer.Option(50.0, help="How long this run should keep polling"),
    interval: int = typer.Option(60, help="Seconds between polls"),
    discover: bool = typer.Option(False, "--discover", help="Refresh the watchlist first"),
) -> None:
    """One bounded collection run — the shape a CI scheduler can actually execute.

    Hosted schedulers fire a job, not a daemon, so instead of one process living
    forever this polls on a timer for a fixed window and exits cleanly. Chaining
    these back to back gives the same 60-second cadence the local collector had,
    which matters because entry speed is itself part of the edge.
    """
    setup_logging()
    init_db()
    logger = logging.getLogger("snapback.collect")
    from snapback.signal import engine as fade_engine

    if discover:
        stats = discover_fade_universe()
        logger.info("discover: %d seen, %d watched", stats["seen"], stats["watched"])

    deadline = time.monotonic() + minutes * 60.0
    polls = opened = closed = 0
    while time.monotonic() < deadline:
        started = time.monotonic()
        markets = watched_markets()
        if not markets:
            logger.warning("no watched markets; run discovery")
            break
        try:
            stats = poll_once(markets)
            result = fade_engine.tick()
            polls += 1
            opened += result["opened"]
            closed += result["closed"]
            for episode in result["episodes"]:
                logger.warning(
                    "FADE %s | jump %+.2f in %.0fmin | breadth %s",
                    episode["question"][:60],
                    episode["jump_logit"],
                    episode["velocity_min"],
                    episode.get("breadth"),
                )
            for exit_ in result["exits"]:
                logger.warning(
                    "EXIT %s %s pnl %+.2f",
                    exit_["market_id"][:40],
                    exit_["exit_reason"],
                    exit_["pnl"] or 0.0,
                )
            logger.info(
                "poll %d: %d/%d quotes, %d bars",
                polls, stats["quotes"], stats["markets"], stats["written"],
            )
        except Exception as exc:  # noqa: BLE001 - a bad tick must not end the run
            logger.warning("tick failed: %s", exc)
        elapsed = time.monotonic() - started
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(max(0.0, min(interval - elapsed, remaining)))

    console.print(
        f"[green]run complete[/green]: {polls} polls, {opened} opened, {closed} closed"
    )


@app.command("web")
def web(
    host: str = typer.Option("127.0.0.1", help="Bind address"),
    port: int = typer.Option(8765),
    open_browser: bool = typer.Option(True, "--open/--no-open"),
) -> None:
    """Serve the local dashboard."""
    setup_logging()
    init_db()
    import uvicorn

    url = f"http://{host}:{port}"
    if open_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    console.print(f"[green]dashboard[/green] {url}  (ctrl-c to stop)")
    uvicorn.run("snapback.web.app:app", host=host, port=port, log_level="warning")


@app.command("status")
def status() -> None:
    """Row counts and bar coverage."""
    setup_logging()
    engine = get_engine()
    with engine.connect() as conn:
        counts = {
            "markets": conn.execute(sa.select(sa.func.count()).select_from(markets_t)).scalar(),
            "watched": conn.execute(
                sa.select(sa.func.count()).select_from(markets_t).where(markets_t.c.watch.is_(True))
            ).scalar(),
            "bars": conn.execute(sa.select(sa.func.count()).select_from(bars_t)).scalar(),
        }
        span = conn.execute(
            sa.select(sa.func.min(bars_t.c.ts), sa.func.max(bars_t.c.ts))
        ).first()

    table = Table("metric", "value")
    for key, value in counts.items():
        table.add_row(key, str(value))
    if span and span[0]:
        table.add_row("bar span", f"{span[0]} -> {span[1]}")
    console.print(table)


if __name__ == "__main__":
    app()
