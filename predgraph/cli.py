from __future__ import annotations

import logging
import threading
import time
import webbrowser
from datetime import UTC, datetime

import sqlalchemy as sa
import typer
from apscheduler.schedulers.blocking import BlockingScheduler
from rich.console import Console
from rich.table import Table

from predgraph.backtest import history
from predgraph.backtest.lag_study import (
    DEFAULT_SOURCES,
    MATERIAL_LOGIT,
    build_cohorts,
    observe,
    positive_control,
    summarize,
    twin_control,
)
from predgraph.config import setup_logging
from predgraph.db import edges as edges_t
from predgraph.db import get_engine, init_db
from predgraph.db import history_bars as hist_t
from predgraph.db import kv as kv_t
from predgraph.db import market_bars as bars_t
from predgraph.db import markets as markets_t
from predgraph.db import nodes as nodes_t
from predgraph.db import paper_trades as trades_t
from predgraph.graph.algo import market_labels, propagate
from predgraph.ingest.runner import discover_and_link, poll_once, watched_markets
from predgraph.ontology import OntologyError, load_ontology, sync_to_db

app = typer.Typer(help="PredGraph — prediction-market intelligence graph", no_args_is_help=True)
db_app = typer.Typer(help="Database operations", no_args_is_help=True)
ontology_app = typer.Typer(help="Ontology operations", no_args_is_help=True)
markets_app = typer.Typer(help="Market discovery and polling", no_args_is_help=True)
graph_app = typer.Typer(help="Graph inspection and propagation", no_args_is_help=True)
backtest_app = typer.Typer(help="Historical study of repricing lag", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(ontology_app, name="ontology")
app.add_typer(markets_app, name="markets")
app.add_typer(graph_app, name="graph")
app.add_typer(backtest_app, name="backtest")

console = Console()


@db_app.command("init")
def db_init() -> None:
    """Create tables (idempotent)."""
    setup_logging()
    console.print(f"[green]schema ready[/green] {init_db()}")


@ontology_app.command("validate")
def ontology_validate() -> None:
    """Check axes, signs, weights and node references without touching the DB."""
    setup_logging()
    try:
        ontology = load_ontology()
    except OntologyError as exc:
        console.print(f"[red]invalid ontology[/red]: {exc}")
        raise typer.Exit(code=1)

    table = Table("domain", "nodes", "edges", "anchors", "kalshi series")
    for domain in ontology.domains:
        table.add_row(
            domain.domain,
            str(len(domain.nodes)),
            str(len(domain.edges)),
            str(len(domain.market_anchors)),
            str(len(domain.kalshi_series)),
        )
    console.print(table)
    console.print("[green]ontology valid[/green]")


@ontology_app.command("sync")
def ontology_sync() -> None:
    """Upsert ontology nodes and edges into the graph."""
    setup_logging()
    init_db()
    stats = sync_to_db(load_ontology())
    console.print(stats)


@markets_app.command("discover")
def markets_discover(
    pages: int = typer.Option(4, help="Polymarket pages (200 markets each)"),
    quarantine: bool = typer.Option(False, help="Record unmatched markets for the curator"),
) -> None:
    """Find markets, link them to driver nodes via ontology anchors."""
    setup_logging()
    init_db()
    ontology = load_ontology()
    sync_to_db(ontology)
    stats = discover_and_link(ontology, poly_pages=pages, quarantine_unmatched=quarantine)

    console.print(
        f"seen [bold]{stats['seen']}[/bold]  linked [bold green]{stats['linked']}[/bold green]  "
        f"watched [bold green]{stats['watched']}[/bold green]  unmatched {stats['unmatched']}"
    )
    if stats["by_anchor"]:
        table = Table("anchor", "watched")
        for anchor, count in sorted(stats["by_anchor"].items(), key=lambda kv: -kv[1]):
            table.add_row(anchor, str(count))
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
        drivers = {
            market_id: names
            for market_id, names in conn.execute(
                sa.select(edges_t.c.dst, sa.func.group_concat(edges_t.c.src))
                .where(edges_t.c.dst.in_([r.id for r in rows]) if rows else sa.false())
                .group_by(edges_t.c.dst)
            ).all()
        }

    table = Table("market", "venue", "question", "closes", "drivers")
    for row in rows:
        table.add_row(
            row.id[:34],
            row.venue,
            (row.question or "")[:52],
            row.close_time.strftime("%Y-%m-%d") if row.close_time else "-",
            (drivers.get(row.id) or "-")[:40],
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
        console.print("[yellow]no watched markets — run 'predgraph markets discover' first[/yellow]")
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


@graph_app.command("impact")
def graph_impact(
    node: str = typer.Argument(..., help="Source node id, e.g. military_escalation_me"),
    direction: int = typer.Option(1, help="+1 or -1 move on the source node's axis"),
    hops: int = typer.Option(3),
    watched_only: bool = typer.Option(True, "--watched/--all"),
    limit: int = typer.Option(15),
) -> None:
    """Which markets a move on this node should push, and in which direction."""
    setup_logging()
    impacts = propagate(node, direction=direction, max_hops=hops)
    market_ids = [i.target for i in impacts if i.target.startswith(("poly:", "kalshi:"))]
    markets = market_labels(market_ids)

    table = Table("effect", "market", "paths", "chain")
    shown = 0
    for impact in impacts:
        info = markets.get(impact.target)
        if info is None:
            continue
        if watched_only and not info["watch"]:
            continue
        arrow = "YES up" if impact.contribution > 0 else "YES down"
        flag = " [red](sign conflict)[/red]" if impact.sign_conflict else ""
        table.add_row(
            f"{impact.contribution:+.3f} {arrow}{flag}",
            (info["question"] or "")[:46],
            str(len(impact.paths)),
            impact.paths[0].describe()[:70] if impact.paths else "-",
        )
        shown += 1
        if shown >= limit:
            break

    console.print(table)
    if not shown:
        console.print("[yellow]no linked markets reachable from that node[/yellow]")


@graph_app.command("nodes")
def graph_nodes(kind: str = typer.Option("", help="Filter by kind")) -> None:
    """List ontology nodes and their axes."""
    setup_logging()
    engine = get_engine()
    query = sa.select(nodes_t.c.id, nodes_t.c.kind, nodes_t.c.axis_def).where(
        nodes_t.c.kind != "market"
    )
    if kind:
        query = query.where(nodes_t.c.kind == kind)
    with engine.connect() as conn:
        rows = conn.execute(query.order_by(nodes_t.c.kind, nodes_t.c.id)).all()
    table = Table("node", "kind", "axis")
    for row in rows:
        table.add_row(row.id, row.kind, (row.axis_def or "-")[:64])
    console.print(table)


def _study_markets(sources: list[str], max_responses: int) -> tuple[list, list[str]]:
    cohorts = build_cohorts(sources)
    ids: set[str] = set()
    for cohort in cohorts:
        ids.update(cohort.triggers)
        ids.update(list(cohort.responses)[:max_responses])
    return cohorts, sorted(ids)


@backtest_app.command("fetch")
def backtest_fetch(
    days: int = typer.Option(90, help="How far back to pull"),
    resolution: int = typer.Option(60, help="Minutes per bar"),
    max_responses: int = typer.Option(40, help="Response markets per source"),
    controls: int = typer.Option(60, help="Unconnected markets for the control cohort"),
) -> None:
    """Backfill venue price history for the lag study."""
    setup_logging()
    init_db()
    _, ids = _study_markets(list(DEFAULT_SOURCES), max_responses)

    engine = get_engine()
    with engine.connect() as conn:
        pool = [
            row.id
            for row in conn.execute(
                sa.select(markets_t.c.id).where(markets_t.c.id.notin_(ids)).limit(controls * 4)
            )
        ]
    control_ids = pool[:controls]
    targets = ids + control_ids

    console.print(f"fetching {len(targets)} markets ({len(ids)} in cohorts, {len(control_ids)} control)")
    stats = history.backfill(targets, days=days, resolution_min=resolution)
    console.print(
        f"[green]done[/green]: {stats['bars']} bars over {stats['markets']} markets "
        f"({stats['empty']} returned nothing)"
    )


@backtest_app.command("lag")
def backtest_lag(
    resolution: int = typer.Option(60),
    z: float = typer.Option(3.0, help="Jump detection threshold in sigmas"),
    max_responses: int = typer.Option(40),
) -> None:
    """Run the lag study and print the verdict table."""
    setup_logging()
    cohorts, ids = _study_markets(list(DEFAULT_SOURCES), max_responses)
    if not cohorts:
        console.print("[red]no cohorts — run 'predgraph markets discover' first[/red]")
        raise typer.Exit(code=1)

    engine = get_engine()
    with engine.connect() as conn:
        control_pool = [
            row.market_id
            for row in conn.execute(
                sa.select(hist_t.c.market_id)
                .where(hist_t.c.market_id.notin_(ids))
                .group_by(hist_t.c.market_id)
            )
        ]

    # Gate first. Cross-venue twins are the calibration standard: the same claim
    # on two venues has no structural reason to disagree, so if this is low the
    # pipeline is broken and nothing below is worth reading.
    twins = twin_control(resolution_min=resolution, z_threshold=z)
    gate = Table("cross-venue twin control", "agreement", "n", "markets")
    if twins and twins.get("agreement_pct") is not None:
        gate.add_row(
            "same claim, both venues",
            f"{twins['agreement_pct']}%",
            str(twins["n"]),
            str(twins["markets"]),
        )
    else:
        gate.add_row("same claim, both venues", "no data", "0", "0")
    console.print(gate)

    twin_score = twins.get("agreement_pct") if twins else None
    trustworthy = twin_score is not None and twin_score >= 85.0
    if trustworthy:
        console.print(f"[green]instrument OK[/green]: twin agreement {twin_score}%\n")
    elif twin_score is None:
        console.print(
            "[yellow]no twin observations[/yellow] — backfill both sides of twins.yaml "
            "before trusting the table below.\n"
        )
    else:
        console.print(
            f"[red]instrument not trustworthy[/red]: twin agreement {twin_score}%, "
            "expected >=85% for the same claim on two venues. Treat the table below "
            "as diagnostics, not a verdict.\n"
        )

    # Secondary diagnostic: peers on the same driver, split by whether they are
    # the same ladder (mechanically linked, ceiling <100%) or only graph-linked.
    control = positive_control(cohorts, resolution_min=resolution, z_threshold=z)
    if control:
        peers = Table("peer agreement (material moves)", "agreement", "n")
        for source, row in sorted(control.items(), key=lambda kv: -kv[1]["agreement_pct"]):
            peers.add_row(source, f"{row['agreement_pct']}%", str(row["n"]))
        console.print(peers)
        console.print()

    observations = observe(
        cohorts, resolution_min=resolution, z_threshold=z, max_responses=max_responses,
        control_pool=control_pool,
    )
    summary = summarize(observations)
    if not summary:
        console.print("[yellow]no observations — check that history was fetched[/yellow]")
        raise typer.Exit(code=1)

    triggers = len({(o.trigger_id, o.jump_ts) for o in observations})
    with engine.begin() as conn:
        payload = {
            "summary": summary,
            "observations": len(observations),
            "trigger_jumps": triggers,
            "sources": len(cohorts),
            "positive_control": control,
            "twin_control": twins,
            "material_floor": MATERIAL_LOGIT,
            "trustworthy": trustworthy,
        }
        exists = conn.execute(sa.select(kv_t.c.key).where(kv_t.c.key == "lag_study")).first()
        if exists:
            conn.execute(kv_t.update().where(kv_t.c.key == "lag_study").values(value=payload))
        else:
            conn.execute(kv_t.insert().values(key="lag_study", value=payload))

    console.print(
        f"\n[bold]{len(observations)} observations[/bold] from {triggers} trigger jumps "
        f"across {len(cohorts)} sources\n"
    )

    horizons = (1.0, 4.0, 24.0, 48.0)
    table = Table(
        "cohort",
        "n",
        *[f"hit {h:g}h" for h in horizons],
        "n material",
        "corr 4h",
        "t½ (h)",
    )
    for key, row in summary.items():
        table.add_row(
            key,
            str(row["n"]),
            *[
                f"{row[f'mhit_{h:g}h']}%" if row[f"mhit_{h:g}h"] is not None else "-"
                for h in horizons
            ],
            str(row["mn_4h"]),
            str(row["magnitude_corr_4h"] if row["magnitude_corr_4h"] is not None else "-"),
            str(row["median_time_to_half_h"] or "-"),
        )
    console.print(table)
    console.print(
        f"\n[dim]hit % = share of responses that moved at least {MATERIAL_LOGIT} logits and did so\n"
        "in the direction the signed path predicted. Sub-threshold moves are excluded: their\n"
        "sign is a coin flip and including them drags every cohort toward 50%.\n"
        "Control is unconnected markets with a random predicted sign and should sit near 50%.\n"
        "corr 4h = correlation between predicted effect size and realized move — sign agreement\n"
        "can pass on noise, this cannot. t half = median hours to complete half the 48h move.[/dim]"
    )


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
    logger = logging.getLogger("predgraph.run")

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
        from predgraph.signal import engine as fade_engine

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
        stats = discover_and_link(load_ontology())
        logger.info(
            "discover: %d seen, %d linked, %d watched", stats["seen"], stats["linked"],
            stats["watched"],
        )

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


@backtest_app.command("minute")
def backtest_minute(
    jumps_per_source: int = typer.Option(40, help="Strongest jumps per source"),
    fetch: bool = typer.Option(True, "--fetch/--no-fetch", help="Fetch missing 1-min windows"),
) -> None:
    """1-minute closure study: sub-hour diffusion + twin lead-lag."""
    setup_logging()
    from predgraph.backtest.lag_study import JUMP_MIN_ABS_LOGIT, _dedupe_jumps
    from predgraph.backtest.minute_study import (
        fetch_planned,
        measure,
        plan_fetch,
        summarize_minutes,
        twin_lead_lag,
    )
    from predgraph.signal.damage import detect_jumps

    cohorts = build_cohorts(list(DEFAULT_SOURCES))
    from predgraph.backtest.lag_study import bar_counts as _bc
    from predgraph.backtest.lag_study import ladder_keys as _lk

    ladder, counts = _lk(), _bc(60)
    jumps_by_source = {}
    for cohort in cohorts:
        triggers = [t for t in cohort.triggers if counts.get(t, 0) >= 200]
        raw = [
            (t, j)
            for t in triggers
            for j in detect_jumps(
                history.load_series(t, 60), z_threshold=3.0, min_abs_logit=JUMP_MIN_ABS_LOGIT
            )
        ]
        deduped = _dedupe_jumps(raw, ladder)
        deduped.sort(key=lambda x: -abs(x[1].z))
        jumps_by_source[cohort.source] = deduped[:jumps_per_source]

    if fetch:
        plan = plan_fetch(cohorts, jumps_by_source)
        console.print(f"fetching {sum(len(w) for w in plan.values())} windows over {len(plan)} markets")
        console.print(fetch_planned(plan))

    observations = measure(cohorts, jumps_by_source)
    summary = summarize_minutes(observations)
    table = Table("horizon", "real: mean signed", "real hit%", "real n(mat)", "placebo: mean", "placebo hit%")
    for horizon, row in summary.items():
        real, placebo = row["real"], row["placebo"]
        table.add_row(
            f"+{horizon}m",
            str(real["mean_signed"]),
            f"{real['hit_material']}%" if real["hit_material"] is not None else "-",
            f"{real['n_material']}/{real['n']}",
            str(placebo["mean_signed"]),
            f"{placebo['hit_material']}%" if placebo["hit_material"] is not None else "-",
        )
    console.print(table)

    twins = twin_lead_lag()
    if twins:
        tt = Table("twin", "xcorr peak (lag min)", "poly→kalshi (med min, n)", "kalshi→poly (med min, n)")
        for row in twins:
            tt.add_row(
                row["note"][:44],
                f"{row['xcorr_peak']} @ {row['xcorr_peak_lag_min']:+d}",
                f"{row['a_to_b_median_min']} ({row['a_to_b_n']})",
                f"{row['b_to_a_median_min']} ({row['b_to_a_n']})",
            )
        console.print(tt)


@app.command("engine")
def engine_cmd(
    once: bool = typer.Option(True, "--once/--loop"),
    interval: int = typer.Option(60),
) -> None:
    """Run the fade engine against current bars (the collector does this too)."""
    setup_logging()
    init_db()
    from predgraph.signal import engine as fade_engine

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
    from predgraph.signal.engine import ledger_summary

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
    uvicorn.run("predgraph.web.app:app", host=host, port=port, log_level="warning")


@app.command("status")
def status() -> None:
    """Row counts and bar coverage."""
    setup_logging()
    engine = get_engine()
    with engine.connect() as conn:
        counts = {
            "nodes": conn.execute(sa.select(sa.func.count()).select_from(nodes_t)).scalar(),
            "edges": conn.execute(sa.select(sa.func.count()).select_from(edges_t)).scalar(),
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
