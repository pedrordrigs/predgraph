from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

import sqlalchemy as sa
import typer
from apscheduler.schedulers.blocking import BlockingScheduler
from rich.console import Console
from rich.table import Table

from predgraph.config import setup_logging
from predgraph.db import edges as edges_t
from predgraph.db import get_engine, init_db
from predgraph.db import market_bars as bars_t
from predgraph.db import markets as markets_t
from predgraph.db import nodes as nodes_t
from predgraph.graph.algo import market_labels, propagate
from predgraph.ingest.runner import discover_and_link, poll_once, watched_markets
from predgraph.ontology import OntologyError, load_ontology, sync_to_db

app = typer.Typer(help="PredGraph — prediction-market intelligence graph", no_args_is_help=True)
db_app = typer.Typer(help="Database operations", no_args_is_help=True)
ontology_app = typer.Typer(help="Ontology operations", no_args_is_help=True)
markets_app = typer.Typer(help="Market discovery and polling", no_args_is_help=True)
graph_app = typer.Typer(help="Graph inspection and propagation", no_args_is_help=True)
app.add_typer(db_app, name="db")
app.add_typer(ontology_app, name="ontology")
app.add_typer(markets_app, name="markets")
app.add_typer(graph_app, name="graph")

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
