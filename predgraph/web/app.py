"""Read-only dashboard.

Deliberately read-only: the collector is a separate process, so a browser tab
can never leave the system in a half-started state. Everything here answers
"what is it doing right now" and "what would move if X happened".
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import sqlalchemy as sa
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse

from predgraph.db import edges as edges_t
from predgraph.db import get_engine
from predgraph.db import history_bars as hist_t
from predgraph.db import kv as kv_t
from predgraph.db import market_bars as bars_t
from predgraph.db import markets as markets_t
from predgraph.db import nodes as nodes_t
from predgraph.graph.algo import market_labels, propagate

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="PredGraph", docs_url=None, redoc_url=None)


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status() -> JSONResponse:
    engine = get_engine()
    with engine.connect() as conn:
        counts = {
            "nodes": conn.execute(sa.select(sa.func.count()).select_from(nodes_t)).scalar(),
            "edges": conn.execute(sa.select(sa.func.count()).select_from(edges_t)).scalar(),
            "markets": conn.execute(sa.select(sa.func.count()).select_from(markets_t)).scalar(),
            "watched": conn.execute(
                sa.select(sa.func.count())
                .select_from(markets_t)
                .where(markets_t.c.watch.is_(True))
            ).scalar(),
            "bars": conn.execute(sa.select(sa.func.count()).select_from(bars_t)).scalar(),
            "history_bars": conn.execute(
                sa.select(sa.func.count()).select_from(hist_t)
            ).scalar(),
        }
        last_bar = conn.execute(sa.select(sa.func.max(bars_t.c.ts))).scalar()
        bars_last_hour = conn.execute(
            sa.select(sa.func.count())
            .select_from(bars_t)
            .where(bars_t.c.ts > _now() - timedelta(hours=1))
        ).scalar()

    age_seconds = (_now() - last_bar).total_seconds() if last_bar else None
    return JSONResponse(
        {
            **counts,
            "last_bar": last_bar.isoformat() if last_bar else None,
            "last_bar_age_s": age_seconds,
            # The collector polls every 60s; 5 minutes of silence means it died.
            "collector_live": age_seconds is not None and age_seconds < 300,
            "bars_last_hour": bars_last_hour,
        }
    )


@app.get("/api/markets")
def markets(limit: int = Query(80, le=500)) -> JSONResponse:
    engine = get_engine()
    with engine.connect() as conn:
        latest = (
            sa.select(bars_t.c.market_id, sa.func.max(bars_t.c.ts).label("ts"))
            .group_by(bars_t.c.market_id)
            .subquery()
        )
        rows = conn.execute(
            sa.select(
                markets_t.c.id,
                markets_t.c.venue,
                markets_t.c.question,
                markets_t.c.close_time,
                bars_t.c.mid,
                bars_t.c.spread,
                bars_t.c.depth_2c,
                bars_t.c.ts,
            )
            .select_from(
                markets_t.join(latest, latest.c.market_id == markets_t.c.id).join(
                    bars_t,
                    sa.and_(
                        bars_t.c.market_id == latest.c.market_id, bars_t.c.ts == latest.c.ts
                    ),
                )
            )
            .where(markets_t.c.watch.is_(True))
            .order_by(bars_t.c.depth_2c.desc())
            .limit(limit)
        ).all()

        drivers: dict[str, list[str]] = {}
        for row in conn.execute(
            sa.select(edges_t.c.dst, edges_t.c.src, edges_t.c.sign).where(
                edges_t.c.dst.in_([r.id for r in rows]) if rows else sa.false()
            )
        ):
            drivers.setdefault(row.dst, []).append(
                f"{'+' if row.sign > 0 else '-'}{row.src}"
            )

    return JSONResponse(
        [
            {
                "id": row.id,
                "venue": row.venue,
                "question": row.question,
                "mid": row.mid,
                "spread": row.spread,
                "depth": row.depth_2c,
                "close_time": row.close_time.isoformat() if row.close_time else None,
                "updated": row.ts.isoformat() if row.ts else None,
                "drivers": drivers.get(row.id, []),
            }
            for row in rows
        ]
    )


@app.get("/api/nodes")
def graph_nodes() -> JSONResponse:
    engine = get_engine()
    with engine.connect() as conn:
        # Latent states first: they are the ones that actually propagate, and an
        # alphabetical list opens on an entity node with no outgoing edges.
        kind_rank = sa.case(
            (nodes_t.c.kind == "latent", 0),
            (nodes_t.c.kind == "indicator", 1),
            (nodes_t.c.kind == "entity", 2),
            else_=3,
        )
        rows = conn.execute(
            sa.select(nodes_t.c.id, nodes_t.c.kind, nodes_t.c.label, nodes_t.c.axis_def)
            .where(sa.and_(nodes_t.c.kind != "market", nodes_t.c.status == "active"))
            .order_by(kind_rank, nodes_t.c.id)
        ).all()
    return JSONResponse(
        [
            {"id": r.id, "kind": r.kind, "label": r.label, "axis": r.axis_def}
            for r in rows
        ]
    )


@app.get("/api/impact")
def impact(
    node: str,
    direction: int = Query(1, ge=-1, le=1),
    watched_only: bool = True,
    limit: int = Query(40, le=200),
) -> JSONResponse:
    impacts = propagate(node, direction=direction or 1)
    market_ids = [i.target for i in impacts if i.target.startswith(("poly:", "kalshi:"))]
    labels = market_labels(market_ids)

    results = []
    for item in impacts:
        info = labels.get(item.target)
        if info is None or (watched_only and not info["watch"]):
            continue
        path = item.paths[0] if item.paths else None
        results.append(
            {
                "market": item.target,
                "question": info["question"],
                "venue": info["venue"],
                "contribution": round(item.contribution, 4),
                "direction": "YES up" if item.contribution > 0 else "YES down",
                "hops": path.hops if path else None,
                "chain": path.describe() if path else "",
                "sign_conflict": item.sign_conflict,
            }
        )
        if len(results) >= limit:
            break
    return JSONResponse(results)


@app.get("/api/study")
def study() -> JSONResponse:
    engine = get_engine()
    with engine.connect() as conn:
        row = conn.execute(sa.select(kv_t.c.value, kv_t.c.updated_at).where(kv_t.c.key == "lag_study")).first()
    if row is None:
        return JSONResponse({"available": False})
    payload = row.value if isinstance(row.value, dict) else json.loads(row.value)
    return JSONResponse(
        {"available": True, "updated_at": row.updated_at.isoformat(), **payload}
    )
