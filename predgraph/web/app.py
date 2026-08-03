"""Read-only dashboard.

Deliberately read-only: the collector is a separate process, so a browser tab
can never leave the system in a half-started state. Everything here answers
"what is it doing right now" and "what would move if X happened".
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import sqlalchemy as sa
from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from predgraph.db import edges as edges_t
from predgraph.db import get_engine
from predgraph.db import history_bars as hist_t
from predgraph.db import kv as kv_t
from predgraph.db import market_bars as bars_t
from predgraph.db import markets as markets_t
from predgraph.db import nodes as nodes_t
from predgraph.db import paper_trades as trades_t
from predgraph.graph.algo import market_labels, propagate

STATIC = Path(__file__).parent / "static"

app = FastAPI(title="PredGraph", docs_url=None, redoc_url=None)

# Once this is deployed the URL is reachable by anyone who guesses it. Setting
# PREDGRAPH_DASHBOARD_TOKEN gates every route behind `?k=<token>`; leaving it
# unset keeps local runs frictionless. The token is a shared secret, not an
# auth system - it only has to stop a stranger stumbling into the numbers.
_TOKEN = os.environ.get("PREDGRAPH_DASHBOARD_TOKEN", "").strip()


@app.middleware("http")
async def _gate(request: Request, call_next):
    if _TOKEN and request.url.path != "/api/health":
        supplied = request.query_params.get("k") or request.headers.get("x-predgraph-key")
        if supplied != _TOKEN:
            return PlainTextResponse("unauthorised", status_code=401)
    return await call_next(request)


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


@app.get("/api/trades")
def trades(limit: int = Query(200, le=1000)) -> JSONResponse:
    """Open and closed paper positions, newest first."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sa.select(
                trades_t.c.id,
                trades_t.c.market_id,
                trades_t.c.strategy,
                trades_t.c.side,
                trades_t.c.entry_ts,
                trades_t.c.entry_mid,
                trades_t.c.entry_price,
                trades_t.c.exit_ts,
                trades_t.c.exit_price,
                trades_t.c.pnl,
                trades_t.c.size,
                trades_t.c.status,
                trades_t.c.thesis,
                trades_t.c.meta,
                markets_t.c.question,
                markets_t.c.venue,
            )
            .select_from(trades_t.join(markets_t, trades_t.c.market_id == markets_t.c.id))
            .order_by(trades_t.c.entry_ts.desc())
            .limit(limit)
        ).all()

        # Mark open positions to market so unrealised PnL is visible.
        latest = {}
        open_ids = [r.market_id for r in rows if r.status == "open"]
        if open_ids:
            sub = (
                sa.select(bars_t.c.market_id, sa.func.max(bars_t.c.ts).label("ts"))
                .where(bars_t.c.market_id.in_(open_ids))
                .group_by(bars_t.c.market_id)
                .subquery()
            )
            for row in conn.execute(
                sa.select(bars_t.c.market_id, bars_t.c.mid, bars_t.c.bid, bars_t.c.ask).join(
                    sub,
                    sa.and_(
                        bars_t.c.market_id == sub.c.market_id, bars_t.c.ts == sub.c.ts
                    ),
                )
            ):
                latest[row.market_id] = row

    out = []
    for row in rows:
        record = {
            "id": row.id,
            "market_id": row.market_id,
            "question": row.question,
            "venue": row.venue,
            "strategy": row.strategy,
            "side": row.side,
            "entry_ts": row.entry_ts.isoformat() if row.entry_ts else None,
            "entry_price": row.entry_price,
            "exit_ts": row.exit_ts.isoformat() if row.exit_ts else None,
            "exit_price": row.exit_price,
            "pnl": row.pnl,
            "size": row.size,
            "status": row.status,
            "thesis": row.thesis,
            "meta": row.meta or {},
        }
        if row.status == "open" and row.market_id in latest:
            quote = latest[row.market_id]
            direction = -1 if row.side == "sell_yes" else 1
            close_at = quote.ask if direction < 0 else quote.bid
            if close_at is not None and row.entry_price is not None:
                record["mark_price"] = close_at
                record["unrealised"] = round(
                    direction * (close_at - row.entry_price) * (row.size or 100.0), 2
                )
        out.append(record)
    return JSONResponse(out)


@app.get("/api/strategies")
def strategies() -> JSONResponse:
    """The rule sets currently trading, so the UI is not hardcoded to two."""
    from predgraph.signal.engine import STRATEGIES

    return JSONResponse([
        {
            "name": r.name, "label": r.label,
            "min_jump_logit": r.min_jump_logit, "max_velocity_min": r.max_velocity_min,
            "price_lo": r.price_lo, "price_hi": r.price_hi, "lockout_h": r.lockout_h,
            "max_open": r.max_open, "max_per_day": r.max_per_day,
        }
        for r in STRATEGIES
    ])


@app.get("/api/performance")
def performance(strategy: str = Query("fade")) -> JSONResponse:
    """Equity curve and headline stats for the paper ledger."""
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sa.select(
                trades_t.c.exit_ts,
                trades_t.c.entry_ts,
                trades_t.c.pnl,
                trades_t.c.status,
                trades_t.c.meta,
                trades_t.c.size,
            ).where(trades_t.c.strategy == strategy)
        ).all()

    closed = sorted(
        [r for r in rows if r.pnl is not None and r.exit_ts is not None],
        key=lambda r: r.exit_ts,
    )
    # The curve is the account balance over time, so it opens at the starting
    # balance rather than zero; `total_pnl` below stays the change from it.
    from predgraph.signal.engine import STARTING_BALANCE, account

    equity, curve, peak, max_dd = 0.0, [], 0.0, 0.0
    curve.append({"ts": None, "equity": STARTING_BALANCE})
    for row in closed:
        equity += row.pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        curve.append(
            {"ts": row.exit_ts.isoformat(), "equity": round(STARTING_BALANCE + equity, 2)}
        )

    with get_engine().connect() as conn:
        acct = account(conn, strategy)

    wins = [r.pnl for r in closed if r.pnl > 0]
    losses = [r.pnl for r in closed if r.pnl <= 0]
    by_reason: dict[str, int] = {}
    for row in closed:
        by_reason[row.status.replace("closed_", "")] = (
            by_reason.get(row.status.replace("closed_", ""), 0) + 1
        )

    # Does breadth predict anything? The question the live ledger exists to settle.
    breadth_split: dict[str, dict] = {}
    for label, test in (
        ("isolated", lambda b: b == 0),
        ("clustered", lambda b: b >= 1),
    ):
        subset = [r.pnl for r in closed if test((r.meta or {}).get("breadth", 0))]
        if subset:
            breadth_split[label] = {
                "n": len(subset),
                "mean": round(sum(subset) / len(subset), 2),
                "win_pct": round(100.0 * sum(1 for p in subset if p > 0) / len(subset), 1),
            }

    return JSONResponse(
        {
            "curve": curve,
            "account": acct,
            "open_count": sum(1 for r in rows if r.status == "open"),
            "closed_count": len(closed),
            "total_pnl": round(equity, 2),
            "win_pct": round(100.0 * len(wins) / len(closed), 1) if closed else None,
            "mean_pnl": round(equity / len(closed), 2) if closed else None,
            "avg_win": round(sum(wins) / len(wins), 2) if wins else None,
            "avg_loss": round(sum(losses) / len(losses), 2) if losses else None,
            "max_drawdown": round(max_dd, 2),
            "by_exit": by_reason,
            "breadth": breadth_split,
            # Graduation bar, fixed in advance: n>=40, mean>0, win>=55%.
            "graduation": {
                "n_needed": 40,
                "n_have": len(closed),
                "mean_ok": bool(closed and equity / len(closed) > 0),
                "win_ok": bool(closed and 100.0 * len(wins) / len(closed) >= 55.0),
            },
        }
    )


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


@app.get("/api/health")
def health() -> JSONResponse:
    """Liveness plus database reachability.

    This is the one route left ungated, so it is the first thing checked when
    something looks wrong. A flat `ok: true` would report healthy while the
    dashboard could not read a single row, which sends the next hour of
    debugging at the wrong layer. The error itself is not echoed back: this
    route is public, and the message would carry the database host.
    """
    try:
        with get_engine().connect() as conn:
            conn.execute(sa.select(sa.literal(1)))
        return JSONResponse({"ok": True, "db": True})
    except Exception as exc:  # noqa: BLE001 - every failure returns the same shape
        return JSONResponse(
            {"ok": False, "db": False, **_db_diagnosis(exc)}, status_code=503
        )


_UNESCAPED_HINT = (
    "The password contains a URL delimiter (@ : / ? #). Percent-encode it "
    "(@ becomes %40), or reset it to one without those characters."
)


def _db_diagnosis(exc: Exception) -> dict:
    """Classify a connection failure without echoing the connection string.

    This route is public, so the message itself can never be returned - it
    carries the host and sometimes the user. But "db: false" alone has twice
    cost a debugging cycle on a setting only the deployment can see, so the
    failure is reduced to a category and a fix instead.
    """
    from predgraph.config import get_settings

    text = str(exc).lower()

    # If the engine cannot even be constructed the fault is the string, not the
    # network. Report its shape - scheme and delimiter counts are not secrets,
    # and they distinguish the three ways this setting actually gets mistyped.
    try:
        raw = get_settings().db_url
    except Exception:  # noqa: BLE001
        raw = ""
    scheme = raw.split("://", 1)[0].lower() if "://" in raw else ""
    body = raw.split("://", 1)[1] if "://" in raw else raw
    # SQLAlchemy splits credentials on the *last* '@', so an unescaped '@' in
    # the password parses without complaint and silently yields a nonsense
    # host - which then fails as a DNS error rather than as the typo it is.
    unescaped = body.count("@") > 1

    host = ""
    try:
        host = get_engine().url.host or ""
    except Exception:  # noqa: BLE001 - a malformed URL is itself the answer
        if scheme in ("http", "https"):
            return {"reason": "web-address",
                    "hint": "This is a dashboard or REST URL, not a DSN. Copy the "
                            "postgresql:// connection string instead."}
        if not scheme:
            return {"reason": "no-scheme",
                    "hint": "PREDGRAPH_DB_URL has no scheme. It should start with "
                            "postgresql://."}
        if unescaped:
            return {"reason": "unescaped-password", "hint": _UNESCAPED_HINT}
        return {"reason": "malformed-url",
                "hint": f"Scheme is '{scheme}' but the DSN could not be parsed."}

    if unescaped:
        return {"reason": "unescaped-password", "hint": _UNESCAPED_HINT}
    if "getaddrinfo" in text or "could not translate" in text or "resolve" in text:
        # The trap that cost an evening: Supabase's direct host publishes only
        # an AAAA record, and most serverless runtimes are IPv4-only.
        if host.startswith("db.") and "supabase" in host:
            return {"reason": "dns-ipv6-only",
                    "hint": "Direct Supabase host is IPv6-only. Use the Session "
                            "pooler DSN (aws-N-<region>.pooler.supabase.com:5432, "
                            "user postgres.<ref>)."}
        return {"reason": "dns", "hint": "Database host does not resolve."}
    if "password authentication failed" in text or "auth" in text:
        return {"reason": "auth",
                "hint": "Password rejected. If it was rotated, update this "
                        "deployment and the collector secret together."}
    if "tenant or user not found" in text or "not found" in text:
        return {"reason": "wrong-tenant",
                "hint": "Pooler does not know this user. Check the region and "
                        "that the username is postgres.<project-ref>."}
    if "timeout" in text or "timed out" in text:
        return {"reason": "timeout", "hint": "Host reachable but not answering."}
    if "ssl" in text:
        return {"reason": "tls", "hint": "TLS negotiation failed."}
    return {"reason": type(exc).__name__,
            "hint": "Unrecognised connection failure; check the collector logs."}
