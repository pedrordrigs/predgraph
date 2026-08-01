"""Storage schema.

v0 runs on SQLite (no Docker on this machine), but every table is defined with
portable SQLAlchemy Core types so moving to Postgres is a DSN change. The only
Postgres-specific thing we will want later is pgvector for the article/event
embeddings used by the M2 dedup cascade; until then embeddings are packed
float32 blobs and similarity is computed in-process.

All timestamps are naive UTC.
"""

from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache

import sqlalchemy as sa
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
)
from sqlalchemy.engine import Engine

from predgraph.config import get_settings


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


metadata = MetaData()

# --- graph -----------------------------------------------------------------

nodes = Table(
    "nodes",
    metadata,
    Column("id", String(160), primary_key=True),
    # market | latent | entity | indicator | event_calendar
    Column("kind", String(32), nullable=False, index=True),
    Column("label", String(256), nullable=False),
    # Required for latent/indicator nodes: "higher = <meaning>". Direction
    # composition along a path is meaningless without it.
    Column("axis_def", Text),
    Column("domain", String(64), index=True),
    # active | provisional | quarantined
    Column("status", String(16), nullable=False, default="active", index=True),
    Column("aliases", JSON, default=list),
    Column("embedding", LargeBinary),
    Column("meta", JSON, default=dict),
    Column("created_at", DateTime, nullable=False, default=utcnow),
    Column("updated_at", DateTime, nullable=False, default=utcnow, onupdate=utcnow),
)

edges = Table(
    "edges",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("src", String(160), ForeignKey("nodes.id"), nullable=False, index=True),
    Column("dst", String(160), ForeignKey("nodes.id"), nullable=False, index=True),
    # Mechanism sign: +1 means "src up drives dst up" on their declared axes.
    Column("sign", Integer, nullable=False),
    Column("weight", Float, nullable=False),
    Column("delay_h", Float, nullable=False, default=0.0),
    Column("halflife_h", Float, nullable=False, default=24.0),
    # structural | co_mention | statistical -- never mix these silently
    Column("edge_class", String(16), nullable=False, default="structural"),
    Column("mechanism", Text),
    Column("valid_from", DateTime),
    Column("valid_until", DateTime),
    Column("provenance", String(64), nullable=False, default="manual"),
    Column("created_at", DateTime, nullable=False, default=utcnow),
    UniqueConstraint("src", "dst", "mechanism", "valid_from", name="uq_edge_identity"),
)

# --- markets ---------------------------------------------------------------

markets = Table(
    "markets",
    metadata,
    # Same id as the graph node, e.g. "poly:0xabc..." / "kalshi:KXFED-26SEP".
    Column("id", String(160), primary_key=True),
    Column("venue", String(16), nullable=False, index=True),
    Column("venue_id", String(160), nullable=False),
    Column("question", Text, nullable=False),
    Column("slug", String(256)),
    Column("event_title", Text),
    Column("outcome", String(32), default="YES"),
    # Polymarket CLOB token id for the YES outcome; null on Kalshi.
    Column("token_id", String(160)),
    Column("status", String(24), default="open", index=True),
    Column("open_time", DateTime),
    Column("close_time", DateTime, index=True),
    Column("tags", JSON, default=list),
    Column("meta", JSON, default=dict),
    Column("watch", Boolean, nullable=False, default=False, index=True),
    Column("created_at", DateTime, nullable=False, default=utcnow),
    Column("updated_at", DateTime, nullable=False, default=utcnow, onupdate=utcnow),
)

market_bars = Table(
    "market_bars",
    metadata,
    Column("market_id", String(160), ForeignKey("markets.id"), primary_key=True),
    Column("ts", DateTime, primary_key=True),
    Column("mid", Float),
    Column("bid", Float),
    Column("ask", Float),
    Column("spread", Float),
    Column("last", Float),
    Column("volume", Float),
    Column("liquidity", Float),
    # Book depth within 2 cents of mid, in USD -- the tradeability gate.
    Column("depth_2c", Float),
)

# --- news ------------------------------------------------------------------

articles = Table(
    "articles",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("url", String(1024), nullable=False, unique=True),
    Column("source", String(160), index=True),
    Column("title", Text),
    Column("lead", Text),
    Column("lang", String(8)),
    Column("published_at", DateTime, index=True),
    Column("fetched_at", DateTime, nullable=False, default=utcnow),
    Column("simhash", String(32), index=True),
    Column("embedding", LargeBinary),
    Column("raw", JSON),
)

events = Table(
    "events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    # {nodes: [...], type: ..., direction: ...} -- hard constraint on merging
    Column("signature", JSON, nullable=False),
    Column("event_type", String(64), index=True),
    Column("headline", Text),
    # When the fact happened vs when it entered our feed. Clustering uses
    # first_seen; temporal reasoning uses event_time.
    Column("event_time", DateTime, nullable=False, index=True),
    Column("first_seen", DateTime, nullable=False),
    Column("last_update", DateTime),
    Column("source_count", Integer, nullable=False, default=1),
    Column("novelty", Float, default=1.0),
    Column("halflife_h", Float, default=12.0),
    Column("status", String(16), default="active"),
    Column("created_at", DateTime, nullable=False, default=utcnow),
)

event_mentions = Table(
    "event_mentions",
    metadata,
    Column("event_id", Integer, ForeignKey("events.id"), primary_key=True),
    Column("article_id", Integer, ForeignKey("articles.id"), primary_key=True),
)

event_impacts = Table(
    "event_impacts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("event_id", Integer, ForeignKey("events.id"), nullable=False, index=True),
    Column("node_id", String(160), ForeignKey("nodes.id"), nullable=False, index=True),
    # +1/-1 relative to the target node's declared axis, never "good/bad news"
    Column("direction", Integer, nullable=False),
    Column("magnitude", Float, nullable=False),
    Column("confidence", Float, nullable=False, default=0.5),
    Column("mechanism", Text),
)

# --- signal ----------------------------------------------------------------

pressure = Table(
    "pressure",
    metadata,
    Column("market_id", String(160), ForeignKey("markets.id"), primary_key=True),
    Column("ts", DateTime, primary_key=True),
    Column("r_signed", Float),
    Column("r_abs", Float),
    Column("r_pct", Float),
    Column("d_1h", Float),
    Column("d_6h", Float),
    Column("d_24h", Float),
    Column("d_pct", Float),
    # Paths disagree on direction -- surface it, never net it away silently.
    Column("sign_conflict", Boolean, default=False),
    Column("quadrant", String(8), index=True),
    Column("top_paths", JSON),
)

alerts = Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("market_id", String(160), ForeignKey("markets.id"), nullable=False, index=True),
    Column("ts", DateTime, nullable=False, index=True),
    Column("quadrant", String(8), nullable=False),
    Column("r_signed", Float),
    Column("r_pct", Float),
    Column("d_pct", Float),
    Column("event_ids", JSON),
    Column("judge", JSON),
    Column("delivered", Boolean, default=False),
    Column("delivered_at", DateTime),
)

paper_trades = Table(
    "paper_trades",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("alert_id", Integer, ForeignKey("alerts.id"), index=True),
    Column("market_id", String(160), ForeignKey("markets.id"), nullable=False),
    # fade | twin
    Column("strategy", String(16), nullable=False, default="fade", index=True),
    # Signal provenance: jump size, velocity, sigma, neighbour sympathy. Kept so
    # a losing run can be sliced by what triggered it, not just by outcome.
    Column("meta", JSON, default=dict),
    Column("side", String(8), nullable=False),
    Column("entry_ts", DateTime, nullable=False),
    Column("entry_mid", Float),
    # Filled at the worse side of the spread -- optimism here would be lying
    # to ourselves about the only number that matters.
    Column("entry_price", Float),
    Column("size", Float, default=100.0),
    Column("thesis", Text),
    Column("invalidation", Text),
    Column("window_h", Float),
    Column("exit_ts", DateTime),
    Column("exit_mid", Float),
    Column("exit_price", Float),
    Column("pnl", Float),
    Column("status", String(16), default="open", index=True),
)

# --- operations ------------------------------------------------------------

history_bars = Table(
    "history_bars",
    metadata,
    Column("market_id", String(160), primary_key=True),
    Column("ts", DateTime, primary_key=True),
    # Kept separate from live bars: backfilled venue history, not what we saw.
    # Resolution is part of the key so a coarse sweep and a fine event window
    # can coexist for the same market.
    Column("resolution_min", Integer, primary_key=True),
    Column("mid", Float),
    Column("bid", Float),
    Column("ask", Float),
)

snapshots = Table(
    "snapshots",
    metadata,
    Column("ts", DateTime, primary_key=True),
    Column("payload", JSON, nullable=False),
)

quarantine = Table(
    "quarantine",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    # node | edge | market | alias
    Column("kind", String(16), nullable=False, index=True),
    Column("payload", JSON, nullable=False),
    Column("rationale", Text),
    Column("source", String(64)),
    # pending | promoted | rejected | merged
    Column("status", String(16), nullable=False, default="pending", index=True),
    Column("created_at", DateTime, nullable=False, default=utcnow),
    Column("resolved_at", DateTime),
    Column("resolution", JSON),
)

curator_log = Table(
    "curator_log",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_date", DateTime, nullable=False, index=True),
    Column("actions", JSON),
    Column("report_md", Text),
)

kv = Table(
    "kv",
    metadata,
    Column("key", String(128), primary_key=True),
    Column("value", JSON),
    Column("updated_at", DateTime, nullable=False, default=utcnow, onupdate=utcnow),
)


@lru_cache
def get_engine() -> Engine:
    url = get_settings().resolved_db_url()
    engine = sa.create_engine(url, future=True)
    if url.startswith("sqlite"):

        @sa.event.listens_for(engine, "connect")
        def _sqlite_pragmas(dbapi_connection, _record):  # pragma: no cover - driver hook
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
            # The collector polls and rediscovers concurrently; without this a
            # write that collides with another just raises "database is locked".
            cursor.execute("PRAGMA busy_timeout=10000")
            cursor.close()

    return engine


def migrate() -> list[str]:
    """Add columns that exist in the model but not yet in the file.

    create_all() only creates missing *tables*, so a column added to a table
    that already exists is silently absent until something fails on it at
    runtime. This keeps a long-lived local database usable across upgrades
    without a migration framework.
    """
    engine = get_engine()
    applied: list[str] = []
    with engine.begin() as conn:
        for table in metadata.sorted_tables:
            existing = {
                row[1]
                for row in conn.exec_driver_sql(f"PRAGMA table_info({table.name})").fetchall()
            }
            if not existing:
                continue  # table itself is new; create_all handles it
            for column in table.columns:
                if column.name in existing:
                    continue
                col_type = column.type.compile(engine.dialect)
                conn.exec_driver_sql(
                    f"ALTER TABLE {table.name} ADD COLUMN {column.name} {col_type}"
                )
                applied.append(f"{table.name}.{column.name}")
    return applied


def init_db() -> str:
    engine = get_engine()
    metadata.create_all(engine)
    if str(engine.url).startswith("sqlite"):
        migrate()
    return str(engine.url)
