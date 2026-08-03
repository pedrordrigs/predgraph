"""Storage schema.

SQLite locally, Postgres in the cloud, one DSN apart. All timestamps are naive
UTC. Five tables, and nothing that is not read by the running bot: the graph,
ontology and news tables were removed once the multi-hop thesis was measured
and rejected.
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
    MetaData,
    String,
    Table,
    Text,
)
from sqlalchemy.engine import Engine

from snapback.config import get_settings


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


metadata = MetaData()

# --- what we watch and what it costs ---------------------------------------

markets = Table(
    "markets",
    metadata,
    # Venue-prefixed, e.g. "poly:0xabc..." / "kalshi:KXFED-26SEP".
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

history_bars = Table(
    "history_bars",
    metadata,
    Column("market_id", String(160), primary_key=True),
    Column("ts", DateTime, primary_key=True),
    # Kept apart from live bars: backfilled venue history, not what we saw.
    # Resolution is part of the key so a coarse sweep and a fine event window
    # can coexist for the same market. Local research only.
    Column("resolution_min", Integer, primary_key=True),
    Column("mid", Float),
    Column("bid", Float),
    Column("ask", Float),
)

# --- what the engine decided ------------------------------------------------

alerts = Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("market_id", String(160), ForeignKey("markets.id"), nullable=False, index=True),
    Column("ts", DateTime, nullable=False, index=True),
    Column("strategy", String(24), nullable=False, index=True),
    # Jump size, velocity, sigma, breadth, capital. Kept so a losing run can be
    # sliced by what triggered it, not only by what it returned.
    Column("detail", JSON, default=dict),
    Column("thesis", Text),
)

paper_trades = Table(
    "paper_trades",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("alert_id", Integer, ForeignKey("alerts.id"), index=True),
    Column("market_id", String(160), ForeignKey("markets.id"), nullable=False),
    # One row per rule set, so two rules can trade the same tape independently.
    Column("strategy", String(24), nullable=False, default="fade", index=True),
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


@lru_cache
def get_engine() -> Engine:
    url = get_settings().resolved_db_url()
    if url.startswith("postgresql"):
        # Supabase's session pooler allows 15 clients across everything we run,
        # and SQLAlchemy's default pool reserves up to 15 per engine on its own
        # (pool_size 5 + max_overflow 10). With the collector and one serverless
        # instance per concurrent request all holding pools, the cap is reached
        # and connections start failing with EMAXCONNSESSION.
        #
        # NullPool opens a connection per checkout and closes it after: the
        # right shape for short-lived functions, and cheap enough for a
        # collector that connects a few times a minute. The pooling that
        # matters is Supavisor's, on the far side.
        # psycopg promotes a repeated query to a prepared statement, which a
        # transaction-mode pooler cannot honour because consecutive statements
        # may land on different backends. Disabling that costs nothing at this
        # query volume and means the 6543 port works if the session pooler's
        # 15-client ceiling ever needs escaping.
        return sa.create_engine(
            url,
            future=True,
            poolclass=sa.pool.NullPool,
            connect_args={"prepare_threshold": None},
        )

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
    inspector = sa.inspect(engine)
    applied: list[str] = []
    present = set(inspector.get_table_names())
    with engine.begin() as conn:
        for table in metadata.sorted_tables:
            if table.name not in present:
                continue  # table itself is new; create_all handles it
            existing = {col["name"] for col in inspector.get_columns(table.name)}
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
    migrate()
    return str(engine.url)
