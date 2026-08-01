"""Vercel entrypoint.

Vercel's Python runtime looks for an ASGI `app`, so the dashboard is served
straight from the existing FastAPI application. This process is read-only by
design: it renders whatever the collector has written to Postgres and never
polls a venue or opens a position itself. A serverless function lives for a few
hundred milliseconds, so it could not hold a position even if asked to.
"""

import sys
from pathlib import Path

# The function's working directory is not guaranteed to be the repo root, and
# `predgraph` is a sibling of this file rather than an installed distribution.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from predgraph.config import get_settings

if get_settings().db_url.startswith("sqlite"):
    # Serverless filesystems are read-only and per-invocation, so a SQLite URL
    # here means PREDGRAPH_DB_URL was never set. Saying so beats the mkdir
    # permission error the default path would raise three frames deeper.
    raise RuntimeError(
        "PREDGRAPH_DB_URL is unset or points at SQLite. The deployed dashboard "
        "needs the hosted Postgres URL the collector writes to."
    )

from predgraph.web.app import app

__all__ = ["app"]
