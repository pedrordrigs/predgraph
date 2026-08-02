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


def _build_app():
    from predgraph.config import get_settings

    if get_settings().db_url.startswith("sqlite"):
        # Serverless filesystems are read-only and per-invocation, so a SQLite
        # URL here means PREDGRAPH_DB_URL was never set on the deployment.
        raise RuntimeError(
            "PREDGRAPH_DB_URL is unset or points at SQLite. The deployed "
            "dashboard needs the hosted Postgres URL the collector writes to."
        )
    from predgraph.web.app import app

    return app


def _error_app(exc: Exception):
    """A stand-in that reports why startup failed.

    Letting the exception escape surfaces only as FUNCTION_INVOCATION_FAILED,
    with the cause in provider logs that need a dashboard login to read - and
    indistinguishable from a routing 404 while debugging.
    """
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    # Configuration errors are ours and safe to echo. Anything else could carry
    # a connection string, so only its type is reported.
    detail = (
        str(exc)
        if isinstance(exc, ValueError | RuntimeError)
        else f"unexpected {type(exc).__name__} while starting up"
    )
    fallback = FastAPI(title="PredGraph (misconfigured)", docs_url=None, redoc_url=None)

    @fallback.get("/{_path:path}")
    def _misconfigured(_path: str) -> JSONResponse:
        return JSONResponse(
            {"ok": False, "error": "deployment not configured", "detail": detail},
            status_code=503,
        )

    return fallback


def _resolve_app():
    try:
        return _build_app()
    except Exception as exc:  # noqa: BLE001 - the reason has to reach the browser
        return _error_app(exc)


# Must stay a plain top-level assignment. Vercel finds the ASGI entrypoint by
# static analysis, so an `app` bound inside a try/except is invisible to it and
# the build fails with PYTHON_ENTRYPOINT_NOT_FOUND.
app = _resolve_app()

__all__ = ["app"]
