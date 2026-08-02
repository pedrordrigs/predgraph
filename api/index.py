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


try:
    app = _build_app()
except Exception as exc:  # noqa: BLE001 - the reason has to reach the browser
    # Raising here would surface only as FUNCTION_INVOCATION_FAILED, with the
    # cause buried in provider logs that need a dashboard login to read. A
    # misconfigured deployment should be diagnosable from the URL itself.
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    # Configuration errors are ours and safe to echo. Anything else could carry
    # a connection string, so only its type is reported.
    _detail = (
        str(exc)
        if isinstance(exc, ValueError | RuntimeError)
        else f"unexpected {type(exc).__name__} while starting up"
    )

    app = FastAPI(title="PredGraph (misconfigured)", docs_url=None, redoc_url=None)

    @app.get("/{_path:path}")
    def _misconfigured(_path: str) -> JSONResponse:
        return JSONResponse(
            {"ok": False, "error": "deployment not configured", "detail": _detail},
            status_code=503,
        )


__all__ = ["app"]
