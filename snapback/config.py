from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent

LEGACY_PREFIX = "PREDGRAPH_"
PREFIX = "SNAPBACK_"


def _adopt_legacy_env() -> None:
    """Accept the old prefix so a rename cannot strand a running deployment.

    The secret lives in GitHub and Vercel, not in this repo, so renaming the
    prefix here and the secret there can never be one atomic change. Reading
    both means the gap is invisible instead of an outage.
    """
    for key, value in list(os.environ.items()):
        if not key.startswith(LEGACY_PREFIX) or not value:
            continue
        current = PREFIX + key[len(LEGACY_PREFIX):]
        # Empty counts as absent, not as a decision. CI sets a missing secret
        # to the empty string, so `setdefault` would happily keep it and the
        # legacy value would never be consulted.
        if not os.environ.get(current):
            os.environ[current] = value


_adopt_legacy_env()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SNAPBACK_",
        env_file=REPO_ROOT / ".env",
        extra="ignore",
    )

    db_url: str = "sqlite:///data/snapback.db"
    dns_servers: str = "1.1.1.1,8.8.8.8"
    dns_override_suffixes: str = "polymarket.com,kalshi.com,kalshi.co"
    http_timeout: float = 20.0
    log_level: str = "INFO"

    @property
    def dns_server_list(self) -> list[str]:
        return [s.strip() for s in self.dns_servers.split(",") if s.strip()]

    @property
    def dns_suffix_list(self) -> list[str]:
        return [s.strip() for s in self.dns_override_suffixes.split(",") if s.strip()]


    def resolved_db_url(self) -> str:
        """Normalise the URL: repo-anchored sqlite paths, psycopg3 for Postgres."""
        # Providers show several URLs on the same page - a dashboard link, a
        # REST endpoint, and the actual DSN - and picking the wrong one
        # otherwise surfaces as "Can't load plugin: sqlalchemy.dialects:https"
        # from four frames down, which names neither the setting nor the fix.
        if self.db_url.startswith(("http://", "https://")):
            raise ValueError(
                f"SNAPBACK_DB_URL is a web address ({self.db_url.split('://')[0]}://...), "
                "not a database connection string. Copy the postgresql:// DSN "
                "from your provider, not the dashboard or REST endpoint URL."
            )

        # Hosted Postgres providers hand out `postgres://` or `postgresql://`,
        # both of which SQLAlchemy maps to psycopg2 - a driver we do not ship.
        # Rewriting the scheme here means the connection string can be pasted
        # from the provider dashboard straight into a secret, unedited.
        for scheme in ("postgres://", "postgresql://", "postgresql+psycopg://"):
            if self.db_url.startswith(scheme):
                url = "postgresql+psycopg://" + self.db_url[len(scheme) :]
                # psycopg defaults to sslmode=prefer, which silently accepts an
                # unencrypted link. The collector runs from CI, so the password
                # and every quote cross the public internet - require TLS unless
                # the DSN deliberately says otherwise.
                if "sslmode=" not in url:
                    url += ("&" if "?" in url else "?") + "sslmode=require"
                return url

        prefix = "sqlite:///"
        if not self.db_url.startswith(prefix):
            return self.db_url
        path = Path(self.db_url[len(prefix) :])
        if not path.is_absolute():
            path = REPO_ROOT / path
        path.parent.mkdir(parents=True, exist_ok=True)
        return prefix + str(path)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def setup_logging(level: str | None = None) -> None:
    logging.basicConfig(
        level=(level or get_settings().log_level).upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # One INFO line per request means 200 lines per poll and six figures of them
    # over a long collection run, which buries the signals and exits we actually
    # need to read. Failures still surface: errors raise rather than log.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
