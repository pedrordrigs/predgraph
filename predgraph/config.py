from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PREDGRAPH_",
        env_file=REPO_ROOT / ".env",
        extra="ignore",
    )

    db_url: str = "sqlite:///data/predgraph.db"
    dns_servers: str = "1.1.1.1,8.8.8.8"
    dns_override_suffixes: str = "polymarket.com,kalshi.com,kalshi.co"
    http_timeout: float = 20.0
    log_level: str = "INFO"
    ontology_dir: str = "ontology"

    @property
    def dns_server_list(self) -> list[str]:
        return [s.strip() for s in self.dns_servers.split(",") if s.strip()]

    @property
    def dns_suffix_list(self) -> list[str]:
        return [s.strip() for s in self.dns_override_suffixes.split(",") if s.strip()]

    @property
    def ontology_path(self) -> Path:
        path = Path(self.ontology_dir)
        return path if path.is_absolute() else REPO_ROOT / path

    def resolved_db_url(self) -> str:
        """Anchor relative sqlite paths to the repo and create the parent dir."""
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
