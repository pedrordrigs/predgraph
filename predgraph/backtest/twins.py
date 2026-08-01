"""Cross-venue twin pairs.

Kept out of the fade line but not deleted: the twin arbitrage analysis found
36/36 profitable episodes held to settlement, and that work should stay
reachable. The full twin/lag study code lives in git history.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from predgraph.config import REPO_ROOT


def load_twins(path: Path | None = None) -> list[dict]:
    file = path or (REPO_ROOT / "twins.yaml")
    if not file.exists():
        return []
    data = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
    return data.get("pairs", [])
