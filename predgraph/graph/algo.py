"""Signed, bounded propagation over the temporal graph.

This is the R side of the signal in its structural form: given that some node
moved in a direction on its own axis, which markets should move, in which
direction, and how strongly. Three deliberate constraints keep it honest:

* hops are capped (default 3) — beyond that everything connects to everything;
* only the top-k strongest paths per target contribute, damped, instead of
  summing every path, which is what makes cycles double-count;
* when paths disagree on direction we flag the conflict instead of netting it
  to a confident-looking number. That ambiguity is a signal, not noise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import sqlalchemy as sa

from predgraph.db import edges as edges_t
from predgraph.db import get_engine
from predgraph.db import markets as markets_t
from predgraph.db import nodes as nodes_t

# 5 hops is set by the structure, not taste: the cross-domain chain we exist to
# catch (escalation -> oil supply -> crude -> inflation -> Fed stance -> market)
# lands at exactly 5. Measured on the live graph, 5 hops enumerates ~1.3k paths;
# 6 jumps to ~3.2k and starts connecting everything to everything.
MAX_HOPS = 5
TOP_K_PATHS = 3
PATH_DAMPING = 0.5


@dataclass(slots=True)
class Edge:
    src: str
    dst: str
    sign: int
    weight: float
    delay_h: float
    halflife_h: float
    mechanism: str | None


@dataclass(slots=True)
class Path:
    nodes: list[str]
    sign: int
    weight: float
    delay_h: float
    edges: list[Edge] = field(default_factory=list)

    @property
    def hops(self) -> int:
        return len(self.edges)

    def describe(self) -> str:
        parts = [self.nodes[0]]
        for edge in self.edges:
            parts.append(f"--[{'+' if edge.sign > 0 else '-'}{edge.weight:.2f}]-->{edge.dst}")
        return " ".join(parts)


@dataclass(slots=True)
class Impact:
    target: str
    contribution: float
    sign_conflict: bool
    paths: list[Path]


def load_adjacency(active_only: bool = True) -> dict[str, list[Edge]]:
    engine = get_engine()
    query = sa.select(
        edges_t.c.src,
        edges_t.c.dst,
        edges_t.c.sign,
        edges_t.c.weight,
        edges_t.c.delay_h,
        edges_t.c.halflife_h,
        edges_t.c.mechanism,
        edges_t.c.valid_until,
    )
    adjacency: dict[str, list[Edge]] = {}
    with engine.connect() as conn:
        for row in conn.execute(query):
            if active_only and row.valid_until is not None:
                continue
            adjacency.setdefault(row.src, []).append(
                Edge(
                    src=row.src,
                    dst=row.dst,
                    sign=int(row.sign),
                    weight=float(row.weight),
                    delay_h=float(row.delay_h),
                    halflife_h=float(row.halflife_h),
                    mechanism=row.mechanism,
                )
            )
    return adjacency


def enumerate_paths(
    source: str,
    adjacency: dict[str, list[Edge]],
    max_hops: int = MAX_HOPS,
) -> dict[str, list[Path]]:
    """All simple paths from `source` within `max_hops`, keyed by target."""
    found: dict[str, list[Path]] = {}

    def walk(node: str, visited: set[str], trail: list[Edge]) -> None:
        if len(trail) >= max_hops:
            return
        for edge in adjacency.get(node, []):
            if edge.dst in visited:
                continue  # simple paths only: a cycle would double-count
            new_trail = trail + [edge]
            sign = 1
            weight = 1.0
            delay = 0.0
            for step in new_trail:
                sign *= step.sign
                weight *= step.weight
                delay += step.delay_h
            path = Path(
                nodes=[source] + [step.dst for step in new_trail],
                sign=sign,
                weight=weight,
                delay_h=delay,
                edges=new_trail,
            )
            found.setdefault(edge.dst, []).append(path)
            walk(edge.dst, visited | {edge.dst}, new_trail)

    walk(source, {source}, [])
    return found


def propagate(
    source: str,
    direction: int = 1,
    magnitude: float = 1.0,
    max_hops: int = MAX_HOPS,
    top_k: int = TOP_K_PATHS,
    damping: float = PATH_DAMPING,
    adjacency: dict[str, list[Edge]] | None = None,
) -> list[Impact]:
    """Structural impact of `source` moving `direction` on its own axis."""
    adjacency = adjacency if adjacency is not None else load_adjacency()
    impacts: list[Impact] = []

    for target, paths in enumerate_paths(source, adjacency, max_hops).items():
        ranked = sorted(paths, key=lambda p: p.weight, reverse=True)[:top_k]
        contribution = sum(
            path.sign * path.weight * (damping**index) for index, path in enumerate(ranked)
        )
        contribution *= direction * magnitude
        signs = {path.sign for path in ranked}
        impacts.append(
            Impact(
                target=target,
                contribution=contribution,
                sign_conflict=len(signs) > 1,
                paths=ranked,
            )
        )

    impacts.sort(key=lambda impact: abs(impact.contribution), reverse=True)
    return impacts


def market_labels(ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sa.select(
                markets_t.c.id, markets_t.c.venue, markets_t.c.question, markets_t.c.watch
            ).where(markets_t.c.id.in_(ids))
        ).all()
    return {row.id: dict(row._mapping) for row in rows}


def node_labels(ids: list[str]) -> dict[str, str]:
    if not ids:
        return {}
    engine = get_engine()
    with engine.connect() as conn:
        rows = conn.execute(
            sa.select(nodes_t.c.id, nodes_t.c.label).where(nodes_t.c.id.in_(ids))
        ).all()
    return {row.id: row.label for row in rows}
