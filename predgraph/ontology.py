"""Ontology loading and validation.

The graph is a small, curated, hand-maintained thing — that is the point. Every
latent/indicator node must declare an *axis* ("higher = ...") because direction
composition along a path is meaningless otherwise, and every edge must declare a
mechanism sign. Extraction later maps news onto these nodes; it never invents them.
"""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import sqlalchemy as sa
import yaml
from pydantic import BaseModel, Field, model_validator

from predgraph.config import get_settings
from predgraph.db import edges as edges_t
from predgraph.db import get_engine, utcnow
from predgraph.db import nodes as nodes_t

log = logging.getLogger(__name__)

AXIS_REQUIRED_KINDS = {"latent", "indicator"}


class NodeSpec(BaseModel):
    id: str
    kind: Literal["latent", "indicator", "entity", "event_calendar"]
    label: str
    axis: str | None = None
    aliases: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _axis_required(self) -> NodeSpec:
        if self.kind in AXIS_REQUIRED_KINDS and not self.axis:
            raise ValueError(
                f"node '{self.id}' is a {self.kind} and must declare an axis "
                '(e.g. axis: "higher = US price pressure rising")'
            )
        return self


class EdgeSpec(BaseModel):
    src: str
    dst: str
    sign: Literal[-1, 1]
    weight: float = Field(gt=0.0, le=1.0)
    delay_h: float = Field(default=0.0, ge=0.0)
    halflife_h: float = Field(default=24.0, gt=0.0)
    mechanism: str
    edge_class: Literal["structural", "co_mention", "statistical"] = "structural"


@lru_cache(maxsize=4096)
def _term_pattern(term: str) -> re.Pattern[str]:
    """Match a term on token boundaries rather than as a bare substring.

    Plain `in` matching is a trap here: a "uk" guard would exclude every
    market mentioning Ukraine, and "deal" would fire on "dealer". Boundaries
    are defined as any non-alphanumeric so "0bps", "u.s." and "CPI-U" work.
    """
    return re.compile(rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])")


def contains_term(text: str, term: str) -> bool:
    return _term_pattern(term).search(text.lower()) is not None


class MatchSpec(BaseModel):
    venue: Literal["polymarket", "kalshi", "any"] = "any"
    any_of: list[str] = Field(default_factory=list)
    all_of: list[str] = Field(default_factory=list)
    none_of: list[str] = Field(default_factory=list)

    def matches(self, venue: str, text: str) -> bool:
        if self.venue != "any" and self.venue != venue:
            return False
        haystack = text.lower()
        if self.none_of and any(contains_term(haystack, term) for term in self.none_of):
            return False
        if self.all_of and not all(contains_term(haystack, term) for term in self.all_of):
            return False
        if self.any_of and not any(contains_term(haystack, term) for term in self.any_of):
            return False
        return bool(self.any_of or self.all_of)


class AnchorSpec(BaseModel):
    """Rule attaching a discovered market to the driver node that moves it."""

    id: str
    driver: str
    sign: Literal[-1, 1]
    weight: float = Field(default=0.8, gt=0.0, le=1.0)
    delay_h: float = Field(default=1.0, ge=0.0)
    halflife_h: float = Field(default=72.0, gt=0.0)
    mechanism: str = ""
    match: MatchSpec


class DomainSpec(BaseModel):
    domain: str
    version: int = 1
    nodes: list[NodeSpec] = Field(default_factory=list)
    edges: list[EdgeSpec] = Field(default_factory=list)
    market_anchors: list[AnchorSpec] = Field(default_factory=list)
    # Kalshi has thousands of open markets (mostly sports); pulling per series
    # is far cheaper and more precise than paging everything and filtering.
    kalshi_series: list[str] = Field(default_factory=list)
    # Polymarket tag ids to pull, so discovery is not limited to whatever is
    # highest-volume right now (that ranking is dominated by sports).
    polymarket_tags: list[str] = Field(default_factory=list)
    # Terms merged into every anchor's none_of in this domain. A US-macro
    # domain must not match "Japan recession" or "UK inflation" — the anchor
    # would wire a foreign market to a US node and feed it US news.
    default_none_of: list[str] = Field(default_factory=list)


class Ontology(BaseModel):
    domains: list[DomainSpec]
    node_domain: dict[str, str] = Field(default_factory=dict)

    @property
    def nodes(self) -> dict[str, NodeSpec]:
        return {node.id: node for domain in self.domains for node in domain.nodes}

    @property
    def edges(self) -> list[EdgeSpec]:
        return [edge for domain in self.domains for edge in domain.edges]

    @property
    def anchors(self) -> list[AnchorSpec]:
        return [anchor for domain in self.domains for anchor in domain.market_anchors]

    @property
    def kalshi_series(self) -> list[str]:
        seen: list[str] = []
        for domain in self.domains:
            for series in domain.kalshi_series:
                if series not in seen:
                    seen.append(series)
        return seen

    @property
    def polymarket_tags(self) -> list[str]:
        seen: list[str] = []
        for domain in self.domains:
            for tag in domain.polymarket_tags:
                if tag not in seen:
                    seen.append(tag)
        return seen

    def match_anchors(self, venue: str, text: str) -> list[AnchorSpec]:
        return [a for a in self.anchors if a.match.matches(venue, text)]


class OntologyError(ValueError):
    pass


def load_ontology(path: Path | None = None) -> Ontology:
    directory = path or get_settings().ontology_path
    files = sorted(directory.glob("*.yaml")) + sorted(directory.glob("*.yml"))
    if not files:
        raise OntologyError(f"no ontology files found in {directory}")

    domains: list[DomainSpec] = []
    node_domain: dict[str, str] = {}
    seen: dict[str, NodeSpec] = {}

    for file in files:
        raw = yaml.safe_load(file.read_text(encoding="utf-8"))
        try:
            domain = DomainSpec.model_validate(raw)
        except Exception as exc:  # pydantic error -> point at the file
            raise OntologyError(f"{file.name}: {exc}") from exc

        # Domain-wide exclusions are merged into each anchor before validation
        # so a single list guards every anchor in the domain.
        for anchor in domain.market_anchors:
            for term in domain.default_none_of:
                if term not in anchor.match.none_of:
                    anchor.match.none_of.append(term)

        for node in domain.nodes:
            existing = seen.get(node.id)
            if existing is not None and existing.model_dump() != node.model_dump():
                raise OntologyError(
                    f"node '{node.id}' declared twice with different definitions "
                    f"(second in {file.name}); declare it once and reference it across domains"
                )
            seen[node.id] = node
            node_domain.setdefault(node.id, domain.domain)
        domains.append(domain)

    ontology = Ontology(domains=domains, node_domain=node_domain)
    _validate_references(ontology)
    return ontology


def _validate_references(ontology: Ontology) -> None:
    known = set(ontology.nodes)
    problems: list[str] = []

    for edge in ontology.edges:
        for side, node_id in (("src", edge.src), ("dst", edge.dst)):
            if node_id not in known:
                problems.append(f"edge {edge.src}->{edge.dst}: unknown {side} node '{node_id}'")
        if edge.src == edge.dst:
            problems.append(f"edge {edge.src}->{edge.dst}: self-loop")

    for anchor in ontology.anchors:
        if anchor.driver not in known:
            problems.append(f"anchor '{anchor.id}': unknown driver node '{anchor.driver}'")
        if not (anchor.match.any_of or anchor.match.all_of):
            problems.append(f"anchor '{anchor.id}': match must set any_of and/or all_of")

    if problems:
        raise OntologyError("; ".join(problems))


def sync_to_db(ontology: Ontology) -> dict[str, int]:
    """Upsert ontology nodes/edges and expire anything the YAML dropped.

    Without the expiry pass the YAML is not actually the source of truth:
    deleting an edge leaves it live in the graph forever, and renaming a
    mechanism creates a duplicate beside the original.
    """
    engine = get_engine()
    stats = {
        "nodes_inserted": 0,
        "nodes_updated": 0,
        "edges_inserted": 0,
        "edges_updated": 0,
        "edges_expired": 0,
        "nodes_retired": 0,
    }
    declared_edges = {(edge.src, edge.dst, edge.mechanism) for edge in ontology.edges}
    declared_nodes = set(ontology.nodes)

    with engine.begin() as conn:
        for node_id, node in ontology.nodes.items():
            values = {
                "kind": node.kind,
                "label": node.label,
                "axis_def": node.axis,
                "domain": ontology.node_domain.get(node_id),
                "status": "active",
                "aliases": node.aliases,
                "meta": {},
                "updated_at": utcnow(),
            }
            exists = conn.execute(
                sa.select(nodes_t.c.id).where(nodes_t.c.id == node_id)
            ).first()
            if exists:
                conn.execute(nodes_t.update().where(nodes_t.c.id == node_id).values(**values))
                stats["nodes_updated"] += 1
            else:
                conn.execute(nodes_t.insert().values(id=node_id, created_at=utcnow(), **values))
                stats["nodes_inserted"] += 1

        for edge in ontology.edges:
            values = {
                "sign": edge.sign,
                "weight": edge.weight,
                "delay_h": edge.delay_h,
                "halflife_h": edge.halflife_h,
                "edge_class": edge.edge_class,
                "provenance": "manual",
            }
            existing = conn.execute(
                sa.select(edges_t.c.id).where(
                    sa.and_(
                        edges_t.c.src == edge.src,
                        edges_t.c.dst == edge.dst,
                        edges_t.c.mechanism == edge.mechanism,
                    )
                )
            ).first()
            if existing:
                conn.execute(edges_t.update().where(edges_t.c.id == existing.id).values(**values))
                stats["edges_updated"] += 1
            else:
                conn.execute(
                    edges_t.insert().values(
                        src=edge.src,
                        dst=edge.dst,
                        mechanism=edge.mechanism,
                        valid_from=utcnow(),
                        created_at=utcnow(),
                        **values,
                    )
                )
                stats["edges_inserted"] += 1

        # Expire hand-curated edges the YAML no longer declares. Anchor edges
        # (provenance "anchor:*") belong to discovery and are left alone, and
        # expiry preserves history rather than deleting rows.
        live = conn.execute(
            sa.select(edges_t.c.id, edges_t.c.src, edges_t.c.dst, edges_t.c.mechanism).where(
                sa.and_(edges_t.c.provenance == "manual", edges_t.c.valid_until.is_(None))
            )
        ).all()
        for row in live:
            if (row.src, row.dst, row.mechanism) not in declared_edges:
                conn.execute(
                    edges_t.update().where(edges_t.c.id == row.id).values(valid_until=utcnow())
                )
                stats["edges_expired"] += 1

        stale_nodes = conn.execute(
            sa.select(nodes_t.c.id).where(
                sa.and_(nodes_t.c.kind != "market", nodes_t.c.status == "active")
            )
        ).all()
        for row in stale_nodes:
            if row.id not in declared_nodes:
                conn.execute(
                    nodes_t.update().where(nodes_t.c.id == row.id).values(status="retired")
                )
                stats["nodes_retired"] += 1

    return stats
