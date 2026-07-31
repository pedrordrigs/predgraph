"""Tests for the load-bearing pure logic: book maths, anchor matching,
signed path composition and watchlist selection. No network, no DB."""

from __future__ import annotations

from datetime import timedelta

import pytest

from predgraph.db import utcnow
from predgraph.graph.algo import Edge, enumerate_paths, propagate
from predgraph.ingest.base import MarketRef, book_metrics
from predgraph.ingest.runner import select_watchlist
from predgraph.ontology import MatchSpec, NodeSpec, load_ontology

# --- order book -------------------------------------------------------------

def test_book_metrics_picks_best_levels_regardless_of_order():
    bids = [(0.40, 100.0), (0.42, 50.0), (0.38, 10.0)]
    asks = [(0.46, 20.0), (0.44, 80.0)]
    m = book_metrics(bids, asks)
    assert m["bid"] == 0.42
    assert m["ask"] == 0.44
    assert m["mid"] == pytest.approx(0.43)
    assert m["spread"] == pytest.approx(0.02)


def test_book_metrics_depth_counts_only_levels_inside_the_band():
    # mid = 0.50; a level 10 cents away must not inflate tradeable depth.
    bids = [(0.49, 100.0), (0.40, 1000.0)]
    asks = [(0.51, 100.0), (0.60, 1000.0)]
    m = book_metrics(bids, asks, band=0.02)
    assert m["depth_2c"] == pytest.approx(0.49 * 100 + 0.51 * 100, rel=1e-3)


def test_book_metrics_survives_an_empty_side():
    m = book_metrics([(0.30, 10.0)], [])
    assert m["mid"] == 0.30 and m["ask"] is None and m["spread"] is None


def test_book_metrics_empty_book_is_all_none():
    assert book_metrics([], [])["mid"] is None


# --- ontology ---------------------------------------------------------------

def test_latent_node_without_axis_is_rejected():
    with pytest.raises(ValueError, match="axis"):
        NodeSpec(id="x", kind="latent", label="X")


def test_entity_node_needs_no_axis():
    assert NodeSpec(id="iran", kind="entity", label="Iran").axis is None


def test_none_of_keeps_ukraine_out_of_middle_east_anchor():
    """The bug this guards: a Ukraine ceasefire market wired to ME escalation."""
    anchor = MatchSpec(any_of=["ceasefire"], none_of=["ukraine", "russia"])
    assert anchor.matches("polymarket", "Israel x Iran ceasefire by August?")
    assert not anchor.matches("polymarket", "Russia Ukraine ceasefire in 2026?")


def test_all_of_requires_every_term():
    spec = MatchSpec(all_of=["federal funds rate"], any_of=["below"])
    assert spec.matches("kalshi", "Will the federal funds rate be below 3%?")
    assert not spec.matches("kalshi", "Will CPI be below 3%?")


def test_venue_scoping():
    spec = MatchSpec(venue="kalshi", any_of=["cpi"])
    assert not spec.matches("polymarket", "CPI above 3%")


def test_terms_match_on_boundaries_not_substrings():
    """A "uk" guard must not exclude Ukraine, and "deal" must not hit "dealer"."""
    assert not MatchSpec(any_of=["deal"]).matches("polymarket", "Top car dealer in 2026?")
    assert MatchSpec(any_of=["deal"]).matches("polymarket", "US-Iran nuclear deal by June?")
    guarded = MatchSpec(any_of=["ceasefire"], none_of=["uk"])
    assert guarded.matches("polymarket", "Ukraine ceasefire in 2026?")


def test_punctuation_counts_as_a_boundary():
    assert MatchSpec(any_of=["cpi"]).matches("kalshi", "12-month change in CPI-U?")
    assert MatchSpec(any_of=["0bps"]).matches("kalshi", "Hike rates by 0bps at the meeting?")


def test_domain_default_none_of_is_merged_into_every_anchor(tmp_path):
    (tmp_path / "d.yaml").write_text(
        """
domain: t
default_none_of: [japan]
nodes:
  - {id: n1, kind: latent, label: N, axis: higher = more}
market_anchors:
  - id: a1
    driver: n1
    sign: 1
    mechanism: m
    match: {any_of: [recession]}
""",
        encoding="utf-8",
    )
    ontology = load_ontology(tmp_path)
    anchor = ontology.anchors[0]
    assert "japan" in anchor.match.none_of
    assert not anchor.match.matches("polymarket", "Japan recession in 2026?")
    assert anchor.match.matches("polymarket", "US recession in 2026?")


def test_shipped_macro_anchors_reject_foreign_markets():
    """Guards the real ontology: Polymarket's recession tag returns Japan markets."""
    ontology = load_ontology()
    foreign = [
        "Japan recession in 2026?",
        "Will the 10-year Japanese government bond yield rise?",
        "Will inflation in Brazil be below 4.00% in Dec 2026?",
        "United Kingdom Unemployment Rate above 5%?",
    ]
    for question in foreign:
        assert not ontology.match_anchors("polymarket", question), question
    assert ontology.match_anchors("kalshi", "Will there be a recession in 2027? Yes")


# --- signed propagation -----------------------------------------------------

def _edge(src, dst, sign, weight=1.0, delay=0.0):
    return Edge(src=src, dst=dst, sign=sign, weight=weight, delay_h=delay,
                halflife_h=24.0, mechanism=f"{src}->{dst}")


def test_signs_compose_along_the_path():
    """Two negative edges make a positive effect: the whole point of the model."""
    adjacency = {
        "escalation": [_edge("escalation", "supply", -1, 0.8)],
        "supply": [_edge("supply", "market", -1, 0.9)],
    }
    impacts = {i.target: i for i in propagate("escalation", adjacency=adjacency)}
    assert impacts["market"].contribution > 0
    assert impacts["supply"].contribution < 0


def test_direction_flips_the_whole_result():
    adjacency = {"a": [_edge("a", "b", 1, 0.8)]}
    up = propagate("a", direction=1, adjacency=adjacency)[0].contribution
    down = propagate("a", direction=-1, adjacency=adjacency)[0].contribution
    assert up == pytest.approx(-down)


def test_conflicting_paths_are_flagged_not_averaged_away():
    adjacency = {
        "src": [_edge("src", "mid1", 1, 0.9), _edge("src", "mid2", -1, 0.9)],
        "mid1": [_edge("mid1", "target", 1, 0.9)],
        "mid2": [_edge("mid2", "target", 1, 0.9)],
    }
    target = {i.target: i for i in propagate("src", adjacency=adjacency)}["target"]
    assert target.sign_conflict is True


def test_agreeing_paths_are_not_flagged():
    adjacency = {
        "src": [_edge("src", "mid1", 1, 0.9), _edge("src", "mid2", -1, 0.9)],
        "mid1": [_edge("mid1", "target", 1, 0.9)],
        "mid2": [_edge("mid2", "target", -1, 0.9)],
    }
    target = {i.target: i for i in propagate("src", adjacency=adjacency)}["target"]
    assert target.sign_conflict is False


def test_cycles_do_not_recurse_forever():
    adjacency = {
        "a": [_edge("a", "b", 1)],
        "b": [_edge("b", "a", 1), _edge("b", "c", 1)],
    }
    paths = enumerate_paths("a", adjacency, max_hops=5)
    assert "c" in paths
    assert all(len(set(p.nodes)) == len(p.nodes) for group in paths.values() for p in group)


def test_extra_paths_are_damped_not_summed_at_full_weight():
    """A second corroborating path should add less than the first."""
    single = {"s": [_edge("s", "t", 1, 0.8)]}
    double = {
        "s": [_edge("s", "t", 1, 0.8), _edge("s", "m", 1, 0.8)],
        "m": [_edge("m", "t", 1, 1.0)],
    }
    one = {i.target: i for i in propagate("s", adjacency=single)}["t"].contribution
    two = {i.target: i for i in propagate("s", adjacency=double)}["t"].contribution
    assert one < two < 2 * one


def test_hop_cap_is_enforced():
    chain = {f"n{i}": [_edge(f"n{i}", f"n{i+1}", 1)] for i in range(8)}
    assert "n3" in enumerate_paths("n0", chain, max_hops=3)
    assert "n4" not in enumerate_paths("n0", chain, max_hops=3)


# --- watchlist selection ----------------------------------------------------

def _ref(market_id, *, days_to_close=30.0, event="e1", oi=1000.0, bid=0.4, ask=0.42):
    return MarketRef(
        id=market_id,
        venue="kalshi",
        venue_id=market_id,
        question=market_id,
        close_time=utcnow() + timedelta(days=days_to_close),
        meta={"open_interest": oi, "volume_24h": 0.0, "yes_bid": bid, "yes_ask": ask,
              "event_ticker": event},
    )


def test_watchlist_respects_the_global_limit():
    refs = [_ref(f"m{i}", event=f"e{i}") for i in range(50)]
    assert len(select_watchlist(refs, limit=10)) == 10


def test_watchlist_caps_one_strike_ladder():
    """Without this, a single CPI ladder eats the whole polling budget."""
    refs = [_ref(f"ladder{i}", event="same") for i in range(20)]
    assert len(select_watchlist(refs, limit=60)) == 3


def test_watchlist_excludes_far_dated_and_expiring_markets():
    assert select_watchlist([_ref("far", days_to_close=900)]) == set()
    assert select_watchlist([_ref("soon", days_to_close=0.5)]) == set()


def test_watchlist_excludes_longshot_tails_and_wide_spreads():
    assert select_watchlist([_ref("tail", bid=0.01, ask=0.02)]) == set()
    assert select_watchlist([_ref("wide", bid=0.30, ask=0.55)]) == set()


def test_watchlist_excludes_illiquid_markets():
    assert select_watchlist([_ref("thin", oi=1.0)]) == set()


def test_watchlist_prefers_the_more_liquid_market():
    chosen = select_watchlist(
        [_ref("thin", event="a", oi=200.0), _ref("deep", event="b", oi=99999.0)], limit=1
    )
    assert chosen == {"deep"}
