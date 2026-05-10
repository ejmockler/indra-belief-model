"""Tests for belief export + model card (Phase 6)."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pytest
from indra.statements import Activation, Agent, Evidence, Phosphorylation, stmts_from_json

from indra_belief.corpus import (
    aggregate_beliefs,
    apply_schema,
    compute_validity,
    export_beliefs,
    ingest_statements,
    model_card,
    score_corpus,
)


def _con():
    con = duckdb.connect(":memory:")
    apply_schema(con)
    return con


def _seed(con, scores: list[float]):
    s = Phosphorylation(
        Agent("MAP2K1", db_refs={"HGNC": "6840"}),
        Agent("MAPK1", db_refs={"HGNC": "6871"}),
        residue="T",
        position="202",
        evidence=[
            Evidence(source_api=f"reach_{i}", text=f"sentence {i}",
                     epistemics={"direct": True})
            for i in range(len(scores))
        ],
    )
    s.belief = 0.5  # deliberately not 1.0 so we can verify replacement
    ingest_statements(con, [s])

    scores_iter = iter(scores)
    def per_evidence(statement, evidence, client):
        return {"score": next(scores_iter), "verdict": "correct",
                "confidence": "high", "reasons": [], "call_log": []}

    return s, score_corpus(con, [s], scorer_version="t", score_evidence=per_evidence)


def test_aggregate_beliefs_mean_default():
    con = _con()
    s, run_id = _seed(con, [0.2, 0.4, 0.6])
    out = aggregate_beliefs(con, run_id)
    assert len(out) == 1
    [(stmt_hash, belief)] = out.items()
    assert belief == pytest.approx(0.4)


def test_aggregate_beliefs_noisy_or():
    con = _con()
    s, run_id = _seed(con, [0.5, 0.5])
    out = aggregate_beliefs(con, run_id, aggregator="noisy_or")
    [(stmt_hash, belief)] = out.items()
    assert belief == pytest.approx(0.75)  # 1 - (1-0.5)*(1-0.5)


def test_export_beliefs_replaces_indra_belief_with_our_score(tmp_path: Path):
    con = _con()
    s, run_id = _seed(con, [0.95])
    out_path = tmp_path / "exported.json"
    written = export_beliefs(con, run_id, out_path)
    assert written.exists()

    data = json.loads(written.read_text())
    assert len(data) == 1
    assert data[0]["belief"] == pytest.approx(0.95)
    assert data[0]["belief"] != 0.5  # original was 0.5, not preserved
    # Lossless on other fields
    assert data[0]["type"] == "Phosphorylation"
    assert data[0]["residue"] == "T"
    assert data[0]["position"] == "202"


def test_export_round_trips_through_indra(tmp_path: Path):
    con = _con()
    s, run_id = _seed(con, [0.95])
    out_path = tmp_path / "rt.json"
    export_beliefs(con, run_id, out_path)

    data = json.loads(out_path.read_text())
    # INDRA can re-load our exported file as Statement objects
    reloaded = stmts_from_json(data)
    assert len(reloaded) == 1
    assert reloaded[0].belief == pytest.approx(0.95)
    assert type(reloaded[0]).__name__ == "Phosphorylation"


def test_export_only_scored_filters_unscored_stmts(tmp_path: Path):
    con = _con()
    s, run_id = _seed(con, [0.95])

    # Add a second statement that is NOT scored
    other = Activation(
        Agent("RAF1", db_refs={"HGNC": "9829"}),
        Agent("MAP2K1", db_refs={"HGNC": "6840"}),
        evidence=[Evidence(source_api="reach", text="x")],
    )
    ingest_statements(con, [other])

    out_path = tmp_path / "only_scored.json"
    export_beliefs(con, run_id, out_path, only_scored=True)
    data = json.loads(out_path.read_text())
    assert len(data) == 1  # only the scored one

    out_path2 = tmp_path / "all.json"
    export_beliefs(con, run_id, out_path2, only_scored=False)
    data2 = json.loads(out_path2.read_text())
    assert len(data2) == 2


def test_model_card_includes_metrics_and_limitations(tmp_path: Path):
    con = _con()
    s, run_id = _seed(con, [0.95])
    compute_validity(con, run_id)
    card_path = tmp_path / "card.json"
    card = model_card(con, run_id, out_path=card_path)

    assert card["run_id"] == run_id
    assert card["status"] == "succeeded"
    assert card["n_stmts_scored"] == 1
    assert "metrics" in card
    assert "limitations" in card
    assert len(card["limitations"]) >= 2

    # File written
    assert card_path.exists()
    on_disk = json.loads(card_path.read_text())
    assert on_disk["run_id"] == run_id


def test_aggregate_beliefs_raises_on_unknown_run_id():
    """Aligned with export_beliefs / compute_validity / model_card."""
    con = _con()
    with pytest.raises(ValueError, match="not found"):
        aggregate_beliefs(con, "nonexistent-run-id")


def test_export_beliefs_raises_on_unknown_run_id(tmp_path: Path):
    """Aligned with model_card: silently writing an empty list on a typo'd
    run_id was a foot-gun (user thinks 'no scored stmts' rather than 'no
    such run')."""
    con = _con()
    out = tmp_path / "test.json"
    with pytest.raises(ValueError, match="not found"):
        export_beliefs(con, "nonexistent-run-id", out)


def test_model_card_carries_format_version():
    """Card must declare its format version so downstream consumers can
    detect breaking shape changes (e.g. iter-72 dict→list metrics)."""
    con = _con()
    s, run_id = _seed(con, [0.95])
    card = model_card(con, run_id)
    assert "card_format_version" in card
    assert card["card_format_version"] == 2
    # v2 contract: metrics is a list, not a dict
    assert isinstance(card["metrics"], list)


def test_model_card_metrics_preserve_multi_truth_set_entries():
    """Same metric_name with different truth_set_id must NOT collapse.

    Pre-fix the metrics dict was keyed by metric_name alone, so registering
    two verdict-grade gold pools on the same run silently lost one set's
    P/R/F1. List form preserves all (metric_name, truth_set_id) pairs.
    """
    from indra_belief.corpus import register_truth_set, load_truth_labels

    con = _con()
    s, run_id = _seed(con, [0.95])
    ev_hash = con.execute(
        "SELECT evidence_hash FROM evidence LIMIT 1"
    ).fetchone()[0]

    # Register two distinct verdict-grade gold pools, each with one label
    for tset_id in ("gold_a", "gold_b"):
        register_truth_set(con, id=tset_id, name=f"Gold {tset_id}")
        load_truth_labels(con, tset_id, [
            {"target_kind": "evidence", "target_id": ev_hash,
             "field": "verdict", "value_text": "correct",
             "provenance": "test"},
        ])
    compute_validity(con, run_id)

    card = model_card(con, run_id)
    metrics = card["metrics"]
    assert isinstance(metrics, list), "metrics shape must be a list (post-fix)"

    # truth_present rows should appear once per (metric_name, truth_set_id)
    f1_entries = [m for m in metrics if m["metric_name"] == "truth_present.aggregate.f1"]
    truth_set_ids = sorted(m["truth_set_id"] for m in f1_entries)
    assert truth_set_ids == ["gold_a", "gold_b"], (
        f"expected both gold pools' f1 metrics, got {truth_set_ids} "
        f"(dict-key collision regression)"
    )


def test_model_card_truth_set_coverage_includes_evidence_and_agent_labels():
    """Coverage must report all target_kinds, not just stmt-level.

    Old SQL `WHERE target_id IN (SELECT stmt_hash FROM scorer_step ...)`
    silently dropped evidence-level (indra_epistemics, demo_gold) and
    agent-level (indra_grounding) labels because their target_id is an
    evidence_hash / agent_hash, not a stmt_hash.
    """
    con = _con()
    s, run_id = _seed(con, [0.95])  # _seed auto-registers INDRA truth_sets

    card = model_card(con, run_id)
    coverage = card["truth_set_coverage"]

    # _seed creates a Phosphorylation with 1 evidence having
    # epistemics={direct: True}; loader auto-registers:
    # - indra_published_belief: 1 stmt-level row
    # - indra_grounding: 2 agent-level rows (one per agent's db_refs)
    # - indra_epistemics: 1 evidence-level row (direct=True)
    # Pre-fix: only indra_published_belief showed up
    # Post-fix: all three should be present
    assert "indra_published_belief" in coverage
    assert coverage["indra_published_belief"] == 1
    assert "indra_grounding" in coverage, (
        "agent-level truth_labels missing from coverage — "
        "target_id IN (stmt_hashes) regression"
    )
    assert coverage["indra_grounding"] == 2
    assert "indra_epistemics" in coverage, (
        "evidence-level truth_labels missing from coverage"
    )
    assert coverage["indra_epistemics"] >= 1
