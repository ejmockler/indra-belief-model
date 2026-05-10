"""Tests for validity-layer metric computation (Phase 4)."""

from __future__ import annotations

import math

import duckdb
import pytest
from indra.statements import Activation, Agent, Evidence, Phosphorylation

from indra_belief.corpus import (
    apply_schema,
    compute_validity,
    ingest_statements,
    score_corpus,
)


def _con():
    con = duckdb.connect(":memory:")
    apply_schema(con)
    return con


def _stmt(belief: float = 0.82, n_ev: int = 1):
    a = Agent("MAP2K1", db_refs={"HGNC": "6840"})
    b = Agent("MAPK1", db_refs={"HGNC": "6871"})
    evs = [
        Evidence(source_api=f"reach_{i}", text=f"sentence {i}",
                 epistemics={"direct": True})
        for i in range(n_ev)
    ]
    s = Phosphorylation(a, b, residue="T", position="202", evidence=evs)
    s.belief = belief
    return s


def _scorer(score: float, verdict: str = "correct", confidence: str = "high"):
    def _fn(statement, evidence, client):
        return {
            "score": score,
            "verdict": verdict,
            "confidence": confidence,
            "reasons": ["match"],
            "call_log": [],
        }
    return _fn


def test_compute_validity_raises_on_unknown_run_id():
    """Aligned with export_beliefs + model_card: silently returning a
    hollow summary dict on a typo'd run_id was a foot-gun.
    """
    con = _con()
    with pytest.raises(ValueError, match="not found"):
        compute_validity(con, "nonexistent-run-id")


def test_compute_validity_is_idempotent():
    """Re-running compute_validity does not double-write metric rows."""
    con = _con()
    s = _stmt(belief=0.82, n_ev=1)
    ingest_statements(con, [s])
    run_id = score_corpus(con, [s], scorer_version="t",
                          score_evidence=_scorer(0.95), with_validity=False)

    compute_validity(con, run_id)
    n1 = con.execute("SELECT COUNT(*) FROM metric WHERE run_id = ?",
                     [run_id]).fetchone()[0]

    compute_validity(con, run_id)
    n2 = con.execute("SELECT COUNT(*) FROM metric WHERE run_id = ?",
                     [run_id]).fetchone()[0]

    assert n1 == n2, f"compute_validity not idempotent: {n1} → {n2}"


def test_calibration_writes_mae_rmse_bias():
    con = _con()
    s = _stmt(belief=0.82, n_ev=1)
    ingest_statements(con, [s])
    run_id = score_corpus(con, [s], scorer_version="t", score_evidence=_scorer(0.95), with_validity=False)

    summary = compute_validity(con, run_id)
    cal = summary["calibration"]
    assert cal["n_stmts"] == 1
    assert cal["mae"] == pytest.approx(0.13)
    assert cal["bias"] == pytest.approx(0.13)
    assert cal["rmse"] == pytest.approx(0.13)

    rows = con.execute(
        "SELECT metric_name, value FROM metric "
        "WHERE truth_set_id = 'indra_published_belief' ORDER BY metric_name"
    ).fetchall()
    names = [r[0] for r in rows]
    assert "indra_belief_calibration.mae" in names
    assert "indra_belief_calibration.rmse" in names
    assert "indra_belief_calibration.bias" in names


def test_inter_evidence_consistency_stdev_for_multi_ev():
    con = _con()
    s = _stmt(belief=0.7, n_ev=3)
    ingest_statements(con, [s])

    # Vary scores per evidence so stdev is non-zero
    scores_iter = iter([0.9, 0.5, 0.7])

    def varying(statement, evidence, client):
        return {"score": next(scores_iter), "verdict": "correct",
                "confidence": "high", "reasons": [], "call_log": []}

    run_id = score_corpus(con, [s], scorer_version="t", score_evidence=varying, with_validity=False)
    summary = compute_validity(con, run_id)
    cons = summary["inter_evidence_consistency"]
    assert cons["n_multi_evidence_stmts"] == 1
    assert cons["mean_stdev"] > 0


def test_singleton_only_writes_unavailable_reason():
    con = _con()
    s = _stmt(belief=0.9, n_ev=1)
    ingest_statements(con, [s])
    run_id = score_corpus(con, [s], scorer_version="t", score_evidence=_scorer(0.85), with_validity=False)

    summary = compute_validity(con, run_id)
    assert summary["inter_evidence_consistency"]["n_multi_evidence_stmts"] == 0
    assert "unavailable_reason" in summary["inter_evidence_consistency"]

    # G4 honest-failure: NaN row written with unavailable_reason in slice_json
    row = con.execute(
        "SELECT value, slice_json::VARCHAR FROM metric "
        "WHERE metric_name = 'inter_evidence_consistency.mean_stdev'"
    ).fetchone()
    assert row is not None
    assert math.isnan(row[0])
    assert "unavailable_reason" in row[1]


def test_truth_present_metrics_compute_pr_f1_against_gold(tmp_path):
    """4a — when gold verdict labels exist, P/R/F1 lands in metric table."""
    from indra_belief.corpus import register_truth_set, load_truth_labels

    con = _con()
    s = _stmt(belief=0.5, n_ev=3)
    ingest_statements(con, [s])

    # Score the 3 evidences as: correct, correct, abstain (mock)
    verdicts = iter(["correct", "correct", "abstain"])
    def mock_v(statement, evidence, client):
        v = next(verdicts)
        return {"score": 0.8 if v == "correct" else 0.5,
                "verdict": v, "confidence": "high",
                "reasons": [], "call_log": []}
    run_id = score_corpus(con, [s], scorer_version="t",
                          score_evidence=mock_v, with_validity=False)

    # Register gold pool. Gold says: correct, incorrect, correct
    # Comparing scorer ↔ gold: correct=correct (TP), correct≠incorrect (FP),
    # abstain≠correct (FN). Expect: precision 1/2=0.5, recall 1/2=0.5, f1=0.5
    register_truth_set(con, id="gold_test", name="test gold")

    ev_hashes = [r[0] for r in con.execute(
        "SELECT evidence_hash FROM evidence ORDER BY evidence_hash"
    ).fetchall()]
    gold_verdicts = ["correct", "incorrect", "correct"]
    labels = [
        {"target_kind": "evidence", "target_id": eh,
         "field": "verdict", "value_text": gv, "provenance": "test"}
        for eh, gv in zip(ev_hashes, gold_verdicts)
    ]
    load_truth_labels(con, "gold_test", labels)

    # Compute validity (which now picks up 4a metrics)
    summary = compute_validity(con, run_id)

    assert "gold_test" in summary["truth_present_metrics"]
    agg_metrics = summary["truth_present_metrics"]["gold_test"]["aggregate"]
    assert agg_metrics["n_compared"] == 3
    assert agg_metrics["tp"] == 1
    assert agg_metrics["fp"] == 1
    assert agg_metrics["fn"] == 1
    assert agg_metrics["precision"] == pytest.approx(0.5)
    assert agg_metrics["recall"] == pytest.approx(0.5)
    assert agg_metrics["f1"] == pytest.approx(0.5)

    # metric rows persisted
    metric_names = {r[0] for r in con.execute(
        "SELECT metric_name FROM metric WHERE truth_set_id = 'gold_test'"
    ).fetchall()}
    assert "truth_present.aggregate.precision" in metric_names
    assert "truth_present.aggregate.recall" in metric_names
    assert "truth_present.aggregate.f1" in metric_names


def test_truth_present_returns_empty_when_no_overlap():
    """4a degrades gracefully — empty summary key when no gold overlap."""
    con = _con()
    s = _stmt(belief=0.5, n_ev=1)
    ingest_statements(con, [s])
    run_id = score_corpus(con, [s], scorer_version="t",
                          score_evidence=_scorer(0.8), with_validity=False)

    summary = compute_validity(con, run_id)
    # No gold registered → empty truth_present_metrics
    assert summary["truth_present_metrics"] == {}


def test_verdict_share_metrics_written():
    con = _con()
    s1 = _stmt(belief=0.8, n_ev=1)
    s2 = Activation(
        Agent("RAF1", db_refs={"HGNC": "9829"}),
        Agent("MAP2K1", db_refs={"HGNC": "6840"}),
        evidence=[Evidence(source_api="reach", text="x")],
    )
    s2.belief = 0.6
    ingest_statements(con, [s1, s2])

    verdicts_iter = iter(["correct", "abstain"])
    def per_v(statement, evidence, client):
        return {"score": 0.8, "verdict": next(verdicts_iter), "confidence": "high",
                "reasons": [], "call_log": []}

    run_id = score_corpus(con, [s1, s2], scorer_version="t", score_evidence=per_v, with_validity=False)
    summary = compute_validity(con, run_id)
    assert summary["verdicts"] == {"correct": 1, "abstain": 1}

    rows = con.execute(
        "SELECT metric_name, value FROM metric "
        "WHERE metric_name LIKE 'verdict_share.%' ORDER BY metric_name"
    ).fetchall()
    assert len(rows) == 2
    for _name, value in rows:
        assert value == pytest.approx(0.5)
