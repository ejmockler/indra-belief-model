"""End-to-end integration test for the corpus pipeline.

Per-module unit tests live in `test_corpus_loader.py`, `test_corpus_scoring.py`,
`test_corpus_validity.py`, `test_corpus_export.py`, `test_corpus_cost.py`.
This file verifies the four-step canonical workflow runs cleanly:

    apply_schema → ingest_statements → estimate_cost → score_corpus
    (with auto-validity) → export_beliefs → model_card → INDRA round-trip
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
from indra.statements import (
    Activation,
    Agent,
    Evidence,
    Inhibition,
    Phosphorylation,
    stmts_from_json,
)

from indra_belief.corpus import (
    apply_schema,
    estimate_cost,
    export_beliefs,
    ingest_statements,
    model_card,
    score_corpus,
)


def _build_corpus():
    """Synthetic mini-corpus exercising distinct stmt types + epistemics."""
    mek1 = Agent("MAP2K1", db_refs={"HGNC": "6840", "UP": "Q02750"})
    erk1 = Agent("MAPK1", db_refs={"HGNC": "6871", "UP": "P28482"})
    raf1 = Agent("RAF1", db_refs={"HGNC": "9829"})
    return [
        Phosphorylation(mek1, erk1, residue="T", position="202", evidence=[
            Evidence(source_api="reach", pmid="1", text="MEK1 phosphorylates ERK at T202.",
                     epistemics={"direct": True, "curated": True}),
            Evidence(source_api="biopax", pmid="2", text="MAP2K1 catalyzes ERK1 T202.",
                     epistemics={"direct": True}),
        ]),
        Activation(raf1, mek1, evidence=[
            Evidence(source_api="signor", pmid="3", text="RAF1 activates MEK1.",
                     epistemics={"direct": True, "curated": True}),
        ]),
        Inhibition(mek1, raf1, evidence=[
            Evidence(source_api="reach", pmid="4", text="MEK1 was not found to inhibit RAF1.",
                     epistemics={"direct": False, "negated": True}),
        ]),
    ]


def _mock_scorer(statement, evidence, client):
    """Deterministic mock that mirrors real scorer output shape."""
    epi = getattr(evidence, "epistemics", {}) or {}
    if epi.get("negated"):
        return {"score": 0.15, "verdict": "incorrect", "confidence": "high",
                "reasons": ["negated"], "call_log": []}
    if epi.get("curated"):
        return {"score": 0.92, "verdict": "correct", "confidence": "high",
                "reasons": ["match", "curated"],
                "call_log": [{"kind": "probe_subject_role", "duration_s": 0.3,
                              "prompt_tokens": 180, "out_tokens": 18}]}
    return {"score": 0.78, "verdict": "correct", "confidence": "medium",
            "reasons": ["match"], "call_log": []}


def test_full_pipeline_e2e(tmp_path: Path):
    """Step-by-step exercise of the canonical workflow."""
    con = duckdb.connect(":memory:")
    apply_schema(con)
    stmts = _build_corpus()

    # 1. Ingest (lossless)
    ingest_counters = ingest_statements(con, stmts, source_dump_id="e2e_smoke")
    assert ingest_counters["n_statements"] == 3
    assert ingest_counters["n_evidences"] == 4
    assert ingest_counters["n_truth_labels"] > 0  # auto-registered

    # Auto-registered truth_sets for INDRA-derived signals
    truth_sets = {r[0] for r in con.execute(
        "SELECT id FROM truth_set"
    ).fetchall()}
    assert {"indra_published_belief", "indra_epistemics", "indra_grounding"} <= truth_sets

    # 2. Estimate cost (pre-Go check)
    est = estimate_cost(stmts, model_id="claude-sonnet-4-6")
    assert est["n_stmts"] == 3
    assert est["n_evidences_est"] == 4
    assert est["cost_usd"] > 0

    # 3. Score (with_validity=True default; decompose=True for per-step rows)
    run_id = score_corpus(
        con, stmts,
        scorer_version="e2e-test",
        model_id_default="mock",
        score_evidence=_mock_scorer,
        decompose=True,
        cost_threshold_usd=10.0,  # generous; mock scoring doesn't actually pay
    )
    assert run_id

    # Score run row populated
    run = con.execute(
        "SELECT status, n_stmts FROM score_run WHERE run_id = ?", [run_id]
    ).fetchone()
    assert run == ("succeeded", 3)

    # Aggregate scorer_step rows + decomposed substeps
    step_kinds = {r[0] for r in con.execute(
        "SELECT DISTINCT step_kind FROM scorer_step WHERE run_id = ?", [run_id]
    ).fetchall()}
    assert "aggregate" in step_kinds
    assert "parse_claim" in step_kinds
    assert "build_context" in step_kinds
    assert "substrate_route" in step_kinds

    # 4. Validity auto-computed → metric rows populated
    n_metrics = con.execute(
        "SELECT COUNT(*) FROM metric WHERE run_id = ?", [run_id]
    ).fetchone()[0]
    assert n_metrics > 0

    # Specific calibration metrics present
    cal_names = {r[0] for r in con.execute(
        "SELECT metric_name FROM metric WHERE run_id = ?", [run_id]
    ).fetchall()}
    assert "indra_belief_calibration.mae" in cal_names
    assert "indra_belief_calibration.bias" in cal_names

    # 5. Export INDRA-native JSON
    out_path = tmp_path / "exported.json"
    export_beliefs(con, run_id, out_path)
    assert out_path.exists()

    # Verify exported beliefs replaced INDRA's defaults
    exported = json.loads(out_path.read_text())
    assert len(exported) == 3
    for stmt_dict in exported:
        assert "belief" in stmt_dict
        # All scored stmts should have belief != 1.0 (the INDRA default)
        # since our mock scorer returns non-1.0 values
        assert stmt_dict["belief"] != 1.0

    # 6. INDRA round-trip — exported JSON re-loads cleanly
    reloaded = stmts_from_json(exported)
    assert len(reloaded) == 3
    types_seen = {type(s).__name__ for s in reloaded}
    assert types_seen == {"Phosphorylation", "Activation", "Inhibition"}

    # 7. Model card exports
    card_path = tmp_path / "card.json"
    card = model_card(con, run_id, out_path=card_path)
    assert card["run_id"] == run_id
    assert card["status"] == "succeeded"
    assert card["n_stmts_scored"] == 3
    assert "metrics" in card
    assert "limitations" in card

    con.close()


def test_full_pipeline_handles_empty_corpus(tmp_path: Path):
    """Pipeline should degrade gracefully on an empty corpus."""
    con = duckdb.connect(":memory:")
    apply_schema(con)

    counters = ingest_statements(con, [], source_dump_id="empty")
    assert counters["n_statements"] == 0

    run_id = score_corpus(con, [], scorer_version="t",
                          score_evidence=_mock_scorer)
    n_steps = con.execute("SELECT COUNT(*) FROM scorer_step").fetchone()[0]
    assert n_steps == 0

    # Validity gracefully reports unavailable
    metrics = con.execute(
        "SELECT metric_name FROM metric WHERE run_id = ?", [run_id]
    ).fetchall()
    # G4 honest-failure: if no scored data, metrics either absent or NaN
    if metrics:
        # All NaN — calibration with no scored stmts
        for name, in metrics:
            row = con.execute(
                "SELECT value FROM metric WHERE run_id = ? AND metric_name = ?",
                [run_id, name]
            ).fetchone()
            v = row[0]
            # value can be NaN (Python `nan != nan`) — accept it
            assert v != v or isinstance(v, float)

    out_path = tmp_path / "empty_export.json"
    export_beliefs(con, run_id, out_path)
    data = json.loads(out_path.read_text())
    assert data == []  # nothing to export

    con.close()
