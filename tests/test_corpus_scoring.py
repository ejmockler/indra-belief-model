"""Tests for score_corpus orchestration (Phase 3.1)."""

from __future__ import annotations

import json

import duckdb
import pytest
from indra.statements import Activation, Agent, Evidence, Phosphorylation

from indra_belief.corpus import (
    apply_schema,
    ingest_statements,
    score_corpus,
)


def _con():
    con = duckdb.connect(":memory:")
    apply_schema(con)
    return con


def _stmt():
    mek1 = Agent("MAP2K1", db_refs={"HGNC": "6840"})
    erk1 = Agent("MAPK1", db_refs={"HGNC": "6871"})
    ev = Evidence(
        source_api="reach",
        text="MEK1 phosphorylates ERK at T202.",
        epistemics={"direct": True},
    )
    s = Phosphorylation(mek1, erk1, residue="T", position="202", evidence=[ev])
    s.belief = 0.82
    return s


def _mock_score(verdict="correct", confidence="high", score=0.95):
    """Returns a `score_evidence`-shaped dict, matching scorer.py docstring."""
    def _fn(statement, evidence, client):
        return {
            "score": score,
            "verdict": verdict,
            "confidence": confidence,
            "tier": "decomposed",
            "grounding_status": "all_match",
            "provenance_triggered": False,
            "tokens": 84,
            "raw_text": "mock trace",
            "reasons": ["match"],
            "rationale": "mock rationale",
            "call_log": [
                {"kind": "probe_subject_role", "duration_s": 0.4,
                 "prompt_tokens": 200, "out_tokens": 20, "finish_reason": "stop"},
                {"kind": "probe_relation_axis", "duration_s": 0.5,
                 "prompt_tokens": 250, "out_tokens": 25, "finish_reason": "stop"},
            ],
        }
    return _fn


def test_score_corpus_writes_run_and_step_rows():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    run_id = score_corpus(
        con, [stmt],
        scorer_version="test-v1",
        model_id_default="mock-model",
        score_evidence=_mock_score(),
    )

    runs = con.execute(
        "SELECT run_id, scorer_version, status, n_stmts FROM score_run"
    ).fetchall()
    assert len(runs) == 1
    assert runs[0][0] == run_id
    assert runs[0][1] == "test-v1"
    assert runs[0][2] == "succeeded"
    assert runs[0][3] == 1

    steps = con.execute(
        "SELECT step_kind, scorer_version, model_id, latency_ms IS NOT NULL,"
        "       prompt_tokens, out_tokens FROM scorer_step"
    ).fetchall()
    assert len(steps) == 1
    assert steps[0][0] == "aggregate"
    assert steps[0][1] == "test-v1"
    assert steps[0][2] == "mock-model"
    assert steps[0][3] is True  # latency_ms recorded
    assert steps[0][4] == 450   # 200 + 250 from call_log
    assert steps[0][5] == 45    # 20 + 25


def test_score_corpus_preserves_full_dict_in_output_json():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])
    score_corpus(con, [stmt], scorer_version="test-v1", score_evidence=_mock_score())

    raw = con.execute("SELECT output_json FROM scorer_step").fetchone()[0]
    out = json.loads(raw)
    assert out["score"] == 0.95
    assert out["verdict"] == "correct"
    assert out["confidence"] == "high"
    assert out["reasons"] == ["match"]
    assert len(out["call_log"]) == 2


def test_score_corpus_handles_evidence_failure_as_abstain():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    def boom(statement, evidence, client):
        raise RuntimeError("LLM transport down")

    run_id = score_corpus(con, [stmt], scorer_version="test-v1", score_evidence=boom)
    runs = con.execute("SELECT status, n_stmts FROM score_run").fetchall()
    assert runs[0] == ("succeeded", 1)  # graceful degradation, not failure
    out = json.loads(con.execute("SELECT output_json FROM scorer_step").fetchone()[0])
    assert out["verdict"] == "abstain"
    assert "LLM transport down" in out["error"]


def test_score_corpus_append_only_across_versions():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    score_corpus(con, [stmt], scorer_version="v1", score_evidence=_mock_score(score=0.75))
    score_corpus(con, [stmt], scorer_version="v2", score_evidence=_mock_score(score=0.95))

    rows = con.execute(
        "SELECT scorer_version, json_extract(output_json, '$.score') "
        "FROM scorer_step ORDER BY scorer_version"
    ).fetchall()
    assert len(rows) == 2
    assert rows[0][0] == "v1"
    assert rows[1][0] == "v2"


def test_score_corpus_on_evidence_callback():
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    seen: list[tuple[str, str, str]] = []
    def cb(stmt_hash, evidence_hash, result):
        seen.append((stmt_hash, evidence_hash, result["verdict"]))

    score_corpus(
        con, [stmt],
        scorer_version="test-v1",
        score_evidence=_mock_score(),
        on_evidence=cb,
    )
    assert len(seen) == 1
    assert seen[0][2] == "correct"


def test_score_corpus_decompose_writes_per_step_rows():
    """Phase 3.4 partial: decompose=True writes rows for parse_claim,
    build_context, substrate_route + per-probe rows for substrate-answered."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    score_corpus(con, [stmt], scorer_version="t",
                 score_evidence=_mock_score(), decompose=True)

    kinds = {row[0] for row in con.execute(
        "SELECT DISTINCT step_kind FROM scorer_step"
    ).fetchall()}
    # Always emitted
    assert "aggregate" in kinds
    assert "parse_claim" in kinds
    assert "build_context" in kinds
    assert "substrate_route" in kinds
    # Probe rows emitted iff substrate resolved them; we don't assert on
    # the specific probes since substrate behavior depends on the stmt+ev
    # content. We just assert no exceptions were swallowed.
    err_rows = con.execute(
        "SELECT step_kind, error FROM scorer_step WHERE error IS NOT NULL"
    ).fetchall()
    assert err_rows == []


def test_score_corpus_cost_threshold_aborts_above():
    """G3b stop-the-line: cost_threshold_usd raises before scoring starts."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    with pytest.raises(ValueError, match=r"exceeds threshold"):
        score_corpus(con, [stmt], scorer_version="t",
                     model_id_default="claude-opus-4-7",  # expensive
                     score_evidence=_mock_score(),
                     cost_threshold_usd=0.000001)  # impossibly tight

    # No score_run row should have been written
    assert con.execute("SELECT COUNT(*) FROM score_run").fetchone()[0] == 0


def test_score_corpus_cost_threshold_passes_below():
    """A generous threshold lets the run proceed normally."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    run_id = score_corpus(con, [stmt], scorer_version="t",
                          model_id_default="claude-haiku-4-5",
                          score_evidence=_mock_score(),
                          cost_threshold_usd=1000.0)
    assert run_id
    assert con.execute("SELECT COUNT(*) FROM scorer_step").fetchone()[0] >= 1


def test_score_corpus_cost_threshold_none_skips_check():
    """Default behavior: no threshold check; expensive runs proceed."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    # No threshold passed → no upfront cost check, no error raised
    run_id = score_corpus(con, [stmt], scorer_version="t",
                          model_id_default="claude-opus-4-7",
                          score_evidence=_mock_score())
    assert run_id


def test_score_corpus_with_validity_default_true():
    """Auto-validity ON by default — score_corpus runs compute_validity at end."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    score_corpus(con, [stmt], scorer_version="t", score_evidence=_mock_score())
    # metric rows should exist for the auto-computed validity
    n_metrics = con.execute("SELECT COUNT(*) FROM metric").fetchone()[0]
    assert n_metrics > 0


def test_score_corpus_with_validity_opt_out():
    """`with_validity=False` skips auto-compute (for tests / cost-sensitive runs)."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    score_corpus(con, [stmt], scorer_version="t", score_evidence=_mock_score(),
                 with_validity=False)
    n_metrics = con.execute("SELECT COUNT(*) FROM metric").fetchone()[0]
    assert n_metrics == 0


def test_score_corpus_decompose_default_false():
    """Backwards-compatible: decompose defaults to False, only aggregate written."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])

    score_corpus(con, [stmt], scorer_version="t",
                 score_evidence=_mock_score())  # no decompose

    kinds = [row[0] for row in con.execute(
        "SELECT step_kind FROM scorer_step"
    ).fetchall()]
    assert kinds == ["aggregate"]


def test_viewer_step_kinds_match_python_emit():
    """Lock the deep-dive's 9-step rail to Python's emitted step_kinds.

    `viewer/src/routes/statements/[stmt_hash]/+page.svelte` hardcodes a
    9-element STEP_KINDS list to render the rail. If Python renames a
    step_kind in `_decompose_steps`, the rail silently fails to light
    that step. This grep-asserts that every step_kind Python actually
    emits (excluding 'aggregate', which is special-cased to light the
    adjudicate tick) is present in the viewer's STEP_KINDS array.
    """
    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    src = (repo_root / "viewer" / "src" / "routes" / "statements"
           / "[stmt_hash]" / "+page.svelte").read_text(encoding="utf-8")

    # Extract the keys (first element of each tuple) from STEP_KINDS
    block_match = re.search(
        r"const STEP_KINDS:\s*Array<\[string,\s*string\]>\s*=\s*\[(.+?)\];",
        src, re.S
    )
    assert block_match, "STEP_KINDS array not found in deep-dive svelte"
    viewer_kinds = set(re.findall(r"\['([a-z_]+)'", block_match.group(1)))

    # Run a synthetic score with decompose=True; collect kinds Python emits
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])
    score_corpus(con, [stmt], scorer_version="t", decompose=True,
                 score_evidence=_mock_score())
    python_kinds = {row[0] for row in con.execute(
        "SELECT DISTINCT step_kind FROM scorer_step"
    ).fetchall()}
    # 'aggregate' is special — rendered via aggForAdjudicate, not in STEP_KINDS
    python_kinds.discard("aggregate")

    missing = python_kinds - viewer_kinds
    assert not missing, (
        f"Python emits step_kinds {missing} that the viewer's STEP_KINDS "
        f"won't render — silent rail bug. Update viewer + this test."
    )


def test_score_corpus_stmt_with_no_evidence_skipped_gracefully():
    con = _con()
    a = Agent("RAF1", db_refs={"HGNC": "9829"})
    b = Agent("MAP2K1", db_refs={"HGNC": "6840"})
    stmt = Activation(a, b, evidence=[])
    ingest_statements(con, [stmt])
    run_id = score_corpus(con, [stmt], scorer_version="test-v1", score_evidence=_mock_score())
    n_steps = con.execute("SELECT COUNT(*) FROM scorer_step").fetchone()[0]
    assert n_steps == 0  # no evidence → no scorer_step rows
    n_runs = con.execute("SELECT n_stmts FROM score_run WHERE run_id = ?", [run_id]).fetchone()[0]
    assert n_runs == 1   # but the statement is counted


def test_score_corpus_raises_when_both_client_and_score_evidence_none():
    """Fail fast: without a ModelClient or a custom scorer, the default
    scorer crashes per-evidence with `client.call(None)` and silently
    yields all-abstain rows. Surface the misuse at the call site instead.
    """
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])
    with pytest.raises(ValueError, match=r"client=.*score_evidence="):
        score_corpus(con, [stmt], scorer_version="t")  # no client, no score_evidence


def test_score_corpus_persists_cost_estimate():
    """cost_estimate_usd was a forever-NULL schema column until iter-90.
    The estimate is computed upfront for the threshold gate; persisting
    it gives the audit trail (visible in model_card hand-off)."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])
    score_corpus(con, [stmt], scorer_version="t",
                 model_id_default="claude-sonnet-4-6",
                 score_evidence=_mock_score())
    cost_est = con.execute(
        "SELECT cost_estimate_usd FROM score_run"
    ).fetchone()[0]
    assert cost_est is not None
    assert cost_est > 0  # Sonnet is non-trivial


def test_score_corpus_persists_cost_actual():
    """cost_actual_usd was the second forever-NULL column; iter-92 wires
    it to sum(prompt_tokens × in_rate + out_tokens × out_rate) across
    aggregate scorer_step rows."""
    con = _con()
    stmt = _stmt()
    ingest_statements(con, [stmt])
    score_corpus(con, [stmt], scorer_version="t",
                 model_id_default="claude-sonnet-4-6",
                 score_evidence=_mock_score())
    cost_actual = con.execute(
        "SELECT cost_actual_usd FROM score_run"
    ).fetchone()[0]
    assert cost_actual is not None
    # Mock returns 450 prompt + 45 out tokens. Sonnet: $3/M in, $15/M out.
    # 1 evidence: 450*3/1M + 45*15/1M = 0.00135 + 0.000675 = 0.002025
    expected = 450 * 3.0 / 1_000_000 + 45 * 15.0 / 1_000_000
    assert abs(cost_actual - expected) < 1e-6, f"got {cost_actual}, expected {expected}"
