"""Tests for J6: ComposedBeliefScorer.score_statement.

The v1 holdout exposed a truth-vs-scoring granularity mismatch — INDRA
tags reflect aggregate-across-N-evidences truth, but score_evidence
runs per-sentence. J6 closes the loop with a statement-native entry
point that aggregates per-evidence verdicts into one belief.

Tests cover:
  - all-correct evidence → belief follows parametric noise model
  - all-incorrect evidence → belief drops (high gating)
  - mixed evidence → at least one correct lifts the statement
  - empty evidence → graceful zero-evidence ComposedScore
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indra_belief.composed_scorer import ComposedBeliefScorer
from indra_belief.model_client import ModelResponse


class _MarkerMockClient:
    """S-phase mock: dispatches by probe kind (subject_role / object_role /
    relation_axis / scope) to produce probe-schema JSON.

    INCORRECT_MARKER in evidence text → scope probe answers "negated"
    (relation explicitly denied). Otherwise → all probes answer the
    default valid value for a clear correct case.
    """

    def __init__(self):
        self.model_name = "mock"
        self.backend = "mock"
        self.config = {"max_tokens": 12000, "timeout": 60}
        self.calls = 0
        self._call_log: list[dict] = []

    def call(self, system, messages, max_tokens=None, temperature=0.1,
             response_format=None, reasoning_effort=None, kind=None,
             **kwargs):
        # Find the user message and check for marker
        user = messages[-1]["content"] if messages else ""
        is_incorrect = "INCORRECT_MARKER" in user

        # Dispatch by probe kind to return appropriate schema
        if kind == "probe_subject_role":
            answer = "present_as_subject"
        elif kind == "probe_object_role":
            answer = "present_as_object"
        elif kind == "probe_relation_axis":
            answer = "direct_sign_match"
        elif kind == "probe_scope":
            answer = "negated" if is_incorrect else "asserted"
        else:
            # grounding / other; return mentioned-style verdict
            answer = "asserted"

        payload = f'{{"answer": "{answer}", "rationale": "mock"}}'
        self.calls += 1
        self._call_log.append({
            "kind": kind, "out_tokens": 10, "duration_s": 0.01,
        })
        return ModelResponse(
            content=payload, reasoning="", tokens=50,
            raw_text=payload, finish_reason="stop",
        )

    def pop_call_log(self) -> list[dict]:
        log = list(self._call_log)
        self._call_log.clear()
        return log


def _stmt_with_evidence(*evidence_texts):
    """Build a Phosphorylation with N evidence sentences."""
    from indra.statements import Phosphorylation, Agent, Evidence
    stmt = Phosphorylation(
        Agent("RPS6KA1"), Agent("YBX1"),
        residue="S", position="102",
    )
    stmt.evidence = [
        Evidence(source_api="reach", text=t) for t in evidence_texts
    ]
    return stmt


# ---------------------------------------------------------------------------
# Empty evidence
# ---------------------------------------------------------------------------

def test_empty_evidence_returns_zero_score():
    from indra.statements import Phosphorylation, Agent
    stmt = Phosphorylation(Agent("A"), Agent("B"))
    stmt.evidence = []
    scorer = ComposedBeliefScorer()
    result = scorer.score_statement(stmt, _MarkerMockClient())
    assert result.n_total == 0
    assert result.belief == 0.0


# ---------------------------------------------------------------------------
# All-correct
# ---------------------------------------------------------------------------

def test_all_correct_evidence_passes_gate():
    """Three correct verdicts → all 3 evidences survive the gate."""
    stmt = _stmt_with_evidence(
        "RSK1 phosphorylates YB-1 at S102.",
        "Phosphorylation of YB-1 Ser102 by RSK1 in vitro.",
        "RSK1-mediated phosphorylation of YB-1 at S102.",
    )
    # Force dec arch off so verdict comes purely from the marker mock
    # (dec arch makes multiple sub-calls per evidence, breaking the
    # 1-call-per-evidence assumption of this mock).
    import indra_belief
    original = indra_belief.score_statement
    def _mono_score(stmt, client, **kw):
        return original(stmt, client)
    indra_belief.score_statement = _mono_score
    try:
        scorer = ComposedBeliefScorer()
        result = scorer.score_statement(stmt, _MarkerMockClient())
        assert result.n_total == 3
        assert result.n_gated == 0
        assert result.n_surviving == 3
        assert result.has_llm_scores is True
    finally:
        indra_belief.score_statement = original


# ---------------------------------------------------------------------------
# All-incorrect
# ---------------------------------------------------------------------------

def test_all_incorrect_gated_out():
    # Both evidences must mention RSK1 and YB-1 so substrate doesn't
    # route subject/object to "absent" (S-phase semantics: no entity
    # → grounding_gap → abstain, not incorrect).
    stmt = _stmt_with_evidence(
        "INCORRECT_MARKER kinase-dead RSK1 failed to phosphorylate YB-1.",
        "INCORRECT_MARKER RSK1 mutant phenotype: no YB-1 S102 phosphorylation.",
    )
    import indra_belief
    original = indra_belief.score_statement
    def _mono_score(stmt, client, **kw):
        return original(stmt, client)
    indra_belief.score_statement = _mono_score
    try:
        scorer = ComposedBeliefScorer()
        result = scorer.score_statement(stmt, _MarkerMockClient())
        assert result.n_total == 2
        assert result.n_gated == 2  # all incorrect → all gated
        assert result.n_surviving == 0
    finally:
        indra_belief.score_statement = original


# ---------------------------------------------------------------------------
# Mixed
# ---------------------------------------------------------------------------

def test_mixed_evidence_lifts_via_correct_signal():
    """One correct + two incorrect → 1 evidence survives the gate."""
    stmt = _stmt_with_evidence(
        "RSK1 phosphorylates YB-1 at S102.",
        "INCORRECT_MARKER kinase-dead RSK1 abolishes YB-1 phosphorylation.",
        "INCORRECT_MARKER no detectable RSK1-mediated YB-1 Ser102 "
        "phosphorylation in vitro.",
    )
    import indra_belief
    original = indra_belief.score_statement
    def _mono_score(stmt, client, **kw):
        return original(stmt, client)
    indra_belief.score_statement = _mono_score
    try:
        scorer = ComposedBeliefScorer()
        result = scorer.score_statement(stmt, _MarkerMockClient())
        assert result.n_total == 3
        assert result.n_surviving == 1  # only the first passed
        assert result.n_gated == 2
    finally:
        indra_belief.score_statement = original
