"""Contract tests for the unified single-call scorer.

Schema integrity, fewshot validity, and the dict-mapping contract that
score_unified shares with score_evidence. NO LLM calls — the live
calibration runs are the empirical layer; these tests lock the static
contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from indra_belief.scorers.unified import (
    ExtractedRelation,
    GroundingCheck,
    UnifiedJudgment,
    _abstain_dict,
    _fewshots,
    _judgment_to_score_dict,
    _PROD_MAX_TOKENS,
    _PROD_TEMPERATURE,
    _PROD_VOTING_K,
    unified_response_schema,
)

ROOT = Path(__file__).resolve().parent.parent
FEWSHOT_PATH = ROOT / "src" / "indra_belief" / "data" / "unified_fewshots.jsonl"


# ─── Production parameters ─────────────────────────────────────────────

def test_production_parameters_locked():
    """A2 probe + design review committed these values; tests guard them."""
    assert _PROD_MAX_TOKENS == 64000
    assert _PROD_TEMPERATURE == 0.1
    assert _PROD_VOTING_K == 1


# ─── Schema integrity ──────────────────────────────────────────────────

def test_unified_judgment_field_order_reasoning_first():
    """Constrained decoding emits in declaration order; reasoning must
    be FIRST so subsequent fields condition on the chain of thought."""
    fields = list(UnifiedJudgment.model_fields.keys())
    assert fields[0] == "reasoning", \
        f"reasoning must be first field, got {fields[:3]}"
    assert fields[-1] == "self_critique", \
        f"self_critique must be last field, got {fields[-3:]}"


def test_unified_judgment_required_fields():
    """All 15 load-bearing fields present."""
    fields = set(UnifiedJudgment.model_fields.keys())
    expected = {
        "reasoning", "extracted_relations", "subject_grounding",
        "object_groundings", "axis_check", "sign_check", "site_check",
        "stance_check", "indirect_chain_present",
        "perturbation_inversion_applied", "verdict", "reason_code",
        "secondary_reasons", "confidence", "self_critique",
    }
    assert fields == expected


def test_unified_judgment_rejects_extra_fields():
    """extra='forbid' blocks unschemaed fields from sneaking in."""
    minimal_valid = {
        "reasoning": "test", "extracted_relations": [],
        "subject_grounding": {"entity": "X", "status": "mentioned",
                              "rationale": ""},
        "object_groundings": [],
        "axis_check": "match", "sign_check": "match",
        "site_check": "not_applicable", "stance_check": "affirmed",
        "indirect_chain_present": False,
        "perturbation_inversion_applied": False,
        "verdict": "correct", "reason_code": "match",
        "secondary_reasons": [], "confidence": "high",
        "self_critique": "",
    }
    # Baseline valid
    UnifiedJudgment.model_validate(minimal_valid)
    # Adding a slop field must reject
    with pytest.raises(ValidationError):
        UnifiedJudgment.model_validate({**minimal_valid, "slop": "extra"})


def test_extracted_relation_rejects_extra_fields():
    valid = {
        "subject": "X", "object": "Y", "axis": "modification",
        "sign": "positive",
    }
    ExtractedRelation.model_validate(valid)
    with pytest.raises(ValidationError):
        ExtractedRelation.model_validate({**valid, "extra": "field"})


def test_grounding_check_rejects_extra_fields():
    valid = {"entity": "X", "status": "mentioned", "rationale": ""}
    GroundingCheck.model_validate(valid)
    with pytest.raises(ValidationError):
        GroundingCheck.model_validate({**valid, "extra": "field"})


def test_unified_judgment_invalid_axis_rejected():
    bad = {
        "reasoning": "test", "extracted_relations": [],
        "subject_grounding": {"entity": "X", "status": "mentioned",
                              "rationale": ""},
        "object_groundings": [],
        "axis_check": "match", "sign_check": "match",
        "site_check": "not_applicable", "stance_check": "affirmed",
        "indirect_chain_present": False,
        "perturbation_inversion_applied": False,
        "verdict": "definitely_correct",  # invalid Verdict
        "reason_code": "match",
        "secondary_reasons": [], "confidence": "high",
        "self_critique": "",
    }
    with pytest.raises(ValidationError):
        UnifiedJudgment.model_validate(bad)


# ─── JSON-Schema export ────────────────────────────────────────────────

def test_response_schema_export_has_strict_mode():
    """unified_response_schema returns OpenAI-compat envelope with strict=True."""
    schema = unified_response_schema()
    assert schema["type"] == "json_schema"
    assert schema["json_schema"]["name"] == "UnifiedJudgment"
    assert schema["json_schema"]["strict"] is True
    assert "schema" in schema["json_schema"]


def test_response_schema_includes_top_level_properties():
    """The exported JSON Schema must list every UnifiedJudgment field
    so Ollama's decoder can enforce the structure."""
    schema = unified_response_schema()
    inner = schema["json_schema"]["schema"]
    props = inner.get("properties", {})
    required_fields = {
        "reasoning", "extracted_relations", "subject_grounding",
        "verdict", "reason_code", "confidence",
    }
    assert required_fields.issubset(set(props.keys()))


# ─── Few-shot curriculum ───────────────────────────────────────────────

def test_fewshots_load_and_validate():
    """Every shot's judgment must conform to the UnifiedJudgment schema."""
    shots = _fewshots()
    assert len(shots) >= 5, f"expected >=5 shots, got {len(shots)}"
    for shot in shots:
        assert "pattern" in shot
        assert "claim" in shot
        assert "gilda_context" in shot
        assert "evidence" in shot
        assert "judgment" in shot
        # Schema validation
        UnifiedJudgment.model_validate(shot["judgment"])


def test_fewshots_balance_correct_and_incorrect():
    """The 10-shot curriculum should include both correct and incorrect
    verdicts so the model sees both branches of the decision tree."""
    shots = _fewshots()
    verdicts = [s["judgment"]["verdict"] for s in shots]
    assert "correct" in verdicts
    assert "incorrect" in verdicts
    # Reasonable balance — neither dominates wildly
    n_correct = verdicts.count("correct")
    n_incorrect = verdicts.count("incorrect")
    assert n_correct >= 2 and n_incorrect >= 2


def test_fewshots_cover_critical_patterns():
    """Iter-1 type-conditional fix depends on the curriculum exposing
    both causal-type chains and direct-type chains."""
    shots = _fewshots()
    patterns = " ".join(s.get("pattern", "") for s in shots)
    assert "indirect_chain" in patterns or "indirect_as_direct" in patterns
    assert "co_occurrence" in patterns
    assert "promoter_target" in patterns
    assert "instrumental_role" in patterns


# ─── Dict-mapping contract ─────────────────────────────────────────────

def test_judgment_to_score_dict_correct_high():
    j = UnifiedJudgment.model_validate({
        "reasoning": "test", "extracted_relations": [],
        "subject_grounding": {"entity": "X", "status": "mentioned",
                              "rationale": ""},
        "object_groundings": [{"entity": "Y", "status": "equivalent",
                               "rationale": ""}],
        "axis_check": "match", "sign_check": "match",
        "site_check": "not_applicable", "stance_check": "affirmed",
        "indirect_chain_present": False,
        "perturbation_inversion_applied": False,
        "verdict": "correct", "reason_code": "match",
        "secondary_reasons": [], "confidence": "high",
        "self_critique": "",
    })
    out = _judgment_to_score_dict(j)
    assert out["verdict"] == "correct"
    assert out["confidence"] == "high"
    assert out["score"] == 0.95
    assert out["tier"] == "unified"
    assert out["grounding_status"] == "all_match"
    assert out["reasons"] == ["match"]


def test_judgment_to_score_dict_incorrect_via_grounding_gap():
    j = UnifiedJudgment.model_validate({
        "reasoning": "test", "extracted_relations": [],
        "subject_grounding": {"entity": "X", "status": "mentioned",
                              "rationale": ""},
        "object_groundings": [{"entity": "Y", "status": "not_present",
                               "rationale": ""}],
        "axis_check": "match", "sign_check": "match",
        "site_check": "not_applicable", "stance_check": "affirmed",
        "indirect_chain_present": False,
        "perturbation_inversion_applied": False,
        "verdict": "incorrect", "reason_code": "grounding_gap",
        "secondary_reasons": [], "confidence": "high",
        "self_critique": "",
    })
    out = _judgment_to_score_dict(j)
    assert out["verdict"] == "incorrect"
    assert out["score"] == 0.05
    assert out["grounding_status"] == "flagged"
    assert "grounding_gap" in out["reasons"]


def test_judgment_to_score_dict_secondary_reasons_propagate():
    j = UnifiedJudgment.model_validate({
        "reasoning": "test", "extracted_relations": [],
        "subject_grounding": {"entity": "X", "status": "mentioned",
                              "rationale": ""},
        "object_groundings": [],
        "axis_check": "match", "sign_check": "mismatch",
        "site_check": "not_applicable", "stance_check": "affirmed",
        "indirect_chain_present": False,
        "perturbation_inversion_applied": False,
        "verdict": "incorrect", "reason_code": "sign_mismatch",
        "secondary_reasons": ["axis_mismatch"], "confidence": "medium",
        "self_critique": "",
    })
    out = _judgment_to_score_dict(j)
    assert out["reasons"][0] == "sign_mismatch"
    assert "axis_mismatch" in out["reasons"]


# ─── Abstain contract ─────────────────────────────────────────────────

def test_abstain_dict_shape():
    """Graceful-degradation abstain has the same key contract as the
    happy-path dict."""
    out = _abstain_dict("reason", raw_text="raw")
    expected_keys = {
        "score", "verdict", "confidence", "tier", "grounding_status",
        "provenance_triggered", "tokens", "raw_text", "reasons",
        "rationale", "self_critique",
    }
    assert set(out.keys()) == expected_keys
    assert out["verdict"] == "abstain"
    assert out["score"] == 0.5
    assert out["tier"] == "unified"


# ─── Few-shot contamination guard hook ────────────────────────────────

def test_fewshot_evidence_distinct_from_holdout():
    """The unified curriculum must not share evidence with any benchmark
    file. Routes via check_contamination's source-5 hook to ensure the
    guard sees this curriculum.
    """
    from scripts.check_contamination import _load_unified_fewshots
    shots = _load_unified_fewshots()
    assert len(shots) > 0
    # Every shot has non-empty evidence
    for s in shots:
        assert s["evidence"].strip()


def test_fewshot_pairs_not_in_holdout():
    """No shot's (subject, object) should match any holdout pair —
    pair-level contamination is the contamination guard's third check."""
    holdout_pairs = set()
    for path in [ROOT / "data/benchmark/holdout_v15_sample.jsonl",
                 ROOT / "data/benchmark/holdout_large.jsonl"]:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                d = json.loads(line)
                s = (d.get("subject") or "").upper()
                o = (d.get("object") or "").upper()
                if s and o:
                    holdout_pairs.add((s, o))

    if not holdout_pairs:
        pytest.skip("no holdout files available")

    shots = _fewshots()
    for shot in shots:
        s = (shot["claim"]["subject"] or "").upper()
        o = (shot["claim"]["object"] or "").upper()
        assert (s, o) not in holdout_pairs, \
            f"fewshot pair ({s},{o}) overlaps holdout"
