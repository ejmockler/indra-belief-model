"""Regression anchors for the scorer pipeline.

Each test pins a specific behavioral contract: a bug class that was
observed, diagnosed, and fixed, captured here so regressions fail loudly
rather than silently. Covers:
  - short-symbol substring false matches in _text_contains
  - pseudogene auto-reject with evidence-text exception
  - verdict extraction with CoT-hypothetical defense + truncation fallback
  - voting confidence calibration under early-stop semantics
  - tool-use entity-lookup target (raw_text, not canonical name)
  - score_evidence and score_statement public API contracts
    (dict shape, value ranges; list-per-evidence iteration)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indra_belief.data.entity import GroundedEntity, _text_contains
from indra_belief.model_client import ModelResponse
from indra_belief.scorers.scorer import _parse_verdict


# ---------------------------------------------------------------------------
# _text_contains — letter-boundary for short symbols
# ---------------------------------------------------------------------------

def test_short_symbol_rejects_letter_neighbor():
    # "ar" must NOT match inside "bar" or "erased"
    assert _text_contains("ar", "the bar was erased", "thebarwaserased") is False


def test_short_symbol_matches_isoform_suffix():
    # "akt" must match in "akt1" (digit suffix is a valid boundary)
    assert _text_contains("akt", "akt1 phosphorylates foxo3", "akt1phosphorylatesfoxo3") is True


def test_short_symbol_rejects_method_false_match():
    # "met" must not match inside "method"
    assert _text_contains("met", "the method used", "themethodused") is False


def test_short_symbol_matches_standalone():
    assert _text_contains("met", "met phosphorylates akt", "metphosphorylatesakt") is True


def test_long_name_uses_substring_with_collapsed_variant():
    # Hyphenated long name with space variant in evidence
    assert _text_contains("beta-arrestin", "the beta arrestin", "thebetaarrestin") is True


def test_empty_needle_returns_false():
    assert _text_contains("", "anything", "anything") is False


# ---------------------------------------------------------------------------
# Pseudogene exception (entity.py:261-270)
# ---------------------------------------------------------------------------

def test_pseudogene_baseline_rejects():
    e = GroundedEntity(name="DVL1P1", verification_status="AMBIGUOUS", is_pseudogene=True)
    should, _ = e.should_auto_reject("DVL1P1 activates FOS in mice")
    assert should is True


def test_pseudogene_transcript_exception():
    e = GroundedEntity(name="DVL1P1", verification_status="AMBIGUOUS", is_pseudogene=True)
    should, _ = e.should_auto_reject("DVL1P1 is a pseudogene transcript")
    assert should is False


def test_pseudogene_lncrna_exception():
    e = GroundedEntity(name="XYZP1", verification_status="AMBIGUOUS", is_pseudogene=True)
    should, _ = e.should_auto_reject("XYZP1 lncRNA expression correlates with disease")
    assert should is False


# ---------------------------------------------------------------------------
# _parse_verdict — content preferred, raw_text fallback
# ---------------------------------------------------------------------------

def _resp(content: str, reasoning: str, finish: str = "stop") -> ModelResponse:
    raw = (reasoning + "\n" + content) if reasoning else content
    return ModelResponse(
        content=content, reasoning=reasoning, tokens=100,
        raw_text=raw, finish_reason=finish,
    )


def test_parse_verdict_prefers_content_over_reasoning():
    # content has the final verdict; reasoning has a contradictory hypothetical
    r = _resp(
        content='{"verdict": "correct", "confidence": "high"}',
        reasoning='Initial thought was {"verdict": "incorrect", "confidence": "low"}',
    )
    assert _parse_verdict(r) == ("correct", "high")


def test_parse_verdict_falls_back_to_raw_text_on_truncation():
    # content is non-empty but verdictless (truncation case)
    r = _resp(
        content="The analysis continues but was cut off",
        reasoning='Therefore {"verdict": "incorrect", "confidence": "high"}',
        finish="length",
    )
    assert _parse_verdict(r) == ("incorrect", "high")


def test_parse_verdict_uses_raw_text_when_content_empty():
    # content is empty (reasoning-only response)
    r = _resp(
        content="",
        reasoning='{"verdict": "correct", "confidence": "medium"}',
    )
    assert _parse_verdict(r) == ("correct", "medium")


def test_parse_verdict_returns_none_when_no_verdict_anywhere():
    r = _resp(content="no structured output", reasoning="")
    assert _parse_verdict(r) == (None, None)


# ---------------------------------------------------------------------------
# Voting confidence (scorer.py: early-stop vs tiebreaker semantics)
# ---------------------------------------------------------------------------

def _pick_conf(samples_taken: int, voting_k: int, correct: int, incorrect: int) -> str | None:
    """Replicates the scorer.py voting confidence logic for isolated testing."""
    total = correct + incorrect
    if total == 0:
        return None
    agreement = max(correct, incorrect) / total
    early_stop_unanimous = (samples_taken < voting_k) and agreement == 1.0
    if early_stop_unanimous:
        return "high"
    if agreement >= 0.6:
        return "medium"
    return "low"


def test_voting_k3_early_stop_unanimous_high():
    # Early-stop at 2/2 → unanimous on first try → high
    assert _pick_conf(2, 3, 2, 0) == "high"
    assert _pick_conf(2, 3, 0, 2) == "high"


def test_voting_k3_tiebreaker_medium():
    # All 3 samples used (first two disagreed) → majority, not unanimous → medium
    assert _pick_conf(3, 3, 2, 1) == "medium"
    assert _pick_conf(3, 3, 1, 2) == "medium"


def test_voting_k5_early_stop_high():
    # k=5, early-stop at samples=3 with 3/0 unanimous → high
    assert _pick_conf(3, 5, 3, 0) == "high"


def test_voting_k5_bare_majority_medium():
    assert _pick_conf(5, 5, 3, 2) == "medium"


# ---------------------------------------------------------------------------
# Tool-use lookups must target entity.raw_text, not entity.name
# ---------------------------------------------------------------------------

class _MockClient:
    """ModelClient stand-in.

    Looks at the user-message evidence text and echoes a verdict keyed to
    it — so iteration tests can prove each Evidence object was actually
    routed through, not the same one repeatedly. Evidence text containing
    "INCORRECT_MARKER" returns incorrect/high; otherwise correct/high.
    """
    def __init__(self):
        self.model_name = "mock"
        self.backend = "mock"
        self.config = {"max_tokens": 2000, "timeout": 60}
        self.seen_evidence_texts: list[str] = []

    def call(self, system, messages, max_tokens=None, temperature=0.1, retries=3):
        user = messages[-1]["content"] if messages else ""
        self.seen_evidence_texts.append(user)
        verdict = "incorrect" if "INCORRECT_MARKER" in user else "correct"
        payload = f'{{"verdict": "{verdict}", "confidence": "high"}}'
        return ModelResponse(
            content=payload, reasoning="", tokens=50,
            raw_text=payload, finish_reason="stop",
        )


def test_score_evidence_public_api_on_synthetic_stmt():
    """Lock in score_evidence: takes (statement, evidence, client) and
    returns the full scoring dict. Shape-contract test — anything that
    breaks argument order, return keys, or import path must fail loudly.
    """
    from indra.statements import Phosphorylation, Agent, Evidence
    from indra_belief import score_evidence

    stmt = Phosphorylation(
        Agent("RPS6KA1"), Agent("YBX1"),
        residue="S", position="102",
    )
    ev = Evidence(
        source_api="reach",
        text="RSK1 phosphorylates YB-1 at S102 in response to stress.",
    )
    result = score_evidence(stmt, ev, _MockClient(), voting_k=1)

    for key in ("score", "verdict", "confidence", "tier",
                 "grounding_status", "provenance_triggered", "tokens"):
        assert key in result, f"missing public API key: {key}"

    assert result["verdict"] in ("correct", "incorrect", None)
    assert result["confidence"] in ("high", "medium", "low", None)
    assert 0.0 <= result["score"] <= 1.0
    assert isinstance(result["tokens"], int)


def test_score_statement_iterates_evidence_list():
    """score_statement must honour INDRA's abstraction: one per-sentence
    dict per Evidence in statement.evidence, in order. Empty evidence
    returns []."""
    from indra.statements import Phosphorylation, Agent, Evidence
    from indra_belief import score_statement

    stmt = Phosphorylation(
        Agent("RPS6KA1"), Agent("YBX1"),
        residue="S", position="102",
    )
    # Middle evidence is the only one tagged INCORRECT_MARKER — the mock
    # returns verdict based on that marker, so the result pattern must be
    # [correct, incorrect, correct]. If the implementation re-scored the
    # same evidence object three times instead of iterating, the pattern
    # would collapse to [correct, correct, correct] or [incorrect]*3.
    stmt.evidence = [
        Evidence(source_api="reach",   text="RSK1 phosphorylates YB-1 at S102."),
        Evidence(source_api="sparser", text="INCORRECT_MARKER kinase-dead RSK1 failed to phosphorylate YB-1 at S102."),
        Evidence(source_api="medscan", text="Phosphorylation of YB-1 Ser102 depends on RSK1 activity."),
    ]

    client = _MockClient()
    results = score_statement(stmt, client, voting_k=1)

    assert isinstance(results, list)
    assert len(results) == len(stmt.evidence)
    verdicts = [r["verdict"] for r in results]
    assert verdicts == ["correct", "incorrect", "correct"], (
        f"expected per-evidence iteration with marker-driven verdicts, got {verdicts}"
    )
    # Each evidence text must have reached the client exactly once, in order
    assert len(client.seen_evidence_texts) == 3
    for ev, seen in zip(stmt.evidence, client.seen_evidence_texts):
        assert ev.text in seen, f"evidence {ev.text!r} never reached the client"

    # Empty-evidence Statement returns [] and makes zero LLM calls
    empty_stmt = Phosphorylation(Agent("A"), Agent("B"))
    empty_stmt.evidence = []
    empty_client = _MockClient()
    assert score_statement(empty_stmt, empty_client, voting_k=1) == []
    assert empty_client.seen_evidence_texts == []


def test_format_entity_lookups_uses_raw_text():
    """Tool-use lookups must target the reader-extracted mention, not the
    already-canonical entity name. Looking up the canonical just confirms
    Gilda's existing decision; the raw_text is what actually needs
    disambiguation. Falls back to name only when raw_text is None.
    """
    captured_args: list[dict] = []

    def fake_lookup(args):
        captured_args.append(args)
        return f'lookup_gene("{args["entity_name"]}"): stub'

    # Monkey-patch through the scorer module import path
    from indra_belief.scorers import scorer as s
    from indra_belief.tools import gilda_tools as gt

    original = gt.lookup_gene_executor
    gt.lookup_gene_executor = fake_lookup
    try:
        # Build a minimal record-like object
        class FakeEntity:
            def __init__(self, name, raw_text):
                self.name = name
                self.raw_text = raw_text

        class FakeRecord:
            subject_entity = FakeEntity("RPS6KA1", "RSK1")
            object_entity = FakeEntity("YBX1", "YB-1")

        block = s._format_entity_lookups(FakeRecord())
        targets = [a["entity_name"] for a in captured_args]
        assert "RSK1" in targets, f"expected raw_text RSK1, got {targets}"
        assert "YB-1" in targets, f"expected raw_text YB-1, got {targets}"
        assert "RPS6KA1" not in targets, f"should not look up canonical RPS6KA1"
    finally:
        gt.lookup_gene_executor = original
