"""Structural tests for the grounding sub-verifier.

Covers verdict construction with Gilda context preserved, crash-to-
uncertain recovery, and prompt contract. LLM reliability on real entities
is measured under task #20 calibration.
"""
import json
from dataclasses import dataclass, field

import pytest

from indra_belief.scorers.commitments import GroundingVerdict
from indra_belief.scorers.grounding import (
    _build_user_message,
    _extract_json_object,
    verify_grounding,
)


# ---------------------------------------------------------------------------
# Test doubles: minimal GroundedEntity-shaped stand-in + mock ModelClient
# ---------------------------------------------------------------------------

@dataclass
class _FakeEntity:
    """Duck-typed GroundedEntity for verify_grounding tests — avoids
    instantiating the real GroundedEntity (which runs Gilda). The grounding
    sub-verifier only reads attributes, never calls methods."""
    name: str
    raw_text: str | None = None
    canonical: str | None = None
    db: str | None = None
    db_id: str | None = None
    aliases: list[str] = field(default_factory=list)
    all_names: list[str] = field(default_factory=list)
    is_family: bool = False
    family_members: list[str] = field(default_factory=list)
    description: str = ""
    is_pseudogene: bool = False
    verification_status: str | None = None
    verification_note: str = ""
    gilda_score: float | None = None
    is_low_confidence: bool = False
    is_known_alias: bool = True
    competing_candidates: list[dict] = field(default_factory=list)
    text_top_name: str | None = None


@dataclass
class _MockResponse:
    content: str
    reasoning: str = ""
    tokens: int = 0
    raw_text: str = ""
    finish_reason: str = "stop"

    def __post_init__(self):
        if not self.raw_text:
            self.raw_text = (self.reasoning + "\n" + self.content).strip()


class _MockClient:
    def __init__(self, responses: list[_MockResponse]):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def call(self, *, system, messages, max_tokens, temperature,
             response_format=None, reasoning_effort=None, **kwargs):
        self.calls.append({
            "system": system, "messages": messages,
            "max_tokens": max_tokens, "temperature": temperature,
            "response_format": response_format,
            "reasoning_effort": reasoning_effort,
        })
        return self.responses.pop(0)


class _RaisingClient:
    def __init__(self, exc):
        self.exc = exc

    def call(self, **_kwargs):
        raise self.exc


def _ok_response(status: str, rationale: str = "reason") -> _MockResponse:
    return _MockResponse(content=json.dumps({
        "status": status, "rationale": rationale,
    }))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestVerifyGrounding:
    def test_mentioned_verdict(self):
        entity = _FakeEntity(name="MAPK1", db="HGNC", db_id="6871")
        client = _MockClient([_ok_response("mentioned", "MAPK1 named directly")])
        v = verify_grounding(entity, "MAPK1 phosphorylates BRAF.", client)
        assert isinstance(v, GroundingVerdict)
        assert v.status == "mentioned"
        assert v.claim_entity == "MAPK1"
        assert v.db_ns == "HGNC"
        assert v.db_id == "6871"
        # rationale is informational only — no assertion on content

    def test_equivalent_verdict_for_family(self):
        entity = _FakeEntity(name="AKT", db="FPLX", db_id="AKT",
                             is_family=True, family_members=["AKT1", "AKT2", "AKT3"])
        client = _MockClient([_ok_response("equivalent", "AKT1 is an AKT family member")])
        v = verify_grounding(entity, "AKT1 was phosphorylated in response to insulin.", client)
        assert v.status == "equivalent"
        assert v.is_family is True

    def test_not_present_verdict(self):
        entity = _FakeEntity(name="TP53")
        client = _MockClient([_ok_response("not_present", "TP53 not referenced")])
        v = verify_grounding(entity, "MAPK1 phosphorylates BRAF.", client)
        assert v.status == "not_present"

    def test_every_valid_status_accepted(self):
        entity = _FakeEntity(name="X")
        for status in ("mentioned", "equivalent", "not_present", "uncertain"):
            client = _MockClient([_ok_response(status)])
            v = verify_grounding(entity, "some text", client)
            assert v.status == status

    def test_structured_gilda_context_always_preserved(self):
        entity = _FakeEntity(
            name="AKT", db="FPLX", db_id="AKT",
            gilda_score=0.85, is_family=True, is_pseudogene=False,
        )
        client = _MockClient([_ok_response("mentioned")])
        v = verify_grounding(entity, "AKT activates mTOR.", client)
        assert v.db_ns == "FPLX"
        assert v.db_id == "AKT"
        assert v.gilda_score == 0.85
        assert v.is_family is True


# ---------------------------------------------------------------------------
# Failure paths — always return a verdict, never raise, never return None
# ---------------------------------------------------------------------------

class TestFailureModes:
    def test_transport_error_returns_uncertain(self):
        entity = _FakeEntity(name="X", db="HGNC", db_id="1")
        client = _RaisingClient(ConnectionError("endpoint down"))
        v = verify_grounding(entity, "some text", client)
        assert v.status == "uncertain"
        # Rationale is informational; contract is only that it's populated
        # (non-empty) on the error path, not its specific content.
        assert v.rationale
        # Gilda context still preserved on error
        assert v.db_ns == "HGNC"
        assert v.db_id == "1"

    def test_timeout_returns_uncertain(self):
        entity = _FakeEntity(name="X")
        client = _RaisingClient(TimeoutError("read timed out"))
        v = verify_grounding(entity, "some text", client)
        assert v.status == "uncertain"

    def test_malformed_json_returns_uncertain(self):
        entity = _FakeEntity(name="X")
        client = _MockClient([_MockResponse(content="not json at all")])
        v = verify_grounding(entity, "some text", client)
        assert v.status == "uncertain"
        assert v.rationale  # populated on error

    def test_out_of_vocabulary_status_returns_uncertain(self):
        entity = _FakeEntity(name="X")
        client = _MockClient([_MockResponse(content=json.dumps({
            "status": "kinda_present", "rationale": "model drift",
        }))])
        v = verify_grounding(entity, "some text", client)
        assert v.status == "uncertain"
        assert v.rationale  # populated on error

    def test_missing_status_field_returns_uncertain(self):
        entity = _FakeEntity(name="X")
        client = _MockClient([_MockResponse(content=json.dumps({
            "rationale": "no status field",
        }))])
        v = verify_grounding(entity, "some text", client)
        assert v.status == "uncertain"

    def test_empty_evidence_returns_uncertain_without_calling(self):
        entity = _FakeEntity(name="X")
        client = _MockClient([])  # would raise on .pop(0) if called
        v = verify_grounding(entity, "", client)
        assert v.status == "uncertain"
        assert client.calls == []

    def test_non_string_rationale_coerces_to_empty(self):
        """A rationale that's not a string should NOT crash verdict creation."""
        entity = _FakeEntity(name="X")
        client = _MockClient([_MockResponse(content=json.dumps({
            "status": "mentioned", "rationale": ["list", "not", "string"],
        }))])
        v = verify_grounding(entity, "X is here", client)
        assert v.status == "mentioned"
        assert v.rationale == ""


# ---------------------------------------------------------------------------
# Prompt contract
# ---------------------------------------------------------------------------

class TestPrompt:
    def test_system_prompt_lists_all_four_statuses(self):
        entity = _FakeEntity(name="X")
        client = _MockClient([_ok_response("mentioned")])
        verify_grounding(entity, "some text", client)
        system = client.calls[0]["system"]
        for status in ("mentioned", "equivalent", "not_present", "uncertain"):
            assert status in system

    def test_user_message_includes_claim_entity_name(self):
        entity = _FakeEntity(name="MAPK1")
        client = _MockClient([_ok_response("mentioned")])
        verify_grounding(entity, "MAPK1 activates BRAF.", client)
        user_msg = client.calls[0]["messages"][0]["content"]
        assert "MAPK1" in user_msg

    def test_user_message_includes_grounding_id_when_available(self):
        entity = _FakeEntity(name="TP53", db="HGNC", db_id="11998")
        client = _MockClient([_ok_response("mentioned")])
        verify_grounding(entity, "TP53 is active.", client)
        user_msg = client.calls[0]["messages"][0]["content"]
        assert "HGNC:11998" in user_msg

    def test_user_message_includes_aliases(self):
        entity = _FakeEntity(name="MAPK1", aliases=["ERK", "ERK2", "p42-MAPK"])
        client = _MockClient([_ok_response("mentioned")])
        verify_grounding(entity, "ERK was phosphorylated.", client)
        user_msg = client.calls[0]["messages"][0]["content"]
        assert "ERK" in user_msg

    def test_user_message_includes_family_members_when_family(self):
        entity = _FakeEntity(name="AKT", is_family=True,
                             family_members=["AKT1", "AKT2", "AKT3"])
        client = _MockClient([_ok_response("equivalent")])
        verify_grounding(entity, "AKT1 was phosphorylated.", client)
        user_msg = client.calls[0]["messages"][0]["content"]
        assert "AKT1" in user_msg

    def test_pseudogene_flag_surfaces_in_prompt(self):
        entity = _FakeEntity(name="HMGB1P1", is_pseudogene=True)
        client = _MockClient([_ok_response("not_present")])
        verify_grounding(entity, "HMGB1 was upregulated.", client)
        user_msg = client.calls[0]["messages"][0]["content"]
        assert "pseudogene" in user_msg.lower()

    def test_low_confidence_gilda_score_in_prompt(self):
        entity = _FakeEntity(name="X", is_low_confidence=True, gilda_score=0.42)
        client = _MockClient([_ok_response("uncertain")])
        verify_grounding(entity, "something", client)
        user_msg = client.calls[0]["messages"][0]["content"]
        assert "0.42" in user_msg

    def test_build_user_message_is_stable(self):
        """Deterministic formatting — used for debugging/logging."""
        entity = _FakeEntity(name="MAPK1", db="HGNC", db_id="6871",
                             aliases=["ERK", "ERK2"])
        msg = _build_user_message(entity, "MAPK1 activates BRAF.")
        assert "Claim entity: MAPK1" in msg
        assert "HGNC:6871" in msg
        assert "ERK" in msg
        assert 'Evidence: "MAPK1 activates BRAF."' in msg

    def test_pseudogene_rule_allows_explicit_symbol(self):
        """Gate-#36 fix: the pseudogene rule must distinguish direct
        pseudogene-symbol mentions (allowed) from parent-symbol-only
        collisions (rejected)."""
        entity = _FakeEntity(name="PTENP1", is_pseudogene=True)
        client = _MockClient([_ok_response("mentioned")])
        verify_grounding(entity, "PTENP1 expression was elevated.", client)
        system = client.calls[0]["system"]
        # The rule must explicitly acknowledge that distinct pseudogene
        # symbols count, not just reject all pseudogene claims.
        lower = system.lower()
        assert "distinct symbol" in lower or "ptenp1" in lower
        assert "parent-gene symbol" in lower or "parent-symbol" in lower
        assert "collision" in lower or "do NOT count" in system or "not a match" in lower

    def test_generic_class_noun_rule_present(self):
        """Task #51: prompt must explicitly handle generic class nouns
        (Histone, Phosphatase, Kinase) so 'TBK1 phosphatase' compound
        terms don't match a 'Phosphatase' claim entity. Mirrors v5
        dual-run #18 (Histone/JDP2 false positive)."""
        entity = _FakeEntity(name="Phosphatase")
        client = _MockClient([_ok_response("not_present")])
        verify_grounding(entity, "PPM1B is a TBK1 phosphatase.", client)
        system = client.calls[0]["system"]
        lower = system.lower()
        assert "generic class noun" in lower or "generic biochemical class" in lower, (
            "Rule 5 (generic class nouns) missing — "
            "'TBK1 phosphatase' will match a Phosphatase claim entity"
        )
        # Compound-term examples must be in the prompt so the model can
        # generalize (not just memorize histone deacetylase).
        assert "histone deacetylase" in lower or "tbk1 phosphatase" in lower

    def test_cross_gene_collision_rule_present(self):
        """Task #51: cross-gene grounding collisions (p97 → GEMIN4 via
        gilda alias bug) must emit not_present, not uncertain. Mirrors
        v5 dual-run #9 (GEMIN4/PTGS2 false positive)."""
        entity = _FakeEntity(name="GEMIN4")
        client = _MockClient([_ok_response("not_present")])
        verify_grounding(entity, "p97 ATPase regulates ER stress.", client)
        system = client.calls[0]["system"]
        lower = system.lower()
        assert "cross-gene grounding collision" in lower or \
               "alias collision is a grounding bug" in lower, (
            "Rule 6 (cross-gene collisions) missing — "
            "p97/GEMIN4 alias bugs will pass as uncertain not not_present"
        )

    def test_case_sensitivity_rule_present(self):
        """Task #51: short uppercase symbols are case-distinct from
        lowercase forms (FAS gene vs FAs focal adhesions). Mirrors v5
        dual-run #13 (Activation(VCL, FAS) false negative)."""
        entity = _FakeEntity(name="FAS", db="HGNC", db_id="11920")
        client = _MockClient([_ok_response("not_present")])
        verify_grounding(entity, "vinculin localizes to FAs.", client)
        system = client.calls[0]["system"]
        lower = system.lower()
        assert "case sensitiv" in lower, (
            "Rule 7 (case sensitivity) missing — "
            "FAs (focal adhesions) will match the FAS gene"
        )
        # Concrete example must be present.
        assert "fas" in lower and "focal adhesion" in lower

    def test_no_grounding_emits_generic_hint(self):
        """When the claim entity has no db grounding (Phosphatase, Kinase),
        the user-message must hint at 'generic class noun' so the model
        applies Rule 5 even without the prompt-side trigger."""
        entity = _FakeEntity(name="Phosphatase", db=None, db_id=None)
        msg = _build_user_message(entity, "PPM1B is a TBK1 phosphatase.")
        assert "generic class noun" in msg.lower(), (
            "no-db hint missing — the model must be told this is "
            "likely a generic class noun, not a specific gene"
        )


# ---------------------------------------------------------------------------
# JSON extraction (shared helper behavior)
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_extracts_last_valid_object(self):
        text = '{"status": "draft"} and then {"status": "mentioned"}'
        obj = _extract_json_object(text)
        assert obj == {"status": "mentioned"}

    def test_returns_none_on_no_valid_json(self):
        assert _extract_json_object("no json") is None
        assert _extract_json_object("") is None


class TestJsonModeOptIn:
    """verify_grounding must opt in to JSON mode on every call.

    Same root cause as #50 in parse_evidence: gemma backends ramble in
    `content` rather than emitting JSON, hitting `finish_reason=length`
    on what should be a one-line status verdict. Forcing
    response_format prevents the prefatory reasoning."""

    def test_passes_json_mode(self):
        entity = _FakeEntity(name="STAT3")
        client = _MockClient([_ok_response("mentioned")])
        verify_grounding(entity, "STAT3 was phosphorylated.", client)
        assert len(client.calls) == 1
        assert client.calls[0]["response_format"] == {"type": "json_object"}

    def test_reasoning_effort_is_none(self):
        """Same rationale as parse_evidence: verify_grounding is pure
        classification (4-way status enum), not judgment. Disable
        reasoning so the backend keeps responses bounded within the
        500-token budget."""
        entity = _FakeEntity(name="STAT3")
        client = _MockClient([_ok_response("mentioned")])
        verify_grounding(entity, "STAT3 was phosphorylated.", client)
        assert client.calls[0]["reasoning_effort"] == "none"
