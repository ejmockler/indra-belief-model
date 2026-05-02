"""Type-validation tests for src/indra_belief/scorers/probes/types.py.

Each closed answer set is exercised across all valid values + at least
one rejection case. Field-to-kind alignment in ProbeBundle is checked
to ensure constructor catches mis-slotted responses.
"""
from __future__ import annotations

import pytest

from indra_belief.scorers.probes.types import (
    ProbeBundle,
    ProbeRequest,
    ProbeResponse,
)


# --- Probe kind / answer-set coverage ---------------------------------------

@pytest.mark.parametrize("answer", [
    "present_as_subject",
    "present_as_object",
    "present_as_mediator",
    "present_as_decoy",
    "absent",
])
def test_subject_role_valid_answers(answer: str) -> None:
    r = ProbeResponse(kind="subject_role", answer=answer, source="substrate")
    assert r.answer == answer


@pytest.mark.parametrize("answer", [
    "present_as_subject",
    "present_as_object",
    "present_as_mediator",
    "present_as_decoy",
    "absent",
])
def test_object_role_valid_answers(answer: str) -> None:
    r = ProbeResponse(kind="object_role", answer=answer, source="llm")
    assert r.answer == answer


@pytest.mark.parametrize("answer", [
    "direct_sign_match",
    "direct_sign_mismatch",
    "direct_axis_mismatch",
    "direct_partner_mismatch",
    "via_mediator",
    "via_mediator_partial",
    "no_relation",
    "abstain",
])
def test_relation_axis_valid_answers(answer: str) -> None:
    r = ProbeResponse(kind="relation_axis", answer=answer, source="substrate")
    assert r.answer == answer


@pytest.mark.parametrize("answer", ["asserted", "hedged", "negated", "abstain"])
def test_scope_valid_answers(answer: str) -> None:
    r = ProbeResponse(kind="scope", answer=answer, source="substrate")
    assert r.answer == answer


# --- Cross-kind rejection ---------------------------------------------------

def test_relation_axis_rejects_subject_role_answer() -> None:
    with pytest.raises(ValueError, match="answer"):
        ProbeResponse(kind="relation_axis", answer="present_as_subject",
                      source="substrate")


def test_scope_rejects_relation_axis_answer() -> None:
    with pytest.raises(ValueError, match="answer"):
        ProbeResponse(kind="scope", answer="direct_sign_match",
                      source="substrate")


def test_subject_role_rejects_scope_answer() -> None:
    with pytest.raises(ValueError, match="answer"):
        ProbeResponse(kind="subject_role", answer="hedged", source="llm")


# --- Invalid kind / source / confidence ------------------------------------

def test_invalid_kind_rejected() -> None:
    with pytest.raises(ValueError, match="kind"):
        ProbeResponse(kind="role", answer="absent", source="substrate")  # type: ignore[arg-type]


def test_invalid_source_rejected() -> None:
    with pytest.raises(ValueError, match="source"):
        ProbeResponse(kind="subject_role", answer="absent",
                      source="oracle")  # type: ignore[arg-type]


def test_invalid_confidence_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        ProbeResponse(kind="subject_role", answer="absent",
                      source="substrate",
                      confidence="certain")  # type: ignore[arg-type]


# --- Perturbation side field ------------------------------------------------

@pytest.mark.parametrize("marker", ["none", "LOF", "GOF"])
def test_perturbation_valid_on_subject_role(marker: str) -> None:
    r = ProbeResponse(kind="subject_role", answer="present_as_subject",
                      source="substrate", perturbation=marker)
    assert r.perturbation == marker


def test_perturbation_invalid_on_object_role() -> None:
    with pytest.raises(ValueError, match="perturbation"):
        ProbeResponse(kind="object_role", answer="present_as_object",
                      source="substrate", perturbation="LOF")


def test_perturbation_invalid_on_relation_axis() -> None:
    with pytest.raises(ValueError, match="perturbation"):
        ProbeResponse(kind="relation_axis", answer="direct_sign_match",
                      source="substrate", perturbation="LOF")


def test_perturbation_invalid_on_scope() -> None:
    with pytest.raises(ValueError, match="perturbation"):
        ProbeResponse(kind="scope", answer="asserted",
                      source="substrate", perturbation="GOF")


def test_perturbation_invalid_marker_value() -> None:
    with pytest.raises(ValueError, match="perturbation"):
        ProbeResponse(kind="subject_role", answer="present_as_subject",
                      source="substrate",
                      perturbation="dominant_negative")  # type: ignore[arg-type]


def test_perturbation_none_is_default() -> None:
    r = ProbeResponse(kind="subject_role", answer="present_as_subject",
                      source="substrate")
    assert r.perturbation is None


# --- ProbeRequest -----------------------------------------------------------

def test_probe_request_valid() -> None:
    req = ProbeRequest(kind="relation_axis",
                       claim_component="MAPK1 -> JUN (activity, positive)",
                       evidence_text="MAPK1 activates JUN",
                       substrate_hint="CATALOG suggests direct sign-match")
    assert req.kind == "relation_axis"
    assert req.substrate_hint is not None


def test_probe_request_no_hint_default() -> None:
    req = ProbeRequest(kind="scope", claim_component="X inhibits Y",
                       evidence_text="X inhibited Y in vitro.")
    assert req.substrate_hint is None


def test_probe_request_invalid_kind() -> None:
    with pytest.raises(ValueError, match="kind"):
        ProbeRequest(kind="grounding", claim_component="X",  # type: ignore[arg-type]
                     evidence_text="...")


# --- ProbeBundle ------------------------------------------------------------

def _r(kind: str, answer: str) -> ProbeResponse:
    return ProbeResponse(kind=kind, answer=answer, source="substrate")  # type: ignore[arg-type]


def test_bundle_valid() -> None:
    bundle = ProbeBundle(
        subject_role=_r("subject_role", "present_as_subject"),
        object_role=_r("object_role", "present_as_object"),
        relation_axis=_r("relation_axis", "direct_sign_match"),
        scope=_r("scope", "asserted"),
    )
    assert bundle.subject_role.answer == "present_as_subject"
    assert bundle.relation_axis.answer == "direct_sign_match"


def test_bundle_rejects_misslotted_subject_role() -> None:
    with pytest.raises(ValueError, match="subject_role slot"):
        ProbeBundle(
            subject_role=_r("scope", "asserted"),
            object_role=_r("object_role", "present_as_object"),
            relation_axis=_r("relation_axis", "direct_sign_match"),
            scope=_r("scope", "asserted"),
        )


def test_bundle_rejects_misslotted_relation_axis() -> None:
    with pytest.raises(ValueError, match="relation_axis slot"):
        ProbeBundle(
            subject_role=_r("subject_role", "present_as_subject"),
            object_role=_r("object_role", "present_as_object"),
            relation_axis=_r("subject_role", "absent"),
            scope=_r("scope", "asserted"),
        )


def test_bundle_rejects_misslotted_scope() -> None:
    with pytest.raises(ValueError, match="scope slot"):
        ProbeBundle(
            subject_role=_r("subject_role", "present_as_subject"),
            object_role=_r("object_role", "present_as_object"),
            relation_axis=_r("relation_axis", "direct_sign_match"),
            scope=_r("relation_axis", "abstain"),
        )


# --- Frozen dataclass invariant --------------------------------------------

def test_response_is_frozen() -> None:
    r = _r("subject_role", "absent")
    with pytest.raises(Exception):
        r.answer = "present_as_subject"  # type: ignore[misc]


def test_request_is_frozen() -> None:
    req = ProbeRequest(kind="scope", claim_component="X", evidence_text="...")
    with pytest.raises(Exception):
        req.substrate_hint = "..."  # type: ignore[misc]
