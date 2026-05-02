"""Typed probe-request and probe-response shapes for the S-phase pipeline.

Each probe is one question with a closed answer set. The architectural
commitment is single-decision-per-call: the parser commits ONE value
from ONE small enum, not a multi-slot extraction. See doctrine §2 + §5
for the four probe definitions and the adjudicator decision table.

Substrate is a question-router. Where it can answer deterministically
(M9 entity-first LOF, exact CATALOG match, explicit hedge marker in
scope), it returns ProbeResponse with source="substrate". Where it has
at most a hint, it returns a ProbeRequest with substrate_hint set; the
LLM answers but the hint never prefills the answer.

Validation is load-bearing: probe-kind ↔ answer-set pairing is
enforced at construction time. A relation_axis response carrying a
scope-axis answer raises ValueError, not a silent miscategorization.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args


# Probe kinds — one per question. Each kind has its own closed answer set.
ProbeKind = Literal["subject_role", "object_role", "relation_axis", "scope"]


# Answer sets per probe kind. The split is the architectural discipline:
# at most ~8 closed values per probe, classifier-friendly with few-shots.

SubjectRoleAnswer = Literal[
    "present_as_subject",   # named as the actor of the relevant event
    "present_as_object",    # named as the target/recipient (role-swap candidate)
    "present_as_mediator",  # named, but as an intermediate node in a chain
    "present_as_decoy",     # named in a different relation than the claim's
    "absent",               # not named (post-alias-resolution)
]

# Same answer set as subject_role; distinct alias for documentation /
# downstream switch-case readability.
ObjectRoleAnswer = SubjectRoleAnswer

RelationAxisAnswer = Literal[
    "direct_sign_match",       # relation present on claim axis with claim sign
    "direct_sign_mismatch",    # relation present on claim axis with opposite sign
    "direct_axis_mismatch",    # relation present but on different axis
    "direct_partner_mismatch", # binding axis present but partner type wrong
                               # (e.g., DNA-binding when claim is Complex/protein)
    "via_mediator",            # relation via intermediate (indirect chain)
    "via_mediator_partial",    # chain detected but terminal entity unresolved
    "no_relation",             # no relation between resolved entities
    "abstain",                 # text underdetermines
]

ScopeAnswer = Literal[
    "asserted",  # unconditionally claimed
    "hedged",    # claimed under hypothesis / may / might / proposed / likely
    "negated",   # explicitly denied ("X did NOT activate Y")
    "abstain",   # text does not commit
]

# Perturbation marker — substrate-only side field on the subject_role probe.
# Detected via M9 entity-first regex: LOF (knockdown / KO / siRNA / inhibitor /
# null mutant / dominant-negative / depletion); GOF (overexpression /
# constitutively-active / forced expression). Never set by an LLM — the
# adjudicator's §5.1 sign-propagation rule consumes it deterministically.
PerturbationMarker = Literal["none", "LOF", "GOF"]

# Provenance: did substrate answer or did LLM (or did the probe abstain)?
ProbeSource = Literal["substrate", "llm", "abstain"]

ProbeConfidence = Literal["high", "medium", "low"]


_VALID_PROBE_KIND = frozenset(get_args(ProbeKind))
_VALID_SUBJECT_ROLE = frozenset(get_args(SubjectRoleAnswer))
_VALID_OBJECT_ROLE = frozenset(get_args(ObjectRoleAnswer))
_VALID_RELATION_AXIS = frozenset(get_args(RelationAxisAnswer))
_VALID_SCOPE = frozenset(get_args(ScopeAnswer))
_VALID_PERTURBATION_MARKER = frozenset(get_args(PerturbationMarker))
_VALID_PROBE_SOURCE = frozenset(get_args(ProbeSource))
_VALID_PROBE_CONFIDENCE = frozenset(get_args(ProbeConfidence))


# Map probe kind → its valid answer set, used by ProbeResponse validation.
_ANSWER_SETS: dict[str, frozenset[str]] = {
    "subject_role": _VALID_SUBJECT_ROLE,
    "object_role": _VALID_OBJECT_ROLE,
    "relation_axis": _VALID_RELATION_AXIS,
    "scope": _VALID_SCOPE,
}


def _reject(field_name: str, value: object, valid: frozenset) -> None:
    if value not in valid:
        raise ValueError(
            f"{field_name}={value!r} is not a valid value. "
            f"Expected one of {sorted(valid)}"
        )


@dataclass(frozen=True)
class ProbeRequest:
    """One probe question to be answered by LLM.

    Carries the question's kind (which probe), the relevant claim
    component (subject for subject_role; object for object_role; the
    pair-axis-sign tuple for relation_axis; the relation-span context
    for scope), and the evidence text.

    `substrate_hint` is set when substrate has narrowed the question
    but cannot answer deterministically — e.g., entities are present
    but their syntactic role is ambiguous. The hint is given to the LLM
    as auxiliary context but does NOT prefill the answer.
    """
    kind: ProbeKind
    claim_component: str
    evidence_text: str
    substrate_hint: str | None = None

    def __post_init__(self) -> None:
        _reject("kind", self.kind, _VALID_PROBE_KIND)


@dataclass(frozen=True)
class ProbeResponse:
    """One probe's answer.

    `answer` is one value from the probe-kind's closed answer set,
    validated at construction. `source` distinguishes substrate-derived
    answers (deterministic, fast-path) from LLM-derived answers
    (escalation) from abstain (no answer possible).

    `perturbation` is set ONLY when kind="subject_role" and substrate
    detected a perturbation marker. None otherwise. Validated.

    `span` is an optional substring of evidence_text identifying which
    region grounded the answer. Informational; not consumed by adjudicator.

    `rationale` is INFORMATIONAL ONLY — decision logic must not read it.
    If something cannot be captured by the typed fields, the answer set
    or side-field set must grow, not the rationale.
    """
    kind: ProbeKind
    answer: str
    source: ProbeSource
    confidence: ProbeConfidence = "medium"
    perturbation: PerturbationMarker | None = None
    span: str | None = None
    rationale: str = ""

    def __post_init__(self) -> None:
        _reject("kind", self.kind, _VALID_PROBE_KIND)
        _reject("source", self.source, _VALID_PROBE_SOURCE)
        _reject("confidence", self.confidence, _VALID_PROBE_CONFIDENCE)
        valid = _ANSWER_SETS[self.kind]
        _reject(f"answer (kind={self.kind})", self.answer, valid)
        if self.perturbation is not None:
            if self.kind != "subject_role":
                raise ValueError(
                    f"perturbation field is only valid for kind='subject_role', "
                    f"got kind={self.kind!r}"
                )
            _reject("perturbation", self.perturbation,
                    _VALID_PERTURBATION_MARKER)


@dataclass(frozen=True)
class ProbeBundle:
    """The four probe responses for a single (claim, evidence) pair.

    The adjudicator consumes a ProbeBundle (plus the EvidenceContext for
    the §5.4 final-arm substrate-fallback) and returns an Adjudication.

    Field-to-kind alignment is enforced: subject_role field must hold a
    kind='subject_role' response, etc. This catches assembly-side bugs
    (passing the relation_axis response into the scope slot) at
    construction rather than as silent miscategorization.
    """
    subject_role: ProbeResponse
    object_role: ProbeResponse
    relation_axis: ProbeResponse
    scope: ProbeResponse

    def __post_init__(self) -> None:
        if self.subject_role.kind != "subject_role":
            raise ValueError(
                f"subject_role slot must hold a kind='subject_role' response, "
                f"got kind={self.subject_role.kind!r}"
            )
        if self.object_role.kind != "object_role":
            raise ValueError(
                f"object_role slot must hold a kind='object_role' response, "
                f"got kind={self.object_role.kind!r}"
            )
        if self.relation_axis.kind != "relation_axis":
            raise ValueError(
                f"relation_axis slot must hold a kind='relation_axis' response, "
                f"got kind={self.relation_axis.kind!r}"
            )
        if self.scope.kind != "scope":
            raise ValueError(
                f"scope slot must hold a kind='scope' response, "
                f"got kind={self.scope.kind!r}"
            )
