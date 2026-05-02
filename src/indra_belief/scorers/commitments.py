"""Typed intermediate commitments for the S-phase scorer.

Each sub-call produces or consumes one of these:
  parse_claim       → ClaimCommitment
  grounding_verify  → GroundingVerdict (per claim entity)
  adjudicate        → Adjudication

The S-phase deleted EvidenceCommitment / EvidenceAssertion (per doctrine
§7 migration discipline) — the parser no longer extracts a multi-slot
schema; the four probes (subject_role, object_role, relation_axis,
scope) commit single-decision answers in their closed sets, and the
adjudicator combines those via a flat decision table.

`rationale` fields on GroundingVerdict and Adjudication remain
INFORMATIONAL ONLY. No decision logic may read them. Reason codes
encode every gating-relevant fact.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, get_args


Axis = Literal[
    "activity",       # functional state change (Activation, Inhibition)
    "amount",         # expression/abundance change (IncreaseAmount, DecreaseAmount)
    "binding",        # physical association (Complex) — intrinsically neutral
    "modification",   # covalent modification (Phos, Dephos, Ubiq, Acetyl, Methyl, …)
    "localization",   # compartment change (Translocation) — intrinsically neutral
    "conversion",     # chemical/structural conversion — intrinsically neutral
    "gtp_state",      # GTP/GDP state change (GtpActivation, Gef, Gap)
    "unclear",        # axis cannot be determined from the text
    "absent",         # no relationship described
]

Sign = Literal[
    "positive",   # activation, increase, modification added, GTP-loaded
    "negative",   # inhibition, decrease, modification removed, GTP-unloaded
    "neutral",    # binding / translocation / conversion — axis is not signed
]

Perturbation = Literal[
    "none",               # agent is the direct regulator; literal sign is final
    "loss_of_function",   # LOF perturbation — adjudicator inverts effective sign
    "gain_of_function",   # GOF perturbation — preserves sign (amplified)
]

GroundingStatus = Literal[
    "mentioned",     # claim entity is named directly in evidence
    "equivalent",    # evidence names an alias, family member, or fragment
    "not_present",   # no referent for the claim entity in the evidence
    "uncertain",     # evidence insufficient to decide
]

# S-phase reason codes (per doctrine §5.5; deferred codes removed).
ReasonCode = Literal[
    "match",                  # asserted relation matches claim
    "axis_mismatch",          # claim and evidence describe different kinds of change
    "sign_mismatch",          # same axis, opposite direction
    "grounding_gap",          # a claim entity is not referenced by evidence
    "role_swap",              # entities present but in swapped agent/target roles
    "hedging_hypothesis",     # evidence hedges the relationship itself
    "absent_relationship",    # evidence does not describe the claimed relationship
    "contradicted",           # evidence explicitly negates the claim
    "indirect_chain",         # claim implies direct; evidence shows an intermediate
    "binding_domain_mismatch", # binding-axis but wrong partner type
    "chain_extraction_gap",   # chain markers present but no extractable mediator
    "regex_substrate_match",  # final-arm CATALOG fallback rescued an abstain
]

Verdict = Literal["correct", "incorrect", "abstain"]
Confidence = Literal["high", "medium", "low"]


_VALID_AXIS = frozenset(get_args(Axis))
_VALID_SIGN = frozenset(get_args(Sign))
_VALID_PERTURBATION = frozenset(get_args(Perturbation))
_VALID_GROUNDING = frozenset(get_args(GroundingStatus))
_VALID_REASON = frozenset(get_args(ReasonCode))
_VALID_VERDICT = frozenset(get_args(Verdict))
_VALID_CONFIDENCE = frozenset(get_args(Confidence))

# Axes that carry a meaningful positive/negative direction. For these, sign
# MUST be "positive" or "negative" — "neutral" is a semantic error.
_SIGNED_AXES = frozenset({"activity", "amount", "modification", "gtp_state"})

# Axes that are intrinsically unsigned. For these, sign MUST be "neutral".
_UNSIGNED_AXES = frozenset({"binding", "localization", "conversion",
                            "unclear", "absent"})


def _reject(field_name: str, value: object, valid: frozenset[str]) -> None:
    if value not in valid:
        raise ValueError(
            f"{field_name}={value!r} is not a valid value. "
            f"Expected one of {sorted(valid)}"
        )


def _require_valid_axis_sign_pairing(axis: str, sign: str) -> None:
    if axis in _SIGNED_AXES and sign == "neutral":
        raise ValueError(
            f"axis={axis!r} is a signed axis; sign must be 'positive' or "
            f"'negative', got {sign!r}"
        )
    if axis in _UNSIGNED_AXES and sign != "neutral":
        raise ValueError(
            f"axis={axis!r} is a neutral axis; sign must be 'neutral', "
            f"got {sign!r}"
        )


@dataclass(frozen=True)
class ClaimCommitment:
    """Normalized structure of what a claim asserts.

    Produced deterministically from an INDRA Statement: statement type
    canonicalizes (axis, sign). The miRNA rule is the one known case
    where subject identity modulates the mapping (Inhibition by miRNA
    subject acts on target amount, not activity).
    """
    stmt_type: str
    subject: str
    objects: tuple[str, ...]
    axis: Axis
    sign: Sign
    site: str | None = None
    location_from: str | None = None
    location_to: str | None = None
    subject_is_mirna: bool = False

    def __post_init__(self) -> None:
        _reject("axis", self.axis, _VALID_AXIS)
        _reject("sign", self.sign, _VALID_SIGN)
        _require_valid_axis_sign_pairing(self.axis, self.sign)


@dataclass(frozen=True)
class GroundingVerdict:
    """Per-entity grounding check with structured Gilda context preserved.

    `rationale` is INFORMATIONAL ONLY — decision logic must not depend on it.
    """
    claim_entity: str
    status: GroundingStatus
    db_ns: str | None = None
    db_id: str | None = None
    gilda_score: float | None = None
    is_family: bool = False
    is_pseudogene: bool = False
    rationale: str = ""

    def __post_init__(self) -> None:
        _reject("status", self.status, _VALID_GROUNDING)


@dataclass(frozen=True)
class Adjudication:
    """Final verdict over typed commitments.

    `reasons` is a structured enum tuple: every disagreement maps to a
    ReasonCode so error stratification can aggregate mechanically.

    `rationale` is INFORMATIONAL ONLY — decision logic must not depend on it.
    """
    verdict: Verdict
    confidence: Confidence
    reasons: tuple[ReasonCode, ...] = field(default_factory=tuple)
    rationale: str = ""

    def __post_init__(self) -> None:
        _reject("verdict", self.verdict, _VALID_VERDICT)
        _reject("confidence", self.confidence, _VALID_CONFIDENCE)
        for r in self.reasons:
            _reject("reason", r, _VALID_REASON)


_VERDICT_SCORE = {
    ("correct", "high"):     0.95,
    ("correct", "medium"):   0.80,
    ("correct", "low"):      0.65,
    ("abstain", "high"):     0.50,
    ("abstain", "medium"):   0.50,
    ("abstain", "low"):      0.50,
    ("incorrect", "low"):    0.35,
    ("incorrect", "medium"): 0.20,
    ("incorrect", "high"):   0.05,
}


def adjudication_to_score(a: Adjudication) -> float:
    """Map an Adjudication to the [0, 1] belief score used downstream."""
    return _VERDICT_SCORE[(a.verdict, a.confidence)]
