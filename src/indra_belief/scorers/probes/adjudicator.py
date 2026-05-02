"""S-phase adjudicator: pure deterministic combinator over ProbeBundle.

Implements the doctrine §5 decision table. Every verdict has exactly one
explicit rule. The table is small (~15 rules with a fall-through default)
and exhaustive against the ProbeBundle's closed answer-set tuples.

Rule ordering encodes precedence:
  §5.1 perturbation pre-rule (sign propagation from subject_role)
  §5.2 canonical decision table
  §5.3 symmetric-binding handling (folded into table)
  §5.4 final-arm substrate-fallback (preserves the M3 hoist)
  §5.6 INDRA semantics: causal claims accept indirect chains
  §5.7 hedged-with-aligned-relation lifts to correct/low

Failure handling: when any probe has source="abstain" (LLM call failed),
the adjudicator emits abstain unless the substrate-fallback final arm
rescues. No retries; no cascades; no claim-aware reasoning beyond the
table.
"""
from __future__ import annotations

from indra_belief.scorers.commitments import (
    Adjudication,
    ClaimCommitment,
    GroundingVerdict,
)
from indra_belief.scorers.context import EvidenceContext
from indra_belief.scorers.probes.types import ProbeBundle, ProbeResponse


# Perturbation propagation: LOF inverts claim sign; GOF preserves; none preserves.
def _effective_claim_sign(claim_sign: str, perturbation: str | None) -> str:
    if perturbation == "LOF":
        if claim_sign == "positive":
            return "negative"
        if claim_sign == "negative":
            return "positive"
    return claim_sign


# INDRA causal claim types — these accept indirect chains (X→Z→Y is a
# valid Activation/Inhibition/IncreaseAmount/DecreaseAmount because the
# claim is about the upstream→downstream regulatory relationship, not a
# direct molecular contact). Direct claim types (Phosphorylation, Complex,
# Translocation, etc.) require X to directly contact Y.
_CAUSAL_STMT_TYPES = frozenset({
    "Activation", "Inhibition",
    "IncreaseAmount", "DecreaseAmount",
})


def _is_causal_claim(stmt_type: str) -> bool:
    return stmt_type in _CAUSAL_STMT_TYPES


def _final_arm_substrate_match(
    claim: ClaimCommitment, ctx: EvidenceContext,
) -> bool:
    """§5.4: return True iff ctx.detected_relations contains an aligned
    CATALOG match for (claim.subject, claim.objects[0], claim.axis)
    with the claim sign (or symmetric-binding equivalent).

    Mirrors the M3 substrate-fallback hoist preserved through R8.
    """
    if not claim.objects:
        return False
    target = claim.objects[0]
    for dr in ctx.detected_relations:
        # Direct alignment
        aligned = (
            dr.agent_canonical == claim.subject
            and dr.target_canonical == target
        )
        # Symmetric binding: (X,Y) ≡ (Y,X) for binding axis
        if claim.axis == "binding" and not aligned:
            aligned = (
                dr.agent_canonical == target
                and dr.target_canonical == claim.subject
            )
        if not aligned:
            continue
        if dr.axis != claim.axis:
            continue
        # Sign match — for binding/translocation axes both are "neutral";
        # for signed axes, sign must match the claim.
        if dr.sign == claim.sign:
            return True
        # Allow "neutral" detected sign on a neutral-axis claim
        if claim.sign == "neutral" and dr.sign == "neutral":
            return True
    return False


def _grounding_uncertain(groundings: tuple[GroundingVerdict, ...]) -> bool:
    return any(g.status == "uncertain" for g in groundings)


def _decide(
    bundle: ProbeBundle, claim: ClaimCommitment,
) -> tuple[str, str | None, str]:
    """Run the canonical decision table. Returns (verdict, reason, rationale).

    `verdict` ∈ {"correct", "incorrect", "abstain"}.
    `reason` is a ReasonCode string or None when no specific code applies.
    `rationale` is a short human-readable note (informational).
    """
    sr = bundle.subject_role.answer
    or_ = bundle.object_role.answer
    ra = bundle.relation_axis.answer
    sc = bundle.scope.answer

    # Any LLM-failure abstain on subject/object/relation/scope → adjudicator
    # cannot commit a verdict from probes alone. Return abstain; final-arm
    # substrate-fallback may still rescue if CATALOG matches.
    sources = (
        bundle.subject_role.source,
        bundle.object_role.source,
        bundle.relation_axis.source,
        bundle.scope.source,
    )
    if "abstain" in sources:
        return "abstain", None, "one or more probes abstained (LLM failure)"

    # Grounding-gap: subject or object not present in evidence.
    if sr == "absent":
        return "abstain", "grounding_gap", "claim subject not in evidence"
    if or_ == "absent":
        return "abstain", "grounding_gap", "claim object not in evidence"

    # Decoy: entities mentioned but not in the claim relation.
    # Treat as no-relation evidence.
    if sr == "present_as_decoy" or or_ == "present_as_decoy":
        return "incorrect", "absent_relationship", \
            "claim entity present only as decoy/control"

    # Mediator: subject or object in the middle of a chain.
    # Indirect chain — abstain (claim implies direct link).
    if sr == "present_as_mediator" or or_ == "present_as_mediator":
        return "abstain", "indirect_chain", \
            "claim entity is a chain mediator, not endpoint"

    # Role-swap: subject in object slot AND object in subject slot.
    # Only fires for non-binding axes (binding is symmetric §5.3).
    if (sr == "present_as_object" and or_ == "present_as_subject"
            and claim.axis != "binding"):
        return "incorrect", "role_swap", \
            "subject and object roles swapped in evidence"

    # From here both probes return present_as_subject and present_as_object
    # (or symmetric-binding equivalent). Now consult relation_axis.

    if ra == "no_relation":
        return "incorrect", "absent_relationship", \
            "no relation between resolved entities"

    if ra == "direct_axis_mismatch":
        return "incorrect", "axis_mismatch", \
            "relation present but on different axis"

    if ra == "direct_partner_mismatch":
        return "incorrect", "binding_domain_mismatch", \
            "binding-axis match but partner type incompatible"

    if ra == "direct_sign_mismatch":
        return "incorrect", "sign_mismatch", \
            "relation present but opposite sign"

    if ra == "via_mediator":
        # §5.6: INDRA semantics. Causal claims (Activation, Inhibition,
        # IncreaseAmount, DecreaseAmount) accept indirect chains —
        # "X activates Y via Z" is valid Activation(X, Y) at the
        # pathway level. Direct claims (Phosphorylation, Complex,
        # Translocation, etc.) require X→Y direct contact.
        if _is_causal_claim(claim.stmt_type):
            # Honor scope for causal indirect chains too.
            if sc == "asserted":
                return "correct", "match", \
                    "causal claim accepts indirect chain"
            if sc == "hedged":
                return "correct", "hedging_hypothesis", \
                    "causal indirect chain, hedged"
            if sc == "negated":
                return "incorrect", "contradicted", \
                    "causal indirect chain, negated"
            return "correct", "match", \
                "causal claim accepts indirect chain"
        return "abstain", "indirect_chain", \
            "direct claim type but evidence shows indirect chain"

    if ra == "via_mediator_partial":
        # Same causal-vs-direct distinction; less confidence.
        if _is_causal_claim(claim.stmt_type):
            return "correct", "chain_extraction_gap", \
                "causal claim with partial chain — accept at lower confidence"
        return "abstain", "chain_extraction_gap", \
            "direct claim, chain markers but no extractable mediator"

    if ra == "abstain":
        return "abstain", None, "relation underdetermined"

    # ra == "direct_sign_match" — consult scope.
    if sc == "asserted":
        return "correct", "match", "asserted relation matches claim"
    if sc == "hedged":
        # §5.7: hedged + matched relation → correct/low rather than
        # abstain. The relation IS asserted; the hedging modulates
        # confidence, not the verdict. Composed scorer can weight
        # low-confidence correct < high-confidence correct.
        return "correct", "hedging_hypothesis", \
            "hedged but relation matches claim"
    if sc == "negated":
        return "incorrect", "contradicted", \
            "relation explicitly negated"
    if sc == "abstain":
        # §5.7 corollary: if relation matches direct + sign and scope
        # is genuinely underdetermined, the most informative verdict
        # is correct/low — the relation evidence is there even if the
        # scope framing is unclear.
        return "correct", "match", \
            "relation matches; scope underdetermined → correct/low"

    # Unreachable — covers all closed-set values.
    return "abstain", None, "unhandled probe-tuple"


def adjudicate(
    claim: ClaimCommitment,
    bundle: ProbeBundle,
    groundings: tuple[GroundingVerdict, ...],
    *,
    ctx: EvidenceContext,
) -> Adjudication:
    """Combine probe responses + grounding into a verdict.

    1. Apply §5.1 perturbation pre-rule (currently informational; the
       relation_axis probe already consumes the effective sign via
       substrate-router and few-shot framing).
    2. Run canonical decision table.
    3. If verdict is abstain AND CATALOG-aligned match exists, apply
       §5.4 final-arm substrate-fallback rescue.
    4. Apply confidence policy: medium by default, downgraded if any
       grounding is uncertain.

    Returns a frozen Adjudication.
    """
    # §5.1 — currently informational; perturbation is propagated in the
    # relation_axis probe via substrate's _effective_sign computation in
    # router._route_relation_axis. This adjudicator receives the
    # already-adjusted answer.
    _ = _effective_claim_sign(
        claim.sign, bundle.subject_role.perturbation,
    )

    verdict, reason, rationale = _decide(bundle, claim)

    # §5.4 final-arm substrate-fallback rescue.
    if verdict == "abstain" and _final_arm_substrate_match(claim, ctx):
        return Adjudication(
            verdict="correct",
            confidence="medium",
            reasons=("regex_substrate_match",),
            rationale=f"final-arm substrate match (was: {rationale})",
        )

    # Confidence policy.
    if verdict == "correct":
        # Hedging, partial chain, or scope-underdetermined matches get
        # downgraded to low confidence so composed scorer can weight
        # them appropriately. Substrate-rescue and uncertain-grounding
        # are both medium.
        if reason in ("hedging_hypothesis", "chain_extraction_gap"):
            confidence = "low"
        elif reason == "regex_substrate_match":
            confidence = "medium"
        elif rationale.startswith("relation matches; scope underdetermined"):
            confidence = "low"
        else:
            confidence = "high"
            if _grounding_uncertain(groundings):
                confidence = "medium"
    elif verdict == "incorrect":
        confidence = "high"
    else:
        confidence = "medium"

    reasons: tuple[str, ...] = (reason,) if reason else ()
    return Adjudication(
        verdict=verdict,
        confidence=confidence,
        reasons=reasons,  # type: ignore[arg-type]
        rationale=rationale,
    )
