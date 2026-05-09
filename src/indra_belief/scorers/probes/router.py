"""Substrate-as-router: route each probe to a deterministic answer
or to an LLM escalation request.

Per doctrine §4, substrate is a function `(claim, ctx) → routings` where
each probe kind maps to either:
  - ProbeResponse (substrate-answered, high-confidence, no LLM)
  - ProbeRequest (LLM-bound, optional substrate_hint to narrow the question)

Substrate is conservative: it answers ONLY when the deterministic
signals are unambiguous. Ambiguous cases hand off to the LLM with the
relevant hint as auxiliary context — the hint never prefills the answer.

The four routing functions encode the substrate's view of each probe's
question. They consume EvidenceContext fields that were populated by
context_builder (CATALOG matches, perturbation markers, hedge markers,
chain signals, alias maps) — substrate detection logic itself is not
duplicated here; this module is the routing layer above it.

No SUBSTRATE PRIORS prompt block is produced anywhere in this module.
The output of substrate_route is consumed by the orchestrator (S6),
NOT by an LLM prompt.
"""
from __future__ import annotations

import re

from indra_belief.scorers.commitments import ClaimCommitment
from indra_belief.scorers.context import DetectedRelation, EvidenceContext
from indra_belief.scorers.probes.types import (
    ProbeKind,
    ProbeRequest,
    ProbeResponse,
)


# Maximum characters between a claim-entity mention and a hedge/negation
# anchor for substrate to attribute the marker to that entity. Mirrors
# the M10 explicit_hedge_marker scope rule.
_LOCAL_WINDOW_CHARS = 50

# Negation cues — verb-negators only. Earlier draft included broad
# lexical cues (no, never, none, neither, nor, absent, lacks) but the
# S4 gate trace showed those firing on adjacent propositions and
# wrongly negating the claim's relation. Tightened to explicit
# verb-negators; the LLM scope probe handles softer cues.
_NEGATION_RE = re.compile(
    r"\b(?:not|cannot|did\s+not|does\s+not|do\s+not|"
    r"is\s+not|are\s+not|was\s+not|were\s+not|"
    r"failed\s+to|fails\s+to|fail\s+to)\b",
    re.IGNORECASE,
)


def _alias_set(name: str, aliases: dict[str, frozenset[str]]) -> frozenset[str]:
    """Return the alias set for a claim entity, including the name itself."""
    if not name:
        return frozenset()
    return aliases.get(name, frozenset()) | {name}


def _find_alias_positions(text: str, alias_set: frozenset[str]) -> list[int]:
    """Return char offsets where any alias of the entity appears in text.

    Case-insensitive, whole-word matching. Aliases shorter than 2 chars
    are skipped (over-match risk; mirrors EvidenceContext.aliases policy).
    """
    if not text or not alias_set:
        return []
    positions: list[int] = []
    for alias in alias_set:
        if len(alias) < 2:
            continue
        # Word-boundary match. Escape regex metacharacters in alias.
        escaped = re.escape(alias)
        pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            positions.append(m.start())
    return sorted(positions)


def _detected_relations_for_pair(
    ctx: EvidenceContext,
    subject_canonical: str,
    object_canonical: str,
) -> tuple[list[DetectedRelation], list[DetectedRelation]]:
    """Partition ctx.detected_relations into:
      - aligned: agent matches subject AND target matches object
      - swapped: agent matches object AND target matches subject (role-swap)
    """
    aligned: list[DetectedRelation] = []
    swapped: list[DetectedRelation] = []
    for dr in ctx.detected_relations:
        if dr.agent_canonical == subject_canonical and dr.target_canonical == object_canonical:
            aligned.append(dr)
        elif dr.agent_canonical == object_canonical and dr.target_canonical == subject_canonical:
            swapped.append(dr)
    return aligned, swapped


def _detected_relations_with_entity_as_agent(
    ctx: EvidenceContext, entity_canonical: str,
) -> list[DetectedRelation]:
    return [dr for dr in ctx.detected_relations
            if dr.agent_canonical == entity_canonical]


def _detected_relations_with_entity_as_target(
    ctx: EvidenceContext, entity_canonical: str,
) -> list[DetectedRelation]:
    return [dr for dr in ctx.detected_relations
            if dr.target_canonical == entity_canonical]


# ---------------------------------------------------------------------------
# subject_role probe
# ---------------------------------------------------------------------------

def _route_subject_role(
    claim: ClaimCommitment,
    ctx: EvidenceContext,
    evidence_text: str,
) -> ProbeResponse | ProbeRequest:
    """Substrate fast-path for the subject_role probe.

    Confidence rules (conservative — answer only when unambiguous):
      - DetectedRelation has agent==subject → present_as_subject
      - DetectedRelation has target==subject AND non-binding axis → present_as_object
        (role-swap candidate; LLM will confirm via shape of evidence)
      - subject's aliases not found in evidence text → absent
      - Otherwise → ProbeRequest with hint listing relevant CATALOG matches
    """
    perturbation = ctx.subject_perturbation_marker
    pert_marker = (
        "LOF" if perturbation == "loss_of_function"
        else "GOF" if perturbation == "gain_of_function"
        else "none"
    )
    claim_component = (
        f"{claim.subject} (axis={claim.axis}, sign={claim.sign}, "
        f"objects={list(claim.objects)})"
    )

    aliases = _alias_set(claim.subject, ctx.aliases)
    positions = _find_alias_positions(evidence_text, aliases)
    if not positions:
        # Substrate did not match any of the known aliases. Escalate to
        # LLM rather than commit to absent — Gilda's alias map is
        # incomplete and the LLM can recognize alternative surface forms
        # (anaphora, paraphrase, family-member references) that substrate
        # missed. Hint says what aliases substrate tried so the LLM
        # doesn't waste attention re-deriving them.
        hint = (
            f"substrate matched no aliases for {claim.subject!r}: "
            f"tried {sorted(a for a in aliases if len(a) >= 2)[:6]}. "
            f"LLM should check for alternative surface forms (anaphora, "
            f"paraphrase, family-member reference); answer 'absent' only "
            f"if confident."
        )
        return ProbeRequest(
            kind="subject_role",
            claim_component=claim_component,
            evidence_text=evidence_text,
            substrate_hint=hint,
        )

    as_agent = _detected_relations_with_entity_as_agent(ctx, claim.subject)
    if as_agent:
        return ProbeResponse(
            kind="subject_role",
            answer="present_as_subject",
            source="substrate",
            confidence="high",
            perturbation=pert_marker,
            rationale=f"CATALOG matches with {claim.subject!r} as agent: "
                      f"{[dr.pattern_id for dr in as_agent]}",
        )

    # Role-swap candidate: subject appears as the TARGET in a non-binding
    # relation. Substrate flags this for the LLM, but the LLM still confirms.
    as_target = _detected_relations_with_entity_as_target(ctx, claim.subject)
    non_binding_swaps = [dr for dr in as_target if dr.axis != "binding"]
    if non_binding_swaps:
        hint = (
            f"substrate observed {claim.subject!r} as TARGET (not agent) of "
            f"non-binding relations: "
            f"{[(dr.pattern_id, dr.agent_canonical) for dr in non_binding_swaps]}. "
            f"Verify subject's role; possible role-swap."
        )
        return ProbeRequest(
            kind="subject_role",
            claim_component=claim_component,
            evidence_text=evidence_text,
            substrate_hint=hint,
        )

    # Mediator candidate: claim has chain signal AND subject sits in
    # chain_intermediate_candidates (an L1 pre-collected list).
    if ctx.has_chain_signal and claim.subject in ctx.chain_intermediate_candidates:
        return ProbeResponse(
            kind="subject_role",
            answer="present_as_mediator",
            source="substrate",
            confidence="medium",
            perturbation=pert_marker,
            rationale="L1 chain signal + subject in intermediate candidates",
        )

    # Substrate sees the entity but can't disambiguate role — escalate.
    hint_parts = [f"{claim.subject!r} mentioned in evidence "
                  f"(positions={positions[:3]})"]
    if pert_marker != "none":
        hint_parts.append(f"perturbation marker: {pert_marker}")
    if ctx.has_chain_signal:
        hint_parts.append("chain signal present in evidence")
    hint = "; ".join(hint_parts)

    return ProbeRequest(
        kind="subject_role",
        claim_component=claim_component,
        evidence_text=evidence_text,
        substrate_hint=hint,
    )


# ---------------------------------------------------------------------------
# object_role probe
# ---------------------------------------------------------------------------

def _route_object_role(
    claim: ClaimCommitment,
    ctx: EvidenceContext,
    evidence_text: str,
) -> ProbeResponse | ProbeRequest:
    """Substrate fast-path for the object_role probe.

    Mirror of _route_subject_role with target/agent inversion.
    Operates on claim.objects[0] (binary statement). Multi-member
    Complex (>2 members) is deferred to v2.
    """
    if not claim.objects:
        # Self-modification (Autophos) or single-agent claim. Object
        # role is bound to subject; substrate confirms.
        return ProbeResponse(
            kind="object_role",
            answer="present_as_object",
            source="substrate",
            confidence="high",
            rationale="single-agent claim; object slot maps to subject",
        )

    obj = claim.objects[0]
    claim_component = (
        f"{obj} (axis={claim.axis}, sign={claim.sign}, "
        f"subject={claim.subject})"
    )

    aliases = _alias_set(obj, ctx.aliases)
    positions = _find_alias_positions(evidence_text, aliases)
    if not positions:
        # Symmetric with subject_role: escalate rather than commit to
        # absent. Gilda alias coverage is the limiting factor; the LLM
        # can recognize anaphora, paraphrase, family references.
        hint = (
            f"substrate matched no aliases for {obj!r}: "
            f"tried {sorted(a for a in aliases if len(a) >= 2)[:6]}. "
            f"LLM should check for alternative surface forms; answer "
            f"'absent' only if confident."
        )
        return ProbeRequest(
            kind="object_role",
            claim_component=claim_component,
            evidence_text=evidence_text,
            substrate_hint=hint,
        )

    as_target = _detected_relations_with_entity_as_target(ctx, obj)
    if as_target:
        return ProbeResponse(
            kind="object_role",
            answer="present_as_object",
            source="substrate",
            confidence="high",
            rationale=f"CATALOG matches with {obj!r} as target: "
                      f"{[dr.pattern_id for dr in as_target]}",
        )

    # Role-swap candidate: object appears as the AGENT of a non-binding
    # relation. Substrate flags for LLM verification.
    as_agent = _detected_relations_with_entity_as_agent(ctx, obj)
    non_binding_swaps = [dr for dr in as_agent if dr.axis != "binding"]
    if non_binding_swaps:
        hint = (
            f"substrate observed {obj!r} as AGENT (not target) of "
            f"non-binding relations: "
            f"{[(dr.pattern_id, dr.target_canonical) for dr in non_binding_swaps]}. "
            f"Verify object's role; possible role-swap."
        )
        return ProbeRequest(
            kind="object_role",
            claim_component=claim_component,
            evidence_text=evidence_text,
            substrate_hint=hint,
        )

    if ctx.has_chain_signal and obj in ctx.chain_intermediate_candidates:
        return ProbeResponse(
            kind="object_role",
            answer="present_as_mediator",
            source="substrate",
            confidence="medium",
            rationale="L1 chain signal + object in intermediate candidates",
        )

    hint_parts = [f"{obj!r} mentioned in evidence (positions={positions[:3]})"]
    if ctx.has_chain_signal:
        hint_parts.append("chain signal present in evidence")
    hint = "; ".join(hint_parts)

    return ProbeRequest(
        kind="object_role",
        claim_component=claim_component,
        evidence_text=evidence_text,
        substrate_hint=hint,
    )


# ---------------------------------------------------------------------------
# relation_axis probe
# ---------------------------------------------------------------------------

# Map between context.py's DetectedRelation.axis values and the canonical
# Axis values used by ClaimCommitment. They're already aligned, but we
# normalize here in case the catalog evolves.
_AXIS_NORMALIZE = {
    "modification": "modification",
    "activity": "activity",
    "amount": "amount",
    "binding": "binding",
    "translocation": "localization",
}


def _normalize_axis(axis: str) -> str:
    return _AXIS_NORMALIZE.get(axis, axis)


def _route_relation_axis(
    claim: ClaimCommitment,
    ctx: EvidenceContext,
    evidence_text: str,
) -> ProbeResponse | ProbeRequest:
    """Substrate fast-path for the relation_axis probe.

    Confidence rules:
      - Aligned DetectedRelation matches axis AND sign → direct_sign_match
      - Aligned matches axis but sign disagrees → direct_sign_mismatch
      - Aligned matches but axis disagrees → direct_axis_mismatch
      - Binding-axis match but partner type incompatible → direct_partner_mismatch
      - Chain signal + intermediate candidates → via_mediator
      - Chain signal but no relation match → via_mediator_partial
      - No alias-resolved entities → escalate (subject/object_role probes already abstain)
      - Else → ProbeRequest with CATALOG hint

    U5 (perturbation effective-sign propagation): when substrate detects
    LOF on the subject, the effective claim sign is inverted. We
    propagate this to BOTH (a) the substrate's own match check (line
    `effective_sign = ...` below) and (b) the LLM probe's claim_component
    string. Without (b), the LLM saw the literal claim sign and the
    substrate saw the inverted one — causing the LLM to emit
    direct_sign_mismatch on records where the substrate would have
    matched. Closes the SGN stratum identified in S-phase v1 limitations.
    """
    obj = claim.objects[0] if claim.objects else claim.subject  # self-mod case

    aligned, swapped = _detected_relations_for_pair(ctx, claim.subject, obj)

    # Symmetric binding: for binding axis, swapped is treated as aligned
    # (Complex(X,Y) ≡ Complex(Y,X) per doctrine §5.3).
    if claim.axis == "binding" and swapped:
        aligned = aligned + swapped

    # Effective claim sign accounts for perturbation propagation. The
    # adjudicator's §5.1 rule does this; substrate honors it here so the
    # match check is consistent.
    effective_sign = claim.sign
    if ctx.subject_perturbation_marker == "loss_of_function":
        effective_sign = _invert_sign(claim.sign)

    # U5: build the LLM-facing claim_component using the EFFECTIVE sign,
    # not the literal claim sign. When the LLM probe has to choose between
    # direct_sign_match and direct_sign_mismatch, it should evaluate
    # against the perturbation-adjusted sign — what the substrate
    # already does for its own matching. This closes the FN cases like
    # "VHL silencing increased vimentin" where claim sign (negative) and
    # effective sign (positive after LOF inversion) diverge.
    perturbation_note = ""
    if ctx.subject_perturbation_marker == "loss_of_function":
        perturbation_note = (
            f" [LOF perturbation on subject: claim_sign={claim.sign} "
            f"inverted → effective_sign={effective_sign}]"
        )
    elif ctx.subject_perturbation_marker == "gain_of_function":
        perturbation_note = (
            f" [GOF perturbation on subject: sign preserved as "
            f"{effective_sign}]"
        )
    claim_component = (
        f"({claim.subject}, {obj}) — claim axis={claim.axis}, "
        f"sign={effective_sign}{perturbation_note}"
    )

    for dr in aligned:
        dr_axis = _normalize_axis(dr.axis)
        if dr_axis == claim.axis:
            if dr.sign == effective_sign:
                # Binding-axis must additionally check partner type. The
                # M3 builder already alias-validates and binds CATALOG
                # entries to claim entities; binding_admissible gates the
                # partner type, but DetectedRelation does not carry the
                # partner-type field directly. The adjudicator's §5.4
                # final-arm rule does the actual partner check; here
                # substrate confirms the axis match conservatively.
                return ProbeResponse(
                    kind="relation_axis",
                    answer="direct_sign_match",
                    source="substrate",
                    confidence="medium",
                    rationale=f"CATALOG aligned: {dr.pattern_id} "
                              f"(axis={dr.axis}, sign={dr.sign})",
                )
            else:
                return ProbeResponse(
                    kind="relation_axis",
                    answer="direct_sign_mismatch",
                    source="substrate",
                    confidence="medium",
                    rationale=f"CATALOG aligned axis but sign mismatch: "
                              f"{dr.pattern_id} (claim_sign={effective_sign}, "
                              f"detected_sign={dr.sign})",
                )

    # Cross-axis match: same entity pair, different axis.
    cross_axis_aligned = [
        dr for dr in (aligned + (swapped if claim.axis != "binding" else []))
        if _normalize_axis(dr.axis) != claim.axis
    ]
    if cross_axis_aligned:
        return ProbeResponse(
            kind="relation_axis",
            answer="direct_axis_mismatch",
            source="substrate",
            confidence="medium",
            rationale=f"CATALOG matched on different axis: "
                      f"{[(dr.pattern_id, dr.axis) for dr in cross_axis_aligned]} "
                      f"(claim axis={claim.axis})",
        )

    # No CATALOG-aligned answer — substrate hands off to LLM.
    #
    # Earlier draft committed via_mediator/via_mediator_partial from
    # chain signals alone; the S4 gate trace showed L1 chain markers
    # firing on nominalizations like "regulates ... via the activation
    # of X" where the relation is direct, not chained. Substrate's
    # CATALOG signals are high-precision; chain signals are not. Only
    # CATALOG-derived answers are emitted; chain signals become hints.
    hint_parts = []
    if aligned:
        hint_parts.append(
            f"CATALOG entries on this pair (axes don't match claim): "
            f"{[(dr.pattern_id, dr.axis, dr.sign) for dr in aligned]}"
        )
    if swapped and claim.axis != "binding":
        hint_parts.append(
            f"CATALOG entries with swapped roles (possible role-swap): "
            f"{[(dr.pattern_id, dr.axis) for dr in swapped]}"
        )
    if ctx.has_chain_signal:
        if ctx.chain_intermediate_candidates:
            hint_parts.append(
                f"L1 chain signal + intermediates: "
                f"{list(ctx.chain_intermediate_candidates)[:3]} "
                f"(may indicate via_mediator OR direct nominalization)"
            )
        else:
            hint_parts.append(
                "L1 chain signal in evidence (may indicate via_mediator "
                "OR nominalization e.g., 'via activation of X')"
            )
    if ctx.nominalized_relations:
        hint_parts.append(
            f"nominalized hints: {list(ctx.nominalized_relations)[:3]}"
        )
    hint = "; ".join(hint_parts) if hint_parts else None

    return ProbeRequest(
        kind="relation_axis",
        claim_component=claim_component,
        evidence_text=evidence_text,
        substrate_hint=hint,
    )


def _invert_sign(sign: str) -> str:
    if sign == "positive":
        return "negative"
    if sign == "negative":
        return "positive"
    return sign  # neutral stays neutral


# ---------------------------------------------------------------------------
# scope probe
# ---------------------------------------------------------------------------

def _route_scope(
    claim: ClaimCommitment,
    ctx: EvidenceContext,
    evidence_text: str,
) -> ProbeResponse | ProbeRequest:
    """Substrate fast-path for the scope probe.

    Confidence rules (conservative):
      - Negation cue within _LOCAL_WINDOW_CHARS of claim entity → negated
      - explicit_hedge_marker present → hedged (M10 already vets scope)
      - Otherwise → ProbeRequest with available cues as hint
    """
    obj = claim.objects[0] if claim.objects else claim.subject
    claim_component = (
        f"relation between {claim.subject} and {obj}"
    )

    subj_aliases = _alias_set(claim.subject, ctx.aliases)
    obj_aliases = _alias_set(obj, ctx.aliases)
    subj_positions = _find_alias_positions(evidence_text, subj_aliases)
    obj_positions = _find_alias_positions(evidence_text, obj_aliases)

    # Negation detection — fires ONLY when a verb-negator cue sits
    # between the subject and object positions in linear order AND
    # within proximity window of at least one. This catches "X did NOT
    # activate Y" but rejects "X activates Z, but [unrelated negation]"
    # where the negator governs a different clause.
    if subj_positions and obj_positions:
        span_lo = min(min(subj_positions), min(obj_positions))
        span_hi = max(max(subj_positions), max(obj_positions))
        for m in _NEGATION_RE.finditer(evidence_text):
            neg_pos = m.start()
            if not (span_lo <= neg_pos <= span_hi):
                continue
            within_subj = any(abs(p - neg_pos) <= _LOCAL_WINDOW_CHARS
                              for p in subj_positions)
            within_obj = any(abs(p - neg_pos) <= _LOCAL_WINDOW_CHARS
                             for p in obj_positions)
            if within_subj or within_obj:
                return ProbeResponse(
                    kind="scope",
                    answer="negated",
                    source="substrate",
                    confidence="medium",
                    rationale=f"negation cue {m.group()!r} between "
                              f"claim entities, within "
                              f"{_LOCAL_WINDOW_CHARS}-char window",
                )

    # M10 explicit hedge markers — already scope-anchored at detection time.
    if ctx.explicit_hedge_markers:
        return ProbeResponse(
            kind="scope",
            answer="hedged",
            source="substrate",
            confidence="medium",
            rationale=f"M10 explicit hedge markers: "
                      f"{sorted(ctx.explicit_hedge_markers)[:3]}",
        )

    # No definitive substrate signal — escalate. The LLM gets a hint
    # listing what substrate did NOT detect, so it doesn't re-derive
    # the negation/hedge analysis from scratch.
    hint_parts = ["no negation cue within local window",
                  "no M10 explicit hedge marker"]
    hint = "; ".join(hint_parts)

    return ProbeRequest(
        kind="scope",
        claim_component=claim_component,
        evidence_text=evidence_text,
        substrate_hint=hint,
    )


# ---------------------------------------------------------------------------
# Top-level routing
# ---------------------------------------------------------------------------

def substrate_route(
    claim: ClaimCommitment,
    ctx: EvidenceContext,
    evidence_text: str,
) -> dict[ProbeKind, ProbeResponse | ProbeRequest]:
    """Route every probe to a deterministic answer or an LLM request.

    Returns a dict keyed by ProbeKind with ProbeResponse where substrate
    answered, ProbeRequest where the LLM must answer (with optional
    substrate_hint to narrow the question).

    Doctrine §4: substrate is a question-router, not a candidate-prefiller.
    The output is consumed by the orchestrator (S6), never by an LLM prompt.
    """
    return {
        "subject_role": _route_subject_role(claim, ctx, evidence_text),
        "object_role": _route_object_role(claim, ctx, evidence_text),
        "relation_axis": _route_relation_axis(claim, ctx, evidence_text),
        "scope": _route_scope(claim, ctx, evidence_text),
    }
