"""S-phase orchestrator: parse_claim → context → substrate_route →
LLM probes (where escalated) → ProbeBundle → adjudicate → output dict.

Replaces the M/N/Q/R-phase decomposed.py orchestration. The output dict
preserves the same keys monolithic + decomposed callers consumed
(score, verdict, confidence, raw_text, tokens, tier, grounding_status,
provenance_triggered, reasons, rationale, call_log) so downstream
composed_scorer code requires no change.

Latency contract (doctrine §6):
  - Substrate-resolved probes never call the LLM
  - LLM probes run sequentially (parallelization deferred to S8 if needed)
  - Median wall ≤ 15s when 50%+ records resolve via substrate alone
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from dataclasses import replace as _dc_replace

from indra_belief.scorers import kg_signal as _kg_signal
from indra_belief.scorers.commitments import (
    Adjudication,
    ClaimCommitment,
    GroundingVerdict,
    adjudication_to_score,
)
from indra_belief.scorers.context import EvidenceContext
from indra_belief.scorers.grounding import verify_grounding
from indra_belief.scorers.parse_claim import parse_claim
from indra_belief.scorers.probes import (
    object_role,
    relation_axis,
    router,
    scope,
    subject_role,
)
from indra_belief.scorers.probes.adjudicator import adjudicate
from indra_belief.scorers.probes.types import (
    ProbeBundle,
    ProbeRequest,
    ProbeResponse,
)

if TYPE_CHECKING:
    from indra_belief.model_client import ModelClient


log = logging.getLogger(__name__)


# Map probe kind → answering module's `answer` function.
_PROBE_HANDLERS = {
    "subject_role": subject_role.answer,
    "object_role": object_role.answer,
    "relation_axis": relation_axis.answer,
    "scope": scope.answer,
}


def _resolve_probe(
    routing: ProbeResponse | ProbeRequest, client: "ModelClient",
    *, reasoning_effort: str = "none",
) -> ProbeResponse:
    """If substrate already answered, return its ProbeResponse. Otherwise
    invoke the probe's LLM module to answer the ProbeRequest.

    U3 selective reasoning: pass reasoning_effort="medium" on escalation.
    Substrate-answered probes are unaffected (no LLM call).
    """
    if isinstance(routing, ProbeResponse):
        return routing
    handler = _PROBE_HANDLERS[routing.kind]
    return handler(routing, client, reasoning_effort=reasoning_effort)


# U3 escalation triggers — refined post-U12 first observation:
# grounding_gap escalation is wasteful (entity absence is structural,
# not reasoning-bound). The escalation is targeted at relation_axis
# and scope reasoning, not subject/object grounding.
#
# Reasons that benefit from reasoning escalation:
_ESCALATE_REASONS_RELATION = frozenset({
    "indirect_chain",            # chain detection benefits from longer reasoning
    "chain_extraction_gap",      # mediator extraction needs more thought
    "relation_underdetermined",  # axis underdetermined → more reading helps
})
_ESCALATE_REASONS_SCOPE = frozenset({
    "hedging_hypothesis",        # hedge classification benefits from reasoning
    "upstream_attribution",      # chain context helps scope decision
})

# Reasons we explicitly DO NOT escalate (structural / unaddressable by reasoning):
_NO_ESCALATE_REASONS = frozenset({
    "grounding_gap",       # entity absence — U6 prompt is the fix, not reasoning
    "absent_relationship", # already a substantive verdict
    "match",               # already correct/high
})


def _escalation_targets(adj: Adjudication) -> set[str]:
    """Return the set of probe kinds to re-run at higher reasoning,
    based on what drove the first-pass uncertainty. Empty set = no escalation.

    Keys: 'subject_role', 'object_role', 'relation_axis', 'scope'.
    """
    targets: set[str] = set()
    if adj.verdict == "abstain":
        # Reasons drive specific probes:
        for r in adj.reasons:
            if r in _NO_ESCALATE_REASONS:
                # Explicit skip — don't escalate this probe.
                continue
            if r in _ESCALATE_REASONS_RELATION:
                targets.add("relation_axis")
                targets.add("scope")
            elif r in _ESCALATE_REASONS_SCOPE:
                targets.add("scope")
        # Abstain with NO reasons (defensive path): escalate all four
        # since we don't know what failed.
        if not adj.reasons:
            targets = {"subject_role", "object_role",
                       "relation_axis", "scope"}
    elif adj.verdict in ("correct", "incorrect") and adj.confidence == "low":
        # Low-confidence verdict — escalate the probe(s) tied to the reason.
        for r in adj.reasons:
            if r in _ESCALATE_REASONS_RELATION:
                targets.add("relation_axis")
            elif r in _ESCALATE_REASONS_SCOPE:
                targets.add("scope")
    return targets


def _resolve_claim_entities(claim: ClaimCommitment, evidence) -> list:
    """Resolve every distinct claim entity via Gilda once. Mirrors the
    decomposed pipeline's helper — kept for grounding's input contract."""
    from indra_belief.data.entity import GroundedEntity

    names: list[str] = []
    seen: set[str] = set()
    for n in (claim.subject,) + claim.objects:
        if not n or n == "?" or n in seen:
            continue
        seen.add(n)
        names.append(n)

    resolved = []
    for n in names:
        try:
            raw = _raw_text_for(n, evidence)
            resolved.append(GroundedEntity.resolve(n, raw))
        except Exception as e:
            log.warning("orchestrator: GroundedEntity.resolve(%r) failed: %s",
                        n, e)
    return resolved


def _raw_text_for(name: str, evidence) -> str | None:
    try:
        agents = evidence.annotations.get("agents", {})
    except AttributeError:
        return None
    names = agents.get("agent_list") or []
    raws = agents.get("raw_text") or []
    for n, rt in zip(names, raws):
        if n == name and rt:
            return rt
    return None


def _format_output(
    adj: Adjudication,
    groundings: tuple[GroundingVerdict, ...],
    bundle: ProbeBundle | None,
    call_log: list[dict],
) -> dict:
    total_out = sum(int(c.get("out_tokens") or 0) for c in call_log)
    return {
        "score": adjudication_to_score(adj),
        "verdict": adj.verdict,
        "confidence": adj.confidence,
        "raw_text": _render_trace(adj, bundle),
        "tokens": total_out,
        "tier": "decomposed",
        "grounding_status": _groundings_to_status(groundings),
        "provenance_triggered": False,
        "reasons": list(adj.reasons),
        "rationale": adj.rationale,
        "call_log": call_log,
    }


def _groundings_to_status(
    groundings: tuple[GroundingVerdict, ...],
) -> str:
    if not groundings:
        return "all_match"
    if any(g.status == "uncertain" for g in groundings):
        return "flagged"
    if any(g.status == "not_present" for g in groundings):
        return "flagged"
    return "all_match"


def _render_trace(adj: Adjudication, bundle: ProbeBundle | None) -> str:
    if bundle is None:
        return f"[S-PHASE] {adj.verdict}/{adj.confidence} reasons={list(adj.reasons)}"
    parts = [
        f"[S-PHASE] {adj.verdict}/{adj.confidence} reasons={list(adj.reasons)}",
        f"  subject_role={bundle.subject_role.answer} "
        f"({bundle.subject_role.source})",
        f"  object_role={bundle.object_role.answer} "
        f"({bundle.object_role.source})",
        f"  relation_axis={bundle.relation_axis.answer} "
        f"({bundle.relation_axis.source})",
        f"  scope={bundle.scope.answer} ({bundle.scope.source})",
    ]
    return "\n".join(parts)


def score_via_probes(statement, evidence, client: "ModelClient") -> dict:
    """Score one (Statement, Evidence) pair via the four-probe pipeline.

    Pipeline:
      1. parse_claim deterministically → ClaimCommitment
      2. Build EvidenceContext (regex/Gilda substrate; no LLM)
      3. substrate_route → ProbeResponse | ProbeRequest per probe kind
      4. For each ProbeRequest, dispatch to the corresponding LLM probe
      5. Assemble ProbeBundle
      6. Verify per-entity grounding
      7. adjudicate → Adjudication
      8. Format output dict (compatible with monolithic + decomposed
         downstream consumers)

    Never raises — sub-call failures degrade gracefully via abstain
    ProbeResponses and the adjudicator's fall-through rules.
    """
    _pop = getattr(client, "pop_call_log", lambda: [])
    _pop()
    evidence_text = getattr(evidence, "text", "") or ""

    # 1. Deterministic claim parse.
    try:
        claim = parse_claim(statement)
    except Exception as e:
        log.warning("orchestrator: parse_claim failed: %s", e)
        adj = Adjudication(
            verdict="abstain", confidence="low",
            rationale=f"parse_claim failed: {type(e).__name__}",
        )
        return _format_output(adj, (), None, _pop())

    # 2. Build EvidenceContext.
    try:
        ctx = EvidenceContext.from_statement_and_evidence(statement, evidence)
    except Exception as e:
        log.warning(
            "orchestrator: build_context failed (%s: %s); "
            "falling back to empty EvidenceContext",
            type(e).__name__, e,
        )
        ctx = EvidenceContext()

    # 2b. U4: KG signal lookup (substrate-only, no LLM call). Confidence
    # modifier only — never overrides verdict. Q-phase failure mode is
    # the line we don't cross.
    try:
        if claim.objects:
            sig = _kg_signal.get_signal(
                claim.subject, claim.objects[0], claim.axis,
            )
            if sig is not None:
                ctx = _dc_replace(ctx, kg_signal=sig)
    except Exception as e:
        # Defensive — KG lookup failures must NEVER break scoring.
        log.warning("orchestrator: kg_signal lookup failed: %s", e)

    # 3. Route every probe through substrate (deterministic answers
    #    where possible; LLM requests otherwise).
    routings = router.substrate_route(claim, ctx, evidence_text)

    # 4. Dispatch LLM probes for any ProbeRequest. Sequential for v1;
    #    parallel dispatch is a S8 optimization if latency demands it.
    bundle = ProbeBundle(
        subject_role=_resolve_probe(routings["subject_role"], client),
        object_role=_resolve_probe(routings["object_role"], client),
        relation_axis=_resolve_probe(routings["relation_axis"], client),
        scope=_resolve_probe(routings["scope"], client),
    )

    # 5. Per-entity grounding verification.
    entities = _resolve_claim_entities(claim, evidence)
    grounding_list: list[GroundingVerdict] = []
    for e in entities:
        try:
            grounding_list.append(verify_grounding(e, evidence_text, client))
        except Exception as gex:
            log.warning("orchestrator: verify_grounding(%r) failed: %s",
                        e, gex)
    groundings = tuple(grounding_list)

    # 6. Pure-function adjudicator (first-pass).
    adj = adjudicate(claim, bundle, groundings, ctx=ctx)

    # 7. U3 selective reasoning escalation. Only re-run probes that
    #    drove the first-pass uncertainty (NOT all 4 — that was wasteful
    #    per U12 first-observation finding). Skip escalation on
    #    grounding_gap (structural absence; reasoning won't help — U6 is
    #    the fix). Substrate-answered probes are unaffected (no LLM call
    #    even on escalation since they have ProbeResponse, not ProbeRequest).
    targets = _escalation_targets(adj)
    if targets:
        log.info(
            "orchestrator: escalating %s (verdict=%s, reasons=%s)",
            sorted(targets), adj.verdict, adj.reasons,
        )
        # Re-resolve only the targeted probes; reuse first-pass for others.
        new_subj = (
            _resolve_probe(routings["subject_role"], client,
                           reasoning_effort="medium")
            if "subject_role" in targets else bundle.subject_role
        )
        new_obj = (
            _resolve_probe(routings["object_role"], client,
                           reasoning_effort="medium")
            if "object_role" in targets else bundle.object_role
        )
        new_ra = (
            _resolve_probe(routings["relation_axis"], client,
                           reasoning_effort="medium")
            if "relation_axis" in targets else bundle.relation_axis
        )
        new_sc = (
            _resolve_probe(routings["scope"], client,
                           reasoning_effort="medium")
            if "scope" in targets else bundle.scope
        )
        bundle_v2 = ProbeBundle(
            subject_role=new_subj, object_role=new_obj,
            relation_axis=new_ra, scope=new_sc,
        )
        adj_v2 = adjudicate(claim, bundle_v2, groundings, ctx=ctx)
        # Use escalated verdict only if it strictly improves uncertainty:
        # - first abstain → second non-abstain: improvement, take v2.
        # - first low-confidence → second medium/high on same verdict:
        #   improvement, take v2.
        # - otherwise: keep first-pass.
        improved = False
        if adj.verdict == "abstain" and adj_v2.verdict != "abstain":
            improved = True
        elif (
            adj.verdict == adj_v2.verdict
            and adj.confidence == "low"
            and adj_v2.confidence in ("medium", "high")
        ):
            improved = True
        if improved:
            adj = adj_v2
            bundle = bundle_v2

    return _format_output(adj, groundings, bundle, _pop())
