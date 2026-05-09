"""Unified single-call scorer schema.

The dec sub-call chain (parse_claim → parse_evidence + verify_grounding ×N
→ adjudicate) is superseded by one LLM call whose structured output
preserves the same interpretability as separate sub-calls. Decomposition
lives in the schema, not the call graph.

Why one call beats four: KV-cache locality + attention-mediated
synthesis. The transformer holds claim, evidence, grounding context, and
intermediate analysis in one residual stream and resolves them jointly.
Decomposing into separate calls forces the model to re-encode the same
context N times and routes synthesis through Python conditionals that
have no view of attention's actual cross-field dependencies.

Reasoning is the FIRST schema field by design. Constrained decoding
emits fields in declaration order; placing free-form reasoning at the
top means subsequent structured fields are autoregressively conditioned
on the model's chain of thought (o1-style structured prefix). With
max_tokens=64000 the model never truncates — reasoning length is
self-sized to the difficulty of the record, paid only when used.

This module defines the schema and the JSON-Schema export. The
implementation function (score_unified) lives in Phase C.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Reuse the existing typed taxonomy from the legacy path — these enums
# are stable, validated by tests, and define the project's canonical
# vocabulary. The unified call must speak the same language so error
# stratification, downstream score mapping, and audit comparisons stay
# coherent across architectures.
from indra_belief.scorers.commitments import (
    Axis,
    Confidence,
    GroundingStatus,
    Perturbation,
    ReasonCode,
    Sign,
    Verdict,
)


class ExtractedRelation(BaseModel):
    """One relation the model reads off the evidence sentence.

    Mirrors EvidenceAssertion's load-bearing fields. Sign is the LITERAL
    direction stated in the sentence; the model captures perturbation as
    a structured flag and the unified call applies inversion in the
    `sign_check` analysis (no separate adjudicator post-pass).
    """
    model_config = ConfigDict(extra="forbid")

    subject: str = Field(description="Underlying agent (not the perturbation phrase).")
    object: str = Field(description="Underlying target.")
    axis: Axis = Field(description="Kind of change asserted.")
    sign: Sign = Field(description="Literal direction stated in evidence.")
    perturbation: Perturbation = Field(
        default="none",
        description="loss_of_function/gain_of_function on the agent; flips effective sign.",
    )
    site: str | None = Field(default=None, description="Modification site if specified.")
    location_from: str | None = Field(default=None)
    location_to: str | None = Field(default=None)
    negation: bool = Field(
        default=False,
        description='True iff sentence asserts the action did NOT happen.',
    )
    hedged: bool = Field(
        default=False,
        description='True iff sentence is exploratory/hypothetical about the relation.',
    )


class GroundingCheck(BaseModel):
    """Per-entity grounding verdict, structured Gilda context preserved.

    Replaces verify_grounding sub-call. The unified call receives the
    entity's GroundedEntity context (db, family_members, aliases,
    is_pseudogene, gilda_score) as injected user-prompt context and
    emits one of these per claim entity (subject + each object).
    """
    model_config = ConfigDict(extra="forbid")

    entity: str
    status: GroundingStatus
    rationale: str = Field(description="One-sentence reason. Informational.")


CheckOutcome = Literal["match", "mismatch", "not_applicable"]


class UnifiedJudgment(BaseModel):
    """The single-call output. Field order is load-bearing — constrained
    decoding emits in declaration order, so free-form reasoning fills
    first and conditions every subsequent structured field.
    """
    # Pydantic v2: forbid extra fields so the decoder can't smuggle
    # unschemaed slop into the JSON. Schema-violation rate must be 0.
    model_config = ConfigDict(extra="forbid")

    # ─── 1. Reasoning prefix ──────────────────────────────────────────
    reasoning: str = Field(
        description=(
            "Free-form chain of thought. Walk the evidence: who acts on whom, "
            "what kind of change, any hedging or negation, indirect-chain risk, "
            "perturbation context, grounding alignment. Length is self-sized."
        ),
    )

    # ─── 2. Parsed evidence (replaces parse_evidence) ─────────────────
    extracted_relations: list[ExtractedRelation] = Field(
        description=(
            "Every distinct relation asserted by the evidence sentence. "
            'Empty list when the sentence asserts no relation between named entities.'
        ),
    )

    # ─── 3. Grounding (replaces verify_grounding) ─────────────────────
    subject_grounding: GroundingCheck
    object_groundings: list[GroundingCheck] = Field(
        description="One per claim object (Statement may have multiple objects).",
    )

    # ─── 4. Adjudication checks (replaces adjudicate's structured pass) ─
    axis_check: CheckOutcome = Field(
        description='Does any extracted relation share the claim\'s axis?',
    )
    sign_check: CheckOutcome = Field(
        description=(
            'Effective sign (after perturbation inversion if any) matches the '
            'claim\'s sign on the matched-axis relation?'
        ),
    )
    site_check: CheckOutcome = Field(
        description='If claim specifies a site/location, evidence agrees?',
    )
    stance_check: Literal["affirmed", "hedged", "negated", "not_applicable"] = Field(
        description=(
            'Epistemic stance of the matched relation. "hedged"/"negated" '
            'are distinct: hedged = exploratory ("we tested whether..."); '
            'negated = denied ("X did not activate Y").'
        ),
    )
    indirect_chain_present: bool = Field(
        description=(
            'True iff evidence shows X→intermediate→Y rather than direct X→Y. '
            'Strict policy: indirect_chain_present + claim implies direct '
            '→ verdict=incorrect, reason_code=indirect_chain.'
        ),
    )
    perturbation_inversion_applied: bool = Field(
        description=(
            'True iff any extracted_relation had perturbation=loss_of_function '
            'and the model applied sign inversion to recover the normal '
            'agent→target relationship.'
        ),
    )

    # ─── 5. Final verdict ─────────────────────────────────────────────
    verdict: Verdict = Field(
        description=(
            'correct: evidence supports the claim. '
            'incorrect: evidence does NOT support (mismatch, contradiction, indirect, hedged, etc.). '
            'abstain: evidence too ambiguous or malformed to decide.'
        ),
    )
    reason_code: ReasonCode = Field(
        description=(
            'The primary reason. "match" iff verdict=correct. Otherwise the '
            'specific failure mode from the canonical taxonomy.'
        ),
    )
    secondary_reasons: list[ReasonCode] = Field(
        default_factory=list,
        description='Additional reason codes when more than one applies.',
    )
    confidence: Confidence
    self_critique: str = Field(
        description=(
            'One paragraph: what could be wrong with this judgment? Edge '
            'cases considered. Used by F1 to validate confidence calibration.'
        ),
    )


def unified_response_schema() -> dict:
    """JSON Schema for the Ollama `format` parameter / OpenAI-compat
    `response_format: json_schema`. Decoder-enforces the structure at
    emit time so invalid JSON is structurally impossible.
    """
    schema = UnifiedJudgment.model_json_schema()
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "UnifiedJudgment",
            "schema": schema,
            "strict": True,
        },
    }


# ─────────────────────────────────────────────────────────────────────
# Implementation: score_unified
# ─────────────────────────────────────────────────────────────────────

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from indra_belief.scorers._unified_prompt import UNIFIED_SYSTEM_PROMPT
from indra_belief.scorers.commitments import (
    Adjudication,
    _VERDICT_SCORE,
)

if TYPE_CHECKING:
    from indra_belief.model_client import ModelClient


_log = logging.getLogger(__name__)


# Production parameters — see project_unified_design_review.md.
_PROD_MAX_TOKENS = 64000   # A2 probe: actual range 549-2148; this is a ceiling
_PROD_TEMPERATURE = 0.1    # slight stochasticity for robustness
_PROD_VOTING_K = 1         # confidence is a structured-output field, not a vote

_FEWSHOT_PATH = Path(__file__).parent.parent / "data" / "unified_fewshots.jsonl"


def _load_fewshots() -> list[dict]:
    """Load the curriculum once. JSONL of {pattern, claim, gilda_context,
    evidence, judgment}."""
    out: list[dict] = []
    if not _FEWSHOT_PATH.exists():
        return out
    with open(_FEWSHOT_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                _log.warning("unified_fewshots: skip malformed line: %s", e)
    return out


_FEWSHOTS_CACHE: list[dict] | None = None


def _fewshots() -> list[dict]:
    global _FEWSHOTS_CACHE
    if _FEWSHOTS_CACHE is None:
        _FEWSHOTS_CACHE = _load_fewshots()
    return _FEWSHOTS_CACHE


def _format_gilda_context(entity) -> dict:
    """Project a GroundedEntity (or fewshot dict) to the compact context
    blob the prompt consumes."""
    if isinstance(entity, dict):
        return entity
    aliases = list(getattr(entity, "aliases", []) or [])
    family_members = list(getattr(entity, "family_members", []) or [])
    return {
        "name": getattr(entity, "name", "?"),
        "db": getattr(entity, "db", None),
        "db_id": getattr(entity, "db_id", None),
        "is_family": bool(getattr(entity, "is_family", False)),
        "family_members": family_members[:8],
        "is_pseudogene": bool(getattr(entity, "is_pseudogene", False)),
        "aliases": aliases[:10],
        "gilda_score": getattr(entity, "gilda_score", None),
    }


def _format_user_message(
    claim_subject: str,
    claim_stmt_type: str,
    claim_objects: tuple[str, ...],
    claim_axis: str,
    claim_sign: str,
    claim_site: str | None,
    claim_location_from: str | None,
    claim_location_to: str | None,
    subject_context: dict,
    object_contexts: list[dict],
    evidence_text: str,
) -> str:
    """Build the user-side prompt body with claim metadata, Gilda
    context, and evidence. Pre-computed fields (axis, sign, aliases) are
    injected as facts so the model doesn't re-derive them."""
    parts = ["# Claim",
             f"  subject: {claim_subject}",
             f"  stmt_type: {claim_stmt_type}",
             f"  object(s): {', '.join(claim_objects) or '-'}",
             f"  axis: {claim_axis}",
             f"  sign: {claim_sign}"]
    if claim_site:
        parts.append(f"  site: {claim_site}")
    if claim_location_from:
        parts.append(f"  location_from: {claim_location_from}")
    if claim_location_to:
        parts.append(f"  location_to: {claim_location_to}")

    parts.append("")
    parts.append("# Subject grounding context")
    parts.append(json.dumps(subject_context, separators=(",", ":")))
    if object_contexts:
        parts.append("")
        parts.append("# Object grounding contexts")
        for oc in object_contexts:
            parts.append(json.dumps(oc, separators=(",", ":")))

    parts.append("")
    parts.append("# Evidence")
    parts.append(f'"{evidence_text}"')
    parts.append("")
    parts.append("Emit one UnifiedJudgment JSON object. Reasoning first.")
    return "\n".join(parts)


def _build_messages(
    user_message: str,
) -> list[dict]:
    """Compose the few-shot curriculum + the actual user message into the
    messages list. Each shot becomes a (user, assistant) pair so the
    model sees the conversational structure of the task."""
    messages: list[dict] = []
    for shot in _fewshots():
        try:
            claim = shot["claim"]
            ctx = shot["gilda_context"]
            shot_user = _format_user_message(
                claim_subject=claim["subject"],
                claim_stmt_type=claim["stmt_type"],
                claim_objects=(claim["object"],) if claim.get("object") else (),
                claim_axis=claim["axis"],
                claim_sign=claim["sign"],
                claim_site=claim.get("site"),
                claim_location_from=claim.get("location_from"),
                claim_location_to=claim.get("location_to"),
                subject_context=ctx["subject"],
                object_contexts=[ctx["object"]] if "object" in ctx else [],
                evidence_text=shot["evidence"],
            )
            messages.append({"role": "user", "content": shot_user})
            messages.append({
                "role": "assistant",
                "content": json.dumps(shot["judgment"], separators=(",", ":")),
            })
        except KeyError as e:
            _log.warning("fewshot skipped (missing key %s)", e)
    messages.append({"role": "user", "content": user_message})
    return messages


def _judgment_to_score_dict(j: UnifiedJudgment) -> dict:
    """Map a UnifiedJudgment to the score_evidence dict shape callers
    already consume. Preserves the existing key contract so dual-run /
    composed scorer pick this up without changes."""
    score = _VERDICT_SCORE.get((j.verdict, j.confidence), 0.5)
    grounding_statuses = ([j.subject_grounding.status]
                          + [g.status for g in j.object_groundings])
    grounding_flag = ("all_match"
                      if all(s in ("mentioned", "equivalent")
                             for s in grounding_statuses)
                      else "flagged")
    rationale = (
        f"axis={j.axis_check}, sign={j.sign_check}, site={j.site_check}, "
        f"stance={j.stance_check}, indirect={j.indirect_chain_present}, "
        f"perturbation_inversion={j.perturbation_inversion_applied}"
    )
    reasons = [j.reason_code, *j.secondary_reasons]
    return {
        "score": score,
        "verdict": j.verdict,
        "confidence": j.confidence,
        "tier": "unified",
        "grounding_status": grounding_flag,
        "provenance_triggered": False,
        "tokens": 0,  # populated by caller from ModelResponse
        "raw_text": j.model_dump_json(),
        "reasons": reasons,
        "rationale": rationale,
        "self_critique": j.self_critique,
    }


def _abstain_dict(reason: str, raw_text: str = "") -> dict:
    """Graceful-degradation abstain when transport / validation fails."""
    return {
        "score": 0.5,
        "verdict": "abstain",
        "confidence": "low",
        "tier": "unified",
        "grounding_status": "flagged",
        "provenance_triggered": False,
        "tokens": 0,
        "raw_text": raw_text,
        "reasons": ["absent_relationship"],
        "rationale": f"unified call abstained: {reason}",
        "self_critique": "",
    }


def score_unified(
    statement,
    evidence,
    client: "ModelClient",
    *,
    max_tokens: int = _PROD_MAX_TOKENS,
    temperature: float = _PROD_TEMPERATURE,
) -> dict:
    """Single-LLM-call evidence scorer. The unified architecture.

    Returns the same dict contract as `score_evidence` so callers swap
    behind a flag. Never raises — sub-call failures degrade to abstain.

    No retries, no cascade, no voting. Confidence is structured output;
    truncation is eliminated by max_tokens=64000; reasoning fills first
    via schema field order.
    """
    # Local imports — keeps INDRA + corpus_index + parse_claim out of the
    # module-level import graph for callers that only need the schema.
    from indra_belief.data.entity import GroundedEntity
    from indra_belief.scorers.decomposed import _raw_text_for
    from indra_belief.scorers.parse_claim import parse_claim

    # 1. Deterministic claim → axis/sign/objects/site
    try:
        claim = parse_claim(statement)
    except Exception as e:
        _log.warning("score_unified: parse_claim failed: %s", e)
        return _abstain_dict(f"parse_claim error: {type(e).__name__}")

    # 2. Resolve Gilda context for every claim entity (subject + each object)
    def _resolve(name: str):
        try:
            return GroundedEntity.resolve(name, _raw_text_for(name, evidence))
        except Exception as e:
            _log.warning("score_unified: GroundedEntity.resolve(%r) failed: %s", name, e)
            return None

    subj_ent = _resolve(claim.subject) if claim.subject and claim.subject != "?" else None
    subj_ctx = _format_gilda_context(subj_ent) if subj_ent is not None else {
        "name": claim.subject, "db": None, "db_id": None,
        "is_family": False, "is_pseudogene": False, "aliases": [],
    }

    obj_ctxs: list[dict] = []
    for o in claim.objects:
        if not o or o == "?":
            continue
        ent = _resolve(o)
        obj_ctxs.append(
            _format_gilda_context(ent) if ent is not None else {
                "name": o, "db": None, "db_id": None,
                "is_family": False, "is_pseudogene": False, "aliases": [],
            }
        )

    # 3. Build the user message + few-shot messages
    evidence_text = (getattr(evidence, "text", None) or "").strip()
    if not evidence_text:
        return _abstain_dict("empty evidence text")

    user_msg = _format_user_message(
        claim_subject=claim.subject,
        claim_stmt_type=claim.stmt_type,
        claim_objects=claim.objects,
        claim_axis=claim.axis,
        claim_sign=claim.sign,
        claim_site=claim.site,
        claim_location_from=claim.location_from,
        claim_location_to=claim.location_to,
        subject_context=subj_ctx,
        object_contexts=obj_ctxs,
        evidence_text=evidence_text,
    )
    messages = _build_messages(user_msg)

    # 4. The single LLM call. Structured output enforced by `format` /
    # response_format. No retries — truncation is gone, malformed-JSON
    # is structurally impossible.
    try:
        response = client.call(
            system=UNIFIED_SYSTEM_PROMPT,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            # json_object (not json_schema strict) — universally supported.
            # Pydantic UnifiedJudgment.model_validate catches schema
            # violations app-side; on backends like vmlx that enforce
            # strict mode by waiting for valid JSON during decoding,
            # strict mode can hang when reasoning_content fills the budget
            # before content emits.
            response_format={"type": "json_object"},
            # reasoning_effort left at config default — model gets to think
        )
    except Exception as e:
        _log.warning("score_unified: client.call failed: %s", e)
        return _abstain_dict(f"transport error: {type(e).__name__}")

    if response.finish_reason == "length":
        # Should not occur at max_tokens=64000 per A2 probe, but log the
        # canary if it does — that's a signal to revisit the budget.
        _log.warning("score_unified: response truncated (finish_reason=length)")

    # 5. Parse + validate
    content = response.content or ""
    try:
        obj = json.loads(content)
    except json.JSONDecodeError as e:
        _log.warning("score_unified: JSON parse failed: %s", e)
        return _abstain_dict("malformed JSON despite schema enforcement",
                             raw_text=content[:1000])

    try:
        judgment = UnifiedJudgment.model_validate(obj)
    except Exception as e:
        _log.warning("score_unified: schema validation failed: %s", e)
        return _abstain_dict(f"schema validation error: {type(e).__name__}",
                             raw_text=content[:1000])

    # 6. Map to score_evidence dict shape
    out = _judgment_to_score_dict(judgment)
    out["tokens"] = response.tokens
    return out
