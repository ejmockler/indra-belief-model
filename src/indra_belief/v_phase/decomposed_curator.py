"""V6g+ decomposed curator — break each probe's single-shot classification
into 1-3 narrower sub-questions, then assemble the final class.

Motivation: single-shot classification asks Gemini Pro to commit to one of
4-8 mutually exclusive classes in one turn, which collapses subtle decisions
(sign vs amount, role vs partner, asserted vs hedged) into the same step.
Decomposed prompting asks crisper sub-questions, then composes deterministic
or near-deterministic answers from the sub-results.

Each per-probe module exposes:
    async def score(rec: dict, call_subq: CallSubQ) -> dict

returning `{"answer": <final_class>, "sub_answers": {...}, "sub_responses": {...}}`
on success, or `{"error": <reason>, "sub_responses": {...}}` on partial failure.

The runner in `scripts/v6g_gemini_validation.py` injects `call_subq` so the
sub-modules don't import the Gemini client directly.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable

# (system_prompt, few_shots, user_msg, class_space) -> {"answer", "rationale"} | {"error"}
CallSubQ = Callable[..., Awaitable[dict[str, Any]]]


# ── Stmt-type → axis / sign (deterministic, no LLM call) ──

_STMT_AXIS: dict[str, str] = {
    # modification axis
    "Phosphorylation": "modification",
    "Dephosphorylation": "modification",
    "Methylation": "modification",
    "Demethylation": "modification",
    "Acetylation": "modification",
    "Deacetylation": "modification",
    "Ubiquitination": "modification",
    "Deubiquitination": "modification",
    "Sumoylation": "modification",
    "Desumoylation": "modification",
    "Hydroxylation": "modification",
    "Dehydroxylation": "modification",
    "Farnesylation": "modification",
    "Defarnesylation": "modification",
    "Glycosylation": "modification",
    "Deglycosylation": "modification",
    "Ribosylation": "modification",
    "Deribosylation": "modification",
    "Palmitoylation": "modification",
    "Depalmitoylation": "modification",
    "Myristoylation": "modification",
    "Demyristoylation": "modification",
    "Geranylgeranylation": "modification",
    "Degeranylgeranylation": "modification",
    "Autophosphorylation": "modification",
    "Transphosphorylation": "modification",
    # activity axis
    "Activation": "activity",
    "Inhibition": "activity",
    # amount axis
    "IncreaseAmount": "amount",
    "DecreaseAmount": "amount",
    # binding axis
    "Complex": "binding",
    "Binding": "binding",
    # localization axis
    "Translocation": "localization",
    # gtp_state axis
    "GtpActivation": "gtp_state",
    "GtpInactivation": "gtp_state",
    # conversion axis
    "Conversion": "conversion",
    "Gef": "gtp_state",
    "Gap": "gtp_state",
    "RegulateActivity": "activity",
    "RegulateAmount": "amount",
}

_STMT_SIGN: dict[str, str] = {
    # positive
    "Phosphorylation": "positive",
    "Methylation": "positive",
    "Acetylation": "positive",
    "Ubiquitination": "positive",
    "Sumoylation": "positive",
    "Hydroxylation": "positive",
    "Farnesylation": "positive",
    "Glycosylation": "positive",
    "Ribosylation": "positive",
    "Palmitoylation": "positive",
    "Myristoylation": "positive",
    "Geranylgeranylation": "positive",
    "Autophosphorylation": "positive",
    "Transphosphorylation": "positive",
    "Activation": "positive",
    "IncreaseAmount": "positive",
    "GtpActivation": "positive",
    "Gef": "positive",
    # negative
    "Dephosphorylation": "negative",
    "Demethylation": "negative",
    "Deacetylation": "negative",
    "Deubiquitination": "negative",
    "Desumoylation": "negative",
    "Dehydroxylation": "negative",
    "Defarnesylation": "negative",
    "Deglycosylation": "negative",
    "Deribosylation": "negative",
    "Depalmitoylation": "negative",
    "Demyristoylation": "negative",
    "Degeranylgeranylation": "negative",
    "Inhibition": "negative",
    "DecreaseAmount": "negative",
    "GtpInactivation": "negative",
    "Gap": "negative",
    # neutral
    "Complex": "neutral",
    "Binding": "neutral",
    "Translocation": "neutral",
    "Conversion": "neutral",
    "RegulateActivity": "neutral",  # underspecified at axis level
    "RegulateAmount": "neutral",
}


def derive_axis(stmt_type: str) -> str:
    return _STMT_AXIS.get(stmt_type, "unknown")


def derive_sign(stmt_type: str) -> str:
    return _STMT_SIGN.get(stmt_type, "neutral")


# ── Per-probe scorer registry ──
# Each scorer is registered by per-probe modules; the dispatcher imports
# them lazily so a missing module doesn't break the others.

_SCORERS: dict[str, Callable[..., Awaitable[dict]]] = {}


def register(probe: str, fn: Callable[..., Awaitable[dict]]) -> None:
    _SCORERS[probe] = fn


def has_decomposed(probe: str) -> bool:
    """True iff a decomposed scorer is registered for `probe`."""
    _ensure_loaded()
    return probe in _SCORERS


async def score_decomposed(probe: str, rec: dict, call_subq: CallSubQ) -> dict:
    """Dispatch to the registered per-probe scorer.

    Raises KeyError if no decomposed scorer is registered for `probe`.
    """
    _ensure_loaded()
    return await _SCORERS[probe](rec, call_subq)


_loaded = False


def _ensure_loaded() -> None:
    """Lazy-import per-probe modules so they self-register."""
    global _loaded
    if _loaded:
        return
    # Import each module; failures are silent so a missing probe just
    # means that probe falls back to single-shot.
    for mod in ("relation_axis", "subject_role", "object_role", "scope"):
        try:
            __import__(f"indra_belief.v_phase.decomposed.{mod}")
        except Exception:
            pass
    _loaded = True
