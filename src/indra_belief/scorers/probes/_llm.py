"""Shared LLM-call helper for the four probe modules.

Each probe module builds a system prompt + user message + few-shot
exemplars, calls _llm_classify, and gets back (answer, rationale)
where answer is validated against the probe's closed answer set.

Failure modes (transport timeout, malformed JSON, out-of-set answer)
all abstain — no retries (consistent with the O-phase no-retry doctrine
in parse_evidence). The probe module then wraps the result in a
ProbeResponse with source="abstain" or source="llm".
"""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from indra_belief.scorers.probes.types import ProbeKind

if TYPE_CHECKING:
    from indra_belief.model_client import ModelClient


log = logging.getLogger(__name__)


def llm_classify(
    *,
    system_prompt: str,
    few_shots: list[tuple[str, str]],
    user_message: str,
    answer_set: frozenset[str],
    kind: ProbeKind,
    client: "ModelClient",
    max_tokens: int = 200,
    temperature: float = 0.1,
    failure_default: str | None = None,
    reasoning_effort: str = "none",
) -> tuple[str, str, bool]:
    """Run a closed-set classification probe via LLM.

    Args:
      system_prompt: prompt stating the probe's question and the closed
        answer set, demanding JSON output {answer, rationale}.
      few_shots: list of (user_question, assistant_answer) pairs threaded
        as alternating user/assistant messages BEFORE the actual question.
        The assistant answer string MUST be valid JSON.
      user_message: the actual question to answer.
      answer_set: closed set of valid answer values; out-of-set answers
        project to failure_default (or to "abstain" if available).
      kind: the probe kind (used for call_log telemetry tag).
      client: the model client.
      max_tokens, temperature: standard knobs.
      failure_default: optional override for the failure-mode answer.
        If provided, used in preference to "abstain" or alphabetical
        fallback. T-phase Fix A: relation_axis passes "no_relation"
        as failure_default after dropping "abstain" from its answer set.
      reasoning_effort: U-phase U3 selective reasoning. Default "none"
        for fast first-pass. Caller can pass "low"/"medium" to escalate
        on hard cases. When escalated, max_tokens auto-extends to 16000
        to leave room for both reasoning and content (gemma-remote
        burns ~12000 reasoning tokens at "medium").

    Returns:
      (answer, rationale, succeeded).
      - succeeded=True when LLM returned a parseable answer in the
        allowed set.
      - succeeded=False on any failure (transport, JSON, schema). In
        that case answer is failure_default (if provided), else
        "abstain" (if in the answer set), else the alphabetically-first
        member of the answer set.
    """
    messages: list[dict] = []
    for shot_q, shot_a in few_shots:
        messages.append({"role": "user", "content": shot_q})
        messages.append({"role": "assistant", "content": shot_a})
    messages.append({"role": "user", "content": user_message})

    # U3 escalation: when reasoning is non-none, extend max_tokens budget
    # to fit ~12000 reasoning tokens + content. Per model_client doctrine,
    # gemma-remote at reasoning="medium" burns 12000+ reasoning tokens
    # before emitting JSON, causing truncation on default 200-token budget.
    effective_max_tokens = max_tokens
    if reasoning_effort != "none":
        effective_max_tokens = max(max_tokens, 16000)

    try:
        response = client.call(
            system=system_prompt,
            messages=messages,
            max_tokens=effective_max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            reasoning_effort=reasoning_effort,
            kind=f"probe_{kind}",
        )
    except Exception as e:
        log.warning("probe(%s): client.call() failed: %s", kind, e)
        return _failure_fallback(
            answer_set, f"transport_error:{type(e).__name__}", failure_default,
        )

    if getattr(response, "finish_reason", None) == "length":
        log.warning(
            "probe(%s, reasoning=%s): response truncated; abstaining",
            kind, reasoning_effort,
        )
        return _failure_fallback(
            answer_set, "response_truncated", failure_default,
        )

    content = (response.content or "").strip()
    if not content:
        # Some backends emit reasoning into raw_text but leave content empty.
        content = (response.raw_text or "").strip()
    if not content:
        return _failure_fallback(answer_set, "empty_response", failure_default)

    obj = _extract_json(content)
    if obj is None:
        return _failure_fallback(
            answer_set, "json_parse_failure", failure_default,
        )

    answer = obj.get("answer")
    if not isinstance(answer, str) or answer not in answer_set:
        # T-phase Fix A: when failure_default is provided AND the LLM
        # returned a parseable but out-of-set string answer, treat it as
        # a successful classification with the answer projected. The LLM
        # did respond meaningfully — we just had to map the response into
        # the closed set. Source stays "llm".
        # Other failure modes (transport, empty, JSON) still source="abstain".
        if (
            failure_default is not None
            and failure_default in answer_set
            and isinstance(answer, str)
        ):
            rationale = obj.get("rationale", "")
            if not isinstance(rationale, str):
                rationale = ""
            log.info(
                "probe(%s): out-of-set answer %r projected to %r",
                kind, answer, failure_default,
            )
            return failure_default, rationale[:200], True
        return _failure_fallback(
            answer_set, f"invalid_answer:{answer!r}", failure_default,
        )

    rationale = obj.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = ""
    return answer, rationale[:200], True


def _failure_fallback(
    answer_set: frozenset[str],
    rationale: str,
    failure_default: str | None = None,
) -> tuple[str, str, bool]:
    """Return a failure-mode triple for transport/JSON/empty failures.

    Out-of-set string answers go through the success path with projection
    (see llm_classify). This function handles only true LLM failures.

    Priority: explicit failure_default (must be in answer_set) >
    "abstain" (if in answer_set) > alphabetical first member.
    """
    if failure_default is not None and failure_default in answer_set:
        return failure_default, rationale, False
    if "abstain" in answer_set:
        return "abstain", rationale, False
    return sorted(answer_set)[0], rationale, False


def _extract_json(text: str) -> dict | None:
    """Extract a JSON object from a model response.

    Tries strict json.loads first; falls back to finding the outermost
    {...} block. Returns None on failure.
    """
    text = text.strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        pass
    # Find first { and last } and try again
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            pass
    return None
