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
        abstain.
      kind: the probe kind (used for call_log telemetry tag).
      client: the model client.
      max_tokens, temperature: standard knobs.

    Returns:
      (answer, rationale, succeeded).
      - succeeded=True when LLM returned a parseable answer in the
        allowed set (even if that answer is "abstain" — legitimate
        abstain for probes whose answer set includes it).
      - succeeded=False on any failure (transport, JSON, schema). In
        that case answer is forced to "abstain" if "abstain" is in the
        answer set; otherwise the caller must provide a fallback.
    """
    messages: list[dict] = []
    for shot_q, shot_a in few_shots:
        messages.append({"role": "user", "content": shot_q})
        messages.append({"role": "assistant", "content": shot_a})
    messages.append({"role": "user", "content": user_message})

    try:
        response = client.call(
            system=system_prompt,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
            reasoning_effort="none",
            kind=f"probe_{kind}",
        )
    except Exception as e:
        log.warning("probe(%s): client.call() failed: %s", kind, e)
        return _failure_fallback(answer_set, f"transport_error:{type(e).__name__}")

    if getattr(response, "finish_reason", None) == "length":
        log.warning("probe(%s): response truncated; abstaining", kind)
        return _failure_fallback(answer_set, "response_truncated")

    content = (response.content or "").strip()
    if not content:
        # Some backends emit reasoning into raw_text but leave content empty.
        content = (response.raw_text or "").strip()
    if not content:
        return _failure_fallback(answer_set, "empty_response")

    obj = _extract_json(content)
    if obj is None:
        return _failure_fallback(answer_set, "json_parse_failure")

    answer = obj.get("answer")
    if not isinstance(answer, str) or answer not in answer_set:
        return _failure_fallback(answer_set, f"invalid_answer:{answer!r}")

    rationale = obj.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = ""
    return answer, rationale[:200], True


def _failure_fallback(
    answer_set: frozenset[str], rationale: str,
) -> tuple[str, str, bool]:
    """Return a failure-mode triple. Uses 'abstain' if it's in the
    answer set; otherwise picks the first sorted member (caller-chosen
    fallback for probes that don't support abstain natively)."""
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
