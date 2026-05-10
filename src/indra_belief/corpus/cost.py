"""LLM-call cost estimation — Phase 0.1 made into a helper.

`estimate_cost(stmts, model_id)` projects how many LLM calls + tokens +
USD a `score_corpus` run will consume. The auditor's natural pre-run
"what will this cost?" check before clicking Go.

Empirical anchor (per memory: substrate-vs-LLM lever):
  - S-phase substrate-resolves only ~1.2% of records to zero LLM calls
    (target was 50%). 68.5% use all 4 LLM probes per evidence.
  - Plus ~1 LLM call per evidence for grounding verification.
  - Avg ~400 tokens per LLM call (~330 in + ~70 out → 5:1 ratio).

Defaults bake the conservative assumption (substrate ≤2%, ~5 LLM calls
per evidence). Override per project.
"""

from __future__ import annotations

import logging
from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from indra.statements import Statement

log = logging.getLogger(__name__)


# Cost per million tokens (USD), as of 2026-05-09 published rates.
# Public list pricing — adjust for your contracted rates.
#
# NOTE: viewer/src/routes/+page.svelte mirrors this table client-side for
# the dashboard cost panel. When rates change, update both — there is no
# build-time check that they match.
MODEL_PRICES_PER_M_TOKENS: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-haiku-4-5":   (0.80,  4.00),
    "claude-sonnet-4-6":  (3.00, 15.00),
    "claude-opus-4-7":   (15.00, 75.00),
    # Google (estimated; see Gemini pricing page for exact)
    "gemini-2.5-flash":   (0.075, 0.30),
    "gemini-2.5-pro":     (1.25,  5.00),
    # OpenAI (estimated)
    "gpt-4o":             (2.50, 10.00),
    "gpt-4o-mini":        (0.15,  0.60),
}


def estimate_cost(
    stmts: Iterable["Statement"],
    *,
    model_id: str = "claude-sonnet-4-6",
    avg_evidences_per_stmt: float | None = None,
    avg_llm_calls_per_evidence: float = 5.0,
    avg_input_tokens_per_call: int = 330,
    avg_output_tokens_per_call: int = 70,
    in_price_per_m: float | None = None,
    out_price_per_m: float | None = None,
) -> dict:
    """Project LLM-call counts + token volume + USD for a `score_corpus` run.

    Args:
        stmts: list/iterable of INDRA Statements (consumed once for counts).
        model_id: looked up in `MODEL_PRICES_PER_M_TOKENS` unless overridden.
        avg_evidences_per_stmt: if None, computed from the actual stmts.
        avg_llm_calls_per_evidence: default 5 — 4 probes (substrate ≤2%
            short-circuit per s_phase_ship.md) + 1 grounding.
        avg_input_tokens_per_call / avg_output_tokens_per_call: typical
            decomposed-probe call shape.
        in_price_per_m / out_price_per_m: override model's rate (e.g. for
            negotiated rates or unlisted models).

    Returns:
        dict with `n_stmts`, `n_evidences_est`, `n_llm_calls_est`,
        `input_tokens_est`, `output_tokens_est`, `cost_usd`,
        `model_id`, `assumptions`.
    """
    stmts = list(stmts)
    n_stmts = len(stmts)

    if avg_evidences_per_stmt is None:
        total_evidences = sum(len(getattr(s, "evidence", []) or []) for s in stmts)
        avg_evidences_per_stmt = (total_evidences / n_stmts) if n_stmts else 0.0
        n_evidences = total_evidences
    else:
        n_evidences = round(n_stmts * avg_evidences_per_stmt)

    n_llm_calls = round(n_evidences * avg_llm_calls_per_evidence)
    input_tokens = n_llm_calls * avg_input_tokens_per_call
    output_tokens = n_llm_calls * avg_output_tokens_per_call

    if in_price_per_m is None or out_price_per_m is None:
        prices = MODEL_PRICES_PER_M_TOKENS.get(model_id)
        if prices is None:
            log.warning(
                "model_id %r unknown to MODEL_PRICES_PER_M_TOKENS; "
                "pass in_price_per_m + out_price_per_m to override",
                model_id,
            )
            in_price_per_m = in_price_per_m or 0.0
            out_price_per_m = out_price_per_m or 0.0
        else:
            in_price_per_m = in_price_per_m or prices[0]
            out_price_per_m = out_price_per_m or prices[1]

    cost_usd = (
        input_tokens * (in_price_per_m / 1_000_000)
        + output_tokens * (out_price_per_m / 1_000_000)
    )

    return {
        "n_stmts": n_stmts,
        "n_evidences_est": n_evidences,
        "n_llm_calls_est": n_llm_calls,
        "input_tokens_est": input_tokens,
        "output_tokens_est": output_tokens,
        "cost_usd": round(cost_usd, 4),
        "model_id": model_id,
        "assumptions": {
            "avg_evidences_per_stmt": round(avg_evidences_per_stmt, 2),
            "avg_llm_calls_per_evidence": avg_llm_calls_per_evidence,
            "avg_input_tokens_per_call": avg_input_tokens_per_call,
            "avg_output_tokens_per_call": avg_output_tokens_per_call,
            "in_price_per_m_tokens_usd": in_price_per_m,
            "out_price_per_m_tokens_usd": out_price_per_m,
        },
    }
