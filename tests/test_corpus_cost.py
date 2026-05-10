"""Tests for cost estimation (Phase 0.1 helper)."""

from __future__ import annotations

import pytest
from indra.statements import Agent, Evidence, Phosphorylation

from indra_belief.corpus import estimate_cost


def _stmt(n_ev: int):
    a = Agent("A", db_refs={"HGNC": "1"})
    b = Agent("B", db_refs={"HGNC": "2"})
    evs = [Evidence(source_api="reach", text=f"sentence {i}") for i in range(n_ev)]
    return Phosphorylation(a, b, evidence=evs)


def test_estimate_zero_stmts():
    out = estimate_cost([])
    assert out["n_stmts"] == 0
    assert out["n_evidences_est"] == 0
    assert out["n_llm_calls_est"] == 0
    assert out["cost_usd"] == 0.0


def test_estimate_uses_actual_evidence_count():
    stmts = [_stmt(2), _stmt(3), _stmt(1)]  # total 6 evidences
    out = estimate_cost(stmts, model_id="claude-sonnet-4-6",
                       avg_llm_calls_per_evidence=5.0)
    assert out["n_stmts"] == 3
    assert out["n_evidences_est"] == 6
    assert out["n_llm_calls_est"] == 30  # 6 * 5
    # Sonnet: $3/M in + $15/M out, 330 in / 70 out per call
    # 30 * 330 = 9900 in tokens; 30 * 70 = 2100 out tokens
    expected = 9900 * (3.00 / 1_000_000) + 2100 * (15.00 / 1_000_000)
    assert out["cost_usd"] == pytest.approx(round(expected, 4))


def test_estimate_unknown_model_warns_and_zeros():
    stmts = [_stmt(1)]
    out = estimate_cost(stmts, model_id="nonexistent-model-9000")
    # Unknown model defaults to 0 prices → 0 cost
    assert out["cost_usd"] == 0.0


def test_estimate_override_prices():
    stmts = [_stmt(1)]  # 1 evidence
    # 5 LLM calls × 330 in × $1/M = $0.00165
    # 5 LLM calls × 70 out × $5/M = $0.00175
    out = estimate_cost(stmts, model_id="custom",
                       in_price_per_m=1.0, out_price_per_m=5.0)
    expected = 5 * 330 * (1.0 / 1_000_000) + 5 * 70 * (5.0 / 1_000_000)
    assert out["cost_usd"] == pytest.approx(round(expected, 4))


def test_estimate_rasmachine_scale_sonnet():
    """Smoke-test against the cost-estimate range in the task graph (0.1)."""
    # 8,724 stmts with avg ~3 evidences, default LLM calls
    out = estimate_cost([_stmt(3) for _ in range(8724)],
                       model_id="claude-sonnet-4-6")
    # Per task graph 0.1 estimate: ~$264 for Sonnet on full rasmachine
    # Allow ±20% for assumption drift
    assert 200 < out["cost_usd"] < 350
    assert out["n_evidences_est"] == 8724 * 3
    assert out["n_llm_calls_est"] == 8724 * 3 * 5


def test_estimate_includes_assumptions():
    out = estimate_cost([_stmt(2)], model_id="claude-sonnet-4-6")
    assert "assumptions" in out
    assert out["assumptions"]["avg_evidences_per_stmt"] == 2.0
    assert out["assumptions"]["avg_llm_calls_per_evidence"] == 5.0
    assert out["assumptions"]["in_price_per_m_tokens_usd"] == 3.00


def test_viewer_cost_panel_in_sync_with_python():
    """Lock the viewer's hardcoded cost-panel constants to Python source.

    `viewer/src/routes/+page.svelte` mirrors `MODEL_PRICES_PER_M_TOKENS` +
    LLM-call assumptions client-side. If Python prices change but the
    viewer doesn't, the dashboard cost projection silently lies. This
    regex-grep enforces that the 4 dashboard models + 3 token assumptions
    match Python source-of-truth.
    """
    import re
    from pathlib import Path

    from indra_belief.corpus.cost import MODEL_PRICES_PER_M_TOKENS

    repo_root = Path(__file__).resolve().parent.parent
    viewer_page = repo_root / "viewer" / "src" / "routes" / "+page.svelte"
    src = viewer_page.read_text(encoding="utf-8")

    # Token-shape constants
    assert "TOKENS_PER_LLM_CALL_IN = 330" in src, "viewer in-tokens drifted"
    assert "TOKENS_PER_LLM_CALL_OUT = 70" in src, "viewer out-tokens drifted"
    assert "LLM_CALLS_PER_EVIDENCE = 5" in src, "viewer LLM-calls drifted"

    # Per-model rates — only the 4 the dashboard surfaces; others in
    # MODEL_PRICES_PER_M_TOKENS aren't on the panel by intent.
    DASHBOARD_MODELS = [
        "gemini-2.5-flash",
        "claude-haiku-4-5",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
    ]
    for model_id in DASHBOARD_MODELS:
        assert model_id in MODEL_PRICES_PER_M_TOKENS, \
            f"{model_id} missing from Python price table"
        in_p, out_p = MODEL_PRICES_PER_M_TOKENS[model_id]
        # Find viewer's tuple for this model: ['Name', 'model-id', in_p, out_p]
        pattern = re.compile(
            rf"\['[^']+',\s*'{re.escape(model_id)}',\s*([\d.]+),\s*([\d.]+)\]"
        )
        m = pattern.search(src)
        assert m, f"viewer cost panel missing entry for {model_id}"
        viewer_in = float(m.group(1))
        viewer_out = float(m.group(2))
        assert viewer_in == in_p, f"{model_id} in-price drift: viewer {viewer_in} vs python {in_p}"
        assert viewer_out == out_p, f"{model_id} out-price drift: viewer {viewer_out} vs python {out_p}"
