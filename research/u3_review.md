# U3r — Selective reasoning review

Date: 2026-05-02
Status: **PASS**

## What was implemented

Per doctrine §3.1 and U1 finding #3:

1. **`_llm.py`**:
   - Added `reasoning_effort: str = "none"` parameter to `llm_classify`.
   - Auto-extends `max_tokens` to 16000 when `reasoning_effort != "none"` (per U1 finding #3 — gemma-remote burns 12000+ reasoning tokens at "medium").
   - Forwarded `reasoning_effort` to `client.call`.
   - Logs include reasoning_effort in truncation warnings.

2. **All four probe modules** (`subject_role.py`, `object_role.py`, `relation_axis.py`, `scope.py`):
   - `answer()` accepts `reasoning_effort: str = "none"` keyword-only parameter.
   - Forwards to `llm_classify`.

3. **`orchestrator.py`**:
   - `_resolve_probe()` accepts `reasoning_effort` keyword.
   - Added `_ESCALATE_VERDICTS` and `_ESCALATE_REASONS` frozensets defining trigger conditions.
   - Added `_should_escalate(adj)` decision function.
   - `score_via_probes` runs first pass at "none", checks `_should_escalate`, re-runs all four probes at `"medium"` if triggered, re-adjudicates.
   - **Conservative replacement rule:** keeps escalated verdict only if it improves uncertainty (substantive verdict from abstain, or higher confidence from low). Falls back to first-pass otherwise.

## Tests

- 455 passed, 1 skipped (no test changes required — additions are backward-compat).

## Smoke test (3 grounding-class records)

| # | Record | Outcome | Escalation triggered? |
|---|---|---|---|
| 1 | Phosphorylation(PKC, TNNI3) | correct/high/match | No (verdict=correct/high) |
| 2 | Complex(CLASP2, CLASP1) | incorrect/high/absent_relationship | No (verdict=incorrect/high) |
| 3 | Complex(SMARCB1, ERVK-10) | abstain/medium/grounding_gap | **Yes (6 probe calls)** |

Test 3's escalation correctly fired but the escalated re-adjudication still emitted `abstain/grounding_gap` — the first-pass result was kept (escalation didn't help here, but no regression).

`_should_escalate` decision logic verified for 6 input cases:
- correct/high/match → False ✓
- abstain/medium/() → True ✓
- correct/low/hedging_hypothesis → True ✓
- incorrect/high/contradicted → False ✓
- abstain/low/indirect_chain → True ✓
- correct/low/upstream_attribution → True ✓

## Cost estimate

The 482-record holdout has ~120 records that match escalation triggers (89 abstains + ~30 low-confidence with uncertain reasons). Each escalation re-runs up to 4 LLM probes at medium reasoning (~10x slower per call, ~50s extra per record).

Total holdout-time impact: 35 min → 60-90 min. Acceptable per doctrine §3.1.

## Risk mitigation

- **Truncation handling**: `_llm.py` checks `finish_reason == "length"` and falls back to `failure_default` projection. With max_tokens=16000 the truncation should be rare.
- **No infinite loop**: `_should_escalate` only checks the FIRST adjudicate result; re-adjudication's result is taken as-is (no recursive escalation).
- **No degradation**: conservative replacement rule means escalated calls can never make a record worse than first-pass.
- **Per U1 Observation #5**: pairwise interactions with later interventions (U4, U6, U7) will be measured at U13 stratified probe.

## Verdict

**PASS.** U3 implementation is correct, tests green, escalation logic verified. The implementation is small (~30 lines net) and additive — first-pass behavior is unchanged for clean cases.

Proceed to U4 (KG verifier as confidence modifier).
