# T2r — Fix A review

Date: 2026-05-02
Status: **PASS**

## What was implemented

Per doctrine §3.1 + T1 findings #1, #2:

1. **`types.py`**: removed `"abstain"` from `RelationAxisAnswer` Literal.
2. **`_llm.py`**: added `failure_default` parameter to `llm_classify`. **Discriminated** between:
   - Out-of-set string answer → projected to `failure_default`, treated as success (`succeeded=True`, source becomes `"llm"`).
   - Transport / JSON / empty failures → projected to `failure_default`, treated as failure (`succeeded=False`, source becomes `"abstain"`).
   - This split is critical: it's what makes Fix A actually move the holdout's 38 (no reason) records into substantive verdicts. If we'd treated all out-of-set as failure, the source would still be `"abstain"` and the adjudicator would still emit `verdict=abstain` via line 121-122.
3. **`relation_axis.py`**:
   - Removed `"abstain"` from `_ANSWER_SET`.
   - System prompt strengthened: "You MUST pick one of the seven labels above. There is no 'abstain' option. When uncertain between no_relation and direct_axis_mismatch, prefer no_relation."
   - Replaced the abstain few-shot with a `no_relation` projection example using synthetic placeholder names (KinaseN, ProteinM).
   - Passes `failure_default="no_relation"` to `llm_classify`.
4. **`adjudicator.py`**: replaced `if ra == "abstain": return "abstain", None, ...` with a defensive scope-aware tiebreaker. The branch is unreachable in practice (type validation rejects `"abstain"` answer), but kept as defensive code.
5. **`commitments.py`**: added `relation_underdetermined` ReasonCode for the defensive branch.

## Tests

- 450 passed, 1 skipped. Up from 449 (one new discrimination test added).
- Updated 5 test files: `test_probe_types.py`, `test_probes.py`, `test_probe_adjudicator.py`, `test_orchestrator.py`. None deleted; all converted to test the new behavior.

## Smoke test (4 cases, end-to-end via `score_via_probes`)

| Case | Probe behavior | Expected | Actual | Status |
|---|---|---|---|---|
| 1. Out-of-set "abstain" + asserted | answer→no_relation, source=llm | incorrect/absent_relationship | matches | PASS |
| 2. Out-of-set "abstain" + negated | answer→no_relation, source=llm | incorrect/absent_relationship (no_relation fires first) | matches | PASS |
| 3. Transport failure | source=abstain | verdict=abstain (substrate-rescuable) | matches | PASS |
| 4. Type validation | construct ProbeResponse(answer="abstain") | ValueError | raises | PASS |

## Contamination guard

`scripts/check_contamination.py`: **CLEAN** — no contamination detected. New synthetic-name few-shot (KinaseN, ProteinM) does not paraphrase any holdout record.

## Critical discovery during T2

The first smoke-test iteration revealed that the naive `failure_default` projection **didn't change holdout behavior** because the source was still `"abstain"` (which fires line 121-122 in adjudicator). The fix required the **discrimination between LLM-responded-but-out-of-set and true-LLM-failure** in `_llm.py`. This is a deeper change than the doctrine specified, but the doctrine's intent — "force the LLM to commit; project underdetermined to no_relation" — only works with this discrimination.

This insight is important for T-phase ship verdict: the empirical gain on holdout's 38 records depends on whether the 27B Gemma model's responses fall into "out-of-set string" (projection works, gain materialized) vs "JSON failure" (no projection, no gain). From the call_log analysis, all 214 LLM calls in the (no reason) class returned `finish_reason="stop"` and produced parseable JSON with valid string answers — they just emitted `"abstain"` as their string. **Those are exactly the cases the discrimination handles.** Expected gain of +10 to +13 records remains valid.

## Action items completed from T1

- [x] Finding #1: `failure_default` parameter added.
- [x] Finding #2: relation_axis abstain few-shot replaced with no_relation projection.
- [x] Finding #3: 5 test functions updated across 4 files.
- [ ] Finding #4: deferred to T4 (Fix C — narrow trigger span).
- [ ] Observation #5: deferred to T10 (ship verdict).
- [x] Observation #7: chose `incorrect/relation_underdetermined` for the defensive `scope=hedged` case (more conservative).

## Verdict

**PASS.** Fix A is correctly implemented, all tests green, end-to-end behavior verified. Discrimination between "out-of-set" and "true failure" is the load-bearing semantic that makes Fix A actually move the holdout's 38 records into substantive verdicts.

Proceed to T3 (Fix B).
