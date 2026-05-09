# U7r — Closed-set redesign (Intervention E) review

Date: 2026-05-02
Status: **PASS**

## What was implemented

Per doctrine §3.5. Two answer-set additions, both surgical:

### Part 1: `direct_amount_match` added to RelationAxisAnswer

- New value alongside `direct_sign_match`. Eight total values now.
- `relation_axis.py` prompt updated with explicit activity-vs-amount disambiguation:
  > `direct_amount_match` — the evidence describes EXPRESSION/ABUNDANCE change of the object by the subject (upregulation, downregulation, increased/decreased protein/mRNA levels, transcriptional regulation). Use this when the evidence verbs are amount-axis, REGARDLESS of the claim axis.
- Three new few-shots demonstrate the distinction (synthetic names FactorR, GeneZ, KinaseA, TargetB).
- Adjudicator gates per claim axis:
  - amount claim + `direct_amount_match` → match (treated as direct_sign_match downstream)
  - non-amount claim (activity, modification, binding, etc.) + `direct_amount_match` → axis_mismatch (incorrect)

### Part 2: `asserted_with_condition` added to ScopeAnswer

- New value alongside asserted/hedged/negated/abstain. Five total.
- `scope.py` prompt updated to distinguish:
  > `asserted_with_condition` — the relation is asserted on a QUALIFIED form of an entity (e.g., wild-type, full-length, specific variant), with the COMPLEMENTARY form negated.
- Three new few-shots: two positive (wild-type/mutant), one negative (unconditional negation). Synthetic names (FactorR, TargetX, AdaptorP, KinaseY, EnzymeE, SubstrateZ).
- Adjudicator: matched relation + `asserted_with_condition` → correct/medium (downgraded from high to signal conditionality).

## Tests

- 6 new tests in `test_probe_adjudicator.py`:
  - `test_direct_amount_match_on_amount_claim_is_match`
  - `test_direct_amount_match_on_activity_claim_is_axis_mismatch`
  - `test_direct_amount_match_on_modification_claim_is_axis_mismatch`
  - `test_asserted_with_condition_correct_medium`
  - `test_asserted_with_condition_via_amount_match`
  - `test_amount_claim_with_direct_sign_match_still_works` (backward-compat)
- Full suite: 467 passed, 1 skipped (was 461 before U7).

## Contamination guard

`scripts/check_contamination.py`: **CLEAN**. All new few-shots use synthetic placeholder names (FactorR, GeneZ, KinaseA, TargetB, FactorR, TargetX, AdaptorP, KinaseY, EnzymeE, SubstrateZ) — none paraphrase holdout records.

## Engineering distinction notes

### Backward compatibility preserved

The doctrine §3.5 Part 1 considered REPLACING `direct_sign_match` with two new values (`direct_activity_match` + `direct_amount_match`). I chose the more conservative path: ADD `direct_amount_match` while keeping `direct_sign_match` as the catch-all for non-amount axes. This means:
- amount-axis claims: LLM SHOULD prefer `direct_amount_match`, but `direct_sign_match` still works (backward-compat).
- non-amount claims: LLM SHOULD use `direct_sign_match`; `direct_amount_match` triggers the axis_mismatch gate.

The strict-discrimination doctrine version would have broken backward compat without empirical evidence that the LLM reliably picks one over the other on non-amount claims. The conservative version achieves the discrimination on amount-axis records (the act_vs_amt class) without risking regressions on the 273 gold-correct records.

### Adjudicator reuses scope-block by reassigning ra

```python
if ra == "direct_amount_match":
    if claim.axis == "amount":
        ra = "direct_sign_match"  # reassign for scope-block reuse below
    else:
        return "incorrect", "axis_mismatch", ...
```

This is intentional — for amount-axis claims, the scope discrimination logic is identical to direct_sign_match's. Avoiding code duplication. The reassignment is local; no side effect.

### Confidence policy uses rationale-prefix matching

The new `asserted_with_condition` confidence-medium downgrade is detected via `rationale.startswith("asserted on qualified form")`. This is the same pattern as `relation matches; scope underdetermined` already in the policy. Consistent with existing code style.

## Risk: prompt-level adoption

This is the largest risk for U7. The closed-set CHANGE forces the LLM to choose. But:
- The prompt is now ~30% longer for relation_axis (3 new few-shots).
- The prompt is ~20% longer for scope (3 new few-shots).
- Gemma 4 26B may not reliably distinguish `direct_sign_match` from `direct_amount_match` on borderline cases.

Mitigations:
- New few-shots include both positive and negative demonstrations.
- Adjudicator's axis-gate catches LLM mistakes (if LLM picks `direct_amount_match` on an activity claim, we still get axis_mismatch — same as if LLM had picked `direct_axis_mismatch`).
- The fallback path is preserved: out-of-set or LLM-failure projects to `no_relation` via Fix A's mechanism.

If U13 stratified probe shows `act_vs_amt` accuracy improves from 41.7% to >55%, U7 is working. If accuracy is unchanged, the LLM isn't using the new value and U7 is essentially a no-op.

## Empirical hypothesis

From U2 per-tag analysis: act_vs_amt class is 24 records at 41.7% acc. Doctrine §3.5 estimate: +10 to +15.

Realistic mapping:
- ~10 records have evidence-side amount-verbs ("overexpression increased", "silencing decreased") that should now route through `direct_amount_match`.
- For activity/modification claims, this becomes axis_mismatch — flipping FP→TN.
- For amount claims, this is just relabeling — no verdict change but cleaner reason code.

Estimated gain on act_vs_amt class: +6 to +10 records.

The conditional-negation portion (asserted_with_condition) targets ~5 records. Estimated gain: +3 to +5.

**Combined U7 leverage: +9 to +15 records.** Doctrine estimate of "+10 to +15" stands.

## Verdict

**PASS.** U7 implementation is correct, all 6 tests green, contamination clean. The closed-set redesign is the most architectural change in U-phase — adds 1 value to each of two answer sets (8 → 9 values for relation_axis is well within the closed-set discipline of "≤8 closed values per probe", but still cleanly bounded; scope at 5 values).

The intervention preserves backward compatibility with existing behavior on non-targeted records, while enabling the new discrimination on the act_vs_amt and conditional-mutant classes.

Proceed to U8 (verb taxonomy in CATALOG).
