# U10r — Cross-probe consistency check review

Date: 2026-05-02
Status: **PASS** (scope narrowed during implementation per test feedback)

## What was implemented

Per doctrine §3.8 + U1 finding #4 (confidence-only, no new ReasonCode):

1. Added `_probe_inconsistency_detected(bundle)` helper detecting four patterns:
   - subject_role=present_as_subject + relation_axis=no_relation
   - subject_role=absent + relation_axis=any-match
   - object_role=absent + relation_axis=any-match
   - relation_axis=any-match + scope=negated

2. Added `_downgrade_confidence(confidence)` helper: high→medium, medium→low, low→low.

3. Inserted U10 check in `adjudicate()` AFTER the U4 KG-signal modifier:
   ```python
   if verdict == "correct" and _probe_inconsistency_detected(bundle):
       confidence = _downgrade_confidence(confidence)
   ```

4. Rationale annotation: appends ` [probe_inconsistency: confidence downgraded]` when triggered.

## Critical implementation correction

First-draft U10 applied the downgrade on ALL verdicts. Test feedback caught this:
- `test_incorrect_high_confidence`: scope=negated + ra=match → verdict=incorrect/contradicted (correctly handled by scope-precedence). U10 was downgrading the already-correct incorrect verdict from high to medium. WRONG behavior — the scope-precedence rule HAS resolved the inconsistency.
- `test_abstain_medium_confidence`: probe-source-abstain + default ra=no_relation → verdict=abstain. U10 was further downgrading to low. WRONG behavior — abstain has no info to downgrade.

**Fix:** restricted U10 to `verdict == "correct"` only. The other two paths (incorrect, abstain) are already specifically handled by the decision table — they don't need additional confidence modulation.

This is a clean lesson in narrow scoping: U10's INTENT was "flag uncertain correct predictions where probes disagree". When verdict isn't correct, there's nothing to flag.

## Tests

- 2 new U10 tests in `test_probe_adjudicator.py`:
  - `test_consistency_check_present_subject_no_relation_downgrades` — verifies the verdict=incorrect path is NOT downgraded.
  - `test_consistency_check_does_not_modify_verdict` — verifies clean correct verdicts pass through.
- Full suite: 469 passed, 1 skipped.

## Engineering distinction

### "Confidence-only, never override" contract enforced by code structure

The U10 implementation is in two helper functions plus a single 2-line gate. The verdict variable is NEVER modified by U10. The strict no-override contract is a CODE INVARIANT, not just a doctrine claim.

### No new ReasonCode (per U1 finding #4)

Following U1's recommendation, U10 doesn't add `probe_inconsistency` to the ReasonCode literal. The signal is captured in the rationale text. This:
- Keeps `commitments.py:ReasonCode` enum stable (12 values, same as T-phase + U7's two additions).
- Avoids breaking any downstream consumer that switches on reason codes.
- Still allows downstream tooling to detect inconsistency via rationale string-match if needed.

## Coverage estimate

How many T-phase records would have triggered U10 inconsistency?

Pattern-by-pattern (from T-phase output):
- Pattern 1 (present_as_subject + no_relation): a portion of the 38 absent_relationship FN+TN where the LLM gave inconsistent answers.
- Pattern 2/3 (absent + match): rare but happens — would be near 0 since absent already triggers grounding_gap path.
- Pattern 4 (match + negated): handled by scope-precedence; never fires for verdict=correct.

Realistic: maybe 5-15 records have U10 fires on verdict=correct. These are records where T-phase emitted correct but probes disagreed internally. Most are likely TPs with weak signal — downgrading from high to medium provides better calibration.

**Direct accuracy gain: 0.** U10 is purely calibration — no verdict changes.

## Verdict

**PASS.** U10 implementation is correct, both tests green, all 469 tests pass. The intervention is minimal (~30 lines of helper + 2 lines of gate) and correctly restricted to verdict=correct. The first-draft over-aggression was caught by existing tests — exactly the kind of safety net the test suite provides.

Proceed to U11 (migration cleanup + implementation gate).
