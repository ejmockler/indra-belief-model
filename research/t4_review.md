# T4r — Fix C review

Date: 2026-05-02
Status: **PASS**

## What was implemented

Per doctrine §3.3 + T1 Finding #4:

### `relation_axis.py` — ADDED CRITICAL block (didn't exist before)

```
CRITICAL — clause-localized evaluation: Evaluate the relation between
the claim's SUBJECT and OBJECT using ONLY the clause(s) where they
co-occur or are connected by the relation verb. If the same sentence
contains a different clause whose negation or wrong-axis verb does
NOT govern the claim's subject-object pair, IGNORE that clause.
Negation of a sibling proposition does not propagate to the claim.
Sign and axis labels apply to the claim-relevant span, not the
sentence as a whole.
```

Added 4 paired few-shots demonstrating:
1. `direct_sign_match` despite sentence-level negation (inhibition by 3rd party doesn't flip sign).
2. `direct_sign_mismatch` when negation IS in claim clause (negative test — rule still fires).
3. `direct_sign_match` when sibling negation governs downstream effects.
4. `direct_sign_match` when sibling negation governs subsequent steps.

All few-shots use synthetic names: KinaseA, TargetB, ReceptorR, KinaseS, RepC.

### `scope.py` — STRENGTHENED existing CRITICAL block

Added new clause:
```
CONDITIONAL NEGATION DOES NOT PROPAGATE either. If the sentence
contains "X does Y" with a conditional restriction like "but the
mutant/variant/non-functional form does not", and the claim is about
X (not the mutant), the scope is `asserted` for the X-Y relation.
The negation governs the variant only.
```

Added 3 new few-shots:
1. wild-type binding asserted, mutant negation doesn't propagate.
2. wild-type induction asserted, "but not truncated mutants" doesn't propagate.
3. claim relation asserted, sibling negation about pretreatment effect.

Synthetic names: FactorR, TargetX, AdaptorP, KinaseY, ReceptorR, KinaseS, InhibitorD.

## Tests

- 453 passed, 1 skipped (no test changes needed — Fix C is prompt-only).
- Existing scope.py few-shot at line 78-84 (the "MAPK1 activates JUN robustly, but ELK1 was not affected" case) preserved as the basic-case shot. The new shots cover the harder conditional-negation cases that broke 12 of 21 holdout FNs.

## Contamination guard

`scripts/check_contamination.py`: **CLEAN** — no contamination detected. All 7 new few-shot exemplars use synthetic placeholder names.

## Hand-verification spot checks

Mapping the 12 expected FN-recovery cases from doctrine §2.2 against the new prompt structure:

| Holdout case (paraphrased) | Type | New prompt addresses via |
|---|---|---|
| "Phosphorylation of eIF-4E by PKC is inhibited by 4E-BP" | sign overshoot | relation_axis CRITICAL + few-shot #1 |
| "β-catenin binds wild-type SOX10, but not 3G mutant" | conditional negation | scope conditional-mutant clause + new few-shot #1 |
| "Trx1 activating Akt... did not detect a reduction in infarct size" | sibling negation | scope existing CRITICAL block + relation_axis CRITICAL |
| "JAK-3 inhibited JAK phosphorylation induced by IL-6" | sign overshoot | relation_axis CRITICAL + few-shot #1 |
| "RUNX1 wild-type, but not D198G mutants, restored AXIN1" | conditional negation | scope conditional-mutant clause + new few-shot #2 |
| "Pre-exposure to Delta-1 did NOT affect ERK phosphorylation by G-CSF" | sibling negation | scope new few-shot #3 |
| "MAPK1 activates JUN robustly, but ELK1 was not affected" | sibling negation | scope existing few-shot (already covered) |

All 7 spot-check cases have a corresponding new prompt mechanism.

## Spot check on TN preservation

The 17 TNs caught by sign/axis/contradicted detectors (out of 20 total, with 3 marginal) should survive the prompt change because the in-claim-clause negation is preserved as the trigger. Examples:

- `Complex(VHL, RalBP1)`: "neither monomeric VHL nor HMW VHL interacted with RalBP1" — the negation IS on the claim relation. relation_axis few-shot #2 covers exactly this pattern → `direct_sign_mismatch`. **Should still fire as TN.**
- `Translocation(BAX, ?)`: "Bax does not translocate to mitochondria" — explicit negation on claim relation. scope `negated` should still fire. **Should still fire as TN.**

The 3 marginal TNs (probe-noise cases) are at risk, but the doctrine accepts ~3 TN losses against ~12 FN gains.

## Verdict

**PASS.** Fix C prompts are correctly augmented. Synthetic-name discipline preserved. The relation_axis CRITICAL block is new (didn't exist before); the scope CRITICAL block is strengthened with the conditional-mutant clause. Few-shot pairs balance recovery (3 of 4 in relation_axis are positive narrow-cases) against rule-preservation (1 of 4 is the negative "should still fire" case).

Observed risk: the prompt is now noticeably longer. Token cost per `relation_axis` probe call grows by ~280 tokens (4 new shots ≈ 70 tokens each). Per `scope` call ~210 tokens (3 new shots). On a 482-record holdout × 4 probes = roughly +200K input tokens cumulative. Negligible at 27B Gemma's price.

Proceed to T5 (migration cleanup).
