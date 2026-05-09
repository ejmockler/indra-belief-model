# U9r — Fix A softening + Fix C tightening review

Date: 2026-05-02
Status: **PASS** (contamination caught and fixed mid-implementation)

## What was implemented

Per doctrine §3.7:

### Fix A softening (relation_axis.py prompt)

Replaced "When uncertain between no_relation and direct_axis_mismatch, prefer no_relation" with a 5-step decision priority list:

```
You MUST pick one of the eight labels above. There is no "abstain"
option. Decision priority:
  1. If the relation between subject and object is explicitly asserted
     (with appropriate axis verb), use direct_sign_match (or
     direct_amount_match for expression-axis evidence).
  2. If the relation is asserted but with a chain mediator, use
     via_mediator (if mediator is named) or via_mediator_partial.
  3. If the relation appears with the OPPOSITE sign, use
     direct_sign_mismatch.
  4. If the evidence describes a different axis verb between the
     entities (e.g., phosphorylation when claim is activity), use
     direct_axis_mismatch.
  5. If both entities are mentioned but NO relation is asserted between
     them, use no_relation.
```

This explicitly orders priorities so the LLM checks ASSERTION first, NEGATION/AXIS-MISMATCH next, NO_RELATION last. Should counter T-phase's 18 absent_relationship overshoots (5 TP regressions + 13 A→F).

Added 2 new few-shots demonstrating assertion-in-dense-evidence:
1. ReceptorR-KinaseS complex with parallel-pathway distractor.
2. AdaptorP activates GeneZ alongside other targets in a list.

Both use synthetic placeholder names.

### Fix C tightening (relation_axis.py CRITICAL block)

Added a second paragraph to the existing CRITICAL block specifically about `direct_axis_mismatch`:

```
DO NOT use direct_axis_mismatch when the claim's axis verb appears
within the claim-relevant clause. The axis_mismatch label is for the
narrow case where the evidence ONLY describes a different axis between
these entities (e.g., the evidence describes phosphorylation but the
claim is activity-axis, AND no activity verb is in the relevant span).
If both axes appear (the claim's axis AND another axis), use
direct_sign_match (or direct_amount_match) on the claim's matching axis.
```

This narrows the trigger for `direct_axis_mismatch` — should counter T-phase's 4 axis_mismatch TP regressions (SRC/KCNMA1, ABL1/BCAR1, TCR/CD3, LEP/KDR).

## Critical event: contamination caught and fixed

**First draft of new few-shot paraphrased holdout's TLR4/HGS record:**

- Holdout: "TLR4 was ubiquitinated and associated with the ubiquitin-binding endosomal sorting protein..."
- First draft few-shot: "ReceptorR was ubiquitinated and associated with the ubiquitin-binding endosomal sorting protein KinaseS..."

Entity names were synthetic but the sentence structure was a direct paraphrase. **`scripts/check_contamination.py` caught this** via the 50-char sliding window paraphrase detector added after the May 2026 contamination incident.

**Fix:** rewrote both new few-shots with substantively different sentence structures. Re-ran contamination guard: CLEAN. Tests: 467/467 passed.

This is the second time the contamination guard has earned its keep (first: original May 2026 incident; second: this U9 draft). The discipline of running the guard BEFORE proceeding is load-bearing.

## Tests

467 passed, 1 skipped (no test changes — prompt-only edits).

## Engineering distinction

### Decision priority is explicit, not implicit

T-phase's "prefer no_relation" was an underspecified hint. The LLM treated it as a blanket bias. U9's numbered priority list gives explicit ordering: assertion FIRST, no_relation LAST. This shifts the LLM's default mode from "bias toward no_relation" to "search for assertion, fall back to no_relation only if absent."

### Fix C's tightening uses the specific failure pattern

The 4 axis_mismatch TP regressions all had the claim's axis verb present in the evidence but the LLM still flagged axis_mismatch. The new CRITICAL paragraph addresses this exact pattern: "DO NOT use direct_axis_mismatch when the claim's axis verb appears within the claim-relevant clause."

## Empirical hypothesis

From T-phase analysis:
- 18 `absent_relationship` FNs from Fix A overshoot (5 TP regressions + 13 A→F).
- 4 `axis_mismatch` TP regressions from Fix C overshoot.

Doctrine §3.7 estimate: +10 to +15.

If U9 recovers ~50% of the 18 absent_relationship FNs (9 records) and ~50% of the 4 axis_mismatch TP regressions (2 records), net gain is ~+11. Within the doctrine estimate.

Risk: the new prompts are longer. Token cost grows ~150 tokens per relation_axis call. At Gemma 4 26B's pricing, negligible.

## Verdict

**PASS.** U9 implementation is correct, contamination caught and fixed, tests green. The decision-priority ordering and Fix C narrowing are both targeted at specific T-phase regression patterns identified in U-phase doctrine §2.

Proceed to U10 (cross-probe consistency check).
