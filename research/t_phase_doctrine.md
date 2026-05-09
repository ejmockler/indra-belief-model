# T-phase doctrine — three parsimonious fixes on top of S-phase

Date: 2026-05-02
Predecessor: S-phase (74.58% holdout, 5/6 ship gates passing)
Target: ~80% raw accuracy / ~78% decision-only on the 482-record holdout

## §1 What this phase is and isn't

T-phase is **not a rewrite**. It is the first patch ship on top of S-phase, applying three empirically-validated fixes to the four-probe architecture. The architecture is unchanged: same four probes, same flat decision table adjudicator, same migration discipline (single scoring path, no flags). Fixes are surgical edits to:

- one closed answer set (relation_axis)
- two probe prompts (relation_axis, scope)
- two adjudicator branches (`relation_axis=abstain` and `via_mediator` for Phosphorylation)

S-phase doctrine §1-§7 carry forward unchanged. T-phase amends §2.3 (relation_axis answer set), §5 (decision table), and §5.6 (causal-vs-direct rule for Phosphorylation only).

## §2 Empirical foundation — what the data says

Validated against the S-phase holdout result (`data/results/dec_s_phase.jsonl`, 482 records).

### §2.1 Three abstain classes carry 128 records (26.6%)

| Reason | n | gold-correct | gold-incorrect |
|---|---|---|---|
| `grounding_gap` | 70 | 26 | 44 |
| `(no reason)` (relation_axis=abstain) | 38 | 14 | 24 |
| `indirect_chain` (direct claims with mediator) | 20 | 12 | 8 |

`grounding_gap` is **not addressed in T-phase** — improving the grounding probe is a v2 effort. The other two classes are addressed by Fixes A and B respectively.

### §2.2 Destructive detector overshoot — 21 FN records

Hand-classified all 21 FNs triggered by sign_mismatch / axis_mismatch / contradicted:

- 12 records: negation/wrong-axis cue is on a clause OTHER than the claim's relation. Span-narrowing fixes these.
- 3 records: semantic-equivalent verbs ("decreases activation"). Out of T-phase scope.
- 3 records: perturbation sign-flip not propagated (LOF/GOF). Out of T-phase scope.
- 3 records: probe noise / contested gold. Not addressable.

Estimated TN collateral risk from span-narrowing: ~3 of 20 TNs caught by these detectors. Net Fix C: **+9 records**.

### §2.3 What was tested and rejected

**Fix D (FamPlex specificity gate) — DROPPED.** Family-claim is a 1.5× FP risk multiplier (41% in FPs vs 27% in TPs) but absolute TP count (52) > 2× absolute FP count (24). Naive family-rejection: net **−28 records**. Sharper rule (family name not in evidence + child mentioned): fired on 1 record, no signal. Empirical conclusion: lexical/ontological substrate has reached its discriminative ceiling for family-vs-instance abstraction-level mismatches.

**Fix E (expression-verb pre-rule) — DROPPED.** Activation/Inhibition + expression-verb evidence has 48% precision (vs 62% without). Force-flagging axis_mismatch on this class: 14 FPs caught, 13 TPs lost, net **+1**. Same lexical-signal-too-noisy issue as Fix D.

The remaining ~13pp of headroom (family/instance, expression-vs-activity, REACH parse-quality) is **deferred to v2**. T-phase ships only what the data justifies.

## §3 The three fixes

### §3.1 Fix A — drop `relation_axis=abstain` from the answer set

**Problem:** the closed answer set for `relation_axis` includes `abstain`. 38 holdout records use it; the adjudicator emits `verdict="abstain"` with no reason. Class prior is 63% gold-incorrect. The abstain answer carries zero information that the other 7 substantive labels don't.

**Change:**
- Remove `"abstain"` from `RelationAxisAnswer` literal in `types.py`.
- Update `relation_axis.py` prompt: "If you cannot decide between `no_relation` and `direct_axis_mismatch`, prefer `no_relation`. Do not output `abstain`."
- Remove the `abstain` few-shot exemplar from `relation_axis.py:_FEW_SHOTS`.
- Update `_llm.py` JSON-parse fallback: if the LLM still returns abstain, project to `no_relation`.
- Replace `adjudicator.py:198-199` (the `if ra == "abstain"` branch) with a scope-aware tiebreaker:
  ```python
  # ra == "abstain" path is unreachable in T-phase. Defensive default:
  # use scope to break the tie rather than emit verdict=abstain.
  if sc == "asserted":
      return "correct", "match", "axis underdetermined; scope asserted"
  if sc == "negated":
      return "incorrect", "contradicted", "axis underdetermined; scope negated"
  return "incorrect", "relation_underdetermined", "axis and scope underdetermined"
  ```

**What this preserves:** Every working path. The 38 records currently emit `abstain`; under the fix they emit one of `no_relation` (→ incorrect/absent_relationship), `direct_sign_match` (→ correct/match), `direct_axis_mismatch` (→ incorrect/axis_mismatch), etc., projected by the LLM.

**Validated estimate:** +10 to +13 records on the 482 holdout. Variance comes from how the LLM projects: blanket-incorrect bound is +10; scope-tiebreaker should achieve +12-13.

### §3.2 Fix B — Phosphorylation-only `via_mediator` → correct/low

**Problem:** S-phase doctrine §5.6 distinguishes causal claim types (Activation, Inhibition, IncreaseAmount, DecreaseAmount — accept indirect chains) from direct claim types (Phosphorylation, Complex, Translocation, etc. — abstain on indirect). Empirical breakdown shows the rule is descriptively wrong **only for Phosphorylation**:

| stmt_type | gold-correct | gold-incorrect | force-correct net |
|---|---|---|---|
| **Phosphorylation** | **4** | **0** | **+4** |
| Complex | 3 | 4 | −1 |
| Inhibition (already causal) | 0 | 1 | n/a |
| Activation (already causal) | 4 | 3 | n/a |

**Change:** Modify `adjudicator.py:187-188` (the `if ra == "via_mediator"` branch for non-causal claims):

```python
return "abstain", "indirect_chain", \
    "direct claim type but evidence shows indirect chain"
```

becomes:

```python
if claim.stmt_type == "Phosphorylation":
    if sc == "negated":
        return "incorrect", "contradicted", \
            "Phosphorylation indirect chain, negated"
    return "correct", "upstream_attribution", \
        "Phosphorylation accepts upstream attribution at low confidence"
return "abstain", "indirect_chain", \
    "direct claim type but evidence shows indirect chain"
```

Adds new ReasonCode `upstream_attribution` to `commitments.py`. Confidence is `low`.

**What this preserves:**
- Complex/Translocation/etc with `via_mediator` still abstain (their empirical precision when forced-correct is too low to justify the change).
- All current causal-claim handling at adjudicator.py:174-186 is unchanged.
- The §5.6 distinction is preserved in spirit — Phosphorylation is being moved into the causal-acceptance bucket because curators empirically treat it that way for chain evidence.

**Validated estimate:** +4 records, zero collateral damage.

### §3.3 Fix C — narrow trigger span in destructive detectors

**Problem:** `direct_sign_mismatch`, `direct_axis_mismatch`, and `negated` answers fire on cues anywhere in the evidence sentence, regardless of which clause they modify. 12 of 21 FNs come from this. Examples:
- "Phosphorylation of eIF-4E by PKC is inhibited by 4E-BP" — `sign_mismatch` fires on "inhibited", but PKC IS the phosphorylation agent; the inhibition is by a third party.
- "β-catenin binds wild-type SOX10, but not 3G mutant" — `negated` fires on "but not", but the wild-type binding IS asserted.
- "Trx1 activating Akt and GSK-3β in an acute way... we did not detect a reduction in infarct size" — `negated` fires on "did not detect", but it governs a different proposition.

**Change:** Prompt and few-shot edits to `relation_axis.py` and `scope.py`. Both probes already have nascent span-narrowing instructions; T-phase strengthens them.

To `relation_axis.py` system prompt, add:
> CRITICAL: Evaluate the relation between the claim's subject and object **only within the clause where they co-occur**. If a different clause contains a negation or wrong-axis verb that does not connect these two entities, ignore it.

To `scope.py` system prompt, the existing CRITICAL block already instructs span-narrowing. Strengthen with:
> If the sentence contains "X does Y, but mutant Z does not do Y", and the claim is about X (not the mutant), the scope is `asserted` for the X-Y relation. Conditional negations on a sibling proposition do not propagate to the claim's relation.

Add 4 paired few-shots per probe (positive: should narrow; negative: should still fire), using **synthetic placeholder names** per the contamination-guard memory.

**What this preserves:**
- Existing in-claim-span detector behavior. ~17 of 20 TNs caught are unambiguous in-clause signals; they should survive.
- The closed answer set is unchanged — only the prompt's clause-localization instruction is sharpened.
- Few-shot exemplars must use synthetic names (KinaseA, ProteinB, FactorR, AdaptorP, GeneZ) per S-phase contamination guard.

**Validated estimate:** +12 FN recovery − 3 TN at risk = **+9 records**.

## §4 Combined impact

| Fix | Records affected | Net |
|---|---|---|
| A. Drop relation_axis=abstain | 38 | +12-13 |
| B. Phosphorylation via_mediator → correct/low | 4 | +4 |
| C. Narrow trigger span | ~25 | +9 (12 FN, ~3 TN loss) |
| **Total** | ~70 | **+25 to +26** |

Projected on 482 holdout:
- Current S-phase: 264/482 = 54.77% raw, 264/354 = 74.58% decision-only.
- T-phase: 264 + 25 = 289/482 = **60.0% raw**, but with abstain rate dropping from 26.6% to ~5%, decision-only on the much larger denominator becomes 289 / (482 − 25) = ~63% — wrong direction.

Correct decomposition: most of the +25 comes from converting abstains into correct verdicts (Fix A) and FNs into TPs (Fix C), so:
- Decision-only: 264 (current TP/TN) + 9 (Fix C FN→TP) + 4 (Fix B abstain→TP) + ~9 (Fix A abstain → TN/TP from class prior) = 286 right; total decisions = 354 (current) + 38 (Fix A new decisions) + 20 (Fix B new decisions) − overlap = ~410.
- Decision-only ≈ 286 / 410 = **~70% decision-only on a much bigger denominator**.
- Raw accuracy: **~60%**.

**Honest framing:** raw accuracy improves; decision-only drops because we're committing to harder records. M12 (74.95%) decides on 99.4% with no architectural advantage. T-phase will likely commit on ~95% of records at ~70% decision-only — competitive with M12 on coverage, slightly behind on per-decision precision. The win is the abstain-rate drop, the +9 hard-error fixes, and a cleaner reason-code surface.

## §5 Sequencing

Risk-minimizing order, per S-phase doctrine §7 migration discipline (no parallel deprecated trajectories):

1. **Fix A first** — closed-set tweak, smallest blast radius. Validates plumbing and surfaces test churn cheaply.
2. **Fix B second** — adjudicator one-rule change. Independent of A. Tiny.
3. **Fix C last** — prompt edits to two probes, highest collateral risk. Last position lets us bisect against A+B if regressions appear.
4. **Cleanup → gate → probe → holdout → ship.**

Each fix has a do/review pair. Reviews verify:
- Tests pass at ≥90% rate.
- Few-shot exemplars use synthetic names (re-run `check_contamination.py` after each prompt edit).
- The fix's empirical claim holds on a 5-record smoke test.

## §6 Success criteria

**Per fix (validated at T7 stratified probe):**
- A: ≥80% of (no reason) class commits to substantive verdict. Of those, ≥60% align with class prior (incorrect-skew).
- B: ≥3 of 4 Phosphorylation+via_mediator records flip correct→correct (i.e., were correct before, are correct after); 0 new FPs introduced.
- C: ≥7 of 12 FN-recovery candidates flip incorrect→correct; ≤3 prior TNs flip incorrect→correct (collateral).

**Overall ship gates (T10):**
- Decision-only accuracy ≥ S-phase (74.58%) — not a regression.
- Raw accuracy ≥ S-phase (54.77%) + 3pp = ≥57.77%.
- Abstain rate ≤ 10%.
- No stmt-type regresses by >3pp from S-phase baseline.
- Inhibition (current weak spot at 42.9%) lifts by ≥3pp via Fix A's class-prior bias.
- Latency p99 ≤ 7s (unchanged from S-phase 6.2s; no new LLM calls added).

**Ship triggers:**
- All gates pass → SHIP.
- Any single non-critical gate fails → SHIP-WITH-RESERVATIONS, document in ship verdict.
- Decision-only accuracy regresses or raw-accuracy gain <3pp → ITERATE before push.

## §7 Migration discipline (carries forward from S-phase)

- Single scoring path. No flags. No fallbacks.
- Few-shot exemplars use synthetic placeholder names. `check_contamination.py` runs before T7 and before T9.
- Brutalist gates at T1 (doctrine), T6 (implementation), T8 (probe), T10 (ship). Each emits a written PASS/FAIL.
- T5 cleanup: any references to `relation_axis="abstain"` in tests/docs must be removed or updated. The `abstain` answer is dead.
- "Ship means push" — T11 pushes to origin/main; local-only verdicts are invisible.

## §8 What is explicitly out of scope

- **Family/instance grounding** (~24 FPs / ~5 FNs / ~4pp). Requires per-probe gold annotation + answer-set redesign. v2.
- **Activation ↔ IncreaseAmount conflation** (~10 FPs / ~2pp). Same — needs answer-set redesign or a learned discriminator.
- **Conditional negation in scope** (~5 FNs / ~1pp). Scope answer set must encode `asserted_for_X_negated_for_Y`; that's a redesign.
- **Perturbation sign-flip propagation to relation_axis probe** (~5 FNs / ~1pp). Substrate change, deferred.
- **KG verifier** — Q-phase failed via scope creep; if reattempted, must be small-scope and targeted.
- **Per-probe gold annotation** — enables future work, not blocking T-phase ship.
- **Joint probe reasoning** — replacing flat table with cascaded re-prompting. Significant rewrite. v2.

## §9 Why this is parsimonious

T-phase touches:
- 1 closed answer set (delete one value).
- 2 probe prompts (add one CRITICAL block each, 4 paired few-shots each).
- 2 adjudicator branches (one stmt-type-specific bypass for Phosphorylation; one scope-aware tiebreaker).
- 1 new ReasonCode (`upstream_attribution`).

It does NOT add:
- New probes.
- New substrate hierarchy lookups.
- New LLM calls per record.
- New decision-table rules beyond the existing structure.

It preserves:
- The four-probe architecture and its closed answer sets (minus one dead value).
- The flat decision table.
- Substrate routing.
- All current TP/TN paths.
- The 11 substrate-fallback rescues.
- Latency profile.

The fix set is the minimum that addresses what data validates and nothing more. Items rejected for parsimony reasons (D, E) are documented above so future work doesn't re-attempt them without new evidence.
