# U-phase doctrine — fundamentals iteration on top of T-phase

Date: 2026-05-02
Predecessor: T-phase (60.17% raw, 73.79% decision-only on 482 holdout)
Target: 65-70% raw, 78-82% decision-only, abstain ≤10%

## §1 What U-phase is — and isn't

U-phase is **not a model upgrade**. We stay on `gemma-remote` (Gemma 4 26B) per user direction. U-phase is **not a parsimonious patch round** like T-phase. It is an architectural fundamentals iteration: nine surgical interventions targeting categories of residual error that T-phase explicitly deferred to v2 in its §8.

The four-probe architecture is preserved. No new probes are added (one is enhanced). The flat decision-table adjudicator is preserved. Closed-answer-set discipline is preserved (one set is split, one set is extended).

T-phase's §7 migration discipline carries forward: single scoring path, no flags, no fallbacks.

## §2 Empirical foundation

T-phase shipped state on the 482-record holdout:
- 290 right (60.17% raw, 73.79% decision-only)
- 103 hard wrong (53 FP + 50 FN)
- 89 abstain (35 lost yields + 54 saved face)

Of the 192 unresolved records, hand-classified by architectural root cause:

| Class | n | Why it's unresolved |
|---|---|---|
| Family/instance grounding (FP `match`) | ~24 | No semantic-hierarchy mechanism; lexical match insufficient |
| `grounding_gap` lost yields | 27 | Grounding probe doesn't bridge anaphora / paraphrase / family-instance |
| Fix A `absent_relationship` overshoot | 18 | T-phase prompt projection too aggressive; no "uncertain but asserted" answer |
| `axis_mismatch` FN (semantic-equivalent verbs) | 10 | "decreases activation of" not mapped to Inhibition; closed set treats axis as discrete |
| Activation ↔ IncreaseAmount FP conflation | ~10 | Closed set conflates activity vs amount; verbs are causal-adjacent |
| Sign-mismatch via perturbation FN | ~5 | Effective-sign computed in substrate but not propagated to LLM probe |
| `indirect_chain` lost yields (non-Phos) | 8 | Empirical precision too low for blanket fix; needs per-record signal |
| Conditional-negation FN | 5 | Scope closed-set has no `asserted_with_condition` |
| `contradicted` FN sibling-clause overshoot | ~3 | Scope CRITICAL block insufficient against information-dense evidence |
| Probe noise / contested gold | ~5 | Not addressable |

These ten categories cover ~115 of the 192 unresolved records. The remainder (~77) are abstains where T-phase's behavior is correct (saved-face on gold-incorrect).

## §3 The nine interventions

### §3.1 U3 — Selective reasoning escalation (Track 3)

**Problem.** Every probe call uses `reasoning_effort="none"` per the model_client doctrine ("medium" with default budget burns 12000 reasoning tokens before emitting JSON, causing truncation). For most records the cheap-reasoning is fine. For records where the LLM gives an uncertain-zone answer, the cheap reasoning is the bottleneck.

**Change.**
1. `_llm.py` accepts `reasoning_effort` parameter (defaults to "none" for backward-compat).
2. Orchestrator runs first pass at "none". After adjudicate, if any of the following triggers fire, escalate:
   - `verdict == "abstain"`
   - `confidence == "low"` AND reason ∈ {`hedging_hypothesis`, `chain_extraction_gap`, `relation_underdetermined`, `upstream_attribution`}
   - reason ∈ {`indirect_chain`, `grounding_gap`, `chain_extraction_gap`}
3. Escalation runs the implicated probes again at `reasoning_effort="medium"` and `max_tokens=12000` (per model_client guidance).
4. Re-adjudicate with new probe answers. Use the more-confident verdict.

**What this preserves.** First-pass behavior on clean cases. Substrate routing. Adjudicator decision table. Existing prompts and few-shots.

**Cost.** ~30-40% of records escalate (the ~120 in the abstain or low-confidence buckets). Each escalation is ~3x latency. Total wall-clock impact: ~1.6x current holdout runtime (35 min → 60 min).

**Estimated leverage: +15 to +25 records.**

### §3.2 U4 — INDRA KG verifier as confidence modifier

**Problem.** INDRA has 894K curated statements. We never query them when scoring. Q-phase tried KG-as-verdict-source and lost 2.65pp because:
- KG noise (low-confidence/erroneous curated statements get used as ground truth)
- Coverage gaps (correct claims for which the KG happens to lack a curated triple)
- KG override of valid evidence-based decisions

**Change.** Add a substrate-level KG check (no LLM call). For each (claim_subject, claim_object, claim_axis) triple:
- If a curated statement of the **same axis** exists in INDRA's KG → boost confidence by one tier (low→medium, medium→high). Reason carries `kg_confirmed` modifier.
- If a curated statement on a **different axis** exists → emit `kg_axis_hint` in adjudicator's confidence rationale; never override.
- If no curated triple exists → no signal (Q-phase failure mode avoided).

**What this preserves.** Verdict logic is untouched. Only confidence is modified. Substrate-only — zero new LLM calls.

**Estimated leverage: +12 to +18 records** (mostly via downstream calibration; some directly via boost-to-medium that recovers low-confidence-incorrect→correct.)

### §3.3 U5 — Effective-sign propagation to relation_axis probe

**Problem.** Substrate detects perturbation markers (LOF/GOF). Adjudicator already computes `_effective_claim_sign(claim.sign, perturbation)` in §5.1. **But the relation_axis probe receives `claim.sign`, not the effective sign.** When the LLM probe sees evidence "VHL silencing increased vimentin" with claim `DecreaseAmount(VHL, VIM)` (sign=negative), it can't reconcile because the LOF inversion isn't propagated.

**Change.** In `router._route_relation_axis`, when building the probe's `claim_component`, use `_effective_claim_sign(claim.sign, ctx.perturbation_marker)` instead of the literal claim sign. The probe then receives:
- `claim_component="(VHL, VIM) — claim axis=amount, sign=positive"` (effective sign after LOF inversion)

Adjudicator continues to use the effective sign in its decision table; this just makes the LLM's view consistent.

**What this preserves.** Existing `_effective_claim_sign` rule. Existing adjudicator §5.1. Substrate detection unchanged.

**Estimated leverage: +4 to +6 records** (SGN stratum from S-phase v1 limitations).

### §3.4 U6 — Grounding probe enhancement (Intervention D)

**Problem.** 27 records have `grounding_gap` lost yields. Inspection shows:
- Anaphora: "this short arrestin-3-derived peptide also binds ASK1" — substrate doesn't bridge "this peptide" to ARR3.
- Greek-letter aliases: β-catenin → CTNNB1, p38 → MAPK14, ERβ → ESR2 — substrate's Gilda alias map covers some but not all.
- Reverse family-instance: claim is `MEK1`, evidence has only "MEK family" — current logic doesn't bridge.
- Bidirectional binding: "Mutually Exclusive Binding of UAP56 and NXF1 to UIF" → claim `Complex(DDX39B, NXF1)` where DDX39B is the canonical name for UAP56.

**Change.** Strengthen `verify_grounding` probe with explicit instructions and 4-6 paired few-shots covering each of the four sub-patterns. Synthetic placeholder names per contamination guard.

**What this preserves.** Grounding probe's existing structure. Closed answer set ({mentioned, equivalent, not_present, uncertain}). The substrate-first routing.

**Estimated leverage: +10 to +15 records** (recover ~half of the 27 grounding_gap lost yields).

### §3.5 U7 — Closed-set redesign (Intervention E) — BLOCKED by U2

**Problem.** Two closed-set conflations bleed accuracy:
1. **`direct_sign_match` conflates activity vs amount.** "X overexpression increased Y protein" gets scored as `Activation(X, Y)` correct because both axes share `direct_sign_match`. Gold demands `IncreaseAmount`. This is the largest unfixed FP class (~10 records).
2. **`scope` answer set** ({asserted, hedged, negated, abstain}) **has no `asserted_with_condition`.** "β-catenin binds wild-type SOX10, but not 3G mutant" → either `asserted` (wrong direction; loses the conditionality) or `negated` (also wrong; false-flags contradiction). Currently the scope probe lands on `negated` for ~5 records, producing `contradicted` FNs.

**Change.**
1. Split `RelationAxisAnswer.direct_sign_match` into:
   - `direct_activity_match` — functional state change (Activation, Inhibition)
   - `direct_amount_match` — expression/abundance change (IncreaseAmount, DecreaseAmount)
   
   Adjudicator §5.2 only accepts axis-aligned matches. `direct_amount_match` against an Activation claim becomes `axis_mismatch`, not match.

2. Extend `ScopeAnswer` with:
   - `asserted_with_condition` — relation is asserted on the qualified entity, negated on the alternative.
   
   Adjudicator treats `asserted_with_condition` like `asserted` for verdict purposes (the relation IS asserted), with confidence downgraded to `medium` to signal the conditionality.

**Prerequisite.** Validating these answer-set changes requires per-probe gold for at least the affected records. U2 derives this from INDRA KG.

**What this preserves.** Existing closed-set discipline. Existing CRITICAL blocks. Existing decision table structure (just wider matching against axis).

**Estimated leverage: +10 to +15 records.**

### §3.6 U8 — Verb taxonomy in CATALOG

**Problem.** 10 `axis_mismatch` FNs come from semantic-equivalent verbs that the relation_axis probe rejects:
- "miR-140 decreases the activation of p38" → semantically Inhibition
- "TLR4 activation enhances TGF-β signaling" → semantically Activation
- "ATF4 enhances expression of MTA1" → IncreaseAmount

The substrate's CATALOG has activation and inhibition patterns but doesn't cover the compound-verb forms.

**Change.** Add CATALOG patterns mapping:
- `(decreases?|reduces?|inhibits?|attenuates?) (the )?activation of` → Inhibition substrate match
- `(enhances?|promotes?|augments?|increases?) (the )?signaling` → Activation substrate match
- `(induces?|drives?|promotes?) (the )?expression of` → IncreaseAmount substrate match
- `(abolishes?|prevents?) (the )?binding` → Complex-negated substrate match

These feed the §5.4 substrate-fallback rescue when the LLM probe emits `axis_mismatch` despite the semantic equivalence.

**What this preserves.** Existing CATALOG patterns. The §5.4 rescue mechanism.

**Estimated leverage: +5 to +10 records.** Risk: over-firing on edge cases (e.g., "enhances signaling" may be a true axis mismatch in some contexts).

### §3.7 U9 — Soften Fix A prompt + tighten Fix C overshoot

**Problem.** T-phase's prompt-level changes introduced 14 hard regressions:
- Fix A's "prefer no_relation" instruction (`relation_axis.py`) over-fires: 5 TP regressions + 13 A→F overshoots = 18 records flipping to `absent_relationship` when the relation IS asserted.
- Fix C's CRITICAL clause-localization (`relation_axis.py`) is too strict: 4 TP regressions where the LLM flagged `axis_mismatch` despite the claim verb being in the evidence's claim-clause span.

**Change.**
1. Replace Fix A's projection bias:
   ```
   When uncertain between no_relation and direct_axis_mismatch, prefer no_relation.
   ```
   with:
   ```
   If the relation between subject and object is asserted in the evidence
   (even briefly, hedged, or as part of a longer description), use the
   appropriate direct_*_match label. Only choose no_relation when the
   entities co-occur without any asserted relation between them.
   ```
   Plus 2 new few-shots showing assertion-detection in dense evidence.

2. Tighten Fix C's `axis_mismatch` trigger: clarify that `direct_axis_mismatch` fires only when the evidence's claim-clause describes a DIFFERENT axis than the claim. If the claim's axis verb appears in the relevant clause, the answer is `direct_*_match`, not `direct_axis_mismatch`.

**What this preserves.** Closed answer sets. The CRITICAL block structure. Fix A's "no abstain" discipline.

**Estimated leverage: +10 to +15 records.**

### §3.8 U10 — Cross-probe consistency check

**Problem.** Probes are independently answered. Inconsistent combinations slip through:
- `subject_role=present_as_subject` + `relation_axis=no_relation` → if the subject is acting, what is it doing?
- `subject_role=absent` + `relation_axis=direct_sign_match` → contradiction.
- `scope=negated` + `relation_axis=direct_sign_match` + `subject_role=present_as_subject` → tension between scope and relation.

**Change.** Adjudicator pre-rule: detect these inconsistency patterns. **Never override the verdict** — only downgrade confidence to "low" and emit a `probe_inconsistency` reason modifier. Composed_scorer can use this as a downstream signal for review or re-prompting.

**What this preserves.** Decision table verdicts. Closed answer sets. Probe independence (the probes still answer separately; just the adjudicator sees them together).

**Estimated leverage.** Doesn't add gains directly. Reduces overconfident-wrong predictions; makes the score calibration cleaner. Foundation for future joint-reasoning work.

### §3.9 U2 — Per-probe gold derivation (DIAGNOSTIC, NOT A FIX)

**Problem.** We have no per-probe ground truth for the 482 holdout records. Every fix is a shot in the dark; we can only measure end-to-end accuracy. U7's closed-set redesign in particular is unsafe without it.

**Change.** Pure analysis script. For each holdout record, query INDRA's `CorpusIndex` (894K statements):
- For (claim_subject, claim_object), find all curated statements between them.
- Classify each: same axis as claim? Same sign? Hedged or asserted?
- Derive expected probe answers:
  - `subject_role`: if INDRA has any curated statement with claim_subject as agent and claim_object as target → expected `present_as_subject`.
  - `relation_axis`: derived from curator's stmt_type vs claim's stmt_type.
  - `scope`: defaulted to `asserted` for high-belief curated statements.
- Save to `data/benchmark/probe_gold_holdout.jsonl`.

Coverage will be partial — not every holdout pair is in INDRA. But the records that ARE covered give probe-level supervision we can use to:
1. Validate U7's split (does the LLM correctly distinguish activity vs amount when the gold says one or the other?)
2. Measure per-probe accuracy historically (which probe is the bottleneck?)
3. Identify probe-noise records vs systematic-failure records.

**What this preserves.** Existing test infrastructure. Read-only — no test changes, no source changes.

**Cost.** ~10 minutes of pure analysis. No LLM calls.

## §4 Sequencing

Risk-minimizing order, with U2 as a parallel diagnostic track:

```
U2 (diagnostic, blocks U7) — runs in parallel
└─► U3 (selective reasoning) — quick win, validates lever
    └─► U4 (KG verifier) — substrate-only, highest single leverage
        └─► U5 (perturbation prop) — small isolated change
            └─► U6 (grounding enhancement, Intervention D) — biggest residual class
                └─► U9 (Fix A softening) — cleanup of T-phase collateral
                    └─► U7 (closed-set redesign, Intervention E) — REQUIRES U2
                        └─► U8 (verb taxonomy) — substrate addition
                            └─► U10 (consistency check) — adjudicator addition
                                └─► cleanup → gate → probe → holdout → ship
```

Parallel optimization: U2 can start before U0/U1 doctrine review since it's pure analysis. If U2 reveals coverage too low (<30% of records have INDRA curated triples), U7 may be deferred to a future phase.

## §5 Combined impact

Independent stacking with conservative collateral:

| Tier | Intervention | Lower est. | Upper est. |
|---|---|---|---|
| 1 | U3 selective reasoning | +15 | +25 |
| 1 | U4 KG verifier | +12 | +18 |
| 2 | U5 perturbation prop | +4 | +6 |
| 2 | U6 grounding (D) | +10 | +15 |
| 2 | U7 closed-set (E) | +10 | +15 |
| 2 | U8 verb taxonomy | +5 | +10 |
| 2 | U9 prompt softening | +10 | +15 |
| 3 | U10 consistency check | indirect | indirect |
| **Sum (independent)** | | **+66** | **+104** |

Realistic with overlap (some U6 records are also U2/U4 records; some U3 escalations would solve U7 cases): **+30 to +50 records**.

Projected on 482 holdout:
- T-phase: 290 right (60.17% raw, 73.79% decision-only)
- U-phase: 320-340 right (66.4-70.5% raw, **~78-82% decision-only**)
- Abstain rate: 18.46% → ~6-10%

This **closes the gap to M12 (74.95% decision-only)** while preserving:
- Four-probe modularity (still attributable per-probe).
- 2.4× latency advantage (selective reasoning slows the hard ~30% of records, not all).
- Substrate-first design (U4, U5, U8 are all substrate-level).

## §6 Success criteria

**Per intervention (validated at U13 stratified probe):**
- U3: ≥40% of escalated records change verdict; ≥60% of changes align with gold.
- U4: ≥80% of records with KG-curated triples get confidence-boost; never overrides verdict.
- U5: ≥3 of 5 SGN-stratum records flip to correct.
- U6: ≥6 of 10 grounding-anaphora candidates recover.
- U7: ≥6 of 10 Activation/IncreaseAmount FPs filter to axis_mismatch; ≥3 of 5 conditional-negation FNs recover.
- U8: ≥5 of 10 axis_mismatch FNs flip to correct via substrate-rescue.
- U9: ≥10 of 18 Fix A overshoots recover (TP regressions + A→F overshoots).
- U10: detects inconsistency on ≥4 of 5 known cross-probe-inconsistent records.

**Overall ship gates (U15):**
- Raw accuracy ≥ 65% (vs T-phase 60.17%).
- Decision-only accuracy ≥ 75% (vs T-phase 73.79%, M12 74.95%).
- Abstain rate ≤ 10% (vs T-phase 18.46%).
- No stmt-type regresses by >3pp from T-phase.
- TP regression rate ≤ 5% (vs T-phase 5.7%).
- Latency p99 ≤ 12s (vs T-phase ~6s; selective reasoning costs are bounded).

## §7 Migration discipline (carries from S/T phases)

- Single scoring path. No flags. No fallbacks.
- All few-shots use synthetic placeholder names; `check_contamination.py` runs before U13 and U14.
- Brutalist gates at U1, U11, U13, U15.
- U2's per-probe gold is for validation, not training; cannot leak holdout structure into prompts.
- U4 NEVER uses KG to override verdict — only confidence. Q-phase failure mode is the line we don't cross.
- U7's closed-set additions must update every consumer (probe modules, adjudicator, tests). No half-done state.

## §8 What is explicitly out of scope (deferred to V-phase or beyond)

- Model upgrade — Anthropic Claude integration deferred per user direction.
- Joint probe reasoning (replacing flat decision table with cascade) — too large for U-phase.
- Cross-record context (anaphora across sentences, paper-level grouping) — separate workstream.
- Score calibration verification — useful but not blocking.
- Self-consistency / probe ensemble (multi-shot per probe) — defer until selective reasoning is measured.
- CATALOG audit (which existing patterns are wrongly firing on holdout) — useful for U8 but not blocking.

## §9 Why U-phase is justified despite scope size

T-phase showed that **architectural fixes (rules in adjudicator) are high-precision; prompt fixes are high-variance**. U-phase concentrates leverage in:
- Substrate additions (U4, U5, U8) — deterministic, no prompt variance.
- Adjudicator additions (U10) — deterministic.
- Closed-set redesigns (U7) — wider expressive capacity, validated by per-probe gold.
- Targeted reasoning (U3) — escalation only on uncertain cases.

Only U6 and U9 are pure prompt edits. T-phase's experience suggests these will yield modest gains. The architectural interventions (U3, U4, U5, U7, U8, U10) are where the real headroom lives.

The user has explicitly authorized "everything in scope". The constraint is staying on Gemma 4 26B. U-phase exhausts the architectural levers available within that constraint.
