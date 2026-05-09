# U13 — Probe results gate

Date: 2026-05-02
Status: **PASS-WITH-CONCERNS** (proceed to U14 full holdout; iterate post-holdout if confirmed)

## Aggregate

T-phase baseline (on this 58-record sample): 23 right, 16 abstain, 19 wrong.
U-phase result: 22 right, 17 abstain, 19 wrong.

**Net delta: −1 right, +1 abstain, 0 wrong.** Essentially neutral on this sample.

## Per-intervention accounting

| Intervention | n | T-right | U-right | Δ | Notes |
|---|---|---|---|---|---|
| u3_selective_reasoning | 8 | 0 | 1 | **+1** | One A→T recovery; 7 stayed abstain |
| u4_kg_boost | 4 | 2 | 2 | 0 | Calibration only (expected) |
| u5_perturbation | 6 | 4 | 3 | **−1** | 2 TP regressions, 1 F→T gain |
| u6_grounding | 8 | 1 | 0 | **−1** | 4 abstains stayed (alias gaps), 1 TP regression |
| u7_act_vs_amt | 8 | 4 | 5 | **+1** | 1 F→T gain via axis_gate |
| u7_conditional | 4 | 2 | 1 | **−1** | 1 TP regression |
| u8_verb_taxonomy | 4 | 0 | 0 | 0 | No movement on small sample |
| u9_overshoot | 6 | 0 | 1 | **+1** | 1 F→T recovery |
| tp_control | 6 | 6 | 6 | 0 | All preserved ✓ |
| tn_control | 4 | 4 | 3 | **−1** | 1 TP regression (Activation(RAS,RAS) self-loop) |

## U3 selective reasoning behavior

Refined escalation works as designed — only escalates when needed:

| n_probe_calls | n records | escalated? |
|---|---|---|
| 1 | 2 | No (substrate-resolved) |
| 2 | 3 | No (some substrate) |
| 3 | 3 | No |
| 4 | 42 | No (first-pass conclusive) |
| 6 | 4 | Yes (1 probe escalated) |
| 8 | 4 | Yes (2 probes escalated) |

**Escalation rate: 14% (8 of 58).** Compared to the first U12 attempt (100% escalation on all u3 records), refined U3 saves 6× per-record cost.

Of 8 escalated records:
- 2 from u3_selective_reasoning (relation_axis-driven, scope-driven)
- 2 from u7_act_vs_amt (relation_axis re-evaluation)
- 3 from u7_conditional (scope re-evaluation)
- 1 from u8_verb_taxonomy

Net escalation gain on the probe: hard to attribute precisely (escalation outcomes hidden in the 8 records' verdicts).

## Critical finding — TP regression pattern

**5 TP regressions, all of shape `T=incorrect/various → U=correct/match`:**

1. `Activation(FAM83B, RAF1)` u5: T saw axis_mismatch (right call — evidence is binding, not activation); U emitted match.
2. `Inhibition(MIR34A, TP53)` u5: T saw contradicted; evidence says "miR-34a inhibits p53" but actor-target attribution is wrong (miR-34a is OCT4's target).
3. `Phosphorylation(PARD6A, PARD3)` u6: T saw axis_mismatch; U emitted match. Actor is Par6-aPKC complex, not PARD6A alone.
4. `Acetylation(HDAC, AR)` u7: T saw absent_relationship (correct — HDAC inhibitors cause AR acetylation, not HDACs themselves); U emitted match.
5. `Activation(RAS, RAS)` tn_ctrl: T saw absent_relationship (correct — self-loop with no actor specified); U emitted match.

**The pattern: U9's prompt softening ("prefer direct_sign_match when relation is asserted, even briefly") is causing the LLM to accept lexical matches without the rigor T-phase had.** This is the inverse failure mode of T-phase's "prefer no_relation" overshoot — we've swung too far in the other direction.

## Critical finding — U6 grounding doesn't address alias gaps

4 of 8 u6_grounding records stayed abstain because their alias gaps are NOT Greek-letter aliases or anaphora — they're common-name aliases:
- KAP1 ↔ TRIM28
- INI1 / hSNF5 ↔ SMARCB1
- SAP ↔ SH2D1A
- p27 Kip1 ↔ (different gene than CIB1)

U6's Rule 9 (Greek-letter aliases) and Rule 8 (anaphora) don't help these. Real biomedical aliases need Gilda alias map enrichment, not prompt rules. **U6 as designed addresses ~half of the grounding-class records.**

## Decision: proceed to U14 with caveats

Per doctrine §6 ship triggers: "If U13 probe gate fails badly (e.g., raw acc < T-phase by >5pp), ITERATE specific failing interventions before U14."

The probe shows -1 record net delta on 58 samples — within the noise floor for a sample this small (95% CI: ±5 records on n=58). Per-intervention deltas of ±1 are all within sample noise.

**Proceed to U14 full holdout** because:
1. Sample size at 58 records is too small to discriminate between "no real effect" and "small effect masked by noise."
2. Multiple interventions (u7 act_vs_amt, u9 overshoot recovery) showed positive direction even on small samples.
3. The TP regression pattern needs validation at scale — if 5 TP regressions on n=6 control scales linearly to 30+ regressions on full holdout, U-phase IS net-negative and ITERATE is mandatory.
4. The U6 alias-gap finding is real but addressable in a future iteration if needed.

## Specific concerns to monitor at U14

- **TP regression rate**: T-phase's was 5.7%. If U-phase's exceeds 10%, ITERATE U9 prompt changes.
- **U6 grounding accuracy**: T-phase was 15.2% on grounding-tag class. If U-phase doesn't lift to ≥30%, the prompt rules aren't internalized — would need alias-map enrichment (deferred to V-phase).
- **U7 closed-set adoption**: if act_vs_amt records show <50% accuracy on the 24-record class, the LLM isn't using direct_amount_match — fall back to T-phase prompt for amount disambiguation.
- **u3 escalation cost**: at 14% rate × ~3x per-record latency, full holdout adds ~8 min runtime. Acceptable.

## What U13 confirms

- The U-phase pipeline runs end-to-end without errors.
- U3 selective reasoning's refined escalation is working (14% rate, only on relation_axis/scope-driven uncertainty).
- U7 closed-set values produce correct adjudicator behavior on test cases.
- TP control on substrate-clean cases is preserved (6/6).

## What U13 cannot confirm

- Whether the per-intervention gains scale with sample size (need 482 records).
- Whether the TP regression pattern is real or sample noise.
- Whether U6's alias gaps cap the ceiling at ~50% recovery.

Proceed to U14.
