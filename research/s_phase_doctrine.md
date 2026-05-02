# S-phase doctrine — surface-shrinkage architecture

Status: DRAFT for S1 brutalist gate
Date: 2026-05-02
Predecessor: R-phase shipped at 72.86% on holdout_v15_sample (M12: 74.95%, Q-phase: 70.21%)

## 1. Unified principle

> Each LLM call should have an information surface and an action surface small enough that its commitments can be reasoned about explicitly. Schema commits are bets; the schema is a place to lose, not a place to win.

Every R-phase failure traces to `parse_evidence` doing too much in one call: claim + text + substrate priors + commitment markers + alias context (large information surface) → typed multi-slot list of EvidenceAssertions (large action surface). Confirmatory cues (substrate priors as candidate list) sum constructively over attention; disconfirmatory cues (negation, role-swap, perturbation, indirect chain) are scattered. Adding more language to the prompt cannot fix this — the action surface IS the bottleneck.

## 2. The four probes

`parse_evidence` is replaced by four probes. Each has a single question and a closed answer set.

### 2.1 subject_role probe

**Question:** "What role does {claim.subject} play in this evidence?"

**Closed answer set:**
| Answer | Meaning |
|---|---|
| `present_as_subject` | named as the actor of the relevant event |
| `present_as_object` | named as the target/recipient (role-swap candidate) |
| `present_as_mediator` | named, but as an intermediate node in a chain |
| `present_as_decoy` | named in a different relation than claim's |
| `absent` | not named (post-alias-resolution) |

**Side field — perturbation:** `Literal["none", "LOF", "GOF"]`. Substrate-only (regex M9 entity-first patterns); never set by LLM. Adjudicator §5.1 uses this to invert claim sign when LOF detected.

**Substrate fast-path:** Gilda alias resolution + entity-first regex (LOF/GOF perturbation patterns from M9). When substrate identifies the subject's syntactic role unambiguously, returns ProbeResponse with `source="substrate"`.

**LLM escalation:** when substrate is ambiguous (multiple role-candidates or cross-sentence anaphora), single closed-set classifier with 3-5 few-shots. Perturbation field stays "none" if substrate didn't set it (LLM is not asked to detect perturbation).

### 2.2 object_role probe

Mirror of 2.1 with `claim.object`. Same answer set and substrate fast-path.

### 2.3 relation_axis probe

**Question:** "Is there a relation between {subject_role-resolved} and {object_role-resolved} on the {claim.axis} axis with the claim's sign?"

**Closed answer set (8 values):**
| Answer | Meaning |
|---|---|
| `direct_sign_match` | relation present on claim axis with claim sign (or claim-neutral) |
| `direct_sign_mismatch` | relation present on claim axis with opposite sign |
| `direct_axis_mismatch` | relation present but on a different axis |
| `direct_partner_mismatch` | binding axis present but partner type wrong (e.g., DNA binding when claim is Complex/protein) |
| `via_mediator` | relation via intermediate (indirect chain) |
| `via_mediator_partial` | chain detected but terminal entity unresolved (chain_extraction_gap) |
| `no_relation` | no relation between resolved entities |
| `abstain` | text underdetermines |

The 8-value set is still closed and small enough for few-shot coverage (one exemplar per value). It absorbs four R-phase reason codes that lacked mappings: sign_mismatch, axis_mismatch, binding_domain_mismatch, chain_extraction_gap.

**Substrate fast-path:** CATALOG (relation_patterns.py) match on resolved entity pair. Sign and partner-type come from CATALOG metadata. If CATALOG match disagrees with claim, returns `direct_sign_mismatch` / `direct_axis_mismatch` / `direct_partner_mismatch` accordingly.

**LLM escalation:** when no CATALOG match but entities co-occur, closed-set classifier given hint = "substrate detected entities but no canonical relation pattern; classify into the 8-value set".

### 2.4 scope probe

**Question:** "Is the relation between {subject} and {object} asserted, hedged, or negated in this evidence?"

**Closed answer set:**
| Answer | Meaning |
|---|---|
| `asserted` | unconditionally claimed |
| `hedged` | claimed under hypothesis/may/might/proposed/likely |
| `negated` | explicitly denied |
| `abstain` | text does not commit |

**Substrate fast-path:** explicit_hedge_markers (M10) + negation lexicon scan within span. Returns ProbeResponse when markers are unambiguously in scope.

**LLM escalation:** when markers are present but scope is unclear (e.g., hedged within one clause but asserted in coordinated clause), closed-set classifier on the relation span.

## 3. Closed answer sets — design rationale

Closed-set classification is where transformers are strongest. Forcing a small, mutually-exclusive enum:

1. Eliminates schema drift (no free-form `axis: str`)
2. Allows few-shots to cover the answer space (5 enum values × 1 example each = compact prompt)
3. Forces abstain to be a first-class citizen, not a degenerate empty list
4. Makes the LLM's commitment incremental — one decision per call, not five

Every probe's answer set is at most 5 values. If a real-world distinction needs a 6th value, the answer set splits into two probes rather than growing.

## 4. Substrate as router, not prefiller

R-phase's contract: substrate detects candidate relations → injected into parse_evidence prompt as "SUBSTRATE PRIORS — candidates the regex layer suspects, verify against text, confirm or reject".

S-phase's contract: substrate is a function `(claim, ctx) → list[ProbeResponse | NarrowedProbeRequest]`.

- Where substrate can answer deterministically (entity-first LOF, exact CATALOG axis-match, explicit hedge in scope), it returns a `ProbeResponse` and that probe **never reaches the LLM**.
- Where substrate has a hint (entities present but ambiguous role), it returns `NarrowedProbeRequest(probe_kind, claim_component, evidence, substrate_hint)` — the hint narrows the LLM's question but does NOT prefill the answer.
- Where substrate is silent, the probe goes to LLM unconditionally with no hint.

**No SUBSTRATE PRIORS prompt block exists in any LLM prompt.** The substrate's output is consumed by the orchestrator (decomposed.py replacement), not by the LLM.

This eliminates the structural cause of Bucket B (over-anchoring): the LLM never sees a list of candidates it's biased to confirm.

## 5. Adjudicator as pure decision table

`adjudicate.py` is rewritten as a flat table over `(subject_role, object_role, relation_axis, scope)` tuples. Each verdict has exactly one rule.

### 5.1 Pre-rule: perturbation-marker sign propagation

Before the canonical table runs, apply perturbation propagation. The subject_role probe returns a side field `perturbation: Literal["none", "LOF", "GOF"]` (substrate-detected via M9 entity-first patterns).

```
if subject_role.perturbation == "LOF":
    effective_claim_sign = invert(claim.sign)
elif subject_role.perturbation == "GOF":
    effective_claim_sign = claim.sign  # GOF preserves
else:
    effective_claim_sign = claim.sign
```

The relation_axis probe is then asked against `effective_claim_sign` rather than `claim.sign`. This recovers the normal agent→target relationship from perturbation experiments (e.g., "ADAM17 RNAi inhibited MMP9" → ADAM17 increases MMP9, which the claim DecAmt(ADAM17, MMP9) contradicts).

Perturbation as a side field on subject_role is a single small carve-out from "one closed enum per probe" — justified because perturbation detection is purely substrate-fast-path (regex M9), never escalated to LLM, so it does not enlarge the LLM's action surface.

### 5.2 Canonical table

| subject_role | object_role | relation_axis | scope | → verdict | reason |
|---|---|---|---|---|---|
| present_as_subject | present_as_object | direct_sign_match | asserted | correct | match |
| present_as_subject | present_as_object | direct_sign_match | hedged | abstain | hedging_hypothesis |
| present_as_subject | present_as_object | direct_sign_match | negated | incorrect | contradicted |
| present_as_subject | present_as_object | direct_sign_mismatch | * | incorrect | sign_mismatch |
| present_as_subject | present_as_object | direct_axis_mismatch | * | incorrect | axis_mismatch |
| present_as_subject | present_as_object | direct_partner_mismatch | * | incorrect | binding_domain_mismatch |
| present_as_subject | present_as_object | via_mediator | asserted | abstain | indirect_chain |
| present_as_subject | present_as_object | via_mediator_partial | * | abstain | chain_extraction_gap |
| present_as_subject | present_as_object | no_relation | * | incorrect | absent_relationship |
| present_as_object | present_as_subject | direct_sign_match | * | incorrect | role_swap |
| present_as_mediator | * | * | * | abstain | indirect_chain |
| absent | * | * | * | abstain | grounding_gap |
| * | absent | * | * | abstain | grounding_gap |
| * | * | abstain | * | abstain | underdetermined |

### 5.3 Symmetric-binding handling

For `claim.axis == "binding"`:
- If `claim.subject == claim.object` (self-binding, e.g., Autophosphorylation, X-X complex): subject_role and object_role probes both query the same entity; both returning `present_as_subject` is sufficient for direct match.
- If subject_role and object_role return swapped values (subject as object, object as subject): treated as direct match, NOT as role_swap. This is because Complex(X, Y) ≡ Complex(Y, X) — INDRA Complex is order-invariant.
- For non-binding axes (Activation, Phosphorylation, etc.), order matters and role_swap fires.

### 5.4 Final-arm substrate fallback (M3 hoist preserved)

After the canonical table emits its verdict:

```
if verdict == "abstain" and ctx.detected_relations has CATALOG match against (claim.subject, claim.object, claim.axis):
    return correct, "regex_substrate_match"
else:
    return verdict
```

This preserves the R8 critical fix that hoisted M3 substrate-fallback above the zero-assertion early-exit, now expressed cleanly as a final-arm rescue rather than a pre-emptive shortcut.

### 5.5 Reason codes deferred in v1

These R-phase reason codes are intentionally NOT in the v1 decision table:
- `internal_contradiction` (5 records) — collapses to abstain via scope=hedged or relation=abstain
- `cascade_terminal_match`, `chain_match` (4+2 records) — collapses to direct_sign_match → match
- `bilateral_ambiguity_downgrade` (8 records) — confidence-only downgrade, no verdict change
- `site_mismatch` (7 records) — defer to v2 (phosphosite sub-probe)

Total deferred: ~26 records (5.4%). Acceptable v1 cost.

If the table grows past ~50 rules in v2+, that's a signal the probe answer sets need re-cutting, not that the table needs more rules.

## 6. Substrate-first per probe — latency contract

Per-probe contract: substrate-first, LLM-as-escalation. Latency budget:

- 50%+ of records resolve all four probes via substrate alone (zero LLM calls)
- Of remaining records, average 2 of 4 probes go to LLM
- Median wall ≤ 15s (vs R-phase 12.5s; modest tax for cleaner architecture)
- p99 wall ≤ 75s (vs R-phase 51.8s; bounded by parallel probe dispatch)

If S8 measurement shows >2s/probe LLM latency, parallel dispatch (asyncio or ThreadPool) compresses per-probe cost.

## 7. Migration discipline — no parallel deprecated trajectories

This is a doctrinal commitment, not a stylistic one. The R-phase pipeline did NOT cleanly retire M-phase artifacts (parse_evidence still carried the SUBSTRATE PRIORS block, EvidenceAssertion still had nullable slots from earlier iterations). Each parallel-trajectory file became a place where regressions hid.

S-phase migration:

| Module | S-phase disposition |
|---|---|
| `scorers/parse_evidence.py` | **DELETED** — replaced by `scorers/probes/` package |
| `scorers/_prompts.py` | **DELETED** — replaced by per-probe prompts in `probes/` |
| `scorers/decomposed.py` | **REWRITTEN** — orchestrates four probes, not parse_evidence |
| `scorers/adjudicate.py` | **REWRITTEN** — decision table over probe tuples |
| `scorers/commitments.py` | **TRIMMED** — EvidenceCommitment, EvidenceAssertion removed; ClaimCommitment, GroundingVerdict, Adjudication retained |
| `scorers/relation_patterns.py` | **REFACTORED** — emits ProbeResponse/NarrowedProbeRequest, not raw assertions |
| `scorers/context_builder.py` | **TRIMMED** — keeps perturbation/hedge/CATALOG detection; removes SUBSTRATE PRIORS rendering |
| `scorers/parse_claim.py` | **PRESERVED** — claim parsing is unchanged |
| `scorers/grounding.py` | **PRESERVED** — entity grounding is orthogonal to evidence parsing |
| `scorers/scorer.py` | **MIGRATED** — top-level entry calls new pipeline; no `use_decomposed` flag (only one path) |
| `scorers/composed_scorer.py` | **MIGRATED** — delegates to new pipeline |

Test files:
- `tests/test_substrate_priors.py` → replaced by `tests/test_probe_routing.py`
- `tests/test_relation_patterns.py` → kept, retargeted to probe outputs
- `tests/test_escalation.py` → DELETED (escalation flag was R6 telemetry, dormant in R7; not part of S-phase)
- All other adjudicate/grounding/parse_claim tests → kept but updated to match new probe-tuple inputs to adjudicator

Scripts:
- `scripts/r*_*.py` → archived under `scripts/archive/r_phase/`
- `scripts/run_n9_holdout.py` → adapted to S-pipeline as `scripts/run_s_holdout.py`

The end state has exactly one scoring path. No flags, no fallbacks to the old extractor, no commented-out code referencing EvidenceAssertion. CI runs against the single path.

## 8. Regression-preservation invariants

The S-phase MUST preserve:

| Invariant | Source | S-phase mechanism |
|---|---|---|
| 14 parser-match gains (R vs M12) | R-phase parse_evidence extracted clean assertions where M12 failed | relation_axis_probe extracts the same `direct` answer when CATALOG matches; for non-CATALOG patterns (e.g., AURKA→TP53 nominalization), few-shot exemplars in relation_axis_probe must include them |
| 19 preserved Q-gains | Q-phase parser changes still hold under R | substrate fast-path resolves these without LLM (Q-phase wins were mostly substrate-detectable) |
| 3 substrate fallback gains (R8 hoist) | M3 zero-assertion early-exit was hoisting substrate match over abstain | adjudicator's final-arm rule (§5) preserves this |
| 4 NEW R-only gains | substrate fast-path on records both M12 and Q missed | substrate-as-router preserves these via NarrowedProbeRequest pathways |

S8 explicitly audits these classes before the holdout.

## 9. Failure-bucket → probe mapping

R-phase regression buckets traced to probe-architecture fixes:

### Bucket A — parser FN on real claims (12 of 28 persistent-Q-regs)
| Pattern | Example | S-phase probe-fix |
|---|---|---|
| Apposition resolution | "cyclin A-CDK2 (Cdk2)" | subject_role probe with Gilda alias + substrate-detected appositive pattern |
| Nominalization | "via the activation of STAT3" | relation_axis probe few-shot exemplar covers nominalization |
| Coordination | "induced release of A, B, C, D" | object_role probe handles distributive answer (each coordinate gets one probe) |
| Causal chain | "X required for Y to inhibit Z" | relation_axis returns `via_mediator` for chains beyond direct |
| Distributive copula | "Like X, Y did Z" | subject_role substrate pattern for "Like X, Y…" extracts both X and Y |

### Bucket B — parser FP from over-anchoring (7 of 28 + 3 of 5 new-regs)
| Pattern | Example | S-phase probe-fix |
|---|---|---|
| Indirect chain via mediator | "Myc co-opts Phlpp2 to suppress Akt" | relation_axis returns `via_mediator`; adjudicator §5.2 emits indirect_chain abstain |
| Role-swap on receptor | "TNF receptor binding of TNFα" | subject_role returns present_as_object, object_role returns present_as_subject; adjudicator §5.2 emits role_swap → incorrect |
| Sign-flip via perturbation | "ADAM17 RNAi inhibited MMP9" | subject_role substrate (M9 entity-first LOF) sets perturbation=LOF; adjudicator §5.1 inverts effective sign; relation_axis returns `direct_sign_mismatch` against effective sign → sign_mismatch → incorrect |
| Axis confusion | "increased promoter activity" claimed as Activation | relation_axis returns `direct_axis_mismatch` (relation is on amount axis, not activity); adjudicator emits axis_mismatch → incorrect |
| Mediator negation | "X through PI3K but not MAPK" | scope probe handles "but not Y" within span; few-shot in S5 covers this |
| Wrong binding partner | substrate detects DNA binding when claim is Complex/protein | relation_axis returns `direct_partner_mismatch`; adjudicator emits binding_domain_mismatch → incorrect |

The probe architecture does not eliminate every Bucket A/B failure mode in v1. It eliminates the *structural cause* — forced premature commitment — and gives each pattern a place to be addressed (a few-shot exemplar in the right probe, or a substrate pattern in the routing layer).

## 10. Out of scope for S-phase

These are deliberate non-goals:

- **Cross-evidence aggregation.** S-phase scores one (statement, evidence) pair. Multi-evidence belief fusion remains in `composed_scorer.py` unchanged.
- **Claim parsing changes.** ClaimCommitment is the input to S-phase, not its output.
- **Grounding rewrite.** Gilda + GroundedEntity stays as-is.
- **Active escalation.** R6's escalation-triggered re-call is dormant (R7: 27B Gemma thinking-on times out). S-phase does not reintroduce it. If a probe abstains, the verdict is abstain — no model retry.
- **Calibration.** S-phase preserves the v15 calibration thresholds. If S-phase shifts the precision-recall curve, recalibration is a follow-up phase, not part of S.
- **Phosphosite verification.** site_mismatch (7 records, R-phase) is deferred to v2. A 5th probe for sub-axis detail (residue identity) is not justified by 7 records.
- **Bilateral ambiguity confidence downgrade.** bilateral_ambiguity_downgrade (8 records) becomes a confidence-only adjustment, not a verdict change. Implementation in S6 is a single confidence-tier rule on adjudicator output.

## 11. S-phase ship gates (recap from S11 task)

1. S beats R-phase by ≥1pp (architectural improvement is real)
2. S within 1pp of M12 (closes the gap)
3. Net delta vs M12 ≥ -3 records
4. Median wall ≤ 25s
5. p99 wall ≤ 90s
6. Substrate fast-path resolves ≥ 50% of records without LLM

If 5/6 gates pass with #2 marginal: SHIP-WITH-RESERVATIONS.
If <4/6 gates: ITERATE.
If S regresses R-phase by ≥1pp: REVERT.
