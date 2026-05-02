# S7 brutalist gate — implementation review

Status: PASS-WITH-CAVEAT (few-shot coverage gap identified; addressed before S8)
Date: 2026-05-02
Method: substrate-only trace on the 13 R-phase records used in S4 + smoke test on real INDRA inputs

## Substrate-only trace (no LLM) — 13 records

### G class — substrate-fallback gains (3 records)

| Record | Target | S-phase verdict | Notes |
|---|---|---|---|
| RRAGA→RRAGC Complex | correct | **correct (regex_substrate_match)** ✓ | Final-arm rescue preserved (was at risk per S4 negation false-positive; verified fix held) |
| CDK1→RSF1 Phos | correct | **correct (regex_substrate_match)** ✓ | Final-arm rescue preserved |
| ?-? (ezrin) Phos | correct | abstain | parse_claim can't bind entities (subject="?"); same loss as R-phase, structural limitation upstream of S-phase |

**Substrate fallback gains: 2/3 preserved by deterministic-only path. The third was always going to be an abstain because there's nothing for the substrate to bind.**

### A class — parser FNs in R-phase (5 records, target=correct)

All 5 substrate-route to abstain (waiting for LLM probes). Substrate can't resolve because:
- CATALOG patterns don't cover apposition ("cyclin A-CDK2 (Cdk2)"), distributive ("Like X, Y did"), nominalization ("via the activation of"), coordination ("induced release of A, B, C, D"), or causal chain ("X required for Y to inhibit Z")
- These are exactly the patterns that should escalate to the LLM probe

**Risk:** if the LLM probe few-shots don't cover these patterns either, the LLM will mirror R-phase's behavior. The few-shots in `relation_axis.py` and `subject_role.py` cover the basic answer set values (direct_sign_match, role_swap, mediator) but DON'T exemplify these specific surface forms.

### B class — parser FPs in R-phase (5 records, target=incorrect)

All 5 substrate-route to abstain. R-phase had these as FP=correct; substrate-only flips to abstain.

**Calibration win:** abstain → score 0.5 (was 0.95 FP via R-phase). Composed scorer treats abstain much more conservatively. Net improvement in calibration even if no accuracy lift.

**For final verdict on these:** LLM scope probe should detect negation/role-swap/indirect chain. Few-shot for "negation governs different clause" already in scope.py (one of 5 shots).

## Implementation correctness

| Doctrine rule | Implementation | Status |
|---|---|---|
| §5.1 perturbation pre-rule | router applies LOF inversion to effective_sign before CATALOG match | ✓ verified by test_relation_perturbation_inverts_effective_sign |
| §5.2 canonical decision table | adjudicator.py `_decide` covers all 13+ rules; closed-set values exhaustive | ✓ verified by 27 adjudicator tests |
| §5.3 symmetric-binding | swapped (X,Y) treated as match for binding axis; role_swap fires only for non-binding | ✓ verified by test_binding_axis_swapped_roles_treated_as_match + test_role_swap_non_binding_axis_incorrect |
| §5.4 final-arm substrate-fallback | adjudicate() runs `_final_arm_substrate_match` on abstain verdicts | ✓ verified by 5 tests including symmetric-binding rescue |

All 13 R-phase reason codes have probe-tuple → reason mappings. Deferred codes (site_mismatch, internal_contradiction, cascade_terminal_match, chain_match) are noted in §5.5; no rule emits them.

## Migration discipline

- `parse_evidence.py`, `_prompts.py`, `decomposed.py`, `adjudicate.py` deleted
- `EvidenceCommitment`, `EvidenceAssertion`, deferred ReasonCodes deleted from commitments.py
- `use_decomposed` flag removed; `score_evidence(stmt, ev, client)` signature is cleaner
- 16 R-phase / N-phase / Q-phase scripts archived to `scripts/archive/r_phase/`
- 16 deprecated test files deleted; 90 new probe-pipeline tests added covering equivalent surface
- `_norm_alias` inlined in context_builder.py (was in deleted adjudicate.py)
- Test suite: 447 passing, 1 skipped — green
- `composed_scorer.py` unchanged (consumes the dict result; shape preserved)
- Public API: `score_evidence(stmt, ev, client)` and `score_statement(stmt, client)` — same shape, fewer params
- End state: one scoring path, no flags, no fallbacks, no parallel deprecated trajectories

## Known gap: few-shot coverage for Bucket A patterns

The five Bucket A patterns will hit the LLM probes without any few-shot exemplars matching them:

1. Apposition — "cyclin A-cyclin dependent kinase 2 (Cdk2)" → CDK2 is the agent
2. Distributive — "Like Sprouty1, Spred inhibited..." → both Sprouty1 and Spred are agents
3. Nominalization — "via the activation of STAT3" → IL10 activates STAT3 (direct, not chain)
4. Coordination — "induced release of A, B, C, and D" → TCR activates each of A, B, C, D
5. Causal chain bridging — "X required for Y to inhibit Z" → X inhibits Z (with Y as the proximate effector)

Without exemplars, the LLM may default to abstain on these patterns. Acceptable for v1 (still better calibration than R-phase FPs) but suboptimal for accuracy.

## Action items before S8

1. Add 2-3 few-shot exemplars covering the high-value Bucket A patterns:
   - One nominalization shot in relation_axis.py (e.g., "via the activation of" → direct_sign_match)
   - One coordination shot in object_role.py (e.g., "induced release of A, B, C, D" → present_as_object for each)
   - One apposition shot in subject_role.py (e.g., "kinase X (canonicalY)" → present_as_subject)
2. Skip distributive ("Like X, Y did") and complex causal chains for v1 — too few records to justify few-shot budget; defer to v2 patterns.

## Verdict

**PASS** — implementation is sound; architecture rules apply correctly; migration is clean. The known few-shot gap is small and addressed before S8 to maximize S-phase v1 accuracy upside.

S8 unblocked.
