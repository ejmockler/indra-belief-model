# T6 — Implementation gate

Date: 2026-05-02
Status: **PASS**

## Pre-probe verification

| Check | Status | Detail |
|---|---|---|
| Test suite (tests/) | PASS | 453 passed, 1 skipped |
| Contamination guard | CLEAN | All eval files OK; no holdout paraphrase in few-shots |
| Imports clean | OK | `RelationAxisAnswer` confirms abstain removed |
| 5-record cross-fix smoke | PASS | A+B+C compose correctly; carve-outs respected |

## Cross-fix smoke results

| # | Probe inputs | Expected | Actual |
|---|---|---|---|
| 1 | LLM emits "abstain" for relation_axis (out-of-set) | non-abstain verdict | incorrect/absent_relationship ✓ |
| 2 | Phos + via_mediator + asserted | correct/low/upstream_attribution | correct/low/upstream_attribution ✓ |
| 3 | Complex + via_mediator + asserted | abstain/indirect_chain (carve-out) | abstain/indirect_chain ✓ |
| 4 | direct_sign_match + asserted | correct/match (clean path unaffected) | correct/match ✓ |
| 5 | Phos + relation_axis="abstain" (out-of-set) | projected to no_relation → incorrect | incorrect/absent_relationship ✓ |

Test 5 is important: it shows that Fix A's projection runs BEFORE Fix B's via_mediator branch. The LLM "abstain" projects to "no_relation" which routes into the `no_relation` adjudicator branch (incorrect/absent_relationship), not into the `via_mediator` branch where Fix B fires. This is correct semantics — projection is a conservative reading, and "absent_relationship" is the most informative verdict when the LLM punted.

## Implementation summary

**Files touched (6):**
- `src/indra_belief/scorers/probes/types.py` — removed `"abstain"` from `RelationAxisAnswer`
- `src/indra_belief/scorers/probes/_llm.py` — added `failure_default` parameter; discriminated out-of-set string projection (success) from transport failure (abstain)
- `src/indra_belief/scorers/probes/relation_axis.py` — added CRITICAL clause-localized block; replaced abstain few-shot; added 4 paired span-narrow few-shots; passes `failure_default="no_relation"`
- `src/indra_belief/scorers/probes/scope.py` — strengthened CRITICAL block with conditional-mutant clause; added 3 new few-shots
- `src/indra_belief/scorers/probes/adjudicator.py` — extended via_mediator branch for Phosphorylation; replaced ra==abstain branch with defensive scope-tiebreaker; updated confidence policy for upstream_attribution
- `src/indra_belief/scorers/commitments.py` — added `relation_underdetermined` and `upstream_attribution` ReasonCodes

**Files NOT touched:**
- `subject_role.py`, `object_role.py` — no Fix-C changes (they don't have sign/axis/scope detectors)
- `router.py`, `orchestrator.py` — pipeline structure unchanged
- All `tests/` files were updated only to align with the new behavior; no semantic intent changes

**Tests touched:**
- `tests/test_probe_types.py`: removed abstain from valid relation_axis values, added rejection test
- `tests/test_probes.py`: replaced legitimate-abstain test with projection test, added transport-failure-discriminated test
- `tests/test_probe_adjudicator.py`: rewrote 6 tests using `relation_source="abstain"` instead of `relation="abstain"`; added 4 new Phosphorylation Fix-B tests
- `tests/test_orchestrator.py`: updated alias-miss-escalates test to use `no_relation` instead of `abstain`

## Migration discipline check (S-phase doctrine §7)

- Single scoring path: ✓ no flags introduced
- No fallbacks: ✓ defensive ra==abstain branch is unreachable (type validation)
- No half-finished implementations: ✓ all three fixes complete + tested
- Tests preserved: ✓ no test deleted, all converted to new contract

## What this gate verifies

- All three fixes are wired through the orchestrator → router → probes → adjudicator pipeline.
- The fixes compose correctly when multiple trigger simultaneously (Test 5).
- Fix B's Phosphorylation-only carve-out is enforced (Test 3).
- The closed-set discipline (S-phase doctrine §2) is preserved — relation_axis still has 7 closed values.
- Contamination guard from May 2026 incident remains in effect.

## What this gate cannot verify

- Empirical behavior on real 27B Gemma responses for Fix A's projection. The mock test 1 confirms the plumbing; the 38 holdout records will validate the behavior. This is the role of T7 stratified probe.
- Fix C's prompt-level effectiveness — the LLM might or might not internalize the new few-shots. T7 will measure.
- Net accuracy gain — only the full holdout (T9) tells.

## Verdict

**PASS.** All static checks pass. Proceed to T7 stratified probe (50 records covering each fix's expected impact).
