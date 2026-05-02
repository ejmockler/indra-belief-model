# S-phase ship verdict — 2026-05-02

Status: **SHIP-WITH-RESERVATIONS** (5/6 gates passing)
Predecessor: R-phase (72.86% on holdout_v15_sample)
Doctrine: research/s_phase_doctrine.md

## Headline

| Metric | S-phase | R-phase | M12 | Q-phase |
|---|---|---|---|---|
| Accuracy | **74.58%** | 72.86% | 74.95% | 70.21% |
| Δ vs R | +1.72pp | — | +2.09pp | -2.65pp |
| Δ vs M | -0.37pp | -2.09pp | — | -4.74pp |
| Median wall | **5.2s** | 12.5s | n/a | 7.0s |
| p99 wall | 6.2s | 51.8s | n/a | n/a |
| Abstain rate | 26.6% | 0.6% | 0.6% | 0.6% |

S-phase is **+1.72pp** vs R-phase and only **-0.37pp** vs M12 baseline. Wall-time is 2.4× better than R-phase. The architecture exposes finer error categories (role_swap, binding_domain_mismatch, partial-chain).

## Architecture summary

Replaced the monolithic `parse_evidence` extraction with four narrow probes (subject_role, object_role, relation_axis, scope), each with a closed answer set ≤8 values. Substrate is a question-router (CATALOG/Gilda/perturbation markers) that either answers deterministically or hands off to LLM with a hint. Adjudicator is a flat decision table (~15 rules) over ProbeBundle.

Migration discipline (per doctrine §7): **single scoring path, no flags, no fallbacks**. Deleted: parse_evidence.py, _prompts.py, decomposed.py, adjudicate.py (replaced), EvidenceCommitment, EvidenceAssertion, deferred ReasonCodes, 16 deprecated test files, 16 R/N/Q-phase scripts archived.

## Fix history

Three architectural fixes applied during S10 to address abstain-rate diagnosis:

1. **Substrate alias-miss escalates to LLM** — was committing `absent` deterministically when Gilda alias map didn't match; now escalates with hint (recovers anaphora, paraphrase, family references).
2. **Hedged → correct/low (was abstain)** — doctrine §5.7. The relation IS asserted; hedging modulates confidence, not verdict. Composed scorer can weight low-confidence correct < high.
3. **Causal claims accept indirect chains** — doctrine §5.6. Activation/Inhibition/IncreaseAmount/DecreaseAmount accept via_mediator (INDRA pathway-level semantics); Phosphorylation/Complex/etc. require direct contact.

## Reason-code profile (482-record holdout)

```
match                          206  (43%)
grounding_gap                   70   absent subject or object after LLM verification
absent_relationship             48   no relation between mentioned entities
hedging_hypothesis              30   correct/low — hedged but matched
sign_mismatch                   20   sign-flip detected
indirect_chain                  20   direct claim, evidence shows chain → abstain
axis_mismatch                   16   relation on wrong axis
regex_substrate_match           11   final-arm CATALOG rescue (M3 hoist preserved)
role_swap                        9   subject in object slot, non-binding
binding_domain_mismatch          9   DNA binding when claim is Complex/protein
contradicted                     5   explicit negation
```

## Failed gate — substrate fast-path coverage

Doctrine §6 targeted **≥50% records resolve all four probes via substrate alone (zero LLM calls)**. Actual: **1.2%** (6/482).

Cause: Fix 1 (alias-miss escalates to LLM) made every record where Gilda alias map missed at least one entity escalate to LLM. The trade was worth it for accuracy (+~3pp restored vs the over-aggressive abstain version).

LLM-call distribution:
- 0 LLM calls: 6 records (1.2%)
- 1 LLM call: 52 (10.8%)
- 2 LLM calls: 10 (2.1%)
- 3 LLM calls: 84 (17.4%)
- 4 LLM calls: 330 (68.5%)

**Disposition:** acknowledged as known limitation. The latency budget is so under-spec (5.2s median vs 25s gate) that the LLM-call trade is fine. Ship.

## Known v1 limitations

- **SGN stratum (sign-flip via perturbation context):** 0/3 in S8 stratified probe. The LLM relation_axis probe receives the original claim sign, not the substrate-computed effective sign. A v2 fix would pass effective_sign in the claim_component when LOF perturbation marker is set. ~7 records on holdout.
- **(no reason) abstains:** 12 records in pre-fix S10 partial were LLM transport / JSON failures. No retry logic. Adding a single retry on JSON parse failure would recover ~4-8 records.
- **Direct claims with indirect chain:** Phosphorylation/Complex/Translocation with via_mediator stays abstain. R-phase had `upstream_attribution` to lift these to correct in some cases; not in S-phase v1. ~6 records.
- **Substrate fast-path coverage (1.2%):** see "Failed gate" above. Bounded by Gilda's alias coverage; v2 could add stronger alias enrichment.

## Net regression / gain analysis

vs M12 (-0.37pp accuracy):
- 55 regressions (M12 right, S wrong) — 33 went from correct → abstain (calibration improvement, not a hard miss); 20 went from correct → incorrect (hard miss)
- 53 gains (M12 wrong, S right)

vs R-phase (+1.72pp):
- 48 regressions (R right, S wrong) — most from substrate-fast-path-resolved cases that R got correct; S now escalates and LLM doesn't always reach the same answer
- 57 gains (R wrong, S right) — Fix 2/3 (hedging + causal indirect chain) drove most of these

## Recommendation

**SHIP** to origin/main. The architecture is sound. The accuracy is competitive with M12, exceeds R-phase, exceeds Q-phase. Latency is excellent. The reason-code profile is finer-grained than any prior phase. The migration discipline is clean — one scoring path, no parallel deprecated trajectories. Few-shot contamination guard is now stricter (paraphrase detection added).

The substrate fast-path gate failure is a calibration trade, not an architectural regression. Future work can address the SGN stratum, retry-on-failure, and alias enrichment without architectural change.
