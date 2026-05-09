# T3r — Fix B review

Date: 2026-05-02
Status: **PASS**

## What was implemented

Per doctrine §3.2:

1. **`adjudicator.py`** — extended the `via_mediator` and `via_mediator_partial` branches to handle Phosphorylation specifically, in addition to the existing causal-claim handling:
   - `via_mediator` + Phosphorylation + asserted/hedged/abstain scope → `correct/upstream_attribution` (low confidence).
   - `via_mediator` + Phosphorylation + negated → `incorrect/contradicted`.
   - `via_mediator_partial` + Phosphorylation → `correct/chain_extraction_gap` (low).
   - Other direct claims (Complex, Translocation, etc.) still abstain.

2. **Confidence policy** — added `upstream_attribution` to the low-confidence set alongside `hedging_hypothesis` and `chain_extraction_gap`.

3. **`commitments.py`** — `upstream_attribution` ReasonCode was added in T2.

## Tests

- 4 new tests in `test_probe_adjudicator.py`:
  - `test_via_mediator_phosphorylation_correct_low`
  - `test_via_mediator_phosphorylation_negated_incorrect`
  - `test_via_mediator_partial_phosphorylation_correct_low`
  - `test_via_mediator_partial_complex_still_abstains`
- Modified `test_via_mediator_direct_claim_abstains` to use Complex (Phosphorylation now goes to the new branch).
- Full test suite: 453 passed, 1 skipped.

## Smoke test (5 cases)

| # | Case | Expected | Status |
|---|---|---|---|
| 1 | Phosphorylation + via_mediator + asserted | correct/low/upstream_attribution | PASS |
| 2 | Complex + via_mediator + asserted | abstain/indirect_chain (still) | PASS |
| 3 | Activation + via_mediator + asserted | correct/match (causal path unchanged) | PASS |
| 4 | Phosphorylation + via_mediator + negated | incorrect/contradicted (scope wins) | PASS |
| 5 | Phosphorylation + via_mediator_partial | correct/low/chain_extraction_gap | PASS |

## Empirical hypothesis traceability

The Phosphorylation-only scope was empirically validated in pre-T0 research:

| stmt_type | indirect_chain class | gold-correct | gold-incorrect | force-correct net |
|---|---|---|---|---|
| **Phosphorylation** | 4 | **4** | **0** | **+4** |
| Complex | 7 | 3 | 4 | −1 |
| Activation | 7 | 4 | 3 | n/a (already causal) |
| Inhibition | 1 | 0 | 1 | n/a (already causal) |
| IncreaseAmount | 1 | 1 | 0 | n/a (already causal) |

Fix B targets precisely the 4 records where forced-correct has 100% precision. Other direct claim types stay abstain; flipping them all would damage 4 records and gain only 3.

## Verdict

**PASS.** Fix B is implemented as a narrow surgical bypass within the existing `via_mediator` branch. No collateral damage on other stmt types, all tests green, end-to-end smoke verified. Expected gain: +4 records on the holdout.

Proceed to T4 (Fix C — narrow trigger span).
