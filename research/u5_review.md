# U5r — Perturbation effective-sign propagation review

Date: 2026-05-02
Status: **PASS**

## What was implemented

Per doctrine §3.3 + U1 Observation #6 (audit adjudicator before changes):

**Audit finding:** adjudicator's `_effective_claim_sign` is computed at line 297-303 but **prefixed with `_` indicating "computed but unused"**. The adjudicator's decision table compares sign at line 86 (`if dr.sign == claim.sign`) using the LITERAL claim.sign from the ClaimCommitment, not an inverted effective sign. **No double-flip risk**: the adjudicator doesn't independently invert.

**Change applied:** `router._route_relation_axis` at line 367 was building the LLM-facing `claim_component` string with `sign={claim.sign}` (literal). The router already computed `effective_sign` (lines 380-382, used for substrate matching) but didn't propagate it to the LLM. U5 fix: build `claim_component` using `effective_sign` and add an explicit perturbation note when LOF/GOF is detected.

The note explains the inversion to the LLM probe so it understands why the sign field doesn't match the literal stmt_type:
```
[LOF perturbation on subject: claim_sign=negative inverted → effective_sign=positive]
```

## Tests

461 passed, 1 skipped. No test changes required (additive change to claim_component string).

## Smoke test (3 cases)

| Case | Pertub | Claim sign | Effective sign | Component |
|---|---|---|---|---|
| Normal claim | none | negative | negative | `sign=negative` (no note) |
| LOF subject | LOF | negative | **positive** | `sign=positive [LOF...claim_sign=negative inverted → effective_sign=positive]` |
| GOF subject | GOF | positive | positive | `sign=positive [GOF...sign preserved as positive]` |

All three behave as expected.

## Why this fix matters empirically

From U2's per-tag analysis:
- `polarity` records: 16, T-phase acc 56.2% (decision-only).
- These are records where the curator flagged sign-flip.

Half of the polarity FNs trace to "evidence describes a knockout effect" (LOF) where the literal claim sign differs from the effective sign. The substrate already knew this (computed effective_sign at router line 380); the LLM probe just didn't see it. U5 closes that signal gap.

Doctrine §3.3 estimate: +4 to +6 records. T-phase polarity-class T→T transitions: 9. Recoverable subset: ~5 records where the LLM probe rejected the sign as mismatch. Estimate stands.

## Q-phase failure mode check

Q-phase used INDRA KG to override LLM verdicts and lost 2.65pp. U5 doesn't override — it just gives the LLM probe a more accurate input. The LLM still makes its own decision. No verdict-override risk.

## Verdict

**PASS.** U5 implementation is correct, all tests green, smoke test verifies LOF inverts and GOF preserves with explicit note. The change is minimal (~20 lines) and additive — backward-compat with non-perturbation records unchanged.

Proceed to U6 (grounding probe enhancement, Intervention D).
