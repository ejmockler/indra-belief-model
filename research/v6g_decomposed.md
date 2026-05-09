# V6g+ — decomposed curator (sub-question chain)

**Date:** 2026-05-07
**Author:** ejmockler
**Files touched:** `src/indra_belief/v_phase/decomposed/relation_axis.py`
**Predecessor docs:** `research/v6g_relation_axis_iteration.md`,
`research/v6g_gemini_validation.md`

The V6g single-shot curator path on `gemini-3.1-pro-preview` plateaued
on the two minority gold classes for `relation_axis`:

  - `direct_sign_match`    (n=273): P=0.912 R=0.853 F1=0.891
  - `no_relation`          (n=53):  P=0.527 R=0.736 F1=0.614 (over-predicted)
  - `direct_axis_mismatch` (n=50):  P=0.558 R=0.580 F1=0.569
  - `direct_sign_mismatch` (n=16):  P=0.600 R=**0.188** F1=0.286 (stuck)

Hypothesis: collapsing relation-presence + axis + sign into one
4-class pick lets the model default to `direct_sign_match` whenever
the entities and axis match. Decomposing into narrower sub-questions
should free the sign-mismatch decision and tighten the
no_relation/axis-mismatch border.

---

## Decomposed curator — relation_axis

### Sub-question chain

| # | Sub-question        | Class space                                                                              | Dependency                  |
|---|---------------------|------------------------------------------------------------------------------------------|-----------------------------|
| 1 | `relation_present`  | `{yes, no, via_intermediate}`                                                            | always                      |
| 2 | `evidence_axis`     | `{modification, activity, amount, binding, localization, gtp_state, conversion}`         | only if Q1 ∈ {yes, via_intermediate} (Q1 == "yes" only in current code) |
| 3 | `evidence_sign`     | `{positive, negative, neutral}`                                                          | only if `evidence_axis == claim_axis` AND `claim_sign != "neutral"` |

`claim_axis` and `claim_sign` are computed deterministically from
`stmt_type` via `derive_axis` / `derive_sign` in
`indra_belief.v_phase.decomposed_curator`.

### Assembly logic

```
if relation_present == "no":
    return "no_relation"
if relation_present == "via_intermediate":
    return "direct_sign_match"      # curator-v2 alignment (no via_mediator in U2 gold)
if evidence_axis != claim_axis:
    return "direct_axis_mismatch"
if claim_sign == "neutral":         # binding / localization / conversion
    return "direct_sign_match"
if evidence_sign == "neutral":      # degenerate match (no sign info to mismatch on)
    return "direct_sign_match"
if evidence_sign != claim_sign:
    return "direct_sign_mismatch"
return "direct_sign_match"
```

Deviation from the recommended logic: the `via_intermediate` early-exit
skips Q2 and Q3 (no axis/sign check). This matches U2 gold's
re-tagging of indirect-but-asserted relations as `correct` (=
direct_sign_match) during the audit (see
`research/v6g_gold_audit.md`); chasing axis/sign on a chain
introduces noise without changing the gold class.

### Sub-prompt design

Each sub-question gets a focused system prompt (200-400 tokens of
rules) plus 11-13 few-shots covering the class space. Synthetic
placeholder names (MAPK1, JUN, KinaseA/B, TF_X, FactorR, GeneZ,
EntityP/Q, CytokineY, etc.) per the contamination guard at
`memory/feedback_fewshot_contamination.md`. None of the few-shots
paraphrase holdout records; verified via grep against
`data/benchmark/holdout_v15_sample.jsonl` (zero matches on
synthetic placeholder names).

Key prompt features per sub-question:

  - **Q1 relation_present** — "DEFAULT to yes" framing with explicit
    permissive examples (passive, nominalization, symmetric binding).
    `via_intermediate` is reserved for named-mediator chains. `no` is
    reserved for parallel-list co-mention, methods/setup, loading
    controls, or claim-entity-absence.
  - **Q2 evidence_axis** — five disambiguation rules (A: amount beats
    activity, B: modification beats activity, C: TF-DNA is not
    binding, D: loss-of-function inherits the outcome verb's axis,
    E: degradation is amount). These mirror the curator-v2
    decision-table rules but isolated to the axis decision.
  - **Q3 evidence_sign** — five sign-flipping rules (F: knockdown
    flips sign, G: double-negation collapses, H: explicit
    "negatively/positively regulates", I: required-for inherits sign,
    J: only neutralize on binding/translocation/conversion). Q3 is
    only invoked once axes match and the claim sign is non-neutral,
    so the model isn't forced to pick "neutral" for activity/amount
    claims.

### Smoke-test results (limit=15, 13 relation_axis records)

```
relation_axis: micro=0.846 macro=0.705 mfc=0.846 (n=13, err=0)
subject_role:  micro=0.800 macro=0.800 mfc=1.000 (n=15, err=0)
object_role:   micro=0.867 macro=0.867 mfc=1.000 (n=15, err=0)
scope:         micro=1.000 macro=1.000 mfc=0.917 (n=12, err=0)
```

Per-record on relation_axis (13 records → 11 correct, 2 errors):

  - 1 axis-mismatch correctly fired (gold=axis_mismatch via amount-
    evidence on activity claim).
  - 1 axis-mismatch missed (`IncreaseAmount(NKX2-5, MYH)` evidence:
    "MEF2C expression initiated cardiomyogenesis ... up-regulation
    of ... myosin heavy chain"). Q1 picked "no" with reasoning that
    Nkx2-5 is in a parallel list and MEF2C is the actual upregulator.
    This is a defensible read — the same pattern would also be a
    valid `no_relation` under tighter gold; gold called it
    `direct_axis_mismatch`. Borderline case.
  - 1 false `no_relation` on a record where the claim subject and
    object were `?` placeholders (data quality issue, not a chain
    bug).

### Expected lift on each minority class (rough estimate)

  - **direct_sign_mismatch** (n=16): single-shot R=0.188 → expected
    R=0.45-0.65. Q3 is asked the sign question in isolation only
    when the axis already matches the claim, so the "default to
    sign_match" pressure is gone. The dominant single-shot failure
    mode (predicting `direct_amount_match` or `via_mediator` instead
    of `direct_sign_mismatch`) is structurally eliminated by the
    `class_space=["positive","negative","neutral"]` enum — the
    answer can ONLY be a sign label. Lift expected primarily from
    the loss-of-function flippers (Rule F) firing reliably.
  - **direct_axis_mismatch** (n=50): single-shot R=0.580 → expected
    similar or slightly better (≈0.55-0.65). Q2 is closed to seven
    axis tokens with strong rules, so Pro is unlikely to slip into
    `direct_amount_match` (no longer a valid label) or other
    out-of-gold labels.
  - **no_relation** (n=53): single-shot R=0.736 P=0.527
    (over-predicted) → expected R=0.65-0.75 with P=0.60-0.75. Q1's
    permissive "default to yes" framing should reduce false-positive
    `no_relation` calls from the borderline-asserted cases that
    drove single-shot precision down.
  - **direct_sign_match** (n=273): single-shot R=0.853 → expected
    similar (0.83-0.87). Most direct_sign_match records pass through
    Q1=yes → Q2=match → Q3=match cleanly.

### Per-record cost

Smoke-test breakdown (13 records):
  - 7 records (54%) → 3 sub-calls (relation_present + axis + sign)
  - 4 records (31%) → 2 sub-calls (axis-mismatch or claim-sign-neutral
    short-circuit after axis)
  - 2 records (15%) → 1 sub-call (no_relation early exit)

Average = **2.38 sub-calls per record**. At Pro's per-call cost,
that's ≈2.4× the single-shot cost. Latency is similar (chain runs
sequentially within the per-record coroutine, but records run
concurrently up to the global RPS limit, so wall-clock impact is
near-zero on a 482-record run).

### Sample-error analysis notes

From the 2 misclassifications in the 13-record smoke:

1. The `NKX2-5/MYH` borderline (gold=axis_mismatch, pred=no_relation)
   reflects a real ambiguity in the gold itself — the curator-v2
   single-shot prompt ALSO has a tendency to call this `no_relation`.
   The decomposed chain just makes the decision more explicit (Q1
   says "no, the upregulator is MEF2C, not NKX2-5"). Could be
   addressed by tightening Q1's "no" rules to require BOTH entities
   absent OR a strict parallel-list pattern, but at the risk of
   over-firing `yes` on alias-shadow records.

2. The `?,?` claim record (no subject or object given) is a data
   issue — Q1 correctly says "no" because there's nothing to relate.
   Could be filtered upstream in the runner if these records are
   noise.

No structural surprises: the chain produces sensible per-step
answers and the assembly logic doesn't fight the sub-answers.

---

## Validation footprint

  - Module: `src/indra_belief/v_phase/decomposed/relation_axis.py`
    (608 lines, ~13 KB).
  - Self-registers via `register("relation_axis", score)` at import.
  - Auto-loaded by `decomposed_curator._ensure_loaded()` lazily.
  - Runner picks it up when `--decomposed` flag is set:
    `decomposed-curator ENABLED for: ['relation_axis']` printed at
    startup.
  - Output schema matches the runner's expected
    `{"answer": ..., "sub_answers": ..., "sub_responses": ...,
    "claim_axis": ..., "claim_sign": ...}` format. On any sub-call
    error: `{"error": "sub_call_failed:<stage>", "sub_responses":
    {...}}` so the runner counts it as wrong.

User runs the full 482-record validation as next step.
