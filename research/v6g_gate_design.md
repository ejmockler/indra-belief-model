# V6g composite gate design

Date: 2026-05-07

The V6g labeling validation replaces the simple `micro ≥ 0.90 AND
macro-recall ≥ 0.75 per probe` gate with a four-criterion composite
gate. This document records the rationale, thresholds, and a worked
example.

## Why the simple gate was inadequate

Per-probe class distributions are heavily imbalanced:

| Probe          | trivial-baseline accuracy (mfc) |
|----------------|-------------------------------|
| relation_axis  | 0.696                         |
| scope          | 0.948                         |
| subject_role   | 1.000 (post-audit, 1-class)   |
| object_role    | 1.000 (post-audit, 1-class)   |

Two independent failure modes go undetected by a single threshold:

1. **Majority-class collapse.** A model can clear `micro ≥ 0.90` on
   `scope` by predicting `asserted` for every record (baseline is 0.948
   already). Micro doesn't distinguish "skill" from "passing through".
2. **Trivially perfect 1-class probes.** When gold collapses to a single
   class (subject_role/object_role after the 2026-05-07 audit fix),
   `micro = recall = 1.0` for any model that tags the dominant class —
   which contains zero discriminative information. PASS in this case is
   misleading; the right verdict is "we don't have gold here".
3. **Hidden minority blowouts.** `relation_axis` had 0.020 recall on
   `direct_axis_mismatch` (50 gold examples). Macro-recall is moved
   somewhat by this, but a single number can't name the offending class.

We need composite criteria that (a) require lift over the trivial
baseline, (b) require minority-class coverage (both via a macro floor
and a per-class floor), and (c) escape gracefully when gold is
single-class.

## Four criteria + 1-class escape

Implemented as `Gate` (`scripts/v6g_gemini_validation.py`) and
`evaluate_gate(...)`. Verdict per probe = AND of the three boolean
criteria, except a 1-class probe escapes to `INSUFFICIENT_GOLD`.

### 1. Lift over trivial baseline

`micro ≥ mfc + delta_mfc`, default `delta_mfc = 0.05`.

Rationale: a model that ties or under-performs the most-frequent-class
baseline isn't earning its $200/run. The 5pp pad covers sampling noise
on n≈300 (95%CI half-width on a Bernoulli at p=0.9 is ≈3pp) and forces
real lift, not just a tied score.

### 2. Macro-recall floor

`macro_recall ≥ macro_floor`, default `macro_floor = 0.65`.

Rationale: macro-recall is an unweighted mean over classes that appear
in gold, so it falls hard when a minority class is missed. 0.65 is high
enough to require non-trivial minority coverage but low enough to be
achievable for the curator on the first try; the previous `0.75` floor
would have been unachievable for relation_axis even after the fixes
shipped in V6e/V6f. Once the curator hits 0.65, we ratchet to 0.70+.

### 3. Per-class recall floor (with min-support guard)

For every gold class with `support ≥ min_support`, require
`recall ≥ min_class_recall`. Defaults: `min_class_recall = 0.30`,
`min_support = 5`.

Rationale: the macro floor is an average — a model can hit 0.65 macro
while one class has 0.02 recall (e.g., `direct_axis_mismatch` in the
current run: macro looks acceptable, min-class is catastrophic).
Per-class floor catches that pattern. The min-support guard exempts
classes with ≤4 examples because their recall has CI half-width ≥ ±0.45
— failing on noisy small-class estimates would punish models for
sample-size, not skill. 0.30 is the recall a strong baseline would hit
on a well-defined minority class; below that, the model is effectively
ignoring the class. Ratchet to 0.50 once the curator clears macro 0.65.

### 4. 1-class escape: `INSUFFICIENT_GOLD`

Detected via `mfc == 1.0 AND macro_recall == micro` (any 1-class probe
forces these to be equal). Verdict is neither `PASS` nor `FAIL` — we
literally have no minority gold to test against. Emitting
`INSUFFICIENT_GOLD` flags the gap in the gold set rather than silently
passing models that just predict the dominant class.

## Per-model verdict aggregation

- All probes `PASS` → model `PASS`.
- Any probe `FAIL` → model `FAIL`.
- No probes fail but at least one is `INSUFFICIENT_GOLD` → model
  `INSUFFICIENT_GOLD` (we cannot certify the model on the current gold
  set, but it didn't fail anywhere either).

`FAIL` dominates `INSUFFICIENT_GOLD` because a documented failure on
one probe is a stronger signal than missing gold on another.

## Worked example: gemini-3.1-pro-preview, current run, defaults

Numbers from `data/v_phase/v6g_responses.jsonl`; defaults
`delta_mfc=0.05, macro_floor=0.65, min_class_recall=0.30, min_support=5`.

### relation_axis (n=392, micro=0.712, macro=0.463, mfc=0.696)

| Criterion        | Threshold                  | Value                              | Pass |
|------------------|----------------------------|-----------------------------------|------|
| Lift             | micro ≥ 0.696+0.05 = 0.746 | 0.712                             | ✗    |
| Macro            | ≥ 0.65                     | 0.463                             | ✗    |
| Per-class recall | ≥ 0.30 if support ≥ 5      | direct_axis_mismatch r=0.020 (n=50); direct_sign_mismatch r=0.188 (n=16) | ✗ |

→ **FAIL** on all three. The model's "skill" over baseline is a single
percentage point spent on `no_relation` recall (0.79); it has
near-zero recall on the two mismatch classes that are the entire point
of this probe.

### subject_role (n=354, micro=0.870, macro=0.870, mfc=1.000)

mfc=1.0 and macro==micro → **INSUFFICIENT_GOLD**. Audit collapsed
gold to a single class (`present_as_subject`); we cannot certify
labeling skill from this run. Need a re-audit that recovers
minority-class examples (decoy / mediator / object / absent) before
this probe can be gated.

### object_role (n=354, micro=0.884, macro=0.884, mfc=1.000)

Same as subject_role → **INSUFFICIENT_GOLD**.

### scope (n=288, micro=0.903, macro=0.711, mfc=0.948)

| Criterion        | Threshold                  | Value                              | Pass |
|------------------|----------------------------|-----------------------------------|------|
| Lift             | micro ≥ 0.948+0.05 = 0.998 | 0.903                             | ✗    |
| Macro            | ≥ 0.65                     | 0.711                             | ✓    |
| Per-class recall | ≥ 0.30 if support ≥ 5      | hedged r=0.500 (n=8), negated r=0.714 (n=7) | ✓ |

→ **FAIL** on lift only. The model shows decent minority-class
coverage (hedged@0.50, negated@0.71) but cannot beat the
predict-everything-asserted baseline. This is informative: scope is
near-saturated by trivial guessing on this distribution; we'd need
either a hedge-/negation-enriched gold split or a different metric to
demonstrate lift.

### Per-model verdict

Two probes FAIL and two are INSUFFICIENT_GOLD → **gemini-3.1-pro-preview:
FAIL**.

## What to tune as the numbers move

- **Curator iteration lifts macro on relation_axis above 0.65.** Raise
  `macro_floor` to 0.70 (next ratchet) and `min_class_recall` to 0.50.
  Once minority recall is real, hold the model to it.
- **Re-audit recovers minority gold for subject/object_role.** The
  1-class escape disappears automatically (mfc < 1.0) and the existing
  thresholds apply. If post-audit support is small (<10/class), keep
  `min_support=5` so the per-class floor still bites; consider raising
  to 10 once balanced.
- **Scope's lift is structurally hard.** If we cannot beat
  `mfc + 0.05 = 0.998` after curator rounds, replace lift with
  `macro ≥ 0.85` for scope only, or compute lift on a class-balanced
  subsample. Document the override in the gate config rather than
  silently lowering `delta_mfc`.
- **Models genuinely tie baseline within sampling noise.** Drop
  `delta_mfc` to 0.03 (covers ±2pp CI on n≈400) only after we have
  shown the model fails meaningfully on harder splits — never as a
  reaction to a single failed gate run.

## Files

- `scripts/v6g_gemini_validation.py` — `Gate` dataclass,
  `evaluate_gate(...)`, refactored `render_report(...)`, four CLI
  flags (`--gate-delta-mfc`, `--gate-macro-floor`,
  `--gate-min-class-recall`, `--gate-min-support`).
- `research/v6g_gemini_validation.md` — regenerated report under the
  default composite gate.
- `research/v6g_gate_design.md` — this document.
