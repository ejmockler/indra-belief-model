# V8c — stratified resample to match U2 holdout class distribution

## Why

V8b preserved the V8a corpus distribution when building the LoRA training
set. The corpus was 41% `direct_sign_match`, 39% `no_relation`, 12%
`direct_axis_mismatch`, 8% `direct_sign_mismatch`. The U2 holdout
(relation_axis tasks, n=392) is heavily skewed toward
`direct_sign_match` at 70%, with only 14% `no_relation`, 13%
`direct_axis_mismatch`, 4% `direct_sign_mismatch`.

V9 LoRA tuned on V8b's corpus distribution learned the prior of the
training set: it over-predicts `no_relation` and under-predicts
`direct_sign_match`. On the holdout this manifests as over-recall on
the minority `no_relation` class and low precision on the dominant
`direct_sign_match` class. V9 scored 54.3% micro / 51.5% macro on the
holdout — losing both to the Pro curator (78.8% micro) and to the
trivial most-frequent-class baseline (69.6%).

The class prior the model absorbs from a SFT corpus dominates its
predictions when evidence is ambiguous. Aligning the train prior with
the holdout prior removes this artifact and lets the rationale-anchored
labels actually drive the decision.

## Bugs fixed (2026-05-07 brutalist round)

Four bugs identified and corrected in `scripts/v8c_stratified_split.py`:

1. **HOLDOUT_PROPORTIONS sum was 1.01.** Raw values 0.70+0.14+0.13+0.04
   = 1.01 caused `compute_quotas(2000)` to return quotas summing to
   2020. Fix: keep the documented raw values for clarity but normalize
   them at module load (`HOLDOUT_PROPORTIONS = {c: v/sum_raw}`); same
   pattern applied to `CORPUS_PROPORTIONS`. Asserts on the normalized
   sum and on `compute_quotas` invariant guarantee the post-allocation
   total equals `target_n` exactly.

2. **Train/val leakage via duplicate `source_hash`.** V8a contains 281
   hashes appearing >1 time. Of these 201 carry meaningful variation
   (different `(subject, object, stmt_type)` claims against the same
   evidence), so deduplication alone would discard signal. Fix:
   `GroupShuffleSplit` semantics — every record sharing a `source_hash`
   goes entirely to train or entirely to val. Within each class we
   shuffle hash-groups, then greedily take whole groups until the
   per-class record quota is reached.

3. **Class shortfall not enforced.** When a class pool was smaller than
   its quota, the script took everything available and the realized
   distribution drifted up to 5pp from target. Fix:
   `feasible_target_n(pool, p) = floor(min(pool[c]/p[c]))` computes
   the largest `target_n_train` at which every per-class quota fits
   its pool. If the requested `--target-n-train` exceeds this, the
   script logs a `WARN` and scales down. Also: hard-fail (exit code 2)
   if any post-allocation quota is 0.

4. **No assertions against hidden overlap.** Fix: after writing the
   JSONL files, the script asserts `train_hashes ∩ val_hashes == ∅`
   and `(train_hashes ∪ val_hashes) ∩ u2_holdout_hashes == ∅`, where
   the U2 holdout hashes are read from `probe_gold_holdout.jsonl`
   (482) and `holdout_v15_sample.jsonl` (also 482, identical set).
   Either failure raises `AssertionError` with up to 10 offending
   hashes for diagnosis.

### Side effect: multi-class hash drop

44 of the 2607 unique hashes are multi-class (their records span >1
distinct `answer`). Keeping these in primary-class buckets would let
secondary-answer records bleed into wrong buckets, distorting the
per-class distribution by ≈1.7pp. They are dropped entirely from V8c
(95 records, 3.2% loss), prioritizing distribution accuracy and clean
group integrity over data volume.

## Resampling math (after fixes, default args)

- **Source**: V8a kept records, n=2956 (3000 minus 44 `rationale_no_anchor`).
- **After multi-class hash drop**: 2861 single-class records across 2563 hashes.
- **Per-class record availability**:
  - direct_sign_match: 1177
  - no_relation: 1124
  - direct_axis_mismatch: 329
  - direct_sign_mismatch: 231
- **Step 1 — class-balanced val held out first** (50 records/class via
  whole-hash groups; 200 total). Pool after val:
  - direct_sign_match: 1127
  - no_relation: 1074
  - direct_axis_mismatch: 279
  - direct_sign_mismatch: 181
- **Step 2 — feasibility scaling**.
  - Bottleneck: direct_sign_match pool 1127 / 0.6931 = 1626. Other
    classes' bounds are higher.
  - Requested `--target-n-train 2000` infeasible → scaled down to
    `feasible_n = 1626`.
- **Step 3 — quotas at target_n=1626** × normalized U2 holdout
  proportions (largest-remainder allocation, sum exactly 1626):
  - direct_sign_match: 1127
  - no_relation: 225
  - direct_axis_mismatch: 209
  - direct_sign_mismatch: 65
- **Step 4 — group-aware sampling** (whole hash-groups only):
  - direct_axis_mismatch fell 2 records short due to group-rounding
    (smallest available hash-group exceeded the remaining slack).

## Resulting distributions

### Train (n=1624)

| class                | count | ratio  | U2 target (norm.) | delta pp |
| -------------------- | ----- | ------ | ----------------- | -------- |
| direct_sign_match    | 1127  | 69.40% | 69.31%            | +0.09    |
| no_relation          | 225   | 13.85% | 13.86%            | -0.01    |
| direct_axis_mismatch | 207   | 12.75% | 12.87%            | -0.12    |
| direct_sign_mismatch | 65    |  4.00% |  3.96%            | +0.04    |

Every class is within ±0.12pp of the (normalized) U2 holdout target.

### Val (n=200, class-balanced)

| class                | count |
| -------------------- | ----- |
| direct_sign_match    | 50    |
| direct_sign_mismatch | 50    |
| direct_axis_mismatch | 50    |
| no_relation          | 50    |

Class-balanced val is appropriate for monitoring per-class loss / F1
during fine-tuning regardless of train prior; it does not need to
mirror the holdout.

### Assertions (all pass)

- `train_hashes ∩ val_hashes = ∅` (1472 train hashes, 180 val hashes,
  zero overlap).
- `(train_hashes ∪ val_hashes) ∩ U2_holdout_hashes = ∅` (482 U2 hashes,
  zero overlap with V8c).

## Files

- Script: `scripts/v8c_stratified_split.py`
- Train: `data/v_phase/train/v8c_relation_axis.jsonl` (1624 records)
- Val:   `data/v_phase/val/v8c_relation_axis.jsonl` (200 records)

## CLI

```
python scripts/v8c_stratified_split.py \
    --match holdout \
    --target-n-train 2000 \
    --val-per-class 50 \
    --seed 42
```

The script auto-detects infeasible `target-n-train` and scales it down
with a `WARN` log. `--match corpus` is also supported for parity with
V8b's distribution if a future ablation needs it.
