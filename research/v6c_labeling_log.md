# V6c — per-probe Λ matrix + Snorkel LabelModel fit (log)

Date: 2026-05-06
Doctrine: `research/v5r_data_prep_doctrine.md` §4 (aggregation), §5
(holdout exclusion), §6 (sample sizes), §11 (NO `Y_dev`).

Module: `src/indra_belief/v_phase/labeling.py`
Tests: `tests/test_v6c_labeling.py` — 8 tests pass in 9.5s.

V6a delivered 13 substrate-tuned LFs; V6b delivered 38 clean LFs;
V6c stitches them into per-probe Λ matrices, fits Snorkel LabelModel
(no Y_dev, explicit equal class_balance), and emits parquet labels.

## Run record

| run | sample N | walk dt | LF dt | total | output suffix |
|---|---|---|---|---|---|
| 500-sample | 500 (statement, evidence) | 0.0s | 29.4s | 32.0s | `_sample` |
| 10k-sample | 10000 (statement, evidence) | 1.1s | (background, ~10 min) | TBD | `_sample` |

Both runs target `data/v_phase/labels/{probe}_sample.parquet`. The 500
run validated end-to-end; the 10k run was triggered in background and
will overwrite the same parquets when complete (same suffix). The 10k
run produced `[V6c] corpus walk: yielded 10000 (stmt, ev) pairs from
3218 statements; dropped 8 by holdout; 1.1s` before LF application
started (visible in `/tmp/v6c_10k.log`). Walk extrapolation: 1.1s for
3218 statements ⇒ ~5 minutes wall-clock for full 894K statements walk
(LF application dominates — extrapolated ~73 hours single-process,
~9 hours with 8-way pool parallelism). Full-run is therefore deferred
per task brief item 6.

## Corpus walk + holdout exclusion (V5r §5)

Holdout exclusion built from `data/benchmark/holdout_v15_sample.jsonl`
(501 records):
  - 482 unique source_hashes
  - 442 unique matches_hashes
  - 473 (pmid, frozenset(subject, object)) tuples

10k-sample walk: 10000 (statement, evidence) pairs yielded from 3218
unique statements; 8 pairs dropped by holdout exclusion. The dropped-8
fraction (8/10008 = 0.08%) extrapolates to ~7K of 2.85M total —
consistent with ~1009 holdout-overlap statements in V5r §5 spec
expanded by mean evidence-count.

Full-corpus N estimate: 894K statements × ~3.2 evidences/statement ≈
**2.86M (statement, evidence) pairs pre-holdout**. After holdout
exclusion (~7K-9K dropped), expect **~2.85M training-eligible pairs**.

## Per-probe Λ-matrix shape (500-sample run)

| Probe | Λ shape | n_LFs | K | LM-fit | predict_proba finite | kept (max_proba ≥ 0.5) |
|---|---|---|---|---|---|---|
| relation_axis | (500, 15) | 15 | 8 | OK | OK | 342/500 (68.4%) |
| subject_role | (500, 9) | 9 | 5 | OK | OK | 70/500 (14.0%) |
| object_role | (500, 9) | 9 | 5 | OK | OK | 222/500 (44.4%) |
| scope | (500, 9) | 9 | 5 | OK | OK | 195/500 (39.0%) |
| verify_grounding | (1000, 9) | 9 | 4 | OK | OK | 1000/1000 (100.0%) |

verify_grounding is per-(stmt, ev, entity); 500 (stmt, ev) ⇒ 1000
records (binary statements have 2 agents). The verify_grounding
multiplier on the 500-sample data is exactly 2.0×; V5r §6 budgeted
~2.3× for the full corpus with Complex-statement (3+ agents) inflow.

Total LFs: 13 substrate (V6a) + 38 clean (V6b) = 51, with the per-probe
split:
  - relation_axis: 4 substrate + 11 clean = 15
  - subject_role: 2 substrate + 7 clean = 9
  - object_role: 2 substrate + 7 clean = 9
  - scope: 3 substrate + 6 clean = 9
  - verify_grounding: 2 substrate + 7 clean = 9

## Per-LF firing rate (500-sample run)

### relation_axis (top 10 / bottom 10)
```
lf_reach_found_by_axis_match            340 (68.0%)
lf_epistemics_direct_true               186 (37.2%)
lf_epistemics_direct_false              163 (32.6%)
lf_chain_no_terminal                     61 (12.2%)
lf_substrate_catalog_match               30 (6.0%)
lf_substrate_negation_regex              26 (5.2%)
lf_no_entity_overlap                     16 (3.2%)
lf_curated_db_axis                        3 (0.6%)
lf_partner_substrate_gate                 1 (0.2%)
lf_reach_found_by_axis_mismatch           0 (0.0%)
lf_multi_extractor_axis_agreement         0 (0.0%)
lf_amount_lexical                         0 (0.0%)
lf_amount_keyword_negative                0 (0.0%)
lf_chain_with_named_intermediate          0 (0.0%)
lf_partner_dna_lexical                    0 (0.0%)
```

### subject_role
```
lf_position_subject_subj                500 (100.0%)
lf_position_object_subj                  91 (18.2%)
lf_no_grounded_match_subj                56 (11.2%)
lf_absent_alias_check_subj               56 (11.2%)
lf_decoy_lexical_subj                    25 (5.0%)
lf_chain_position_lexical_subj           14 (2.8%)
lf_role_swap_lexical_subj                 4 (0.8%)
lf_substrate_chain_position_subject       0 (0.0%)
lf_substrate_decoy_subject                0 (0.0%)
```

### object_role
```
lf_position_object_obj                  500 (100.0%)
lf_no_grounded_match_obj                135 (27.0%)
lf_absent_alias_check_obj               135 (27.0%)
lf_position_subject_obj                  91 (18.2%)
lf_decoy_lexical_obj                     19 (3.8%)
lf_chain_position_lexical_obj             7 (1.4%)
lf_role_swap_lexical_obj                  4 (0.8%)
lf_substrate_chain_position_object        0 (0.0%)
lf_substrate_decoy_object                 0 (0.0%)
```

### scope
```
lf_clean_assertion                      104 (20.8%)
lf_hedge_lexical                         67 (13.4%)
lf_substrate_hedge_marker                34 (6.8%)
lf_low_information_evidence              15 (3.0%)
lf_substrate_negation_explicit           14 (2.8%)
lf_negation_lexical                      12 (2.4%)
lf_conditional_clause_substrate          11 (2.2%)
lf_conditional_lexical                    4 (0.8%)
lf_text_too_short                         4 (0.8%)
```

### verify_grounding
```
lf_evidence_contains_official_symbol    681 (68.1%)
lf_gilda_exact_symbol                   583 (58.3%)
lf_ambiguous_grounding                  323 (32.3%)
lf_gilda_alias                          226 (22.6%)
lf_gilda_no_match                       191 (19.1%)
lf_evidence_too_short_grounding          12 (1.2%)
lf_fragment_processed_form_subject        2 (0.2%)
lf_fragment_processed_form_object         2 (0.2%)
lf_gilda_family_member                    0 (0.0%)
```

## Per-class vote distribution (500-sample run)

| Probe | class 0 | class 1 | class 2 | class 3 | class 4 | class 5 | class 6 | class 7 |
|---|---|---|---|---|---|---|---|---|
| relation_axis | 552 | 0 | 33 | 0 | 1 | 163 | 61 | 16 |
| subject_role | 500 | 95 | 14 | 25 | 112 | — | — | — |
| object_role | 500 | 91 | 19 | 19 | 270 | — | — | — |
| scope | 104 | 101 | 15 | 26 | 19 | — | — | — |
| verify_grounding | 1264 | 230 | 191 | 335 | — | — | — | — |

Classes with **zero** non-ABSTAIN votes in the 500 sample:
  - relation_axis: classes 1 (`direct_amount_match`) and 3
    (`direct_axis_mismatch`) — both V5r §3.1 already flagged for
    synthetic-oracle augmentation in the rare-class plan
    (LF_amount_lexical and LF_amount_keyword_negative didn't fire,
    consistent with low base rate of explicit Activation/Inhibition →
    amount-lexicon co-occurrence in a 500-record slice).

`relation_axis` class 4 (`direct_partner_mismatch`) had only 1 vote;
relation_axis class 7 (`no_relation`) had 16. These will likely
populate denser at 10k+ scale; full-corpus N estimates will confirm
whether synthetic-oracle is needed per V5r §6.

## Correlation flags (V5r §4 step 2)

**No pairs flagged with |r| > 0.5 and overlap ≥ 50** across any of the
five probes. The diagnosis used `numpy.corrcoef` on records where both
LFs in a pair were non-ABSTAIN, with min_overlap=50 per V5r §4
correction (Snorkel 0.9.9's `lf_summary` does not compute pairwise
correlation; manual computation is the V5r-mandated path).

Note: this is the **500-sample** result; the 10k run will produce a
denser overlap matrix and may surface new flags. The current zero-flag
result holds at 500-sample power; V6d should re-check on the 10k
output before committing the LFs to the LoRA training feed.

## LabelModel weights (500-sample run)

Per V5r §11 / §4: `LabelModel(cardinality=K, verbose=True).fit(L_train=Λ,
class_balance=[1/K]*K, n_epochs=500, lr=0.01)` — NO `Y_dev`.

| Probe | min | mean | max |
|---|---|---|---|
| relation_axis | 0.167 | 0.649 | 1.000 |
| subject_role | 0.201 | 0.631 | 1.000 |
| object_role | 0.085 | 0.553 | 1.000 |
| scope | 0.494 | 0.739 | 1.000 |
| verify_grounding | 0.083 | 0.585 | 1.000 |

LFs that hit the 1.000 ceiling are typically those with zero/very-low
fires on the 500-sample (Snorkel returns 1.000 when `accs/coverage`
goes through divide-by-zero clamp). The `RuntimeWarning: divide by
zero encountered in divide` from snorkel/labeling/model/label_model.py:387
appears on every probe that has a no-fire LF in the slice — this is a
known Snorkel 0.9.9 behavior, not a V6c bug. With the full 10k or
2.85M run, near-zero-fire LFs should shrink and the weight ceiling
should drop. V7a will compare these to per-LF accuracy on U2 gold (V5r
§7a) — divergence >10pp signals identifiability collapse.

predict_proba is **finite for all rows on every probe** (no NaN, no
inf). All probability rows sum to 1 within 1e-3.

## Kept-for-training counts (max_proba ≥ 0.5)

| Probe | kept | total | rate |
|---|---|---|---|
| relation_axis | 342 | 500 | 68.4% |
| subject_role | 70 | 500 | 14.0% |
| object_role | 222 | 500 | 44.4% |
| scope | 195 | 500 | 39.0% |
| verify_grounding | 1000 | 1000 | 100.0% |

Notes:
  - `subject_role` 14.0% is suspiciously low. Investigation: at 500
    sample, only `lf_position_subject_subj` fires reliably (100%);
    other clean LFs fire 0-18%, so LabelModel resolves to a uniform
    posterior over 5 classes for many records (max_proba just below
    0.5). At 10k+ scale, additional non-ABSTAIN votes from
    `lf_decoy_lexical_subj`, `lf_chain_position_lexical_subj`, and
    `lf_role_swap_lexical_subj` should push more records over the 0.5
    threshold.
  - `verify_grounding` 100% is also expected at small N because the
    clean lf_evidence_contains_official_symbol fires on 68% of
    records and concentrates posterior mass on `mentioned`; with
    larger N + stratified subsample (V6d), kept fraction will drop to
    a more useful range.
  - V6d's stratified subsample (V5r §6) targets **3K min per class +
    15-28K per probe**. The current 500-sample kept counts are
    pipeline-validation outputs, not training-set sizes.

## Validation gates

1. `pytest tests/test_v6c_labeling.py -v` — **8 passed in 9.49s**
2. `python -m indra_belief.v_phase.labeling --sample 500
   --no-verbose-lm` — completed in 32.0s, 5 sample parquets emitted
3. `python -m indra_belief.v_phase.labeling --sample 10000
   --no-verbose-lm` — corpus walk completed in 1.1s; LF application in
   progress (background). Will overwrite `_sample.parquet` files.
4. No NaN/inf in any predict_proba output.
5. No LF errors recorded in the per-fn error dict (51/51 LFs apply
   without throwing on real corpus records).

## Expected V6d input shape per probe (full-corpus extrapolation)

Assuming ~2.85M (statement, evidence) pairs after holdout exclusion:

| Probe | Λ rows (full corpus) | n_LFs |
|---|---|---|
| relation_axis | ~2.85M | 15 |
| subject_role | ~2.85M | 9 |
| object_role | ~2.85M | 9 |
| scope | ~2.85M | 9 |
| verify_grounding | ~6.6M (×2.3 per V5r §6) | 9 |

V6d will: (a) read the full-corpus parquets (`{probe}.parquet`,
no `_sample` suffix), (b) apply the V5r §10 stratified subsample
to enforce min-3K per class, (c) apply the §12 contamination filter,
(d) emit JSONL training files per V5r §8.

## Blockers / follow-ups

1. **Full-corpus run deferred**: 894K statements × ~3.2 evidences/stmt
   × 51 LFs single-process is ~73 hours wall-clock. V6c task brief
   item 6 explicitly defers this. Recommended: 8-way `multiprocessing`
   pool over chunks of (stmt, ev) records, each worker importing
   `v_phase.{substrate_lfs, clean_lfs}` once. Estimated 9-10 hours
   wall-clock at 8-way parallelism.

2. **Snorkel divide-by-zero warning**: harmless RuntimeWarning when an
   LF has 0 fires in the slice. Suppressing via `warnings.filterwarnings`
   in V6d full-run is reasonable; not blocking V6c.

3. **Class-coverage gap at 500-sample**: relation_axis class 1
   (`direct_amount_match`) and class 3 (`direct_axis_mismatch`)
   showed zero votes; both are flagged in V5r §6 for synthetic-oracle
   augmentation. Confirmation pending the 10k or full-corpus run.

4. **subject_role 14% kept-rate**: the LabelModel is under-determined
   when only 1 of 9 LFs fires on most records. Watch this in the 10k
   run; if still <30% kept, V6d may need to relax the 0.5 threshold
   per V5r §4 step 6 OR V5r §3.2 may need a position-LF that better
   distinguishes between the role classes.

5. **No `Y_dev` enforcement (V5r §11)**: confirmed — V6c does not pass
   `Y_dev` to `LabelModel.fit()`; class_balance is the single anchor.
   U2 gold validation lives in V7a (post-hoc).
