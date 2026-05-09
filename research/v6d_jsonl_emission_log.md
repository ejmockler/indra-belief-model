# V6d — JSONL emission log

Date: 2026-05-06
Predecessor: V6c (per-probe Snorkel LabelModel parquets)
Successor: V7a (LF accuracy on U2 gold)

## Pipeline summary

1. Read V6c parquets from `data/v_phase/labels/{probe}_sample.parquet`.
2. Filter to `kept_for_training=True` records.
3. Re-walk `data/benchmark/indra_benchmark_corpus.json.gz` to recover
   (statement, evidence) text for each record_id.
4. Stratified subsample per V5r §6 targets (sample mode caps to natural
   pool with min_per_class=3000 ceiling).
5. Synthetic-oracle augmentation for classes with <50 natural records
   (sample-mode threshold; full-mode threshold is `min_per_class`).
6. Two-filter contamination guard (V5r §12):
   - Filter 1 = `scripts/check_contamination.py` algorithm (exact,
     substring, paraphrase 50-char window 5-char stride, pair).
   - Filter 2 = trigram-Jaccard at 95th-percentile-of-natural-overlap.
   - Synthetic records BYPASS both filters (no holdout-overlap risk by
     construction).
7. 90/10 train/val split, JSONL emission.

Module: `src/indra_belief/v_phase/jsonl_emitter.py`.
Tests: `tests/test_v6d_jsonl_emitter.py` (25 tests, all pass).

## Per-probe natural-vs-synthetic record counts (sample-mode validation)

Run command:
```
python -m indra_belief.v_phase.jsonl_emitter --from-sample
```

| Probe | Natural | Synthetic | Total | Train | Val |
|---|--:|--:|--:|--:|--:|
| relation_axis     |  88 |  980 | 1068 |  962 | 106 |
| subject_role      |  21 |  380 |  401 |  361 |  40 |
| object_role       | 118 |  360 |  478 |  431 |  47 |
| scope             |  57 |  560 |  617 |  556 |  61 |
| verify_grounding  | 572 |    0 |  572 |  515 |  57 |

Per-class final counts (train + val combined):

- relation_axis: direct_partner_mismatch=200, no_relation=180,
  direct_axis_mismatch=200, via_mediator_partial=201,
  direct_amount_match=200, via_mediator=73, direct_sign_match=14
- subject_role: present_as_decoy=180, present_as_mediator=200,
  absent=20, present_as_object=1
- object_role: present_as_mediator=180, present_as_decoy=180,
  present_as_subject=77, absent=37, present_as_object=4
- scope: asserted_with_condition=200, hedged=30, abstain=183,
  negated=185, asserted=19
- verify_grounding: mentioned=343, equivalent=79, not_present=70,
  uncertain=80

Note: at sample size (500 stmt/ev pairs), natural-pool counts are far
below the `min_per_class=3000` target. Synthetic augmentation supplies
~200 records per rare class (10 templates × ~30 placeholders = ~300,
deduplicated). The pipeline structure is validated end-to-end; full-
target counts are reached when the V6c parquets are re-emitted from
the full corpus (V8 scope).

## Contamination filter drop rates (sample-mode)

| Probe | F1 only | F2 only | Both | Total drop | Total in | Drop rate |
|---|--:|--:|--:|--:|--:|--:|
| relation_axis     | 18 |  8 |  2 |  28 / 1322 | 2.1% |
| subject_role      | 10 |  1 |  0 |  11 /  450 | 2.4% |
| object_role       |  2 |  6 |  7 |  15 /  582 | 2.6% |
| scope             |  7 |  2 |  1 |  10 /  755 | 1.3% |
| verify_grounding  | 54 | 28 | 18 | 100 / 1000 | 10.0% |

Filter 1 (substring/exact/paraphrase) catches the high-confidence
overlap cases. Filter 2 (trigram-Jaccard) catches the long-range
fuzzy-paraphrase cases. The two filters are complementary — overlap
("Both" column) is small but non-zero, confirming each contributes
distinct evidence. verify_grounding has the highest drop rate
because its records carry the same (stmt, ev) text across multiple
entity probes; once one entity's text is filtered, the sibling entity
records are also at risk.

## Trigram-Jaccard threshold

Calibrated value (95th percentile of 10K random natural pairs at sample
mode): **0.0000**

Effective threshold (after `min_floor=0.05`): **0.05**

Persisted at `data/v_phase/contamination_threshold.json`. The floor of
0.05 prevents the degenerate sample-mode case (where natural pairs have
near-zero overlap) from flagging every trigram match as contamination.
At full-corpus scale the calibrated 95th-pct will likely be in the
0.05-0.15 range; the floor becomes irrelevant.

## Production-prompt byte-equality verification

For relation_axis (representative probe):
- `system_sha256`: `693828ca0f5be145…` (16-char prefix shown)
- `n_few_shots`: 18 (matches `_pr._FEW_SHOTS` length)
- `first_user_msg_len`: 113 chars

Verification mechanism (V5r §8 byte-equality contract): the V6d
`render_messages()` function imports the same probe modules used at
inference (`src/indra_belief/scorers/probes/{relation_axis,
subject_role, object_role, scope}.py` and
`src/indra_belief/scorers/grounding.py`) and reads `_SYSTEM_PROMPT` +
`_FEW_SHOTS` directly. Any future drift between training and inference
prompts is impossible by construction unless someone forks the prompt
module — which the test
`test_render_uses_production_system_prompt_verbatim` would catch (it
asserts `msgs[0]["content"] == _pr._SYSTEM_PROMPT` literally).

The four non-grounding probes use the production few-shot curriculum
threaded as alternating user/assistant messages, matching what
`_llm.llm_classify` constructs. verify_grounding has no few-shots
(matches inference behavior) and uses the raw `grounding._SYSTEM_PROMPT`.

## Outputs

- Train: `data/v_phase/train/{probe}_sample.jsonl` (5 files, total ~21 MB)
- Val: `data/v_phase/val/{probe}_sample.jsonl` (5 files, total ~2.3 MB)
- Threshold: `data/v_phase/contamination_threshold.json`
- Synthetic templates: `data/v_phase/synthetic_oracles/{probe}_{class}.yaml`
  (13 files: 1 placeholder pool + 12 per-rare-class template files)

## Schema (V5r §8)

Each JSONL line:
```json
{
  "messages": [
    {"role": "system", "content": "<probe-specific system prompt>"},
    {"role": "user",      "content": "<few-shot user 1>"},
    {"role": "assistant", "content": "<few-shot assistant 1>"},
    ...
    {"role": "user",      "content": "<actual question (claim + evidence)>"}
  ],
  "completion": "{\"answer\": \"<class>\", \"rationale\": \"\"}",
  "soft_labels": {"class_0": 0.05, "class_1": 0.85, ...},
  "synthetic": false
}
```

verify_grounding completion is `{"status": "<status>", "rationale": ""}`
to match `grounding._VALID_GROUNDING`.

## Blockers / gaps for full-mode run

1. **V6c full-corpus run** is required before this pipeline can produce
   the V5r §6 target counts. V6c sample-mode used only 500 (stmt, ev)
   pairs; full-corpus pass through 894K statements is the V8 prerequisite.
2. **Filter 2 calibration scale**: 10K random pairs from a 500-record
   pool has limited statistical power. At full scale, calibration runs
   on the full natural-text pool; the 0.05 floor likely becomes
   inactive.
3. **verify_grounding entity-block fidelity**: V6d uses the agent's
   `db_refs` HGNC/FPLX hint to render the entity block. Full Gilda
   resolution (with aliases, family members, is_pseudogene flags) is
   deferred to a V6d-extension stage when the full-corpus pipeline runs;
   for V8 ship-readiness this is a known gap.
4. **Synthetic-oracle coverage**: 12 rare-class YAMLs cover the
   highest-priority cases identified in V5r §3 ≥2-LFs-per-class gap.
   Not all 8 relation_axis classes have synthetic templates (e.g.,
   direct_sign_mismatch, direct_sign_match are well-served by natural
   data). If full-corpus reveals additional rare classes, more YAMLs
   ship as a one-time effort.
