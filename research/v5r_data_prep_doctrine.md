# V5r — data prep doctrine, revised after gate FAIL

Date: 2026-05-06
Predecessor: V5 (research/v5_data_prep_doctrine.md, superseded); V5 gate findings (research/v5_gate.md)
Purpose: specify how training data for each of the five probe LoRA adapters is derived from the 894K-statement INDRA corpus using multi-source weak supervision, with no use of the existing belief score and with quantified substrate-contamination guards.

This revision addresses the 8 fixes from V5 gate (4 BLOCKING + 4 of 5 MAJOR). Sections marked **[REVISED]** changed from V5; unchanged sections inherit the V5 wording.

## §1 The five probes and their label spaces (unchanged from V5)

Per `src/indra_belief/scorers/probes/types.py` and `grounding.py`:

| Probe | Classes | Count |
|---|---|---|
| **subject_role** | `present_as_subject`, `present_as_object`, `present_as_mediator`, `present_as_decoy`, `absent` | 5 |
| **object_role** | (same as subject_role) | 5 |
| **relation_axis** | `direct_sign_match`, `direct_amount_match`, `direct_sign_mismatch`, `direct_axis_mismatch`, `direct_partner_mismatch`, `via_mediator`, `via_mediator_partial`, `no_relation` | 8 |
| **scope** | `asserted`, `hedged`, `asserted_with_condition`, `negated`, `abstain` | 5 |
| **verify_grounding** | `mentioned`, `equivalent`, `not_present`, `uncertain` | 4 |

V5 gate confirmed these match `types.py:43-79` and `grounding.py:42-47` exactly.

## §2 Training-record schema **[REVISED — verify_grounding multiplicity]**

Each training record is a (claim, evidence_text, probe_kind, label_distribution) tuple where:

- `claim` = an INDRA Statement: `(subject_entity, object_entity, relation_type, …)` rendered as a JSON-shaped instruction prompt
- `evidence_text` = a single Evidence's `text` field (≤512 tokens after tokenization)
- `probe_kind` ∈ {subject_role, object_role, relation_axis, scope, verify_grounding}
- `label_distribution` = probabilistic label per class from Snorkel LabelModel

**Per-(statement, evidence) record count**:
- subject_role: 1 record
- object_role: 1 record
- relation_axis: 1 record
- scope: 1 record
- verify_grounding: **N records** where N = count of distinct claim agents (typically 2 for binary statements; 3+ for Complexes). Each record is keyed on (statement, evidence, entity).

So a typical binary-statement (subject + object) (statement, evidence) pair generates up to **6 training records** (4 single-call probes + 2 verify_grounding records).

The **prompt format** matches the production prompt for that probe verbatim. For verify_grounding, V6 must reconstruct `GroundedEntity` per agent — running Gilda lookup with the same thresholds used at inference (V6 dry-run validates byte-equality on a holdout-excluded sample).

## §3 Per-probe labeling functions **[REVISED — ≥2 LFs per class]**

Each LF emits a vote ∈ {one of the probe classes} ∪ {ABSTAIN}. ABSTAIN means the LF cannot judge for this record.

LFs are tagged `[clean]` (V0 §4.4 audit-clean — no holdout-tuned features) or `[substrate-tuned]` (consumes substrate code with holdout-derived entries; gated by V7c contamination measurement).

### §3.1 relation_axis (8 classes)

| LF | Tag | Inspects | Vote |
|---|---|---|---|
| LF_curated_db_axis | clean | source_api ∈ {hprd, biopax, signor, bel, trrust} | `direct_sign_match` |
| LF_reach_found_by_axis_match | clean | REACH `annotations.found_by` parses to claim's stmt_type axis+sign | `direct_sign_match` (if matches) or `direct_sign_mismatch` (if axis matches but sign opposite) |
| LF_reach_found_by_axis_mismatch | clean | REACH `found_by` parses to a stmt_type axis ≠ claim's stmt_type axis (e.g., Translocation parse on Activation claim, Binding parse on Phosphorylation claim) | `direct_axis_mismatch` |
| LF_epistemics_direct_true | clean | `epistemics.direct == True` | `direct_sign_match` |
| LF_epistemics_direct_false | clean | `epistemics.direct == False` | `via_mediator` |
| LF_multi_extractor_axis_agreement | clean | ≥2 distinct source_apis converge AND types match | `direct_sign_match` |
| LF_amount_lexical | clean | Claim is Activation/Inhibition AND evidence text contains amount-axis lexicon (`expression`, `abundance`, `level(s)`, `protein levels`, `transcript`, `mRNA`, `upregulat`, `downregulat`) within 50 chars of claim entities | `direct_amount_match` |
| LF_amount_keyword_negative | clean | Claim is IncreaseAmount/DecreaseAmount AND evidence describes activity-axis (`phosphorylat`, `activates`, `binds`) NOT amount | `direct_axis_mismatch` |
| LF_substrate_catalog_match | substrate-tuned | CATALOG verb regex anchored on claim entities matches evidence | `direct_sign_match` |
| LF_substrate_negation_regex | substrate-tuned | Verb-negation pattern (M9/M10) anchored on claim entities | `direct_sign_mismatch` |
| LF_chain_no_terminal | substrate-tuned | M11/M12 detects mediator language (`thereby`, `leads to`, `mediated by`, `via`, `through`) without resolved terminal entity | `via_mediator_partial` |
| LF_chain_with_named_intermediate | clean | Lexical pattern `<claim_subj> {verb} <X>{ thereby|that|which then } {verb} <claim_obj>` with X != claim entities, X is a recognizable biomedical entity | `via_mediator` |
| LF_no_entity_overlap | clean | Neither claim entity (post-Gilda alias resolution) appears in evidence text within token window of ±20 of any verb | `no_relation` |
| LF_partner_dna_lexical | clean | Claim is Complex/Activation with protein-protein implied AND evidence text shows binding to DNA-binding-element lexicon (`promoter`, `enhancer`, `binding site`, `motif`, `consensus sequence`, `DNA-binding`) on claim subject | `direct_partner_mismatch` |
| LF_partner_substrate_gate | substrate-tuned | M11 `_binding_admissible_for(stmt_type)` returns False (substrate already gates partner-mismatched binding patterns) | `direct_partner_mismatch` |

**Class coverage check**:
- `direct_sign_match`: 4 LFs ✓
- `direct_amount_match`: 1 LF (LF_amount_lexical). **Synthetic-oracle augmentation per §10 if natural count <3K**
- `direct_sign_mismatch`: 2 LFs (reach_found_by_axis_match-opposite-sign, substrate_negation_regex) ✓
- `direct_axis_mismatch`: 2 LFs (reach_found_by_axis_mismatch, amount_keyword_negative) ✓
- `direct_partner_mismatch`: 2 LFs (partner_dna_lexical, partner_substrate_gate) ✓
- `via_mediator`: 2 LFs (epistemics_direct_false, chain_with_named_intermediate) ✓
- `via_mediator_partial`: 1 LF (chain_no_terminal). **Synthetic-oracle if <3K**
- `no_relation`: 1 LF (no_entity_overlap). Plus implicit from LabelModel ABSTAIN handling. **Synthetic-oracle if <3K**

### §3.2 subject_role / object_role (5 classes)

| LF | Tag | Inspects | Vote |
|---|---|---|---|
| LF_position_subject | clean | Claim entity matches subject position in source-API extracted statement | `present_as_subject` |
| LF_position_object | clean | Claim entity matches object position | `present_as_object` |
| LF_role_swap_lexical | clean | Lexical pattern `<claim_obj> {active-verb} <claim_subj>` (claim subject is grammatical object of evidence) | `present_as_object` (for the claim subject role) / `present_as_subject` (for object role) |
| LF_substrate_chain_position | substrate-tuned | Substrate detects entity as middle node in path A→entity→B with both A and B in evidence | `present_as_mediator` |
| LF_chain_position_lexical | clean | Lexical: claim entity appears in `via X`, `through X`, `mediated by X`, `X-dependent` patterns, with non-empty A and B context | `present_as_mediator` |
| LF_substrate_decoy | substrate-tuned | Substrate detects entity in unrelated relation pattern (different verb, different partner) | `present_as_decoy` |
| LF_decoy_lexical | clean | Claim entity present in evidence but in a relation pattern that does NOT match claim's stmt_type axis (e.g., entity binding something else when claim is Phosphorylation, entity expression-changing when claim is Activation) | `present_as_decoy` |
| LF_no_grounded_match | clean | Gilda finds no match for entity name (or alias) in evidence text | `absent` |
| LF_absent_alias_check | clean | Per-statement deterministic check: claim entity name + ALL aliases (HGNC + UniProt + commonly-used variants) not found in evidence text via word-boundary regex | `absent` |
| LF_alias_match | clean | Gilda matches entity via alias only (not exact symbol) — used as supplemental signal alongside position LFs to disambiguate role | (votes match position LF: `present_as_subject` or `present_as_object`) |

**Class coverage**:
- `present_as_subject`: 2 LFs (position_subject, role_swap_lexical-inverse) ✓
- `present_as_object`: 2 LFs (position_object, role_swap_lexical) ✓
- `present_as_mediator`: 2 LFs (substrate_chain_position, chain_position_lexical) ✓
- `present_as_decoy`: 2 LFs (substrate_decoy, decoy_lexical) ✓
- `absent`: 2 LFs (no_grounded_match, absent_alias_check) ✓

### §3.3 scope (5 classes)

| LF | Tag | Inspects | Vote |
|---|---|---|---|
| LF_substrate_hedge_marker | substrate-tuned | M10 hedge detector fires (may, might, suggest, propose, hypothesize, putative, …) | `hedged` |
| LF_hedge_lexical | clean | Lexical regex on a fixed open-source hedge lexicon (LingScope-derived list of ~150 hedge cues) anchored on claim verb | `hedged` |
| LF_substrate_negation_explicit | substrate-tuned | M9 negation pattern (no, not, never, fail to, did not, lack of) anchored on claim verb | `negated` |
| LF_negation_lexical | clean | Standalone regex with `(?:no|not|never|failed?\s+to|did\s+not|lack(?:ed)?\s+(?:of|to))` within 8 tokens of claim verb | `negated` |
| LF_conditional_clause_substrate | substrate-tuned | Substrate detects "in <condition>", "wild-type vs mutant", "in the presence of X" pattern | `asserted_with_condition` |
| LF_conditional_lexical | clean | Lexical: `(?:in|under|upon|during|after|with)\s+(?:wild[\s-]type|mutant|presence\s+of|absence\s+of|knockout|knockdown)` | `asserted_with_condition` |
| LF_clean_assertion | clean | No hedge, no negation, no condition match AND claim entities resolved AND verb present | `asserted` |
| LF_low_information_evidence | clean | Evidence < 12 tokens OR matches boilerplate patterns (`This is consistent with`, `Previous studies suggest`, `Future work will`) | `abstain` |
| LF_text_too_short | clean | Evidence < 6 tokens after tokenization (catches cases where extractor pulled a fragment) | `abstain` |

**Class coverage**:
- `asserted`: 1 LF (clean_assertion). **Plus implicit: when no other scope LF fires AND claim is in evidence, default to `asserted` via LabelModel calibration**
- `hedged`: 2 LFs (substrate_hedge, hedge_lexical) ✓
- `asserted_with_condition`: 2 LFs (conditional_substrate, conditional_lexical) ✓
- `negated`: 2 LFs (substrate_negation, negation_lexical) ✓
- `abstain`: 2 LFs (low_information, text_too_short) ✓

### §3.4 verify_grounding (4 classes, per-entity)

| LF | Tag | Inspects | Vote |
|---|---|---|---|
| LF_gilda_exact_symbol | clean | Gilda match with score ≥ 0.95 AND exact symbol (case-folded) appears in evidence | `mentioned` |
| LF_evidence_contains_official_symbol | clean | Entity's HGNC/UniProt official symbol appears literally in evidence (regex word-boundary match) | `mentioned` |
| LF_gilda_alias | clean | Gilda match via known alias (score 0.6-0.95, alias not equal to symbol) | `equivalent` |
| LF_gilda_family_member | clean | Claim entity is a family head (HGNC group) AND evidence names a member of that family | `equivalent` |
| LF_fragment_processed_form | substrate-tuned | Evidence text contains processed-form indicator (`cleaved <X>`, `phosphorylated <X>`, `<X> peptide`, `Aβ` for APP, `<X>-CTD`) for the claim entity. **Note**: lexicon includes holdout-derived entries (Aβ→APP from S-phase audit). Subject to V7c contamination measurement. | `equivalent` |
| LF_gilda_no_match | clean | Gilda returns no match AND entity symbol/aliases not present in evidence | `not_present` |
| LF_evidence_too_short | clean | Evidence < 8 tokens | `uncertain` |
| LF_ambiguous_grounding | clean | Gilda returns multiple matches with similar scores (top-1 - top-2 < 0.1) AND none is the claim entity uniquely | `uncertain` |

**Class coverage**:
- `mentioned`: 2 LFs (gilda_exact_symbol, evidence_contains_official_symbol) ✓
- `equivalent`: 3 LFs (gilda_alias, gilda_family_member, fragment_processed_form) ✓
- `not_present`: 1 LF (gilda_no_match). **Plus implicit when no other LF fires** ✓
- `uncertain`: 2 LFs (evidence_too_short, ambiguous_grounding) ✓

## §4 Aggregation strategy **[REVISED — Snorkel API + class_balance]**

For each (record, probe), aggregate the LF votes into a probabilistic label using **Snorkel's LabelModel** (data-programming aggregator), pinned to `snorkel==0.9.9`:

1. Build the (n_records × n_LFs) label matrix Λ — entries ∈ {ABSTAIN, class_0, …, class_K}.
2. Diagnose LF correlation. Snorkel 0.9.9's `LFAnalysis.lf_summary()` reports Polarity/Coverage/Overlaps/Conflicts but NOT pairwise correlation (verified against snorkel 0.9.9 source). Compute pairwise correlation manually:
   ```python
   import numpy as np
   # Λ: (n_records, n_LFs), -1 = ABSTAIN
   non_abstain = (Λ != -1)
   # For each pair (i,j): compute correlation on records where both fire
   for i, j in itertools.combinations(range(n_LFs), 2):
       both_fire = non_abstain[:, i] & non_abstain[:, j]
       if both_fire.sum() < 50: continue  # insufficient overlap
       corr = np.corrcoef(Λ[both_fire, i], Λ[both_fire, j])[0, 1]
       if abs(corr) > 0.5: ...
   ```
   If any pair has |correlation| > 0.5 on records where both fire (≥50 overlap): merge into a single LF (logical AND/OR semantics) OR drop the redundant member. Also use `lf_summary()`'s Conflicts column as a secondary signal: high Overlap + low Conflict ≈ correlated.
3. Decide `class_balance`: pass an explicit array reflecting either:
   - **Natural distribution** (use V4.5 corpus ratios): preserves prior; LoRA adapter inherits corpus class skew
   - **Equal distribution** (`[1/K]*K`): targets balanced training; LoRA adapter calibrated to balanced output

   **Decision per probe** (V5r commitment, all equal for V8-gate consistency):
   - relation_axis, subject_role, object_role, scope, verify_grounding: **equal distribution** at training time. Per-class V8 accuracy floors require balanced training data; verify_grounding is no exception.
4. Fit `LabelModel(cardinality=K, verbose=True).fit(L_train=Λ, class_balance=[1/K]*K, n_epochs=500, lr=0.01)`. **Do NOT pass Y_dev** — Snorkel 0.9.9's Y_dev is used only to compute class_balance via `_set_class_balance`, redundant when class_balance is explicit. U2 gold validation happens AFTER the fit, in V7a, by computing per-LF accuracy on the holdout-overlap subset against U2 gold (no leakage into LabelModel internals).
5. For each record, output `predict_proba(Λ_record)` → probability vector over the K classes.
6. Hard-label threshold at training time: argmax with probability ≥ 0.5; otherwise drop the record from this probe's training set.

Cardinality 8 (relation_axis) realism: Snorkel's identifiability depends on per-LF accuracy being recoverable from agreement structure. With ≥2 LFs per class (V5r §3.1) and explicit equal class_balance, the identifiability is well-conditioned. V7a measures the resulting per-LF accuracy on U2 gold post-hoc; if any LF's measured accuracy diverges >10pp from `LabelModel.get_weights()`, that's an identifiability failure signal.

## §5 Holdout exclusion (unchanged from V5)

Drop from training any statement whose:
- `source_hash` ∈ holdout source_hashes (~990 records)
- OR `matches_hash` ∈ holdout matches_hashes (~442 records)
- OR `(pmid, entity_pair)` ∈ holdout (paper-level near-dupes, ~1,009 records)

`pmid`-only overlap (8,816 records) is NOT excluded.

## §6 Sample size targets **[REVISED — verify_grounding multiplier + class_balance]**

| Probe | K | Min-class target | Total target | Notes |
|---|---|---|---|---|
| subject_role | 5 | 3K | 15K | balanced via class_balance + stratified subsample |
| object_role | 5 | 3K | 15K | (same) |
| relation_axis | 8 | 3K | 24K | balanced |
| scope | 5 | 3K | 15K | balanced |
| **verify_grounding** | 4 | 3K | **28K** | per-(stmt,ev,entity); ~2.3× multiplier from average claim-agent count |

Total training records (with overlap across probes): ~80-100K unique (statement, evidence) pairs.

If Snorkel produces severe imbalance for a probe, AND class_balance was set to equal: stratified subsample at the JSONL emission step (V6d) to enforce min-3K per class.

Rare-class augmentation: classes with <1K natural records get **synthetic-oracle generation** of 100-300 placeholder examples per class (down-sized from the original 200-500 per class to budget the curator effort). Generation method: **LLM-assisted templating** — define ~10 template structures per class (e.g., for `direct_partner_mismatch`: "{ENT_A} binds the {DNA_ELEMENT} of {GENE_X}", with ~20 synthetic placeholder ENT_A names and 5 DNA_ELEMENT variants → 1000 records per template). The templating rules + placeholder lists are committed to `data/v_phase/synthetic_oracles/{probe}_{class}.yaml`; V6d generates from templates deterministically. Synthetic examples are tagged `synthetic=true` in the training record and excluded from contamination filter (since they have no holdout-overlap risk by construction).

## §7 V7 hand-validation protocol **[REVISED — sample size + stratification + U2 gold]**

Three components (V7a + V7b + V7c subdivision in task graph):

### V7a — LF accuracy on U2 per-probe gold (zero labeling cost)

For each LF, compute accuracy on the 482-record holdout's U2-derived per-probe gold (`data/benchmark/probe_gold_holdout.jsonl`). This is ZERO-COST validation since the labels already exist.

For each LF: compare LF vote vs U2 gold class on records where LF fires (non-ABSTAIN). Report accuracy + n_fires + 95% CI.

Compare LF accuracy to the LabelModel's learned weight (`label_model.get_weights()`). Disagreement >10pp on any LF = identifiability collapse signal.

### V7b — hand-validate ≥250 records stratified

Per probe: sample **50 records** from training set, stratified:
- **50% rare-class** (5 records from each of the rarest classes; for relation_axis spread across 5 rarest of 8 classes)
- **30% borderline** (proba 0.5-0.7) — 15 records per probe
- **20% confident** (proba > 0.9) — 10 records per probe

Total: 50 × 5 = **250 hand-labels**.

V7b GATE thresholds (per probe), computed on **POOLED 50 records** (not per-bucket — bucket-level CI at N=10/15 is too noisy):
- **Pooled LF accuracy**: Wilson 95% lower CI bound ≥ 60% (at N=50, p=0.75 → Wilson lower ≈ 0.62, gates pass; p=0.85 → Wilson lower ≈ 0.74, comfortable)
- **No class point-estimate accuracy < 30%** — bucket-level point estimates without CI requirement on tiny buckets
- **No class entirely missing from training set** — verified by counting non-zero-record classes in V6c output

V7a's U2-gold validation (N=482 records with per-probe gold) provides the high-power LF-accuracy signal. V7b is a sanity check that the LFs generalize beyond the holdout-overlap region. The two together (high-power U2-gold + lower-power hand-labeled) form the complete LF accuracy picture.

### V7c — substrate-contamination delta (Path B for Finding D)

Per V6a, run all `[substrate-tuned]` LFs in **two modes** against the holdout-overlap subset:
- **tuned mode**: current substrate code (post-S/T/U-phase tuning) as it lives on disk
- **baseline mode**: substrate with the following holdout-derived constants explicitly emptied/reset:

  | File | Symbol | Baseline state |
  |---|---|---|
  | `context_builder.py` | `_CYTOKINE_LIGAND_HGNC` | `frozenset()` (empty) |
  | `context_builder.py` | `_SITE_DENYLIST` | `frozenset()` (empty) |
  | `context_builder.py` | `_HEDGE_MARKERS` | restrict to ~5 universal cues: `{"may", "might", "suggest", "propose", "hypothesize"}` (drop diagnosis-fix additions) |
  | `context_builder.py` | `_LOF_PATTERNS` | empty list (drop M9 perturbation patterns) |
  | `context_builder.py` | M13 60-char window for hedging | revert to a default 30-char window (no diagnosis-tuned size) |
  | `relation_patterns.py` | CATALOG `RelationPattern` entries | retain only entries whose docstring does NOT cite a "Diagnosis FN" / "Diagnosis FP" / specific holdout statement reference; drop everything else |
  | `relation_patterns.py` | M11 `_binding_admissible_for(stmt_type)` | retain core gate (binding requires Complex/Phosphorylation/Methylation), drop refinements citing diagnosis cases |
  
  V6a implements baseline mode as a context-manager / monkey-patch that swaps these constants at import time, runs the LFs, restores the original state. The baseline state is REPRODUCIBLE FROM CODE, not from a non-existent git commit. The doctrine commits to this enumeration as the canonical "M-baseline".

Compute per-class **vote-share delta** (the LF's per-class vote distribution: `count(LF_votes_for_class) / count(LF_fires)`). NOT accuracy delta — vote-share captures the LF's selective firing pattern shift.

Verdict per LF, per class:
- **All classes Δ < 10pp**: that LF ACCEPTED with documented contamination floor
- **Any class Δ ≥ 10pp**: that LF must be REPLACED with from-scratch lexical version in V6b before V8 passes (the holdout-overlap subset is the audit slice; baseline-mode votes are what generalizes; tuned-mode votes are what's optimistic)

V7c output: `research/v7c_substrate_contamination.md` with per-LF, per-class vote-share delta table + verdict.

## §8 Output format for training (unchanged from V5)

```jsonl
{
  "messages": [
    {"role": "system", "content": "<probe-specific system prompt verbatim from production>"},
    {"role": "user", "content": "<probe-specific user prompt with claim + evidence>"}
  ],
  "completion": "{\"answer\": \"<class>\", \"rationale\": \"<optional, can be empty>\"}",
  "soft_labels": {"class_0": 0.05, "class_1": 0.85, ...},
  "synthetic": false
}
```

## §9 Implementation sequence **[REVISED — V6 subdivided]**

V6a: Implement substrate-LFs in dual mode (tuned + baseline); V6b: implement clean LFs; V6c: build Λ matrix + LFAnalysis + LabelModel.fit (per probe, with Y_dev + class_balance); V6d: stratified subsample + JSONL emission + contamination filter.

V7a: LF accuracy on U2 gold (zero cost); V7b: 250 hand-labels; V7c: substrate contamination delta.

V8: gate against all V7 components plus environment health.

## §10 Risks specific to data prep **[REVISED — substrate-tuning audit]**

| Risk | Severity | Mitigation |
|---|---|---|
| Snorkel LabelModel underfits with single-LF classes | resolved | V5r §3 ensures ≥2 LFs per class for all but 3 classes; those 3 use synthetic-oracle augmentation |
| Confident class minority (rare classes <1K natural records) | medium | V5r §6 explicit synthetic-oracle generation of 200-500 placeholder examples per rare class, tagged `synthetic=true` |
| Highly correlated LFs vote together → over-confidence | medium | V5r §4 step 2: `LFAnalysis.lf_summary()` diagnoses correlation > 0.5; remediation by LF merge or drop, NOT a fictional `dependencies=` parameter |
| Substrate LFs consume holdout-tuned code | high | **V7c quantifies delta vs baseline-substrate; per-class Δ ≥ 10pp triggers LF replacement** (Path B per V5 gate Finding D) |
| Training data contamination via paraphrase from holdout | high | V5r §10 contamination filter: trigram (n=3) Jaccard overlap, lowercase + punctuation stripped + biomedical-stopword list, threshold = 95th percentile of natural corpus pairwise overlap (computed from a 10K random-pair sample). Any record exceeding threshold against ANY holdout evidence dropped. |
| LF accuracy estimates are over-optimistic on holdout-overlap records | medium | V5r §11 explicit Y_dev = U2-gold subset of holdout that is NOT used for L_train (V5r §5 holdout exclusion ensures these records aren't in training corpus); Y_dev anchors LabelModel without leaking labels into LoRA training |
| LF accuracy validation thresholds are statistical noise | resolved | V5r §7 V7b at N=50/probe gives 95% CI half-width ≤ 14pp; thresholds use lower CI bounds, not point estimates |

## §11 U2 per-probe gold use (NEW SECTION) **[REVISED — Y_dev path dropped]**

`data/benchmark/probe_gold_holdout.jsonl` carries U2-derived per-probe labels for the 482 holdout records. Per V0r §4.4 ("strict: do NOT use INDRA's belief score") and the broader "hold the holdout for evaluation" commitment:

- **PERMITTED**: Use U2 gold for V7a LF accuracy measurement, post-hoc, after `LabelModel.fit()` returns. This is the high-power signal feeding the V8 gate's per-LF accuracy check. Zero training-set cost.
- **PERMITTED**: Cross-validate `LabelModel.get_weights()` (Snorkel's learned per-LF accuracy) against the V7a U2-gold accuracy. Disagreement >10pp on any LF flags identifiability collapse.
- **FORBIDDEN**: Use U2 gold as a label source for any record entering V6d's training JSONL files.
- **FORBIDDEN**: Pass U2 gold via `Y_dev` argument to `LabelModel.fit()`. **Reason**: per Snorkel 0.9.9 source, Y_dev is used only to compute `class_balance` via `_set_class_balance`; it does not anchor per-LF mu-parameter optimization. Since V5r §4 step 3 commits explicit class_balance values, Y_dev is redundant. Worse, passing Y_dev would create a hidden filtering path: U2 gold → class_balance → predict_proba → 0.5 threshold → which records survive. We avoid that path entirely by NOT passing Y_dev.
- **FORBIDDEN**: Use U2 gold to select/filter training records (e.g., "drop records where U2 disagrees with LabelModel").

This satisfies V0r §4.4 (no curator gold leaks into training) while still using U2 gold for the high-power V7a evaluation. The LabelModel is fit on Λ alone; U2 gold is consumed only post-hoc.

## §12 Contamination guard specification **[REVISED — adopt existing + add trigram]**

V5r commits to a **two-filter** approach: drop a training record if EITHER filter flags it. The first filter is the existing well-tested algorithm; the second is a complementary fuzzy-paraphrase detector.

**Filter 1: existing `scripts/check_contamination.py` algorithm** (preserves working behavior, CI-guarded, validated by 2026-05-02 paraphrase incident catch):
- `_norm`: collapse whitespace, strip trailing punctuation, casefold (NO stopword removal)
- Checks: exact match, substring containment, paraphrase_overlap (sliding 50-char window with 5-char stride), pair match
- Behavior unchanged from current production. V6d invokes this script as a library function.

**Filter 2: trigram-Jaccard fuzzy-paraphrase** (new; complements Filter 1's substring focus):
- **n=3** trigram
- **Normalization**: lowercase, strip punctuation, drop English stopwords + biomedical-stopword list (PubMed-derived top-50 by frequency: `cell`, `cells`, `protein`, `proteins`, `expression`, `level`, `study`, `result`, `data`, `figure`, `analysis`, `using`, `shown`, `observed`, `effect`, `effects`, etc.)
- **Overlap measure**: Jaccard similarity on trigram set
- **Threshold**: derived from a calibration step in V6d — sample 10,000 random pairs of evidence texts from the corpus, compute trigram-Jaccard distribution, set threshold at the **95th percentile of natural overlap**. Any training record exceeding this against any holdout evidence is dropped.

A training record passes only if both filters return "not contaminated". This is conservative-failing (more drops than either filter alone) but preserves the working substring-detection capability while adding paraphrase-fuzzy detection. The algorithms are orthogonal: Filter 1 catches verbatim quotes and prefix/suffix overlap; Filter 2 catches re-worded passages with different surface syntax but shared content trigrams.

V6d's contamination-filter step:
1. Compute Filter 1 once via `scripts/check_contamination.py` library call against all holdout records.
2. Compute Filter 2's threshold via 10K-pair calibration (one-time per V6d invocation).
3. Apply Filter 2 to each training record against all holdout records.
4. Drop record if Filter 1 flags OR Filter 2 flags.

## §13 What V5r explicitly does NOT specify

- Exact Snorkel LabelModel hyperparameters beyond `cardinality, n_epochs=500, lr=0.01, class_balance=…` — V6c is empirical
- Exact prompt wording for training records — copies production prompts as-of V0r commit
- Tokenizer settings — uses Gemma 4 E4B tokenizer
- LoRA training hyperparameters — those belong in V9 not V5r
- Active learning strategy beyond V7's sample-and-validate

## §14 V5r ship criteria (this revised doctrine becomes binding when:)

1. The 5-probe label space is committed (§1) ✓
2. ≥2 LFs per class for non-trivial classes (§3) ✓ (rare-class exceptions covered by §6 synthetic-oracle)
3. Aggregation method is fixed: Snorkel 0.9.9 LabelModel with explicit class_balance and Y_dev (§4)
4. Holdout exclusion is unambiguous (§5) ✓ (unchanged)
5. V8 thresholds are CI-based and testable (§7)
6. Substrate-tuning quantification gate (V7c) is specified (§7c)
7. Output schema is a JSONL contract (§8) ✓ (unchanged + `synthetic` flag)
8. U2 gold use is explicit (§11)
9. Contamination guard is parametrized (§12)

V5g (re-gate) verifies items 2-9 against the gate findings before V6 unblocks.

## §15 Revision history (vs V5)

- **2026-05-06 V5**: original doctrine.
- **2026-05-06 V5 gate**: FAIL with 4 BLOCKING (LF coverage, Snorkel API, substrate-tuning, verify_grounding shape) + 5 MAJOR (class_balance, contamination spec, V8 noise, U2 gold ignored, §1-§3 LF mismatch).
- **2026-05-06 V5r initial draft**: addressed all 8 fixes. Most consequential: §3 rewritten for ≥2 LFs per class, §4 corrected Snorkel API + class_balance, §7 split into V7a/V7b/V7c with U2 gold + substrate-tuning quantification, §11 explicit U2 gold use clause, §12 contamination guard parametrized.
- **2026-05-06 V5g re-gate**: CONDITIONAL with 2 BLOCKING (Snorkel `lf_summary()` doesn't compute correlation, M5-baseline not git-recoverable) + 4 MAJOR (Y_dev semantics, Y_dev/L_train alignment, hidden filtering path, contamination filter direction) + 3 MINOR.
- **2026-05-06 V5r post-gate fixes (this file)**: corrected §4 step 2 to use np.corrcoef; §4 step 4 dropped Y_dev path (resolves Issues 2/3/4 by using only explicit class_balance); §7c enumerated baseline-substrate features explicitly (table of constants to reset); §7b changed to pooled CI with per-class point-estimate floors (resolves Wilson math at small bucket N); §11 forbade Y_dev with reasoning; §12 adopted two-filter approach (existing script + trigram-Jaccard); §6 committed to LLM-templating for synthetic oracles with 100-300 budget; §3.4 retagged LF_fragment_processed_form as substrate-tuned; §3.2 promoted LF_absent_alias_check to named row.
