# V5 — data prep doctrine: per-probe label derivation

Date: 2026-05-06
Status: **SUPERSEDED by V5r** (research/v5r_data_prep_doctrine.md) after brutalist gate FAIL — see research/v5_gate.md. Original kept for diff reference.
Predecessor: V4.5 corpus inventory (PASS); V0r doctrine revision (E4B base)
Purpose: specify how training data for each of the five probe LoRA adapters is derived from the 894K-statement INDRA corpus using multi-source weak supervision, with no use of the existing belief score.

## §1 The five probes and their label spaces

Per `src/indra_belief/scorers/probes/types.py` and `grounding.py`:

| Probe | Classes | Count |
|---|---|---|
| **subject_role** | `present_as_subject`, `present_as_object`, `present_as_mediator`, `present_as_decoy`, `absent` | 5 |
| **object_role** | (same as subject_role) | 5 |
| **relation_axis** | `direct_sign_match`, `direct_amount_match`, `direct_sign_mismatch`, `direct_axis_mismatch`, `direct_partner_mismatch`, `via_mediator`, `via_mediator_partial`, `no_relation` | 8 |
| **scope** | `asserted`, `hedged`, `asserted_with_condition`, `negated`, `abstain` | 5 |
| **verify_grounding** | `mentioned`, `equivalent`, `not_present`, `uncertain` | 4 |

Each LoRA adapter is a closed-set classifier over its probe's label space.

## §2 Training-record schema

Each training record is a (claim, evidence_text, probe_kind, label_distribution) tuple where:

- `claim` = an INDRA Statement: `(subject_entity, object_entity, relation_type, …)` rendered as a JSON-shaped instruction prompt
- `evidence_text` = a single Evidence's `text` field (≤512 tokens after tokenization)
- `probe_kind` ∈ {subject_role, object_role, relation_axis, scope, verify_grounding}
- `label_distribution` = probabilistic label per class from Snorkel LabelModel (NOT a hard label; capturing aggregation uncertainty)

Each (statement, evidence) pair generates up to 5 training records — one per probe — though probes that have no firing LFs for a given pair are dropped from the training set for that probe.

The **prompt format** matches the production prompt for that probe verbatim, so the LoRA adapter trains exactly the format it serves at inference. The training target is the JSON answer corresponding to the argmax-probability class, with optional confidence-distillation (see §6).

## §3 Per-probe labeling functions

Drawing from V4.5's signal inventory. Each LF emits a vote ∈ {one of the probe classes} ∪ {ABSTAIN}. ABSTAIN means the LF cannot judge for this record.

### §3.1 relation_axis (highest priority — V9 feasibility check)

| LF | What it inspects | Vote |
|---|---|---|
| LF_curated_db_axis | source_api ∈ {hprd, biopax, signor, bel, trrust} | `direct_sign_match` (curated DBs are direct-relation by convention) |
| LF_reach_found_by_axis | REACH `annotations.found_by` pattern parser | parse pattern ID → axis+sign vote (Positive_activation_*→`direct_sign_match`, Negative_activation_*→`direct_sign_mismatch`, Acetylation_*→`direct_sign_match`, Translocation→`direct_axis_mismatch`, etc.) |
| LF_epistemics_direct_true | `epistemics.direct == True` | `direct_sign_match` (high precision) |
| LF_epistemics_direct_false | `epistemics.direct == False` | `via_mediator` (extractor-flagged indirection) |
| LF_multi_extractor_agreement | ≥2 distinct source_apis converge | `direct_sign_match` (only when statement type matches) |
| LF_substrate_catalog_match | Statement's relation type + entity pair matches CATALOG verb regex on evidence text | `direct_sign_match` |
| LF_substrate_negation_regex | Verb-negation pattern anchored to claim entities | `direct_sign_mismatch` |
| LF_chain_no_terminal | Substrate sees mediator language but no resolved terminal entity | `via_mediator_partial` |
| LF_no_entity_overlap | Neither claim entity appears in evidence text post-Gilda | `no_relation` |

Notes:
- `direct_amount_match` (U7 closed-set) gets its own LF: `LF_amount_lexical` matches "expression", "abundance", "level(s)", "transcript" patterns paired with claim relation Activation/Inhibition AND the evidence describes amount change.
- `direct_partner_mismatch` requires DNA-binding vs Complex/protein detection — substrate-driven LF using M11 signal.

### §3.2 subject_role / object_role

| LF | What it inspects | Vote |
|---|---|---|
| LF_position_in_extractor | Claim entity matches subject position of source-API's extracted statement | `present_as_subject` |
| LF_position_swap | Claim entity matches object position | `present_as_object` |
| LF_substrate_chain_position | Substrate detects the entity as middle node in a path A→entity→B | `present_as_mediator` |
| LF_substrate_decoy | Entity present in evidence but in unrelated relation pattern | `present_as_decoy` |
| LF_no_grounded_match | Gilda finds no match for the entity in evidence text | `absent` |
| LF_alias_match_only | Gilda matches via alias (not exact symbol) | `present_as_subject` or `present_as_object` (per position) |

### §3.3 scope

| LF | What it inspects | Vote |
|---|---|---|
| LF_substrate_hedge_marker | M10 hedge detector fires (may, might, suggest, propose, hypothesize, …) | `hedged` |
| LF_substrate_negation_explicit | Explicit negation regex (no, not, never, fail to, did not, …) anchored to claim verb | `negated` |
| LF_conditional_clause | Substrate detects "in <condition>" or "of mutant/wild-type" pattern paired with claim relation | `asserted_with_condition` |
| LF_clean_assertion | No hedge, no negation, no condition, claim entities present | `asserted` |
| LF_indecisive_text | Evidence too short or generic ("This is consistent with …") | `abstain` |

### §3.4 verify_grounding

| LF | What it inspects | Vote |
|---|---|---|
| LF_gilda_exact_symbol | Claim entity matches evidence exact symbol (case-folded) | `mentioned` |
| LF_gilda_alias | Gilda alias match | `equivalent` |
| LF_gilda_family_member | Claim is a family head and evidence names a member (per HGNC group hierarchy) | `equivalent` |
| LF_gilda_no_match | Gilda returns nothing | `not_present` |
| LF_evidence_too_short | Evidence < 10 tokens or boilerplate | `uncertain` |

## §4 Aggregation strategy

For each (record, probe), aggregate the LF votes into a probabilistic label using **Snorkel's LabelModel** (the standard data-programming aggregator):

1. Build the (n_records × n_LFs) label matrix Λ — entries ∈ {ABSTAIN, class_0, …, class_K}.
2. Fit `LabelModel(cardinality=K).fit(L_train=Λ, n_epochs=500, lr=0.01)` — learns per-LF accuracy and per-pair correlation purely from agreement structure (no gold labels needed).
3. For each record, output `predict_proba(Λ_record)` → probability vector over the K classes.
4. Threshold for hard labels at training time: argmax with probability ≥ 0.5; otherwise drop the record from this probe's training set (insufficient agreement).

Training records below the agreement threshold are dropped, not voted-up to ABSTAIN. We want the LoRA adapter to learn from confidently-aggregated examples, not from low-confidence ones (those become V7 hand-validation candidates instead).

## §5 Holdout exclusion (extends V0 doctrine §7)

Drop from training any statement whose:
- `source_hash` ∈ holdout source_hashes (~990 records)
- OR `matches_hash` ∈ holdout matches_hashes (~442 records)
- OR `(pmid, entity_pair)` ∈ holdout (paper-level near-dupes, ~1,009 records)

`pmid`-only overlap (8,816 records) is NOT excluded.

Implementation: build a Python set of forbidden hashes/keys at the start of V6 derivation; `.discard()` matching records during the corpus walk before label aggregation.

## §6 Sample size targets

Empirical floor for fine-tuning a closed-set classifier of K classes at LoRA rank 16: ~3K examples per class × min-class. For our probes:

| Probe | K | Min-class target | Total target |
|---|---|---|---|
| subject_role | 5 | 3K | 15K |
| object_role | 5 | 3K | 15K |
| relation_axis | 8 | 3K | 24K |
| scope | 5 | 3K | 15K |
| verify_grounding | 4 | 3K | 12K |

Total training records (with overlap across probes): ~80K unique (statement, evidence) pairs.

If the LF aggregation produces severe class imbalance (e.g., relation_axis with 90% `direct_sign_match`), use **stratified subsampling** to enforce min-3K per class. The held-out validation set is a stratified sample of 1K records per probe, drawn from the post-aggregation pool BEFORE class balancing.

The corpus has 894K statements × ~3.2 evidences/stmt ≈ 2.9M (statement, evidence) pairs. We subsample to ~80K, far below the supply. No data scarcity risk.

## §7 V7 hand-validation protocol

Sample 50-100 (record, probe, predicted_class) tuples from the post-aggregation training set, weighted across:
- 30% from confident (proba > 0.9) predictions — verify the easy case is correct
- 50% from borderline (0.5 ≤ proba ≤ 0.7) predictions — verify the LabelModel disagreement-handling is sane
- 20% from class-rare predictions (e.g., `direct_partner_mismatch`, `via_mediator_partial`) — verify rare classes aren't artifacts

Hand-label by the user (or a careful inspection by the agent against U2's per-probe gold). Compute LF accuracy (predicted-class == hand-labeled-class).

**V8 GATE thresholds:**
- Per-probe LF accuracy ≥ 75% on confident predictions
- Per-probe LF accuracy ≥ 50% on borderline predictions
- No class with < 30% accuracy
- No class entirely missing from training set

If any threshold fails: iterate LF set, re-run aggregation, re-validate. Do NOT proceed to V9 until V8 passes.

## §8 Output format for training

```jsonl
{
  "messages": [
    {"role": "system", "content": "<probe-specific system prompt verbatim from production>"},
    {"role": "user", "content": "<probe-specific user prompt with claim + evidence>"}
  ],
  "completion": "{\"answer\": \"<class>\", \"rationale\": \"<optional, can be empty>\"}",
  "soft_labels": {"class_0": 0.05, "class_1": 0.85, ...}  // for distillation loss option
}
```

The `completion` is the hard target; `soft_labels` enables KL-distillation if we want the LoRA to learn the LabelModel's full distribution (better calibration). Both are emitted; V9 chooses one.

## §9 Implementation sequence

V6 (build derivation script):
1. Load corpus; apply holdout exclusion → filtered statement list
2. For each (statement, evidence) pair, compute all LF votes (parallel via multiprocessing)
3. Per probe: build Λ matrix, fit LabelModel, output probabilistic labels
4. Stratified subsample to per-probe targets
5. Format as JSONL training files: `data/v_phase/train/{probe}.jsonl` + `val/{probe}.jsonl`

V7 (hand-validate):
1. Sample per §7 protocol; render probe-prompt + hand-label form
2. User labels; compute accuracy metrics
3. If thresholds met: V8 PASS → V9 unblocks
4. Else: identify failing LFs, iterate

## §10 Risks specific to data prep

| Risk | Severity | Mitigation |
|---|---|---|
| Snorkel LabelModel underfits with small LF count per probe | medium | min 4 LFs per probe; verify pairwise agreement covariance is non-singular |
| Confident class minority (e.g., `direct_partner_mismatch` < 1K natural records) | medium | targeted oracle generation: hand-craft 200-500 synthetic examples for ultra-rare classes (with placeholder names per contamination guard) |
| LF correlation underestimated (highly correlated LFs vote together → over-confidence) | medium | LabelModel models pairwise dependencies; declare known-correlated pairs explicitly via `dependencies=` argument |
| Substrate LF uses code paths that themselves consume the belief score | high | audit V6 implementation: each LF must be traceable to V4.5 inventory signals; reject any LF that touches `belief` field |
| Training data contamination via paraphrase from holdout | high | V6 includes a contamination check pass: any training record whose source text shares ≥30% n-gram overlap with a holdout record's evidence is dropped |

## §11 What V5 explicitly does NOT specify

- Exact Snorkel LabelModel hyperparameters (n_epochs, lr, dependencies) — V6 is empirical
- Exact prompt wording for training records — copies production prompts as-of V0r commit
- Tokenizer settings — uses Gemma 4 E4B tokenizer (vocab 262144, BOS=2, EOS=1)
- LoRA training hyperparameters — those belong in V9 not V5
- Active learning strategy beyond V7's sample-and-validate — that would be a future V-phase enhancement

## §12 V5 ship criteria (this doctrine becomes binding when:)

1. The 5-probe label space is committed (§1).
2. Per-probe LF set is named (§3) — at least 4 LFs per probe with non-singular agreement structure.
3. Aggregation method is fixed (§4) — Snorkel LabelModel.
4. Holdout exclusion is unambiguous (§5).
5. V8 thresholds are testable (§7).
6. Output schema is a JSONL contract (§8).
7. No LF anywhere consults the existing belief score (§10 audit).

This doctrine is in scope for V1-style brutalist review before V6 implementation begins.
