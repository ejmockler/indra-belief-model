# V4.5 — INDRA corpus signal inventory

Date: 2026-05-05
Corpus: `data/benchmark/indra_benchmark_corpus.json.gz` — 894,939 Statements, 2,847,196 evidence pieces.
Purpose: enumerate the INDEPENDENT signals available for V-phase weak-supervision labeling, EXCLUDING `belief` (deprecated per V0 doctrine — V-phase is its replacement).

## Headline counts

| Signal | Statements | Coverage |
|---|---|---|
| Total | 894,939 | 100.00% |
| Has curated-DB source | 186,338 | **20.82%** |
| Curated-DB ONLY (no text-mining) | 160,731 | **17.96%** ← strongest gold anchor |
| Curated + text-mining (both) | 25,607 | 2.86% |
| Multi-extractor agreement (≥2 source_apis) | 155,890 | **17.42%** |
| Multi-extractor INCLUDING curated | 30,854 | 3.45% |
| Multi-paper convergence (≥3 distinct pmids) | 87,545 | **9.79%** |
| Has `epistemics.direct` annotation | (1.54M evidence) | 54% of evidence |
| Has REACH `found_by` pattern | (1.29M evidence) | 45% of evidence |
| Has `supported_by` non-empty | 340,144 | 38.01% |

## (a) Source API distribution

### Evidence-level (2.85M pieces)

| source_api | count | % | Class |
|---|---|---|---|
| reach | 973,191 | 34.18% | text-mining |
| sparser | 904,119 | 31.75% | text-mining |
| medscan | 427,886 | 15.03% | text-mining |
| **hprd** | 198,154 | 6.96% | **curated DB** |
| rlimsp | 148,014 | 5.20% | text-mining |
| **biopax** | 71,245 | 2.50% | **curated DB** (Reactome) |
| trips | 45,602 | 1.60% | text-mining |
| **signor** | 44,069 | 1.55% | **curated DB** |
| **bel** | 24,713 | 0.87% | **curated DB** |
| **trrust** | 6,120 | 0.21% | **curated DB** |
| isi | 4,083 | 0.14% | text-mining |

**Total curated DB evidence: 344,301 (12.09%). Total text-mining: 2,502,895 (87.91%).**

Classification rationale:
- `hprd` (Human Protein Reference Database) — manually curated PPI database from literature
- `biopax` — pathway databases (Reactome) imported via BioPAX format
- `signor` — SIGNOR signaling network, expert-curated
- `bel` — Biological Expression Language, curated assertions
- `trrust` — Transcriptional Regulatory Relationships database, curated

These five constitute our **gold anchor** — assertions validated by domain experts, not extracted from text. They're the cleanest source for LF_curated_db_source.

### Statement-level multi-source structure

| # distinct source_apis per statement | # statements | % |
|---|---|---|
| 1 (single extractor) | 739,049 | 82.58% |
| 2 | 113,589 | 12.69% |
| 3 | 30,414 | 3.40% |
| 4 | 8,587 | 0.96% |
| 5 | 2,416 | 0.27% |
| 6+ | 884 | 0.10% |

**17.42% (156K) of statements have multi-extractor agreement** — independent labeling signal.

## (b) Curator-correction annotations

**None found at statement level.** The corpus does NOT contain a `curator_label`, `is_correct`, or similar field that would record human-validated correctness on individual statements.

Implication: our gold anchors are
1. **Curated-DB-sourced statements** (186K, indirect curator validation via database import)
2. **The 482-record holdout** (direct curator labels, MUST stay held out from training)

This means we cannot bootstrap from per-statement curator interventions. Weak supervision via independent signals is the only feasible path.

## (c) Extractor-internal signals (independent of belief)

Two underused signals that are NOT belief-derived:

### `epistemics.direct` (extractor's own direct/indirect annotation)

- 1,544,436 evidence pieces have this field set (54% of all evidence)
- Of those, 349,784 are `direct=True` (22.6%)
- This is the EXTRACTOR's claim that the relation is direct contact (not chained)
- Strong LF for relation_axis: `direct=True` → vote `direct_sign_match`; `direct=False` → vote `via_mediator`

### REACH `annotations.found_by` patterns

- 1,287,126 evidence pieces have a `found_by` pattern ID
- Patterns explicitly carry axis/sign:
  - `Positive_activation_syntax_1_verb` (39K)
  - `Negative_activation_syntax_1_verb` (7K)
  - `Acetylation_syntax_1a_noun` (1K)
  - etc.
- Pattern ID parsing gives us extractor-level (axis, sign) at near-perfect precision for the patterns that fired
- **Strong LF for relation_axis**: parse `found_by` to vote axis+sign

These two signals together cover ~50-60% of evidence with extractor-derived axis/sign labels INDEPENDENT of belief.

### Note on hedgings/negated fields

INDRA's `epistemics` schema includes `hedgings` and `negated` fields but they are **all empty in this corpus** (0 evidence pieces have either populated). This means we cannot use extractor hedge/negation directly for the scope probe — must rely on M10's substrate hedge detector + new regex for explicit negation.

## (d) Multi-paper convergence (matches_hash → distinct pmids)

| # papers per statement | # statements | cumulative % |
|---|---|---|
| 1 | 686,228 | 77.88% |
| 2 | 97,654 | 88.96% |
| 3 | 36,054 | 93.06% |
| 4 | 17,926 | 95.09% |
| 5 | 10,730 | 96.31% |
| 6-10 | 17,688 | 98.54% |
| >10 | 12,847 | 100.00% |

**22% of statements have evidence in ≥2 distinct papers; 9.79% in ≥3 papers.** Multi-paper convergence is independent of single-paper extractor behavior — a strong reliability signal.

The long tail (~12.8K statements with >10 paper convergence) are the most-corroborated assertions in the corpus.

## (e) Knowledge-graph structure: `supported_by` / `supports`

- 340,144 statements (38%) have non-empty `supported_by` (other statements that corroborate)
- 122,903 (13.7%) have non-empty `supports` (reverse direction)

`supported_by` lists are produced by INDRA's preassembler grouping — statements that share `pa_hash`/`matches_hash` AND have compatible structure are linked. This is **structural identity grouping**, not belief assembly — usable as an independent signal of "this statement has corroborating equivalents in the corpus."

LF_supported_by: a statement with ≥3 supported_by entries has structural redundancy in the corpus → vote with proportional confidence.

## (f) Holdout overlap (for V5 exclusion logic)

| Overlap criterion | Statements to exclude |
|---|---|
| source_hash ∈ holdout (482 source_hashes) | **990** records |
| matches_hash ∈ holdout (442 unique) | **442** records |
| pmid + entity-pair ∈ holdout (paper-level near-dupes) | **1,009** records |
| pmid only ∈ holdout (looser) | 8,816 records |

After de-duplicating the union of the first three criteria: approximately **2,000-2,500 records** to exclude (~0.3% of corpus). The pmid-only filter is too aggressive (excludes 1% of corpus, drops genuinely independent statements that just happen to share a paper).

**V5 holdout-exclusion policy:** drop statements meeting ANY of (source_hash, matches_hash, pmid+entity_pair) overlap. Keep the rest, including pmid-only co-occurrences.

## Summary: signals available for weak-supervision LFs

Ranked by trust + coverage:

| LF | Coverage | Independent of belief? | Trust |
|---|---|---|---|
| LF_curated_db_source (HPRD/biopax/SIGNOR/BEL/TRRUST) | 186K stmts | ✓ | very high |
| LF_multi_extractor_agreement (≥2 source_apis) | 156K stmts | ✓ | high |
| LF_multi_paper_convergence (≥3 pmids) | 88K stmts | ✓ | high |
| LF_epistemics_direct (`direct=True/False`) | 1.54M evidence | ✓ | high (extractor-direct) |
| LF_reach_found_by (pattern → axis/sign) | 1.29M evidence | ✓ | very high (high-precision per pattern) |
| LF_supported_by (≥3 corroborators) | (subset of above) | ✓ | medium |
| LF_substrate_catalog_match | (substrate-determined) | ✓ | high |
| LF_substrate_negation | (substrate-determined) | ✓ | medium-high |
| LF_substrate_hedge_marker (M10) | (substrate-determined) | ✓ | medium |
| LF_alias_grounding_clean (Gilda) | (substrate-determined) | ✓ | high |

**Combined high-trust pool** (any of: curated, multi-extractor, ≥3 papers, multi-LF agreement): conservatively **300-400K statements**. Far more than needed for QLoRA training (30-50K per probe).

## What this means for V5/V6 design

1. **Curated-DB anchor is real and large** (186K). Use it as Snorkel's near-gold anchor — the LF generative model will calibrate other LFs against this.

2. **`found_by` pattern parsing is unexpectedly powerful.** REACH/Sparser tag each extraction with their internal pattern ID, which encodes axis+sign at high precision. We've never used this. Should be the primary labeling function for relation_axis.

3. **`epistemics.direct` is the cleanest signal for `via_mediator` vs `direct_sign_match`** — extractor's own assessment of indirection. Should be a primary LF for relation_axis.

4. **No curator gold at scale.** The 482-record holdout is the only direct curator validation. We MUST hold it out from training and only use it for V8 derivation-validation cross-check + V22 ship calibration metric.

5. **Holdout exclusion is cheap** (~2.5K records, 0.3% of corpus). Not a budget concern.

6. **`hedgings` and `negated` epistemics fields are empty.** Scope probe's hedged/negated classes must derive from M10 substrate + new regex, NOT from extractor metadata.

7. **`supported_by` knowledge-graph structure** (340K with ≥1 corroborator) provides an additional independent labeling signal that's pure structural — not derived from belief.

## V5 doctrine inputs (ready to write)

The corpus inventory supports the multi-source weak-supervision framing. Key empirical inputs:

- **Curated anchor pool size: 186K** — large enough to seed Snorkel
- **REACH found_by patterns: 1.29M** — rich axis/sign labels we've ignored
- **epistemics.direct: 1.54M** — direct vs indirect relation labels
- **Holdout exclusion: ~2.5K records** — clean evaluation possible

V5 can be written. V4.5 acceptance: **PASS**. The corpus has enough independent signal to support multi-source weak supervision; we don't need belief or LLM-based labeling to bootstrap.
