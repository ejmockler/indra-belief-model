# indra-belief-model

LLM-based evidence quality scoring for [INDRA](https://github.com/gyorilab/indra) biomedical text-mining extractions.

## What this does

INDRA's NLP readers extract structured biological relationships from scientific papers. For example, from the sentence:

> *"The kinase-dead RSK1 mutant, however, was unable to phosphorylate YB-1 at S102."*

a reader might extract: **RPS6KA1 [Phosphorylation] YBX1 @S102**

This scorer judges whether such extractions are correct. Here, the extraction is *incorrect* — the sentence describes a negative result (the mutant was **unable** to phosphorylate).

### Input

The scorer receives what INDRA's pipeline produces for each extraction:

| Field | Example | Source |
|-------|---------|--------|
| **Claim** | `RPS6KA1 [Phosphorylation] YBX1 @S102` | Statement type + agents + modification site |
| **Evidence** | *"The kinase-dead RSK1 mutant..."* | Source sentence from paper |
| **Entity aliases** | RSK1, YB-1, p90Rsk... | [Gilda](https://github.com/gyorilab/gilda) grounding service |
| **Directness** | `[indirect evidence]` | `epistemics.direct` from INDRA evidence metadata |

### Output

```json
{"verdict": "correct", "confidence": "high"}
```

Mapped to a continuous score: `{correct+high: 0.95, correct+medium: 0.80, ..., incorrect+high: 0.05}`.

### Model serving

The scorer calls any OpenAI-compatible API endpoint. Models are configured in `model_client.py`:

```python
LOCAL_MODELS = {
    "gemma-moe": {..., "base_url": "http://localhost:8085/v1"},   # Local MLX
    "gemma-remote": {..., "base_url": "http://host:11434/v1"},    # Remote Ollama
    # Add any vLLM/mlx-lm/ollama endpoint here
}
```

It also supports Anthropic models natively (`ModelClient("claude-sonnet-4-6")`).

## Results

Evaluated on a 200-record held-out set from the [INDRA assembly Benchmark Corpus](https://doi.org/10.5281/zenodo.7559353) (Gyori et al. 2023). Holdout is balanced (100 correct, 100 incorrect extractions) with tags proportional to natural error frequency.

| Version | Accuracy | Architecture |
|---------|----------|-------------|
| v5 baseline | 80.2% | 8 contrastive examples |
| v8 | 85.4% | + hedging scope rules, alias filter, epistemics.direct |
| v10 | 85.9% | + deterministic gilda grounding verification (Tier 1) |
| **v11** | **90.0%** | + gilda confidence threshold, pseudogene detection, structured provenance |

All versions evaluated on the same 200 records with paired McNemar tests. Model: gemma-4-26b-a4b MoE (local, 8-bit quantized).

A large-scale evaluation on 4,647 records (half-corpus) is in progress to measure generalization beyond the 200-record holdout.

## How it works

### Contrastive few-shot prompting

The scorer uses 9 contrastive example pairs (18 examples) that teach the LLM to distinguish correct from incorrect extractions. Each pair shares surface similarity but has opposite verdicts:

| Pair | Teaches |
|------|---------|
| Complex with/without signal | "interacted with" as fact vs. metaphor |
| Activity vs. amount | "activation of enzyme" vs. "up-regulated expression" |
| Logical inversion | Knockdown/blockade implies normal function |
| **Hedging scope** | "found X could interact with Y *to inhibit Z*" (correct) vs. "to test whether X could interact" (incorrect) |
| Discourse trap | "TRAIL activated caspase-3" (direct) vs. "independent of caspase-8" (negated) |
| Modification site | @S12 matches text vs. @S77 doesn't |
| Loss-of-function epithet | "HIF-1alpha activates p53" vs. "loss of p53 upregulates HIF1A" |
| Degradation mechanism | Direct activity inhibition vs. proteasome-mediated degradation |

### Claim enrichment

Before prompting, claims are enriched with INDRA Statement metadata:

- **Modification sites**: `AURKB [Phosphorylation] ATXN10` → `AURKB [Phosphorylation] ATXN10 @S77`
- **Mutations**: Agent mutation annotations from the Statement
- **Entity aliases**: Gilda resolves canonical names + synonyms (e.g., ARHGEF12 → aliases: LARG, KIAA0382)
- **Family warnings**: "STAT (family — if text names a specific member, the claim should use that member)"

### Deterministic grounding verification (v10–v11)

Before the LLM scores, a deterministic tier compares what the NLP reader extracted (`raw_text`) against the claim entities using [gilda](https://github.com/gyorilab/gilda):

| Status | Action | Example |
|--------|--------|---------|
| **MISMATCH** | Auto-reject (no LLM call) | "RhoA" → RHOA ≠ ARHGEF25 |
| **LOW_CONFIDENCE** | Auto-reject (gilda score ≤ 0.53) | "CagA" → S100A8 (0.521, cross-species bacterial protein) |
| **PSEUDOGENE** | Auto-reject | "DVL" → DVL1P1 (pseudogene, not functional protein) |
| **AMBIGUOUS** | LLM judges with grounding context | "9G8" → SRSF7 (0.556) = SLU7 (0.556), tied |
| **MATCH** | Pass to LLM text comprehension | "FAK" → PTK2 (1.0, confirmed alias) |

The corpus index handles source_hash collisions — multiple INDRA statements can share one evidence sentence. `lookup_evidence_meta()` finds the matching entry by statement entities, preventing wrong `raw_text` from reaching the scorer.

### Structured provenance (v11)

When the grounding check finds a MISMATCH, structured provenance is shown to the LLM:

```
Extraction provenance:
  Subject: NLP extracted "IAPP" → IAPP (exact match)
  Object: NLP extracted "amyloid" → mapped to IAPP (MISMATCH — "amyloid" is a DIFFERENT entity, gilda: 0.76)
  Reader: sparser, pattern: BIO-FORM
```

Provenance triggers on ~7% of records (targeted to avoid the attention-dilution regressions observed when extraction context was shown broadly).

## Error profile

### 200-record holdout (v11)

20 remaining errors. v11's deterministic tier resolved 5 grounding errors that were stable across v5–v10 (CagA cross-species, ActA cross-species, DVL1P1 pseudogene ×2, IAPP/amyloid concept mismatch).

| Category | Count | Root cause |
|----------|-------|-----------|
| **Sentence comprehension** | 6 FP | Third-party agent (galectin-1→Ras/ERK), construct names (Myc-EGFR), "potentiates effect of" ≠ activates, signaling ≠ physical binding |
| **Domain knowledge** | 5 FP | Cross-species (Sir2→SIRT1), entity boundary truncation, tied gilda scores, multi-protein complex mislabeling |
| **Benchmark debatable** | 5 FN | Model correctly rejects: reversed direction (Cyclin/E2F1), explicit uncertainty ("still unclear"), multi-step indirect chain, contradictory clauses |
| **Hedging calibration** | 3 FN | "if associated", "may partially retain" — curators accept, model rejects |
| **Stochastic** | 1 FN | Flips between runs (contradictory discourse) |

### Large-scale evaluation (in progress)

A 4,647-record evaluation (half-corpus, excluding few-shot examples) reveals performance degrades on the broader corpus: ~78% at 1,800 records scored. The 18 contrastive few-shot examples — 66% of input tokens — were tuned on patterns from the 200-record holdout and don't generalize to the full diversity of INDRA extraction errors (act_vs_amt: 40% miss rate, hypothesis: 38%, grounding: 38%).

## Setup

```bash
pip install gilda indra

# Download the benchmark corpus (460MB, not included in repo)
# Place at data/benchmark/indra_benchmark_corpus.json.gz
# Source: https://doi.org/10.5281/zenodo.7559353
```

## Usage

```python
from indra_belief.scorers.v11_scorer import score_record
from indra_belief.model_client import ModelClient
from indra_belief.data.claim_enricher import build_corpus_index_v8

client = ModelClient("gemma-remote")  # or "gemma-moe" for local MLX
indexes = build_corpus_index_v8()

result = score_record(
    client=client,
    subject="AURKB",
    stmt_type="Phosphorylation",
    obj="ATXN10",
    evidence_text="Aurora B phosphorylates Ataxin-10 at S12.",
    source_hash=2197027780787608736,
    corpus_index=indexes["statements"],
    evidence_meta=indexes["evidence_meta"],
)
# result["verdict"] → "correct"
# result["score"]   → 0.95
# result["tier"]    → "llm_comprehension"
```

Command-line evaluation:
```bash
PYTHONPATH=src python -m indra_belief.scorers.v11_scorer \
    --model gemma-remote \
    --holdout data/benchmark/holdout.jsonl \
    --output data/results/v11.jsonl
```

## Project structure

```
src/indra_belief/
  model_client.py          # LLM client (OpenAI-compat + Anthropic, local + remote)
  scorers/
    v11_scorer.py          # v11 — deterministic grounding + provenance + LLM (90.0%)
    v10_scorer.py          # v10 — deterministic grounding + LLM (85.9%)
    evidence_scorer.py     # v8 — prompt-only scorer (85.4%, foundation for Tier 2)
    agentic_scorer.py      # v9 — two-pass with gilda tool use (experimental)
  tools/
    grounding_verifier.py  # Deterministic entity verification with gilda scores
    gilda_tools.py         # Gilda tool schema for agentic scorer
  data/
    claim_enricher.py      # Corpus index, claim enrichment, provenance context

data/
  benchmark/
    holdout.jsonl          # 200-record balanced evaluation set
    holdout_large.jsonl    # 4,647-record half-corpus evaluation set
  results/                 # Evaluation results (v5–v11)
```

## References

- Gyori et al. (2023). "Automated assembly of molecular mechanisms at scale from text mining and curated databases." *Molecular Systems Biology*, e11325. [Benchmark corpus: Zenodo 7559353](https://doi.org/10.5281/zenodo.7559353)
- [Gilda](https://github.com/gyorilab/gilda) — Biomedical entity grounding with contextual disambiguation
- [INDRA](https://github.com/gyorilab/indra) — Integrated Network and Dynamical Reasoning Assembler
