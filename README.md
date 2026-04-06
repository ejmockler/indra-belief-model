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
    "gemma-moe": {
        "base_url": "http://localhost:8085/v1",
        "model_id": "mlx-community/gemma-4-26b-a4b-it-8bit",
    },
    # Add any vLLM/mlx-lm/ollama endpoint here
}
```

It also supports Anthropic models natively (`ModelClient("claude-sonnet-4-6")`).

## Results

Evaluated on a 200-record held-out set from the [INDRA assembly Benchmark Corpus](https://doi.org/10.5281/zenodo.7559353) (Gyori et al. 2023). Holdout is balanced (100 correct, 100 incorrect extractions) with tags proportional to natural error frequency.

| Version | Accuracy | AUPRC | Architecture |
|---------|----------|-------|-------------|
| v5 baseline | 80.2% | 0.804 | 8 contrastive examples |
| v7 | 84.9% | 0.851 | + hedging scope rules, alias filter, new examples |
| **v8** | **85.4%** | **0.833** | + epistemics.direct marker, softened Rule 3 |
| v9 | 84.9% | 0.817 | + two-pass agentic tool use (gilda) |

All versions evaluated on the same 200 records with paired McNemar tests. Model: gemma-4-26b-a4b MoE (local, 8-bit quantized). AUPRC is limited by the discrete 7-value score grid.

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

### Agentic grounding verification (v9, experimental)

A two-pass architecture where the LLM can call `lookup_gene()` to verify entity mappings:

```
Pass 1: Text comprehension (v8 prompt + few-shot examples)
         → verdict: "correct"

Pass 2: If extraction provenance shows mismatch AND pass 1 accepted:
         → LLM calls lookup_gene("RhoA")
         → Tool returns: "RHOA — Ras Homolog Family Member A"
         → LLM sees RHOA ≠ ARHGEF25 → overrides to "incorrect"
```

The tool returns enriched descriptions with functional annotations, pseudogene tags, and alias provenance — domain knowledge the LLM lacks parametrically.

**Current limitation**: On a 26B model, pass 2 confirms rather than challenges pass 1's verdict. The architecture is validated but needs a frontier model to leverage tool results for disambiguation.

## Error profile

30 remaining errors on v8 (200-record holdout):

| Category | Count | Root cause | Path forward |
|----------|-------|-----------|-------------|
| **Grounding** | 8 FP | INDRA mapped ambiguous entity names to wrong genes (9G8→SLU7 should be SRSF7) | Frontier model for tool-based disambiguation |
| **Discourse** | 6 FP | Complex sentence structure: negative results, construct names, partial negation | More contrastive examples (diminishing returns) |
| **Hedging/indirect** | 8 FN | LLM rejects correct extractions with hedging language the benchmark accepts | Calibration gap between prompt rules and curator standard |
| **Biological inference** | 3 FN | Multi-step reasoning: Aβ is a fragment of APP, direction inversion | Domain knowledge beyond text |
| **Benchmark noise** | 5 | LLM is arguably more correct than the ground truth label | Unfixable |

## Setup

```bash
pip install gilda indra

# Download the benchmark corpus (460MB, not included in repo)
# Place at data/benchmark/indra_benchmark_corpus.json.gz
# Source: https://doi.org/10.5281/zenodo.7559353
```

## Usage

```python
from indra_belief.scorers.evidence_scorer import score_record
from indra_belief.model_client import ModelClient
from indra_belief.data.claim_enricher import build_corpus_index_v8

client = ModelClient("gemma-moe")  # or any OpenAI-compatible endpoint
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
# result["confidence"] → "high"
```

## Project structure

```
src/indra_belief/
  model_client.py          # LLM client (OpenAI-compat + Anthropic + Gemma 4 tool calling)
  scorers/
    evidence_scorer.py     # v8 — best prompt-only scorer (85.4%)
    agentic_scorer.py      # v9 — two-pass with gilda tool use (experimental)
  tools/
    gilda_tools.py         # Gilda tool: enriched descriptions, pseudogene tags, alias provenance
    gene_lookup.py         # Entity matching and grounding utilities
  data/
    claim_enricher.py      # Corpus index, claim enrichment, entity alias context

scripts/
  evaluate.py              # AUPRC and accuracy evaluation
  build_holdout.py         # Reproducible holdout construction (seed=42)
  run_baseline.py          # v5 baseline scorer
  ablation_runner.py       # Three-condition ablation framework

data/
  benchmark/               # Benchmark dataset + holdout
  results/                 # Evaluation results (v5–v9)
```

## References

- Gyori et al. (2023). "Automated assembly of molecular mechanisms at scale from text mining and curated databases." *Molecular Systems Biology*, e11325. [Benchmark corpus: Zenodo 7559353](https://doi.org/10.5281/zenodo.7559353)
- [Gilda](https://github.com/gyorilab/gilda) — Biomedical entity grounding with contextual disambiguation
- [INDRA](https://github.com/gyorilab/indra) — Integrated Network and Dynamical Reasoning Assembler
