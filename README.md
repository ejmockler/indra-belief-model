# indra-belief-model

LLM-based evidence quality scoring for INDRA biomedical text-mining extractions.

Given an INDRA Statement (e.g., `AURKB [Phosphorylation] ATXN10`) and the evidence sentence it was extracted from, judges whether the extraction is correct.

## Results

Evaluated on a 200-record held-out benchmark from [Gyori et al. 2023](https://doi.org/10.5281/zenodo.7559353) (INDRA assembly Benchmark Corpus).

| Version | Accuracy | Architecture | Notes |
|---------|----------|-------------|-------|
| v5 baseline | 80.2% | 8 contrastive examples | Production prototype |
| v7 | 84.9% | + refined hedging rules, alias filter | Fresh holdout |
| **v8** | **85.4%** | + epistemics.direct, softened Rule 3 | **Best prompt-only** |
| v9 | 84.9% | + two-pass agentic tool use (gilda) | No gain on 26B model |

All versions evaluated on the same 200-record holdout with paired McNemar tests. Model: gemma-4-26b-a4b MoE (local, 8-bit quantized).

## Architecture

### v8 (prompt-only scorer)
- Enriched claims with INDRA Statement metadata (residue/position, mutations)
- Entity alias context via gilda (canonical names + synonyms)
- 9 contrastive example pairs covering: act_vs_amt, hypothesis scope, discourse, epithet, degradation-as-mechanism, modification site verification
- `epistemics.direct` marker for indirect evidence
- System prompt with 5 key rules for text comprehension

### v9 (agentic scorer — experimental)
- Two-pass design: Pass 1 = v8 text comprehension, Pass 2 = gilda tool-calling for grounding verification
- Enriched tool output: functional descriptions, pseudogene tags, alias provenance
- Alias-aware filter reduces unnecessary tool calls by 75%
- Infrastructure validated but 26B model can't leverage tool results for disambiguation

## Error Profile (30 remaining errors on v8)

| Category | Count | Addressable by |
|----------|-------|---------------|
| Grounding (ambiguous entity mapping) | 8 FP | Frontier model or deterministic pipeline |
| Discourse (complex sentence structure) | 6 FP | More contrastive examples (diminishing returns) |
| FN on correct records (hedging/indirect) | 8 FN | Partially benchmark noise |
| Biological inference (multi-step reasoning) | 3 FN | Domain knowledge |
| Benchmark noise/disagreement | 5 | Unfixable |

## Key Findings

1. **Context engineering on a 26B model plateaus at ~85%** for this task
2. **Tool use adds zero accuracy** on gemma-26b — the model calls gilda but can't interpret ambiguous results
3. **Enriched tool descriptions work in isolation** (5/6 grounding errors caught) but fail in the two-pass architecture (model confirms rather than challenges)
4. **The remaining errors require domain knowledge** the 26B model doesn't have (gene alias disambiguation, cross-species mapping, pseudogene detection)
5. **The architecture is validated** — ready for a frontier model that can leverage it

## Setup

```bash
# Requires gilda and INDRA (for ontology)
pip install gilda indra

# Download the benchmark corpus (460MB)
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
print(result["verdict"])  # "correct" or "incorrect"
```

## Project Structure

```
src/indra_belief/
  model_client.py          # Unified LLM client (OpenAI-compat + Anthropic + tool calling)
  scorers/
    evidence_scorer.py     # v8 prompt-only scorer (best accuracy)
    agentic_scorer.py      # v9 two-pass with gilda tool use
  tools/
    gilda_tools.py         # Gilda grounding tool with enriched descriptions
    gene_lookup.py         # Entity matching and grounding utilities
  data/
    claim_enricher.py      # Corpus index, claim enrichment, alias context

scripts/
  evaluate.py              # AUPRC and accuracy evaluation
  build_holdout.py         # Reproducible holdout construction
  run_baseline.py          # v5 baseline on holdout
  ablation_runner.py       # Three-condition ablation framework

data/
  benchmark/               # Benchmark dataset files
  results/                 # Evaluation results (v5-v9)
```

## References

- Gyori et al. (2023). "Automated assembly of molecular mechanisms at scale from text mining and curated databases." *Molecular Systems Biology*, e11325. [Benchmark corpus: Zenodo 7559353](https://doi.org/10.5281/zenodo.7559353)
- [Gilda](https://github.com/gyorilab/gilda) — Biomedical entity grounding
- [INDRA](https://github.com/gyorilab/indra) — Integrated Network and Dynamical Reasoning Assembler
