# indra-belief-model

LLM-based evidence quality scoring for [INDRA](https://github.com/gyorilab/indra) biomedical text-mining extractions.

## What this does

INDRA's NLP readers extract structured biological relationships from scientific papers. For example, from the sentence:

> *"The kinase-dead RSK1 mutant, however, was unable to phosphorylate YB-1 at S102."*

a reader might extract: **RPS6KA1 [Phosphorylation] YBX1 @S102**

This scorer judges whether such extractions are correct. Here, the extraction is *incorrect* — the sentence describes a negative result (the mutant was **unable** to phosphorylate).

### Input

Native INDRA Statement + Evidence objects, resolved through `ScoringRecord`:

| Field | Example | Source |
|-------|---------|--------|
| **Claim** | `RPS6KA1 [Phosphorylation] YBX1 @S102` | Statement type + agents + modification site |
| **Evidence** | *"The kinase-dead RSK1 mutant..."* | Source sentence from paper |
| **Entity aliases** | RSK1, YB-1, p90Rsk... | [Gilda](https://github.com/gyorilab/gilda) grounding via `GroundedEntity.resolve()` |

### Output

```json
{"verdict": "correct", "confidence": "high"}
```

Mapped to a continuous score: `{correct+high: 0.95, correct+medium: 0.80, ..., incorrect+high: 0.05}`.

## Results

| Version | Holdout (200) | Large (3,754) | Architecture |
|---------|---------------|---------------|-------------|
| v5 | 80.2% | — | 8 contrastive examples |
| v8 | 85.4% | — | + hedging rules, alias filter |
| v10 | 85.9% | — | + deterministic gilda verification |
| v11 | 90.0% | 78.7% | + gilda confidence threshold, pseudogene, provenance |
| **v12** | *eval pending* | *eval pending* | Removed LOW_CONFIDENCE/provenance, expanded bank, voting |

Model: gemma-4-26b (Ollama remote or local MLX 8-bit).

### v11 large-scale findings

The 200-record holdout (90.0%) overstated performance. At 3,754 records:
- Overall: **78.7%** (2,956/3,754)
- LOW_CONFIDENCE auto-reject: **53.6% precision** (32 false rejections) — disabled in v12
- Provenance: **72.2%** when triggered vs 78.9% baseline — removed in v12
- Worst by volume: Activation (74.2%, 236 errors), Complex (84.6%, 200 errors)

### Failed experiments

| Experiment | Result | Why it failed |
|-----------|--------|--------------|
| Decomposed 3-call | 65.9% | Semantic gap: natural-language extraction can't bridge INDRA's soft ontology boundaries |
| Agentic tool-use (v9) | 84.9% | Model ignores tool results after committing in pass 1 |
| Structured provenance | -6.7pp | Attention dilution on 26B model |
| Graduated warnings | 3 regressions per 1 fix | Redirects attention from sentence comprehension |
| Indirect evidence marker | +5pp FN | Prejudices model toward rejection |

## How it works

### Two-tier architecture (v12)

**Tier 1: Deterministic grounding** (no LLM call)

| Status | Action | Example |
|--------|--------|---------|
| **MISMATCH** | Auto-reject | "RhoA" → RHOA != ARHGEF25 |
| **PSEUDOGENE + AMBIGUOUS** | Auto-reject | "DVL" → DVL1P1 (pseudogene) |
| **AMBIGUOUS** | Pass to Tier 2 | "9G8" → SRSF7/SLU7 (tied scores) |
| **MATCH** | Pass to Tier 2 | "FAK" → PTK2 (confirmed alias) |

**Tier 2: LLM text comprehension**

- v8 system prompt (6 rules)
- 7 adaptive contrastive pairs (14 examples) selected by statement type
- Optional self-consistency voting (k=3 or k=5, majority vote at temperature=0.6)

### Adaptive few-shot selection

The example bank has 12 type-specific contrastive pairs. For each record, 7 pairs are selected by priority:

1. **Own type** from bank (e.g., Activation pairs for an Activation claim)
2. **Adjacent types** from `TYPE_ADJACENCY` map (e.g., IncreaseAmount for Activation)
3. **Universal patterns** (logical inversion, hedging scope)
4. **Fill** from v8 base examples

Types with bank examples: Activation (2 pairs), Inhibition (2), Phosphorylation, Complex, IncreaseAmount, DecreaseAmount, Dephosphorylation, Autophosphorylation, Translocation, Ubiquitination.

### Self-consistency voting

Model confidence scores are useless (100% report "high"). Self-consistency uses implicit confidence via agreement across k independent samples at temperature=0.6:

```bash
PYTHONPATH=src python -m indra_belief.scorers.scorer --voting-k 3
```

## Setup

```bash
pip install gilda indra

# Download the benchmark corpus (460MB, not included in repo)
# Place at data/benchmark/indra_benchmark_corpus.json.gz
# Source: https://doi.org/10.5281/zenodo.7559353
```

## Usage

```python
from indra_belief.scorers.scorer import score
from indra_belief.model_client import ModelClient
from indra_belief.data.corpus import CorpusIndex

client = ModelClient("gemma-remote")
index = CorpusIndex()
records = index.build_records("data/benchmark/holdout.jsonl")

result = score(client, records[0])
# result["verdict"] → "correct" or "incorrect"
# result["score"]   → 0.95 (correct+high) to 0.05 (incorrect+high)
# result["tier"]    → "llm_comprehension" or "deterministic_mismatch"
```

Command-line evaluation:
```bash
PYTHONPATH=src python -m indra_belief.scorers.scorer \
    --model gemma-remote \
    --holdout data/benchmark/holdout_large.jsonl \
    --output data/results/v12_large.jsonl \
    --voting-k 1 \
    --resume data/results/v12_large.jsonl  # resume interrupted runs
```

## Project structure

```
src/indra_belief/
  model_client.py          # LLM client (OpenAI-compat + Anthropic)
  scorers/
    scorer.py              # v12 — current: native INDRA, adaptive bank, voting
    evidence_scorer.py     # v8 foundation (SYSTEM_PROMPT, contrastive examples)
    v11_scorer.py          # v11 — dict-based scorer (legacy)
    v10_scorer.py          # v10 — first deterministic tier (legacy)
    decomposed_scorer.py   # Failed 3-call experiment
    agentic_scorer.py      # Failed tool-use experiment
  data/
    entity.py              # GroundedEntity: single gilda resolution per entity
    scoring_record.py      # ScoringRecord: wraps INDRA Statement + Evidence
    corpus.py              # CorpusIndex: lazy source_hash → Statement lookup
    example_bank.json      # 12 type-specific contrastive pairs
    claim_enricher.py      # Legacy corpus index (v10/v11 dict-based)

data/
  benchmark/
    holdout.jsonl          # 200-record balanced evaluation set
    holdout_large.jsonl    # 4,625-record half-corpus evaluation
    example_pairs.json     # 36 entity pairs excluded from holdouts
  results/                 # Evaluation results
```

## References

- Gyori et al. (2023). "Automated assembly of molecular mechanisms at scale from text mining and curated databases." *Molecular Systems Biology*, e11325. [Benchmark corpus: Zenodo 7559353](https://doi.org/10.5281/zenodo.7559353)
- [Gilda](https://github.com/gyorilab/gilda) — Biomedical entity grounding
- [INDRA](https://github.com/gyorilab/indra) — Integrated Network and Dynamical Reasoning Assembler
