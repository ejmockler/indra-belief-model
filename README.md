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

## How it works

Model: gemma-4-26b (Ollama remote or local MLX 8-bit).

### Two-tier architecture

**Tier 1: Deterministic grounding** (no LLM call)

| Status | Action | Example |
|--------|--------|---------|
| **MISMATCH** | Auto-reject | "RhoA" → RHOA != ARHGEF25 |
| **PSEUDOGENE + AMBIGUOUS** | Auto-reject | "DVL" → DVL1P1 (pseudogene) |
| **AMBIGUOUS** | Pass to Tier 2 | "9G8" → SRSF7/SLU7 (tied scores) |
| **MATCH** | Pass to Tier 2 | "FAK" → PTK2 (confirmed alias) |

**Tier 2: LLM text comprehension**

- Six-rule system prompt (negation, hedging, family/member equivalence, etc.)
- Seven adaptive contrastive pairs (14 examples) selected by statement type
- Self-consistency voting (k=3 default, majority vote at temperature=0.6)

### Adaptive few-shot selection

The example bank has type-specific contrastive pairs. For each record, 7 pairs are selected by priority:

1. **Own type** from bank (e.g., Activation pairs for an Activation claim)
2. **Adjacent types** from `TYPE_ADJACENCY` map (e.g., IncreaseAmount for Activation)
3. **Universal patterns** (logical inversion, hedging scope)
4. **Fill** from the base contrastive pair set

Types with bank examples: Activation (2 pairs), Inhibition (2), Phosphorylation, Complex, IncreaseAmount, DecreaseAmount, Dephosphorylation, Autophosphorylation, Translocation, Ubiquitination.

### Self-consistency voting

Model confidence scores are useless (100% report "high"). Self-consistency uses implicit confidence via agreement across k independent samples at temperature=0.6:

```bash
PYTHONPATH=src python -m indra_belief.scorers.scorer --voting-k 3
```

## Design decisions we already paid for

Earlier iterations measured the following approaches and rejected them. If you're considering a change that resembles one of these, check the data before re-proposing:

| Approach | Outcome | Why it fails |
|---|---|---|
| Decomposed 3-call scorer | 65.9% accuracy | Natural-language extraction can't bridge INDRA's soft ontology boundaries — requires three LLMs to agree on a fuzzy contract |
| Native tool-calling (agentic lookup) | 84.9%, below baseline | Model ignores tool results after committing to a verdict in its first pass |
| Structured provenance, full population | -6.7pp accuracy | Attention dilution on 26B model outweighs disambiguation benefit — selectively enabling provenance only for flagged-grounding records preserves the signal without the cost |
| Graduated warnings for every grounding quirk | 3 regressions per 1 fix | Redirects attention from sentence comprehension; now limited to PSEUDOGENE and LOW_CONFIDENCE |
| Indirect-evidence marker in the prompt | +5pp false negatives | Prejudices model toward rejection; removed |
| LOW_CONFIDENCE auto-reject (blanket) | 53.6% precision at scale (32 false rejections on 3,754 records) | The gilda score threshold is too noisy to gate on deterministically; the signal is still available to the LLM as context |

Headline baselines measured during iteration: gemma-4-26b + adaptive bank + voting reaches ~84% accuracy on the 501-record stratified sample. Small-holdout numbers (200 records) overstate by ~4-5pp relative to large-scale evaluation (3,000+ records) — check the larger set before celebrating.

## Setup

### Dependencies

```bash
pip install gilda indra openai

# Download the benchmark corpus (460MB, not included in repo)
# Place at data/benchmark/indra_benchmark_corpus.json.gz
# Source: https://doi.org/10.5281/zenodo.7559353
```

### Model configuration

The scorer calls an LLM via `ModelClient(model_name)`. Model names map to
entries in `model_client.py`'s `LOCAL_MODELS` dict, or to Anthropic model
IDs (any string starting with `claude-`).

**Local Ollama (recommended for getting started):**

```bash
# Install Ollama: https://ollama.com
ollama pull gemma3:27b          # or any model you prefer
ollama serve                    # starts on localhost:11434
```

Then add an entry to `LOCAL_MODELS` in `src/indra_belief/model_client.py`:

```python
"ollama-local": {
    "base_url": "http://localhost:11434/v1",
    "model_id": "gemma3:27b",
    "reasoning_in_content": False,
    "max_tokens": 1000,
    "timeout": 120,
},
```

Use it: `ModelClient("ollama-local")` or `--model ollama-local` from the CLI.

**Remote Ollama (e.g., a beefy server on your network):**

Same as above but point `base_url` at the remote host. The `gemma-remote`
entry in the registry shows this pattern — it targets an Ollama instance
over Tailscale.

**Anthropic API:**

```bash
export ANTHROPIC_API_KEY=sk-...
```

```python
client = ModelClient("claude-sonnet-4-20250514")
```

Any `claude-*` model name routes to the Anthropic backend automatically.

**Key `LOCAL_MODELS` fields:**

| Field | Purpose |
|-------|---------|
| `base_url` | OpenAI-compatible endpoint (Ollama serves this at `/v1`) |
| `model_id` | Model name as known to the server (`ollama list` to check) |
| `reasoning_in_content` | `True` if CoT appears in `content` (Qwen CRACK); `False` for models with a separate `reasoning_content` field (Gemma 4) or no reasoning |
| `max_tokens` | Completion token budget — reasoning models need more (8000+) |
| `num_ctx` | Ollama-specific: context window size (passed via `extra_body`) |
| `timeout` | Seconds before retry — increase for large models or slow hardware |

## Usage

### Score a Statement's evidence

An INDRA `Statement` bundles a list of `Evidence` objects. `score_statement`
mirrors that abstraction: one per-sentence verdict per evidence, returned
in order.

```python
from indra.statements import Phosphorylation, Agent, Evidence
from indra_belief import ModelClient, score_statement

stmt = Phosphorylation(
    Agent("RPS6KA1"), Agent("YBX1"),
    residue="S", position="102",
)
stmt.evidence = [
    Evidence(source_api="reach",
             text="RSK1 phosphorylates YB-1 at S102 in response to stress."),
    Evidence(source_api="sparser",
             text="The kinase-dead RSK1 mutant was unable to phosphorylate YB-1 at S102."),
]

client = ModelClient("gemma-remote")
verdicts = score_statement(stmt, client)
# verdicts is list[dict], one per evidence:
#   verdicts[i]["verdict"]    → "correct" | "incorrect" | None
#   verdicts[i]["score"]      → 0.95 (correct+high) … 0.05 (incorrect+high)
#   verdicts[i]["confidence"] → "high" | "medium" | "low"
#   verdicts[i]["tier"]       → which scoring path produced the verdict
```

To score just one evidence of a Statement (skipping the rest of `stmt.evidence`), use `score_evidence(stmt, ev, client)`.

### Composition with INDRA belief

`score_statement` is the per-sentence comprehension layer. The edge-level
question — *given all evidence for a statement, what is the belief?* — is
answered by composing per-sentence verdicts with INDRA's parametric noise
model. The two layers chain directly:

```python
from indra_belief import score_statement
from indra_belief.composed_scorer import ComposedBeliefScorer, EvidenceRecord
from indra_belief.noise_model import RECALIBRATED_PRIORS

verdicts = score_statement(stmt, client)  # list[dict], one per stmt.evidence
records = [
    EvidenceRecord(source_api=ev.source_api, verdict=v["verdict"])
    for ev, v in zip(stmt.evidence, verdicts)
]
belief = ComposedBeliefScorer(priors=RECALIBRATED_PRIORS).score_edge(records)
# belief.belief           → composed edge belief
# belief.parametric_only  → belief before LLM gating (for ablation)
# belief.n_gated          → evidence removed by the gate
```

Gate semantics: `verdict="correct"` passes; unscored evidence
(`verdict=None`) passes by default (`gate_unscored=True` to tighten);
`"incorrect"` and any other string — including `"ambiguous"` or parse
failures — are removed. Priors live in `noise_model.py` (`INDRA_PRIORS`,
`RECALIBRATED_PRIORS`). See `scripts/benchmark_composition.py` for the
benchmark used to pick them.

### Benchmark evaluation against a holdout file

```bash
PYTHONPATH=src python -m indra_belief.scorers.scorer \
    --model gemma-remote \
    --holdout data/benchmark/holdout_large.jsonl \
    --output data/results/run.jsonl \
    --voting-k 3 \
    --resume data/results/run.jsonl  # resume interrupted runs
```

## How we iterate

Contributor-facing rules to keep the repository legible:

- **`main` is the canonical state.** Every "ship" decision ends with `git push`. Local ship decisions don't count.
- **Version labels don't belong in source.** Version numbers appear in PR titles, CHANGELOG entries, and benchmark-run output filenames (`data/results/<run>.jsonl`). They do *not* appear in source comments, docstrings, or identifier names. `scripts/check_no_version_labels.py` enforces this.
- **Public API is `score_statement(statement, client)` + `score_evidence(statement, evidence, client)`.** `score_statement` mirrors INDRA's abstraction (a Statement owns a list of Evidence) and returns one dict per evidence. `score_evidence` is the atomic per-sentence call. `score(client, record, …)` is the benchmark-harness path used by `indra_belief.scorers.scorer.main`; treat it as internal.
- **Comments explain current constraints, not past versions.** If a reader needs history, `git log` is the source of truth. "Provenance is selectively enabled because full-population provenance dilutes attention" is legitimate. "Removed in v12" is not.

## Project structure

```
src/indra_belief/
  model_client.py          # Model transport (OpenAI-compat + Anthropic)
  noise_model.py           # INDRA SimpleScorer (parametric belief from source priors)
  composed_scorer.py       # LLM verdict → hard gate over the parametric noise model
  scorers/
    scorer.py              # The scorer — native INDRA, adaptive bank, voting
    _prompts.py            # System prompt, contrastive examples, verdict parser
  data/
    entity.py              # GroundedEntity: single gilda resolution per entity
    scoring_record.py      # ScoringRecord: wraps INDRA Statement + Evidence
    corpus.py              # CorpusIndex: source_hash → Statement lookup
    example_bank.json      # Type-specific contrastive pairs
  tools/
    gilda_tools.py         # Entity lookup helper (pre-computed, injected into prompt)

data/
  benchmark/
    holdout.jsonl          # 200-record balanced evaluation set
    holdout_large.jsonl    # 4,625-record half-corpus evaluation
    example_pairs.json     # Entity pairs excluded from holdouts
  results/                 # Evaluation results

scripts/
  check_contamination.py        # Pre-eval gate: examples must not overlap holdout
  check_no_version_labels.py    # CI guard: no v{n} labels in src, tests, scripts

.github/workflows/
  ci.yml                        # pytest + both guards on every push and PR
```

## References

- Gyori et al. (2023). "Automated assembly of molecular mechanisms at scale from text mining and curated databases." *Molecular Systems Biology*, e11325. [Benchmark corpus: Zenodo 7559353](https://doi.org/10.5281/zenodo.7559353)
- [Gilda](https://github.com/gyorilab/gilda) — Biomedical entity grounding
- [INDRA](https://github.com/gyorilab/indra) — Integrated Network and Dynamical Reasoning Assembler
