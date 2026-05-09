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

Model: gemma-4-26b (Ollama remote or local MLX 8-bit) is the headline target; the registry in `model_client.py` carries other Gemma variants and Anthropic Claude is wired through the same `ModelClient` interface.

### Decomposed pipeline (production)

`score_evidence(statement, evidence, client)` routes every (Statement, Evidence) pair through five steps:

```
parse_claim → build_context → parse_evidence → verify_grounding → adjudicate
    ↓              ↓                 ↓                  ↓                ↓
 deterministic deterministic     1 LLM call       1 LLM call       deterministic
   (no LLM)    (Gilda + regex)                    (per entity)
```

Each step emits a typed commitment (`ClaimCommitment`, `EvidenceContext`, `EvidenceCommitment`, `GroundingVerdict`). The adjudicator reconciles them and returns a verdict with structured reason codes — no free-text rationalization. Disagreement surfaces as a typed reason (`absent_relationship`, `sign_mismatch`, `binding_domain_mismatch`, `hedging_hypothesis`, …), not a hidden vote tie.

The decomposition is the load-bearing design choice: each sub-call is independently checkable, so the agreement signal that self-consistency voting was approximating comes from structural redundancy instead of sample averaging. Self-consistency voting was erased (P-phase) for the same reason.

### Substrate-expansion doctrine (L / M / N phases)

At 27B scale the LLM is reliable for local extraction but unreliable for: chains, counting, world-knowledge priors, cross-sentence antecedents, nominalization, and (under prompt pressure) attention to non-prominent subjects. Each LLM call also carries JSON-mode reasoning degradation (Tam 2024 ~10–15pp) and attention budget cost.

Successive phases push signal out of the LLM and into a deterministic substrate threaded through `EvidenceContext`. The doctrine is recorded in `scorers/context.py` (top-of-module docstring). Briefly:

- **L-phase**: chain-signal regex, subject/object semantic class (Gilda), bilateral-ambiguity precision, nominalization detection, multi-site union.
- **M-phase**: relation surface-form catalog (`relation_patterns.py`), cascade-terminal detection, substrate-driven sign inversion under perturbation markers, alias normalization (Greek↔Latin, hyphen-strip), hypothesis-marker carve-out.
- **N-phase**: written-form FPLX expansion (PKC/PKA/PKG/HDAC), hedge clause-bound proximity, substrate-driven inhibitor agent rewrite, **verifier-asymmetry break in `parse_evidence`** — the parser receives the claim entities as TOPICS OF INTEREST (an attention focus cue), not a verdict prompt; the schema still enforces literal extraction with literal signs.

The N-phase doctrine break is documented at length in `scorers/context.py`; the architectural concession is that strict claim-blindness was lossy under attention pressure at 27B scale.

### Monolithic scorer (research/ablation only)

The single-call scorer still ships (`use_decomposed=False` to invoke directly, or `--use-monolithic` from the CLI). It uses a six-rule system prompt and an adaptive contrastive-pair bank keyed on `stmt_type`. It is preserved as the reference baseline; it is not the production path. Cross-path cascade was erased (P-phase) — the decomposed path stands alone.

## Design decisions we already paid for

Earlier iterations measured the following approaches and rejected them. If you're considering a change that resembles one of these, check the data before re-proposing:

| Approach | Outcome | Why it fails / what replaced it |
|---|---|---|
| Free-text decomposed scorer (3 LLMs, NL-mediated handoffs) | 65.9% on 50 records | Natural-language extraction couldn't bridge INDRA's soft ontology boundaries. **Replaced** by the typed-commitment decomposed pipeline (current production): each sub-call emits a frozen dataclass; the adjudicator runs deterministically on the structured output, not on free text. |
| Native tool-calling (agentic lookup) | 84.9%, below baseline | Model ignores tool results after committing to a verdict in its first pass |
| Structured provenance, full population | −6.7pp accuracy | Attention dilution on 26B model outweighs disambiguation benefit — selectively enabling provenance only for flagged-grounding records preserves the signal without the cost |
| Graduated warnings for every grounding quirk | 3 regressions per 1 fix | Redirects attention from sentence comprehension; now limited to PSEUDOGENE and LOW_CONFIDENCE |
| Indirect-evidence marker in the prompt | +5pp false negatives | Prejudices model toward rejection; removed |
| LOW_CONFIDENCE auto-reject (blanket) | 53.6% precision at scale (32 false rejections on 3,754 records) | The Gilda score threshold is too noisy to gate on deterministically; the signal is still available to the LLM as context |
| Self-consistency voting (k=3 default) | Compute 3× for stability, not accuracy | **Erased (P-phase)**. Demoted to k=1 default during M-phase, then deleted entirely. The agreement signal voting was approximating now comes from structural redundancy across the decomposed sub-calls. |
| Cross-path cascade as default (`dec abstain → mono fallback`) | Erased the abstain signal | **Erased (P-phase)**. Cascade plumbing deleted entirely; decomposed abstention is calibrated semantic uncertainty, and routing it to the claim-aware monolithic path replaced principled abstain with a claim-biased guess. The monolithic path remains reachable for ablation via `use_decomposed=False`, but no longer serves as a fallback target. |
| Strict claim-blind `parse_evidence` (verifier asymmetry) | Lossy under attention pressure at 27B scale | **N-phase doctrine break**: parser receives claim entities as TOPICS OF INTEREST (an attention focus cue, not a verdict prompt). Schema unchanged; literal extraction enforced; `assertions=[]` remains a valid output when the sentence asserts no topic-relevant relation. See `scorers/context.py` for the full doctrine. |
| 3-tier `parse_evidence` retry (informed retry + decomposition-hint retry) | Catastrophic on degraded endpoints (12h, 112/501 records, 50 min/record on N9 holdout 2026-04-29 due to retries-on-timeout amplifying queue pressure) | **O-phase one-shot**: parse_evidence makes exactly one LLM call per evidence (or N calls per multi-clause input via per-clause structural decomposition). Retries gone — the L/M/N substrate-fallback bind in adjudicate covers the failure classes the retries were compensating for. Mirrors H3's demotion of self-consistency voting: reliability comes from structural redundancy, not sample-level retry. See `scorers/context.py` O-phase doctrine. |

Headline baseline: the monolithic scorer with the adaptive contrastive-pair bank reaches ~84% accuracy on the 501-record stratified sample (gemma-4-26b). The decomposed pipeline's value proposition is **interpretability** (structured reason codes, attributable sub-call failures) at parity-or-better; per-phase holdout numbers and verdicts live in `data/results/`. Small-holdout numbers (200 records) overstate by ~4–5pp relative to large-scale evaluation (3,000+ records) — check the larger set before celebrating.

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
    --resume data/results/run.jsonl  # resume interrupted runs
```

The decomposed pipeline (parse_claim → context → parse_evidence → grounding → adjudicate) runs by default. Pass `--use-monolithic` to run the legacy single-call baseline for ablation.

## How we iterate

Contributor-facing rules to keep the repository legible:

- **`main` is the canonical state.** Every "ship" decision ends with `git push`. Local ship decisions don't count.
- **Version labels don't belong in source.** Version numbers appear in PR titles, CHANGELOG entries, and benchmark-run output filenames (`data/results/<run>.jsonl`). They do *not* appear in source comments, docstrings, or identifier names. `scripts/check_no_version_labels.py` enforces this.
- **Public API is `score_statement(statement, client)` + `score_evidence(statement, evidence, client)`.** `score_statement` mirrors INDRA's abstraction (a Statement owns a list of Evidence) and returns one dict per evidence. `score_evidence` is the atomic per-sentence call. `score(client, record, …)` is the benchmark-harness path used by `indra_belief.scorers.scorer.main`; treat it as internal.
- **Comments explain current constraints, not past versions.** If a reader needs history, `git log` is the source of truth. "Provenance is selectively enabled because full-population provenance dilutes attention" is legitimate. "Removed in v12" is not.

## Project structure

```
src/indra_belief/
  model_client.py            # Model transport (OpenAI-compat + Anthropic)
  noise_model.py             # INDRA SimpleScorer (parametric belief from source priors)
  composed_scorer.py         # LLM verdict → hard gate over the parametric noise model
  scorers/
    scorer.py                # Public API: score_evidence / score_statement (dec default)
    _prompts.py              # Monolithic system prompt + contrastive bank (research/ablation)
    # --- Decomposed pipeline (production) ---
    commitments.py           # Typed sub-call schemas (ClaimCommitment, EvidenceCommitment, …)
    parse_claim.py           # Deterministic claim → ClaimCommitment (no LLM)
    context.py               # EvidenceContext + L/M/N-phase doctrine
    context_builder.py       # build_context (Gilda + regex substrate; no LLM)
    parse_evidence.py        # 1-LLM-call evidence parse → EvidenceCommitment
    grounding.py             # Per-entity grounding verifier
    relation_patterns.py     # Regex catalog for relation surface forms (M-phase)
    adjudicate.py            # Deterministic verdict reconciliation + reason codes
    decomposed.py            # score_evidence_decomposed: orchestrates the pipeline
  data/
    entity.py                # GroundedEntity: single Gilda resolution per entity
    scoring_record.py        # ScoringRecord: wraps INDRA Statement + Evidence
    corpus.py                # CorpusIndex: source_hash → Statement lookup
    example_bank.json        # Monolithic contrastive pairs (still used by mono path)
  tools/
    gilda_tools.py           # Entity lookup helper (pre-computed, injected into prompt)

data/
  benchmark/
    holdout.jsonl            # 200-record balanced evaluation set
    holdout_v15_sample.jsonl # 501-record stratified sample (mono v15 baseline)
    holdout_large.jsonl      # 4,625-record half-corpus evaluation
    calibration_*.jsonl      # FN/FP calibration sets (per-pattern accuracy)
    example_pairs.json       # Entity pairs excluded from holdouts
  results/                   # Evaluation results (one file per locked baseline)

scripts/
  check_contamination.py        # Pre-eval gate: examples must not overlap holdout
  check_no_version_labels.py    # CI guard: no v{n} labels in src, tests, scripts
  d4_overfit_check.py           # Held-back overfit gate (dec path; --use-monolithic for ablation)
  dual_run.py                   # Side-by-side mono vs dec on a stratified sample
  m13_analyze.py                # M-phase holdout analysis (pre-registered ship verdict)

research/
  unified/                      # Archived single-call unified scorer (B/C/D phases)

.github/workflows/
  ci.yml                        # pytest + both guards on every push and PR
```

## References

- Gyori et al. (2023). "Automated assembly of molecular mechanisms at scale from text mining and curated databases." *Molecular Systems Biology*, e11325. [Benchmark corpus: Zenodo 7559353](https://doi.org/10.5281/zenodo.7559353)
- [Gilda](https://github.com/gyorilab/gilda) — Biomedical entity grounding
- [INDRA](https://github.com/gyorilab/indra) — Integrated Network and Dynamical Reasoning Assembler
