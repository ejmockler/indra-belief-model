# V-phase greenfield design brief

**Audience:** an agent or team designing a clean implementation of the V-phase
LoRA training pipeline from scratch, given full knowledge of the failures
the current iteration has hit.

**Context window:** this document is intended to be self-contained. Read it
end-to-end before writing code. Every section exists because we tripped on
something real.

---

## 1. The actual problem (not the stated one)

**Stated problem:** "Replace the LLM-prompted relation_axis / subject_role /
object_role / scope / verify_grounding probes with LoRA-fine-tuned
Gemma 4 E4B-it adapters that match or beat the prompted Gemini Pro curator
on the U2 holdout."

**Real problem:**

1. The U2 holdout (482 records, hand-derived `gold_tag` labels) is small,
   stratified for hard cases, **and itself contains ~5–10% noise**. Multiple
   tracks (V6g, V8c) confirmed this — Pro disagrees with U2 on records where
   re-reading the evidence makes Pro look correct.
2. The U2 gold mapping (`GOLD_TAG_TO_PROBE_CLASS`) projects 11 tags onto
   per-probe classes via a static dict. This mapping is a **doctrine
   choice**, not ground truth. It collapses some legitimate distinctions
   (amount-axis claim with matching evidence → `direct_sign_match` regardless
   of axis) and skips others (multi-tag records map to None for some probes).
3. Gemini 3.1 Pro Preview is the curator. It hits ~78.8% / 0.594 macro on
   the U2 holdout (relation_axis), with class imbalance dragging the metric
   numbers around. **That is the upper bound** any LoRA can be measured
   against on this gold, because the LoRA's labels come from Pro.
4. The trivial baseline (most-frequent class on U2 holdout) is **69.6%**
   for relation_axis. Beating this is the lower bar.
5. We have one consumer GPU (Radeon RX 7900 XTX, gfx1100, ~25 GB VRAM via
   ROCm 6.4). It's shared with a llama.cpp inference container that
   normally serves the production scorer at port 8081.

So the success criterion isn't "match Pro." It's:

> **Produce a LoRA-fine-tuned Gemma 4 E4B-it adapter (per probe) that
> beats trivial-baseline by ≥3pp micro AND has macro-recall ≥0.65 on
> the U2 holdout, with reproducible training runs and a validation
> protocol immune to the failure modes documented in §6.**

If your design doesn't hit that, escalate to (a) DAPT on biomedical
text, (b) a larger base model, or (c) an ensemble curator. Don't iterate
prompts or data sampling further.

---

## 2. Hard constraints

- **Compute**: one Radeon RX 7900 XTX (gfx1100), 25 GB VRAM. ROCm 6.4.
  Stop the `llm-gateway-llama-1` Docker container before training (frees
  ~21 GB). Restart after training.
- **Python env**: `~/venvs/v-phase/bin/python` on host `100.97.101.59`.
  Has `torch 2.9.1+rocm6.4`, `peft 0.19.1`, `bitsandbytes 0.48.0.dev0`
  (ROCm fork, see `memory/reference_bnb_rocm_fork.md`), `transformers 5.8.0`.
  **No tensorboard.** Don't depend on it; degrade gracefully.
- **Model**: `google/gemma-4-E4B-it`. ~7.95B params, multimodal config
  (Gemma4Config wraps text/vision/audio). bnb 4-bit (NF4 + double-quant +
  bf16 compute) is the only working config. LoRA target regex
  `^model\.language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$` —
  vision/audio towers must be excluded.
- **Holdout**: `data/benchmark/probe_gold_holdout.jsonl` (482 records,
  hand-derived gold) joined with `data/benchmark/holdout_v15_sample.jsonl`
  (subject/object/evidence) on `source_hash`. The 4-class effective
  taxonomy for `relation_axis` is `{direct_sign_match,
  direct_sign_mismatch, direct_axis_mismatch, no_relation}` — production
  has 8 but U2 only labels 4 (a doctrine collapse).
- **Curator** is Gemini 3.1 Pro Preview via `google-genai`, key persisted
  at `.env`. Pro hits ~$15 per 5K records labeling cost. Pro's curator
  prompts live in `src/indra_belief/v_phase/curator_prompts.py`.
- **Contamination guard**: training records' `source_hash` MUST be disjoint
  from U2 holdout `source_hash` set. Train MUST be disjoint from val by
  hash group, not by row position. Failure modes documented in §6.

---

## 3. Data assets currently on disk

| Asset | Path | Records | What it is |
|---|---|---|---|
| Pro-labeled corpus (V8a) | `data/v_phase/v8a_relation_axis_labels.jsonl` | 2956 kept (out of 3000) | Pro's `(answer, rationale)` for `relation_axis` on stratified corpus walk. 2 of 7 axes covered (modification + activity). 39% are `no_relation` — over-represented vs holdout. |
| Pro-labeled corpus (V8a2) | `data/v_phase/v8a2_relation_axis_labels.jsonl` | ~7K (in flight) | Extended axis coverage (amount, binding, localization, gtp_state, conversion). Skip-set excludes V8a-labeled hashes. Schema-compatible — `cat v8a*.jsonl` works. |
| V8b chat-format (corpus dist) | `data/v_phase/{train,val}/v8b_relation_axis.jsonl` | 2662/294 | Stratified by class, distribution preserved (41/39/12/8 — does NOT match holdout). |
| V8c chat-format (holdout-mirrored) | `data/v_phase/{train,val}/v8c_relation_axis.jsonl` | 1626/200 | Group-split by `source_hash`. Distribution mirrors holdout (70/14/13/4 within 0.7pp). Multi-class hashes (95 records) DROPPED — controversial; see §6. |
| U2 gold | `data/benchmark/probe_gold_holdout.jsonl` + `holdout_v15_sample.jsonl` | 482 | Hand-derived gold tags + evidence. The eval set. |
| Production probe prompts | `src/indra_belief/scorers/probes/{relation_axis,subject_role,object_role,scope}.py` and `grounding.py` | — | The 8-class production prompts. **Do not modify** — they're the live scorer. |
| Curator prompts (V6g) | `src/indra_belief/v_phase/curator_prompts.py` | — | The Pro-tuned curator prompts (4-class collapse for relation_axis). Used by V8a/V8a2 labelers. |
| Adapters trained so far | `data/v_phase/lora/relation_axis_v9{,b}/` | — | V9 (V8b-trained, eval_loss 0.169 best) and V9b (V8c-trained, eval_loss 0.329 best). Both underperform; see §4. |
| Telemetry from V9 runs | `data/v_phase/lora/relation_axis_v9b/{metrics,predictions}.jsonl`, `run.json` | — | Use these to understand what went wrong before redesigning. |

---

## 4. What's been tried, what the numbers were, and why it fell short

### Pro curator (V6g — the upper bound on this gold)

After 3 prompt-iteration rounds + composite gate redesign + role-gold
content-conditional resolver:

| Probe | n | micro | macro | minority recall (smallest class) |
|---|---|---|---|---|
| relation_axis | 392 | 0.788 | 0.594 | direct_sign_mismatch (n=16): 0.188 |
| subject_role | 460 | 0.828 | 0.686 | absent (n=15): 0.533 |
| object_role | 460 | 0.876 | 0.829 | absent (n=18): 0.778 |
| scope | 288 | 0.896 | 0.709 | hedged (n=8): 0.500; negated (n=7): 0.714 |

Pro **fails the simple ≥90% gate** on every probe. The composite gate
(saturating lift + macro-recall floor + per-class recall floor) reframes
it but the underlying numbers are these. Three observations:

- High-mfc probes (role probes after audit, scope) have `micro < mfc`
  whenever the curator emits any minority predictions, because gold is
  ~95% one class. This makes "lift over baseline" structurally hard.
- `direct_sign_mismatch` (n=16) is too small to measure stably.
- Pro's labels are noisy: V6g `no_relation` precision ~0.53 means
  ~half of Pro's "no_relation" predictions disagree with U2 gold.

### V9 LoRA (V8b corpus distribution, no rebalancing)

```
micro=0.543  macro=0.515  mfc=0.696  Δ vs mfc=-15.3pp
```

Per-class recall: `direct_sign_match=0.549, no_relation=0.755,
direct_axis_mismatch=0.320, direct_sign_mismatch=0.438`.

**Failure mode:** trained on 41/39/12/8 distribution; emitted at that
distribution; got penalized on 70/14/13/4 holdout. Over-predicted
`no_relation`, under-predicted `direct_sign_match`.

### V9b LoRA (V8c holdout-mirrored data, hash-grouped)

```
micro=0.571  macro=0.496  mfc=0.696  Δ vs mfc=-12.5pp
```

Per-class recall: `direct_sign_match=0.590, no_relation=0.623,
direct_axis_mismatch=0.520, direct_sign_mismatch=0.250`.

**Failure mode:** mirroring holdout starved `direct_sign_mismatch`
(219 → 65 train examples). Recall on that class dropped 18.8pp despite
the data being "correctly" distributed.

### V10c — V9 with inference-time logit-prior shift

```
micro=0.651  macro=0.534  mfc=0.696  Δ vs mfc=-4.5pp
```

Best result so far. Apply `log(p_holdout / p_train)` to per-class log-probs
at inference (no retrain). Approach used scoring (one short forward per
class) instead of greedy generation. Recovers most of V9's distribution
bias **without training-data manipulation**. Strategy that the brutalist
called out before any retrain happened.

**Still 4.5pp below trivial.** The LoRA fundamentally underperforms the
Pro labels it was trained on, even after fixing distribution bias.

---

## 5. Probable root causes (rank-ordered)

1. **The base model is too small for this task at bnb-4bit.** Gemma 4 E4B
   has ~8B params; quantized to 4-bit is ~3 GB on disk. The task (4-way
   classification with subtle linguistic distinctions: sign-flip, axis-
   mismatch, mediator-vs-direct) is genuinely hard. Pro is much larger.
   The LoRA may not have the capacity to compress Pro's discrimination
   into rank-16 adapters on attention projections only.
2. **Training data is single-axis-biased**. V8a covered only 2 of 7 stmt
   axes; V8a2 is fixing this but wasn't yet merged when measurements were
   made. Holdout has all axes.
3. **Pro labels are 5–10% noisy** (per V6g per-class precision). Noise-
   robust training (label smoothing, confidence-weighted loss, multi-
   pass agreement filtering) was discussed but not implemented.
4. **Loss masking only covers the assistant turn but weights both `answer`
   and `rationale` tokens equally.** Class-balanced loss therefore weights
   rationale style by the answer class — leaks the bias into prose.
5. **No DAPT on biomedical text.** Gemma 4 was trained on web/books
   with biomedical content but not weighted toward it. Continued
   pretraining on PubMed could buy 2–5pp.

These compound. Don't try to fix #1 by piling on prompt tricks for #4.

---

## 6. Pitfalls — every one tripped during this iteration

Read these before writing any code. Each one cost real time.

### 6.1 Asyncio + concurrent API calls return out of order

**Symptom:** `for fut in asyncio.as_completed(coros): results.append(...)`
appends results in completion order, not request order. If your callers
index `results[i]` to find task `i`'s result, you're comparing record A's
gold to record B's response.

**Fix:** wrap each coroutine in an `indexed_call(idx, ...)` helper that
returns `(idx, result)`, write to `results[idx]`. Or use `asyncio.gather`
which preserves input order.

This wasted ~2 hours and produced misleading curator-quality numbers.

### 6.2 `apply_chat_template(tokenize=True)` returns `BatchEncoding`, not `list[int]`

**Symptom:** `len(tok.apply_chat_template(...))` returns `2` (number of
fields in BatchEncoding: input_ids + attention_mask), so seq-length checks
spuriously fire. Or `model.generate(enc, ...)` fails with
`AttributeError: 'BatchEncoding' has no attribute 'shape'`.

**Fix:** always `enc.input_ids if hasattr(enc, "input_ids") else list(enc)`.

### 6.3 Gemini Pro requires thinking budget and large output budget

**Symptom:** `gemini-3.1-pro-preview` returns empty/truncated output if
`max_output_tokens=1024` because the thinking phase consumes the budget
before any JSON is produced. `thinking_budget=0` is rejected
(`Budget 0 is invalid. This model only works in thinking mode.`).

**Fix:** `max_output_tokens=4096` AND
`thinking_config=types.ThinkingConfig(thinking_budget=512)`.

### 6.4 Gemini's `response_schema` enum needs `types.Schema(...)` AND token budget

**Symptom:** dict-style schema may parse but enum constraint doesn't bind;
or model returns out-of-set values when token budget is too tight.

**Fix:** use `types.Schema(type='OBJECT', properties={...,
'answer': types.Schema(type='STRING', enum=[...])}, required=[...])`,
combined with a generous budget.

### 6.5 `prepare_model_for_kbit_training` clobbers `use_reentrant=False`

**Symptom:** you call `model.gradient_checkpointing_enable(
gradient_checkpointing_kwargs={"use_reentrant": False})`, then
`prepare_model_for_kbit_training(model)` — and PEFT silently re-enables
grad-checkpointing with **default** kwargs (use_reentrant=True), undoing
the fix and increasing VRAM.

**Fix:** pass kwargs through:
```python
model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)
```

### 6.6 `Gemma4Config.use_cache` doesn't exist at the top level

**Symptom:** `model.config.use_cache` raises `AttributeError` on multimodal
Gemma 4. The text sub-config has `use_cache`, the wrapping config
doesn't.

**Fix:** `getattr(model.config, "use_cache", None)` and skip the toggle
gracefully if it's None.

### 6.7 Hash-group leakage: same `source_hash` in train and val

**Symptom:** `source_hash` can repeat in the curated corpus because the
same evidence is labeled in different runs or with different claims
against the same evidence. Splitting by record (not by hash group) puts
the same evidence in both train and val. Val accuracy is then optimistic.

**Fix:** dedup by hash OR `GroupShuffleSplit` semantics. Hard-assert
`set(train_hashes) & set(val_hashes) == set()` before writing files.

### 6.8 Holdout exclusion failing open

**Symptom:** `load_u2_holdout_hashes` warns and continues if a U2 path
is missing. Result: contamination assertion vacuously passes, training
data may overlap with eval gold.

**Fix:** hard-fail (`sys.exit(2)`) if any U2 source missing OR if loaded
hash set is empty. Treat the contamination guard as the boundary
invariant it is.

### 6.9 Greedy decode without KV cache is O(prefix²) per token

**Symptom:** `model.generate()` during eval-time prediction trace runs
~10× slower than expected because gradient checkpointing forces
`use_cache=False`.

**Fix:** wrap eval in:
```python
prev_use_cache = getattr(model.config, "use_cache", None)
model.gradient_checkpointing_disable()
if prev_use_cache is not None:
    model.config.use_cache = True
try:
    # ... generate ...
finally:
    if prev_use_cache is not None:
        model.config.use_cache = prev_use_cache
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
```

### 6.10 SSH stdout buffering hides progress

**Symptom:** `ssh host 'long_command 2>&1 | tail -60'` shows nothing until
the command finishes. Live training progress invisible.

**Fix:** write structured telemetry to disk (`metrics.jsonl`,
`predictions.jsonl`, atomically flushed per line). `tail -f` from a
separate connection gives you live progress.

### 6.11 GPU contention with `llm-gateway-llama-1`

**Symptom:** training OOMs at random points because the production
inference container is using ~21 GB.

**Fix:** `docker stop llm-gateway-llama-1` before training. Restart with
`docker start llm-gateway-llama-1` after. **Do not skip this step.**

### 6.12 V8c "match holdout exactly" is methodologically wrong

**Symptom:** rebalancing training data to mirror holdout class
distribution starves minority classes (cuts `direct_sign_mismatch` from
219 examples → 65). LoRA recall on those classes drops accordingly.

**Fix:** train on whatever distribution maximizes information per
minority example (the original corpus distribution is fine). Apply
inference-time logit-prior shift `log(p_target / p_train)` at decode.
This is the textbook approach for class-imbalanced fine-tuning.

### 6.13 Class-balanced loss with the wrong denominator

**Symptom:** `weighted_sum / unweighted_count` produces a per-batch loss
whose scale fluctuates by up to 5× (the cap) depending on class
composition. Implicit per-step learning rate multiplier between 1× and 5×.

**Fix:** standard weighted mean: `weighted_sum / weighted_count` =
`(rec_w * tok_loss * valid).sum() / (rec_w * valid).sum().clamp(min=1)`.

### 6.14 Train and eval objectives diverging

**Symptom:** training optimizes class-weighted CE; `eval_loss()` returns
unweighted CE; "best checkpoint" is selected by the unweighted metric.
Saves a checkpoint that's best at the wrong objective.

**Fix:** apply the same weighting in `eval_loss()` OR drop weighting
entirely. Don't let the two diverge.

### 6.15 Multi-class hash drop discards minority signal

**Symptom:** ~44 hashes have multiple labels (same evidence, different
claims, different answers). V8c drops all 95 records of these on the
grounds that they "distort distribution." But these are precisely the
claim-conditional reasoning examples (the model has to read the claim
to disambiguate). Dropping them teaches the model "evidence determines
label" — a bias toward the failure mode they mitigate.

**Fix:** keep multi-class hashes, assign the entire group to one split
based on hash bucketing. The "distribution distortion" is small
(~3% of records).

### 6.16 `--no-class-balance-weights` doesn't disable the math, just the application

**Symptom:** the script computes class weights regardless and only
gates whether to apply them at the loss layer. Logging is fine, but
when `--eval-weighted` is set without `--class-balance-weights`, the
flag silently no-ops. Reproducibility log records both flags as ON
when one is meaningless.

**Fix:** when `class_balance_weights=False`, also force
`eval_weighted=False` and warn if the user passed inconsistent flags.

### 6.17 No git SHA / dirty marker / env-var snapshot in run config

**Symptom:** an adapter trained yesterday can't be reproduced because
nothing recorded the script state, ROCm env vars, or HuggingFace model
revision. ROCm-specific perf is highly env-var-dependent
(`MIOPEN_FIND_MODE`, `PYTORCH_TUNABLEOP_ENABLED`, `HSA_OVERRIDE_GFX_VERSION`,
etc — see `_env_snapshot()` for the seed list).

**Fix:** write `run.json` at startup with: `git rev-parse HEAD`,
`git status --porcelain | wc -l`, all `^(HIP|HSA|ROCM|ROCR|MIOPEN|PYTORCH|
TORCH|HF|HUGGINGFACE|BNB|TRANSFORMERS|TOKENIZERS|CUDA|OMP|MKL|AMD)_`
env vars, library versions, file hashes (sha256). Add `timeout=5` on
git subprocess calls so a slow filesystem can't hang training startup.

### 6.18 `parse_answer` substring fallback is order-dependent

**Symptom:** `for c in CLASSES: if c in text: return c` returns the
first class whose name is found in the generated text. Free-text
generations like "this is not a `direct_sign_match` case" misclassify.
(Note: `direct_sign_match` is NOT a substring of `direct_sign_mismatch`
— I verified this. But the fallback is still order-sensitive.)

**Fix:** parse JSON properly. If JSON parse fails, refuse to fallback
to substring matching — count as `unparseable` and exclude from
accuracy. Don't pretend you got an answer when you didn't.

### 6.19 Telemetry imports are heavy; tensorboard isn't always installed

**Symptom:** `from torch.utils.tensorboard import SummaryWriter` raises
`ModuleNotFoundError: No module named 'tensorboard'` on minimal
environments. Training crashes at startup before any work.

**Fix:** wrap the import in `try/except` and fall back to a no-op
writer that still satisfies the SummaryWriter API. Always emit JSONL
telemetry as the source of truth; tensorboard is decorative.

### 6.20 The brutalist round 1 found bugs that round-1 fix agents missed

When you spawn fix agents, run a brutalist round on the fixes. Bugs
introduced by fix agents include: `break` instead of `continue` in
greedy quota loops, `prepare_model_for_kbit_training` clobbering
gradient-checkpointing kwargs, `model.config.use_cache` raising
AttributeError on Gemma4. The cost of the brutalist roast is one
codex+claude critic pair (~$0 to $1, several minutes); the cost of
shipping bug-fixes-with-bugs is hours of training time.

---

## 7. Things that worked

Catalog these too — they're not just absence of bugs, they're choices
worth preserving.

- **Gemini 3.1 Pro Preview as curator** is the right tier. 2.5 Flash and
  3 Flash Preview both fail on minority classes. Pro hits ~78%/0.59 macro;
  Flash hit ~13%/0.46 even after schema constraint and prompt iteration.
- **Curator-path prompts** (separate from production probe prompts) let
  us iterate on Pro without touching the live scorer. Keep this
  separation in the greenfield design.
- **Composite gate**: lift over mfc + macro-recall floor + per-class
  recall floor with 1-class escape hatch. The simple ≥90% gate was
  meaningless on imbalanced gold.
- **Decomposed curator** (sub-questions for axis, sign, role) showed a
  small win on one minority class but hurt others. Not worth the
  complexity at Pro scale; might be worth more at smaller-model scale.
- **Inference-time logit-prior shift** (V10c approach) is the cleanest
  fix for class-imbalanced fine-tuning. Don't manipulate training data
  if you can correct at decode.
- **Hash-grouped train/val split** is correct. Just enforce it earlier
  and don't drop multi-class hashes.
- **Streaming corpus walker** (`labeling.py:iter_pairs`) handles the
  439 MB gzipped corpus correctly. Holdout exclusion via `source_hash`
  + `matches_hash` + `(pmid, entity_pair)` is well-tested.
- **JSONL metrics + per-eval prediction trace + run.json snapshot** is
  the minimum viable telemetry. Tensorboard is optional. Predictions
  trace is what tells you the model is actually learning the task,
  not just the loss.
- **bnb 4-bit + bf16 compute + gradient checkpointing (use_reentrant=False)
  + LoRA rank 16 on attention projections** fits Gemma 4 E4B in ~18.5 GB
  during training. Don't try to be clever about quantization; this works.

---

## 8. What a clean greenfield design should look like

These are the load-bearing architectural choices. Get them right, and
the rest is plumbing.

### 8.1 Single source of truth for class space, prompt format, and parser

Right now `RELATION_AXIS_CLASSES`, the system prompt, the user-message
format, and the JSON parser are duplicated across V8b, V8c, V9, V10,
V10c, and the curator_prompts module. When one changes, the others
silently drift. **Pick one module** (`src/indra_belief/v_phase/relation_axis_spec.py`?)
and import from it everywhere.

### 8.2 Group-based train/val/holdout separation as a typed boundary

A single function `split_records(records, holdout_path) -> (train, val)`
that:
- Rejects records whose `source_hash` is in the holdout set (hard-fail
  if holdout missing or empty).
- Group-splits by `source_hash` (no leakage).
- Returns a `dataclass(frozen=True) Split` with explicit `train_hashes`,
  `val_hashes`, `holdout_hashes` sets — so downstream code can assert
  on properties.
- Writes a manifest sidecar `.split.manifest.json` with `sha256(input)`,
  seed, and the realized class distributions.

### 8.3 Training script that does ONE thing

The current `v9_train_relation_axis.py` is 1000+ lines and runs
argparse, dataset loading, model loading, LoRA setup, training loop,
eval, telemetry, prediction tracing, and reporting in one
`def main()`. It's untestable. Refactor:

```
src/indra_belief/v_phase/train/
  __init__.py
  data.py        # JSONL → tokenized batches, group-aware
  model.py       # bnb + LoRA + grad-checkpointing wired together
  loss.py        # class-balanced CE with correct denominator + tests
  eval.py        # eval_loss, prediction-trace, prior-shift scoring
  telemetry.py   # JSONL writer + optional TB + run.json
  trainer.py     # the loop, ~100 lines
scripts/
  v_phase_train.py  # CLI wrapper
  v_phase_eval.py   # CLI wrapper
```

Each module is testable. Add unit tests for `compute_class_weights`,
`weighted_loss`, `parse_answer`, and `apply_prior_shift` BEFORE shipping.

### 8.4 Eval is decoupled from training

A trained adapter is a frozen artifact. Eval reads:
- The adapter
- The holdout file
- The training distribution (from `run.json` or the data file)
- A flag for prior-shift on/off

Eval produces a single markdown report and a JSONL of per-record
predictions. The training loop should NOT include any "best
checkpoint" logic that depends on accuracy — it should save N
checkpoints, and let an external script choose which is best by
running eval on each. Decoupling eval makes training shorter and
makes "the best checkpoint by metric M" a post-hoc choice, not
a baked-in assumption.

### 8.5 Inference-time prior shift as a first-class feature

Add `--prior-shift {auto,off,custom}` to eval. `auto` reads the train
distribution from `run.json` and the target distribution from the
holdout file; `custom` takes a JSON path. The shift is applied at the
class-scoring step (not autoregressive generation). Results in a
multiplier on per-class log-probs; argmax over `score(c) +
log(p_target/p_train)`.

### 8.6 Confidence-weighted training is standard, not bespoke

Rather than custom class-balanced loss math, use one of:
- **Per-record loss weighting** with provably correct denominator (and
  unit tests for the uniform-weight case)
- **Oversample minority classes** in the DataLoader (cleaner — the
  optimizer never sees a weight)
- **Label smoothing** for noise robustness (Pro labels are 5–10% noisy)

Pick one. Document why. Don't combine.

### 8.7 Reproducibility is not optional

Every training run writes `run.json` with: git SHA + dirty flag,
script sha256, env vars (regex-captured), library versions, file
hashes, GPU info, full CLI args, train class distribution, and the
random seeds (torch + numpy + python random). If `git rev-parse`
hangs, time out at 5s — don't block training startup.

### 8.8 Brutalist gate before each phase ships

After data split, after first training, after first eval — run a
brutalist roast (codex + claude pair, ~$0). Iterate. The cost is
small; the cost of shipping a bug to a 45-minute training run is
not.

---

## 9. Decisions the agent must make (and document the reasoning for)

For each of these, a clean greenfield design picks ONE answer and
sticks with it. The current iteration tried multiple and ended up
ambiguous.

1. **Train distribution**: corpus / holdout-mirror / soft-mix /
   minority-oversampled. Pick one. Apply prior shift at inference if
   needed.
2. **Class-balanced loss**: on / off / replaced by oversampling.
   Don't combine with data rebalancing.
3. **Multi-class hashes**: keep (assign to one split via hash bucket)
   or drop. Document.
4. **LoRA target modules**: just attention (current) or +MLP. Test on
   one probe before committing all 5.
5. **LoRA rank**: 16 (current) or higher. Higher = more capacity,
   more VRAM, more risk of overfitting on 2K-example datasets.
6. **DAPT before LoRA**: skip (current) or do a 8-12h subsample DAPT
   on PubMed first. Big lever if base capacity is the bottleneck.
7. **Multi-pass curator + agreement filter**: skip (current) or run
   Pro 2-3× per record and keep only agreement. 2-3× cost, cleaner
   labels, smaller usable subset.
8. **All 4 probes from one adapter or 4 separate adapters**: separate
   (current plan) is simpler but 4× VRAM at inference. One adapter
   with task-specific instructions is cleaner but harder to train.

For each decision, write the reasoning into `research/<phase>_decisions.md`.
Future-you and future-others need to know why.

---

## 10. Success criteria, verbatim

The redesigned pipeline ships when, on the U2 holdout (482 records,
4-class effective taxonomy for relation_axis):

- **Adapter beats trivial baseline by ≥3pp micro** (currently
  V10c is at -4.5pp, V9b at -12.5pp, V9 at -15.3pp)
- **Macro-recall ≥0.65** (currently V10c is at 0.534)
- **Per-class recall ≥0.30** for every class with support ≥5
- **No data contamination** (assertion-enforced — train ∩ val ∩
  holdout = ∅ at hash level)
- **Reproducible**: any run can be re-run from `run.json` and
  produce a bit-identical adapter (or document why not)
- **Telemetry emits structured JSONL** during training; eval emits
  per-record prediction trace
- **Three-phase brutalist gate clean**: data split, training,
  evaluation each pass a brutalist round before merging

If, after one clean iteration, the adapter still doesn't clear these
gates: the bottleneck is base-model capacity. Escalate to DAPT or
larger model. Don't iterate prompts further.

---

## 11. Things to NOT do

- **Don't** mirror the holdout class distribution in training data.
- **Don't** combine data rebalancing with class-balanced loss.
- **Don't** use record-level train/val splits.
- **Don't** silently `continue` on a missing holdout file.
- **Don't** ship a script with a 1000-line `main()`.
- **Don't** trust an SSH `tail` to give you mid-run progress.
- **Don't** spawn fix agents and skip the brutalist round on their work.
- **Don't** assume the curator is correct because Pro is expensive —
  Pro labels are 5–10% noisy and the U2 gold itself has known errors.
- **Don't** treat the simple ≥90% accuracy gate as load-bearing on
  imbalanced gold.
- **Don't** modify the production probe prompts — they're the live scorer.
- **Don't** depend on tensorboard being installed.
- **Don't** generate full JSON for class scoring — use the scoring
  approach (one short forward per class) for eval. 10× faster.

---

## 12. References (paths to read before designing)

In rough order of importance:

```
research/v6g_gold_audit.md               # gold-mapping issues
research/v6g_gemini_validation.md        # curator quality numbers
research/v6g_prompt_iteration.md         # how Pro prompts evolved
research/v6g_relation_axis_iteration.md  # decomposed-curator experiment
research/v6g_gate_design.md              # composite gate rationale
research/v6g_decomposed.md               # sub-question chain
research/v8c_stratification.md           # data-rebalance rationale (and its flaws)
research/v10b_relation_axis_eval.md      # V9b numbers
research/v10c_v9_prior.md                # V10c (best so far)
data/v_phase/lora/relation_axis_v9b/run.json       # config snapshot example
data/v_phase/lora/relation_axis_v9b/metrics.jsonl  # training telemetry example

src/indra_belief/v_phase/curator_prompts.py        # curator prompts
src/indra_belief/v_phase/decomposed_curator.py     # decomposed-curator infra
src/indra_belief/v_phase/labeling.py                # corpus walker (well-tested)
src/indra_belief/scorers/probes/relation_axis.py   # production prompt
scripts/v8a_label_corpus.py                         # baseline labeling pipeline
scripts/v9_train_relation_axis.py                   # current training script (1000+ LOC)
scripts/v10c_eval_with_prior.py                     # scoring + prior-shift eval

memory/feedback_contamination_guard.md             # always run check_contamination
memory/feedback_fewshot_contamination.md           # never paraphrase holdout records
memory/feedback_substrate_vs_llm_lever.md          # push signal to substrate
memory/reference_bnb_rocm_fork.md                  # ROCm-specific bnb fork
memory/feedback_load_then_convert_antipattern.md   # don't load full bf16 to CPU
```

The brutalist outputs (round 1 and round 2) are not on disk but are in
the conversation log they came from. If they're available, read them —
they catch things this brief misses.

---

## 13. The one-paragraph pitch for a fresh agent

You're building per-probe LoRA adapters on Gemma 4 E4B-it for an INDRA
biomedical evidence scoring system. The previous attempt produced
adapters that underperform a trivial-baseline classifier on the holdout.
The fixes are mostly mechanical (correct loss math, group-based splits,
proper grad-checkpointing kwargs, tensorboard-optional, scoring-based
eval with logit-prior shift) but the dominant question is whether
Gemma 4 E4B at bnb-4bit has the capacity for this task or whether
DAPT/larger model is needed. Read §6 (pitfalls), implement §8
(architecture), avoid §11 (anti-patterns), run §10 (success
criteria). Brutalist-roast at every phase boundary. If the redesigned
pipeline still misses §10 after one clean iteration, escalate to
DAPT on PubMed before iterating prompts further.
