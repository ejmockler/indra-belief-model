# V-phase doctrine — calibrated replacement for INDRA BeliefEngine

Date: 2026-05-05 (revised 2026-05-06 for E4B pivot — see §13)
Predecessor: T-phase (60.17% raw, 73.79% decision-only); U-phase (regressed by 2.08pp, ITERATE verdict)
Constraint: AMD Radeon RX 7900 XTX (24 GB VRAM, gfx1100, ROCm 6.4.1) host. Base model is `gemma-4-E4B-it` (dense ~8B effective) — pivoted from 26B-A4B after M6 attempts confirmed the load-then-convert flow does not fit in 62 GB host RAM. The 26B-A4B path is documented backlog (stream-load redesign).

## §1 What V-phase IS

V-phase replaces the prompted closed-set probes (the entire S/T/U-phase implementation strategy) with **fine-tuned LoRA adapters on Gemma 4 E4B-it**, trained via multi-source weak supervision against curator-aligned labels.

The deliverable framing is decisive: we are not building a more accurate classifier. We are building a **calibrated replacement for INDRA's BeliefEngine**. The existing belief score is empirically miscalibrated against curator gold (~40% gold-incorrect at high belief, per the 482-record holdout). Our scorer's verdict + confidence, when calibrated against curator labels, becomes the new per-record correctness probability that downstream consumers (INDRA's belief layer) can trust.

That changes both the success metric and the labeling strategy. Both are detailed below.

## §2 What V-phase IS NOT

- **Not** a model-class upgrade. The base is a smaller member of the same Gemma 4 family (E4B vs 26B-A4B) — chosen for hardware tractability on the 7900 XTX, not for perceived capability.
- **Not** a prompt-engineering iteration. T-phase showed prompts are high-precision when applied to closed-set rules + adjudicator decision tables; U-phase showed prompts are high-variance when applied to LLM behavioral biases. V-phase converts the LLM-behavioral parts to learned parameters.
- **Not** a rewrite of the four-probe architecture. Subject_role / object_role / relation_axis / scope / verify_grounding remain. Their CALL SITE in the orchestrator is unchanged. What changes is the inference path: prompted JSON classification becomes fine-tuned adapter classification.
- **Not** a curator-gold-required project. Per V4.5 inventory, no per-statement curator-correction annotations exist in the 894K corpus. The 482-record holdout's curator labels are too small to train on AND must remain held out for evaluation. Training data comes from multi-source weak supervision over INDEPENDENT signals.

## §3 Why we got here

Three convergent findings drive V-phase:

### §3.1 The U-phase regression (-2.08pp on partial holdout)

U-phase added 9 architectural and prompt-level interventions. The architectural pieces (U4 KG verifier, U7 closed-set adjudicator, U8 verb taxonomy, U10 consistency) all behaved as designed: zero raw-accuracy change, modest calibration gain. The prompt pieces (U6 grounding rules, U9 Fix A softening) introduced 18 hard regressions where the LLM treated narrow prompt instructions as broad biases.

The lesson: **prompt edits are high-variance at 27B**. Each rewrite is a hypothesis to be tested, not a specification. We have empirical evidence that prompt-only iteration plateaus at this scale.

### §3.2 The research literature

The biomedical relation extraction literature (BioRED, ChemProt, BC5CDR, DDI leaderboards) consistently shows:
- Fine-tuned BERT-class models (110M-340M params) outperform zero-shot LLMs at 70B+ by 5-15 F1 points.
- Hybrid neuro-symbolic (LLM + KG + closed-set probes) tops dense LLMs by another 3-5 points.
- Few-shot prompting at <30B parameters teaches OUTPUT FORMAT, not abstract rules (Min et al 2022, "Rethinking the Role of Demonstrations").
- Position bias in answer lists (Lu et al 2022) explains U9's failure pattern.

The architectural pattern that wins: **per-relation-type fine-tuned classifier + symbolic grounding constraints + uncertainty quantification.** Our four-probe + adjudicator architecture is the right shape. The inference method (prompted zero-shot) is what's wrong.

### §3.3 The empirical signal inventory (V4.5)

The corpus has rich INDEPENDENT signals we've never used as labels:
- **186K curated-DB-anchored statements** (HPRD, biopax, SIGNOR, BEL, TRRUST) — gold proxy at scale
- **1.29M evidence pieces with REACH `found_by` patterns** that explicitly encode axis+sign
- **1.54M evidence pieces with `epistemics.direct` annotations** (extractor's direct/indirect)
- **156K multi-extractor agreement** statements (≥2 distinct source_apis)
- **88K multi-paper convergence** statements (≥3 distinct pmids)

These are sufficient to bootstrap weak supervision without ever consulting the existing belief score.

## §4 The architectural commitment

V-phase commits to:

### §4.1 Fine-tuning, not prompting

For each of the five probe questions (subject_role, object_role, relation_axis, scope, verify_grounding), train a LoRA adapter on Gemma 4 E4B-it that learns the closed-set classification from labeled training data. Inference: load base + adapter, generate JSON answer.

### §4.2 Approach B: separate per-probe LoRA adapters

Each probe gets its own LoRA. Adapter swap at inference time. Five adapters total.

Approach A (single multi-task adapter) was considered but rejected because:
- Negative interference between probes is a real risk at rank-16 capacity
- Per-probe calibration matters for the belief-score deliverable; one global temperature is worse than five fitted ones
- Modular iteration (improve one probe without retraining all) matches our debugging workflow

The 14-22 hours of additional sequential training time is amortized across future iterations.

### §4.3 Multi-source weak supervision for labels

Labels come from aggregating 8-10 INDEPENDENT labeling functions per probe (Snorkel-style data programming):
- LF_curated_db_source — curated DBs (HPRD, biopax, SIGNOR, BEL, TRRUST) act as gold anchors
- LF_reach_found_by — parse REACH's pattern IDs to get extractor-level (axis, sign)
- LF_epistemics_direct — extractor's direct-vs-indirect annotation
- LF_multi_extractor_agreement — when 2+ distinct extractors converge
- LF_multi_paper_convergence — when ≥3 papers support
- LF_substrate_catalog_match — CATALOG regex confirmation
- LF_substrate_negation — verb-negation regex anchored to claim entities
- LF_alias_grounding_clean — Gilda confirms entity grounding

A generative aggregation model (Snorkel's LabelModel) fits per-LF accuracies from inter-LF agreement structure (no gold needed). Output: probabilistic label distributions per record, not hard labels. This gives us **explicitly calibrated training data**, not just thresholded weak labels.

### §4.4 STRICT: do NOT use INDRA's belief score

The belief score is what we're replacing. Using it as label, soft target, or filter would bake the existing miscalibration into our trained model.

We use the underlying primitives (multi-extractor agreement, multi-paper convergence, substrate signals) but never the belief assembly itself. This is the load-bearing constraint of V-phase.

## §5 Memory feasibility (Gemma 4 E4B on 24 GB VRAM)

Per host inventory: AMD Radeon RX 7900 XTX, 24 GB VRAM, gfx1100 (RDNA3), ROCm 6.4.1. The 21 GB held by `llama-server` (running `gemma-4-26B-A4B-it-Q4_K_M.gguf` for the gemma-remote endpoint) must be freed before V-phase training; we restart the container after each training run.

QLoRA budget for Gemma 4 E4B (dense ~8B params, 42 layers):
- 4-bit NF4 base weights (with double-quant): ~5 GB
- LoRA adapter (rank 16, attention q/k/v/o on 42 layers): ~9M trainable params × 4 bytes = ~36 MB
- 32-bit AdamW optimizer state on LoRA params only: ~70 MB
- Activations (batch=4, seq=512, gradient checkpointing): ~3-5 GB
- Framework + cuda runtime: ~1-2 GB
- **Training peak: ~17-18 GB** — empirically verified at M6' (peak 17.2 GiB at batch=4 seq=512 lora_rank=16)
- **Inference peak: ~6-8 GB** — comfortable (allows 26B-A4B GGUF inference to run alongside the E4B adapters at serving time, total ~21+8 = 29 GB which exceeds 24 GB; but at training time only E4B is loaded)

Training cost per probe: empirically ~0.4 s/step at batch=4 seq=512 on the M6' smoke. For a 30-50K-sample training run with effective batch 64 over 3 epochs: ~30-45 minutes of compute per probe (vs the 4-6 hours estimated for 26B-A4B). Total for all 5 probes: ~3-5 hours wall clock.

The 26B-A4B base remains feasible on this hardware via stream-load (init_empty_weights + per-tensor load-quantize-place from safetensors directly). That path is documented as a backlog item; we revisit if E4B's accuracy is insufficient.

## §6 Calibration as the primary success metric

V22 ship verdict measures the deliverable as a calibrated belief score, not just a classifier:

| Metric | Target |
|---|---|
| Brier score on 482 holdout vs curator gold | ≤ 0.18 (T-phase: ~0.22 estimated) |
| Expected Calibration Error (ECE) | ≤ 0.08 |
| Brier-skill score vs INDRA BeliefEngine baseline | ≥ +15% |
| Reliability diagram | diagonal recovered within 10pp per bin |
| Per-stmt-type ECE | ≤ 0.15 (no class with worse calibration) |
| Raw accuracy (secondary) | ≥ 65% |
| Decision-only accuracy (secondary) | ≥ 75% |

The skill score vs BeliefEngine is the strongest empirical claim we can make: "our scorer's per-record correctness probability is more calibrated than INDRA's existing one." That's the V-phase ship.

## §7 Holdout protection (extended from S/T/U-phases)

For evaluation-set integrity, exclude from training:
- All `source_hash` ∈ holdout (~990 records, ~482 unique source_hashes plus near-dupes)
- All `matches_hash` ∈ holdout (~442 records)
- All `(pmid, entity_pair)` ∈ holdout (~1,009 paper-level near-dupes)

Total exclusion: ~2,000-2,500 records (~0.3% of corpus).

`pmid`-only overlap (8,816 records) is NOT excluded. These are independent statements from the same papers as holdout records — too aggressive to drop, would lose ~1% of corpus including unrelated extractions.

## §8 Migration discipline (carries from S/T/U-phases)

- Single scoring path. After V-phase ships, the prompted-probe code path is removed (not held as a fallback flag). The orchestrator calls fine-tuned adapters; that's the only path.
- All training data uses synthetic placeholder names where appropriate (contamination guard from May 2026 incident continues to apply).
- Brutalist gates at V1 (doctrine), V8 (data + env), V11 (feasibility), V18 (impl), V20 (probe), V22 (ship). Each emits a written PASS / FAIL.
- Test changes track behavior changes 1:1 — no test deletion, only conversion to test new behavior.

## §9 What V-phase explicitly does not do

- **No model-class change** — within the Gemma 4 family. We are NOT switching to a different model family (Llama, Qwen, Mistral). The E4B / 26B-A4B difference is a capability/hardware tradeoff within Gemma 4.
- **No fine-tuning beyond LoRA** — full fine-tuning is out of memory budget (24 GB) and would overfit.
- **No multi-task LoRA** — per-probe adapters per §4.2.
- **No replacement of the four-probe architecture** — we keep the closed-set discipline and the flat adjudicator.
- **No replacement of the substrate** — Gilda, CATALOG, M9, M10 stay as deterministic preprocessing.
- **No new probes** — the five we have are sufficient if calibrated.
- **No INDRA BeliefEngine integration** beyond consuming our scorer's output. Whether INDRA's belief layer fully migrates to our score is downstream.
- **No public release of fine-tuned adapters** — V-phase ship pushes code to origin/main; adapter weights stay on the host or are versioned via HuggingFace Hub privately.

## §10 V-phase task structure (24 tasks total)

```
Phase 1 — Setup (V0-V4.5)
  V0  doctrine ─► V1 brutalist gate
  V2 sync repo ─► V3 ROCm-PyTorch env ─► V4 download base model
  V4.5 corpus inventory (DONE — empirical foundation for V5)

Phase 2 — Data engineering (V5-V8)
  V5 multi-source LF doctrine ─► V6 build derivation+aggregation
  ─► V7 hand-label active-learning seed ─► V8 data + env GATE

Phase 3 — Feasibility (V9-V11)
  V9 train relation_axis (4-6h, highest priority) ─► V10 validate
  ─► V11 SCALE / ITERATE / ABORT decision

Phase 4 — Per-probe training (V12-V16) — only if V11 SCALE
  V12 subject_role + V13 object_role + V14 scope + V15 grounding
  ─► V16 per-probe validation review

Phase 5 — Integration + ship (V17-V23)
  V17 integrate adapters ─► V18 impl gate
  V19 stratified probe ─► V20 probe gate
  V21 full holdout ─► V22 ship verdict ─► V23 push
```

Wall-clock estimate: 8-11 days focused work, with V11 acting as a kill-switch if the feasibility check fails.

## §11 Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| ROCm + bitsandbytes + PEFT compatibility breakage | resolved | M1 + M2 + M6' verified end-to-end on gfx1100 with AMD's `ROCm/bitsandbytes:rocm_enabled` fork (commit 4fa939b3) |
| E4B's ~8B parameter capacity is insufficient for the closed-set probe distinctions | medium | V11 acts as kill-switch; if relation_axis adapter underperforms T-phase prompted by ≥2pp, escalate to 26B-A4B via stream-load redesign |
| Vision/audio towers in E4B confuse PEFT target_modules matching | resolved | M6' confirmed regex `^model\.language_model\.layers\.\d+\.self_attn\.[qkvo]_proj$` correctly isolates language tower (vision/audio Gemma4ClippableLinear are skipped) |
| Label derivation produces noisy labels despite multi-source aggregation | medium | V7 active learning loop; V8 calibration on curated subset |
| LoRA training instability at rank 16 | low | M6' showed loss decreased monotonically 14.38→8.47 over 5 steps with no instability |
| llama-server downtime during training (gemma-remote endpoint offline) | low | Restart container after each training run; pre-announce windows |
| Adapter swap latency at inference exceeds 5s p50 budget | low | Profile in V17; merge adapters into base if needed |
| HF Gemma license requires manual click-through | resolved | User accepted, token persisted, downloads of 26B-A4B and E4B both succeeded |
| Stream-load implementation needed if 26B-A4B is reconsidered | medium-low (only if V11 fails) | Documented backlog item: write `from accelerate import init_empty_weights` + per-tensor safetensors loader → quantize → place flow. ~1 day work. |

## §12 What success looks like, concretely

V-phase ships when V22 verdict shows:
1. Brier score on 482 holdout improves by ≥15% over INDRA BeliefEngine baseline.
2. ECE ≤ 0.08 (well-calibrated).
3. Raw accuracy on 482 holdout ≥ 65% (vs T-phase 60.17%).
4. No stmt-type calibration regression (per-class ECE ≤ 0.15).
5. Adapter inference path runs end-to-end without code-path fallbacks.
6. Memory in `project_v_phase_ship.md` documents the calibration-replacement framing for downstream consumers.

If V11 ABORTs (relation_axis fine-tuning doesn't beat T-phase prompted by ≥2pp), V-phase reverts to a fallback: U-phase rollback to architectural pieces only (drop U6/U9 prompt changes), ship at T-phase parity, document the empirical ceiling for prompted approach at 26B scale.

## §13 What this document commits us to

After V1 brutalist review, this doctrine becomes the binding contract for V-phase. Changes to the architectural commitments (§4) require writing a new doctrine; tactical refinements (V5/V6/V7 implementation specifics) can iterate within the framework laid out here.

Specifically, V-phase commits to:
- Fine-tuning, not prompting
- Approach B (separate adapters)
- Multi-source weak supervision (no belief)
- Calibration as primary metric
- Strict holdout protection
- Migration discipline (single path, no flags)

These are the load-bearing decisions. Everything else is implementation.

### Revision history

- 2026-05-05: initial doctrine; base committed to Gemma 4 26B-A4B-it.
- 2026-05-05 (V1 gate): identified bitsandbytes 3D-MoE blocker; recommended Option B (E4B pivot) but the user requested deeper investigation.
- 2026-05-06 (M3 / M6 attempts): unfuse-experts path validated bit-exact in isolation, but the load-then-convert flow on 26B-A4B exhausted host CPU RAM (62 GB) and stalled in swap thrashing after 2.5 hours. Ruled this naive flow out; the alternative is stream-load redesign (~1 day work).
- 2026-05-06 (M6' pass): pivoted base model to Gemma 4 E4B-it. Verified end-to-end: bnb 4-bit load (8.8 GB VRAM), PEFT LoRA on 168 attention projections (9M trainable params), 5 training steps with monotonic loss decrease 14.38→8.47, peak 17.2 GB VRAM. The 26B-A4B path is preserved as backlog if V11 says E4B capacity is insufficient.
- §1, §2, §4.1, §5, §9, §11 updated to reflect E4B as base. Other sections unchanged.
