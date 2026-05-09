# V1 brutalist gate — V-phase doctrine review

Date: 2026-05-05
Status: **FAIL ON CRITICAL FINDING** — doctrine §5 (memory feasibility) is empirically blocked. Doctrine must be revised before V2.

## Summary

The V-phase doctrine commits to QLoRA on Gemma 4 26B-A4B with bitsandbytes 4-bit quantization fitting in 24 GB VRAM. **This path does not exist as of May 2026.** Multiple independent sources confirm:

> "Gemma 4's MoE 26B-A4B uses 3D fused expert tensors that bitsandbytes can't quantize yet — blocking QLoRA on 32 GB GPUs."

> "Unsloth, which has custom MoE quantization that handles Qwen 3.5's fused tensors, deliberately skipped the Gemma 4 26B-A4B for their bnb-4bit releases, publishing quantized versions for the dense Gemma 4 models (E2B, E4B, 31B) but not the MoE one."

> "The 3D tensor layout is genuinely harder to handle... the fundamental issue is that bitsandbytes has not yet implemented support for the 3D fused tensor format used by Gemma 4 MoE models."

Vllm's MXFP4 quantization path has the same bug ([vllm issue #39000](https://github.com/vllm-project/vllm/issues/39000)). AWQ works for INFERENCE only — does not enable LoRA training.

The V-phase commitment to "fine-tune the same model we're already running for inference" cannot proceed on this hardware as written.

## Verifiable host claims (V0 §5)

| Claim | Status | Evidence |
|---|---|---|
| ROCm 6.4 with rocBLAS, hipBLAS, hipBLASLt available | ✓ TRUE | confirmed via `ls /opt/rocm/lib/` |
| PyTorch ROCm 6.2 wheel available at pytorch.org | ✓ TRUE | curl returns 200 |
| 24 GB VRAM on 7900 XT | ✓ TRUE | confirmed via sysfs |
| 21 GB held by llama-server | ✓ TRUE | confirmed via mem_info_vram_used |
| 596 GB free disk | ✓ TRUE | confirmed via df |
| HF token persisted at ~/.cache/huggingface/token (mode 600) | ✓ TRUE | curl whoami-v2 succeeds as ejmockler |
| PEFT installable | ✓ ASSUMED — not yet installed | requires V3 |
| **bitsandbytes 4-bit can quantize Gemma 4 26B-A4B** | ✗ **FALSE** | empirical blocker — see above |
| **QLoRA fits in 19-22 GB on 24 GB GPU** | ✗ **FALSE** | premise broken — bnb cannot 4-bit-quantize this model |

## What this means for the doctrine

V-phase has three options, ranked:

### Option A — Pivot to Gemma 4 31B (dense)

- **Pros**: same model family, has working bnb-4bit (Unsloth-published), well-tested QLoRA path
- **Cons**: dense 31B has FULL activation memory (no MoE benefit). 4-bit base ~16 GB + dense activations ~10-15 GB at batch=1 seq=2048 → **27-32 GB total** — does NOT fit on the 24 GB 7900 XT
- **Verdict: NOT VIABLE on this host**

### Option B — Pivot to Gemma 4 E4B (smaller dense, ~4B effective)

- **Pros**:
  - Dense; bnb-4bit support exists; well-tested
  - Fits comfortably in 24 GB at fp16 (~8 GB base + ~3 GB activations + LoRA = ~12 GB peak training)
  - Can train at full bf16 precision — no QLoRA brittleness
  - Faster training (~2-3 hours per probe vs 4-6 for 26B-A4B)
  - No need to stop llama-server during training (E4B fits alongside the 21 GB hold)
- **Cons**:
  - Lower ceiling — 4B parameters vs 26B
  - Different model than the gemma-remote inference endpoint, so deployment story changes (we'd serve E4B for probes alongside or replacing 26B-A4B)
  - Capacity may be insufficient for fine-grained probe distinctions (e.g., direct_amount_match vs direct_sign_match)
- **Verdict: VIABLE; capability uncertain**

### Option C — Wait / abort

- **Wait**: bitsandbytes upstream may add MoE-3D support eventually. No timeline.
- **Abort V-phase**: fall back to U-phase iteration with calibration discipline. Concede the empirical ceiling at prompted 26B.
- **Verdict: VIABLE as a defined fallback**

## Other findings from the doctrine review

### Finding #2 — `epistemics.hedgings` and `epistemics.negated` are empty in this corpus

Doctrine §3.3 cites `epistemics.direct` as a primary LF for relation_axis (correct — 1.54M evidence pieces have it). But it implicitly assumed the parallel `hedgings` and `negated` fields would be available for the scope probe. **They are empty across all 2.85M evidence pieces** (V4.5 confirmed). The scope probe must derive hedged/negated from M10 substrate + new regex, not from extractor metadata. This is a real but addressable limitation; doctrine §4.3 LF list should drop those two and emphasize substrate-only signals for scope.

### Finding #3 — REACH `found_by` patterns are extractor-specific

Doctrine §3.3 identifies `found_by` as a primary LF. But only REACH-extracted evidence has these patterns (1.29M of 2.85M = 45%). Sparser, MedScan, EIDOS use different (or no) annotation conventions. The LF will only fire on REACH evidence. Need to handle this distribution explicitly in V5.

### Finding #4 — `supported_by` is preassembler-derived, not belief-derived

Doctrine §3.3 hesitated on `supported_by` ("derived from belief"). Re-checking: `supported_by` comes from INDRA's preassembler statement-equivalence grouping, not from the BeliefEngine. Statements with shared `pa_hash`/`matches_hash` and compatible structure get linked. This IS a legitimate independent signal we can use as an LF.

### Finding #5 — Unsloth published GGUF for 26B-A4B but not bnb-4bit

The host already has `unsloth/gemma-4-26B-A4B-it-GGUF` cached (config-only, weights pending). GGUF is an INFERENCE format (llama.cpp), not a training-compatible quant. We cannot LoRA on the GGUF weights directly; would need the safetensors. This was already noted in V4 task description and is not a NEW issue.

### Finding #6 — Doctrine's success metrics are sound but unmeasurable until V21

Doctrine §6 specifies Brier score, ECE, Brier-skill score. These are correctly chosen. But measuring them requires a calibrated INDRA BeliefEngine baseline on the same 482 holdout. We need to compute that baseline number (BeliefEngine's Brier vs gold) BEFORE V21 to know what "+15% skill" target means. This should be part of V21 setup, not a deferred concern.

### Finding #7 — MoE LoRA on router weights is itself fraught

Even setting aside the bnb-quantization issue, doctrine §4.1 specifies "rank 16, attention + FFN + router". LoRA on MoE router weights is supported in PEFT v0.12+ but is documented as "experimental — may degrade routing quality." Some published Gemma 4 26B-A4B fine-tunes (e.g., dealignai/Gemma-4-26B-A4B-JANG_4M-CRACK seen in earlier search) appear to have done this successfully via custom paths, but the standard PEFT path is unproven.

## Critical findings to address before V2

1. **Block #1 (CRITICAL):** revise doctrine §5 to choose Option A (abort), Option B (pivot to E4B), or Option C (wait). Cannot proceed with V2 sync until base model is decided.

2. **Findings #2-#4 are tactical** and addressed in V5 (data prep doctrine).

3. **Finding #6 needs a sub-task**: compute INDRA BeliefEngine baseline Brier/ECE on 482 holdout. This was implicit; should be explicit. Either insert as a V0.5 task or as part of V21 setup.

4. **Finding #7**: if we pivot to E4B (Option B), router-weight LoRA isn't an issue (E4B is dense, no MoE). If we somehow find a path back to 26B-A4B, this needs validation in V3 toy run.

## My recommendation

**Pivot to Option B (Gemma 4 E4B).** Specifically:

- E4B is the dense Gemma 4 model best-suited to consumer GPUs and was designed for exactly this use case
- Training without quantization is more reliable than QLoRA — fewer moving parts
- The lower capability is offset by the closed-set classification task being relatively simple (5-8 classes per probe)
- 2-3 hour iterations enable fast feedback during V11 feasibility check
- No conflict with running llama-server in parallel (E4B fits in spare VRAM)

If E4B's accuracy on V11 feasibility check is insufficient, we have time to pivot to Option C (abort V-phase, fall back to T-phase with U-phase architectural pieces).

If you want to KEEP 26B-A4B as the inference base, there's a natural deployment pattern: train E4B for probes, keep 26B-A4B as the held-out reference for any other LLM-shaped task. Two models on the same GPU during inference (~8 GB E4B + ~13 GB 26B-A4B 4-bit) just barely fits.

## Verdict

**FAIL on §5 memory feasibility claim.** Doctrine must be revised before V2. Three options laid out; my recommendation is Option B (E4B). User decision required before any host-side work proceeds.

## Sources (V1 verification)

- [Gemma 4 LoRA Fine-Tuning on RTX 5090: What Works and What Doesn't (ai.rs)](https://ai.rs/ai-developer/gemma-4-lora-fine-tuning-rtx-5090) — confirms bnb-MoE-3D limitation
- [Fine-Tune Gemma 4 with LoRA & QLoRA: Complete Guide (Lushbinary)](https://lushbinary.com/blog/fine-tune-gemma-4-lora-qlora-complete-guide/) — confirms Unsloth skipped 26B-A4B
- [vllm issue #39000](https://github.com/vllm-project/vllm/issues/39000) — same 3D-tensor issue in vllm
- [cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit · Hugging Face](https://huggingface.co/cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit) — AWQ inference works, training does not
- [Gemma 4 Re-Quantization Guide (dasroot.net 2026-05)](https://dasroot.net/posts/2026/05/gemma-4-re-quantization-guide/) — recent quantization status
- [google/gemma-4-E4B · Hugging Face](https://huggingface.co/google/gemma-4-E4B) — pivot target
