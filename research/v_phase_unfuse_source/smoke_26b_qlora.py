"""M6 smoke: Gemma 4 26B-A4B unfuse + bnb 4-bit + LoRA training feasibility check.

Pipeline:
  1. Load bf16 weights to CPU (no GPU memory yet)
  2. Apply unfuse (Gemma4TextExperts → UnfusedGemma4TextExperts in-place)
  3. Replace nn.Linear with bnb.nn.Linear4bit (still on CPU, marked for 4-bit)
  4. Move to GPU — bnb actually quantizes here
  5. Add LoRA via PEFT
  6. Run N small forward+backward+step iterations with synthetic data
  7. Report peak VRAM, loss trajectory

Requires:
  - llama-server stopped (frees ~22 GB VRAM)
  - Gemma 4 26B-A4B-it safetensors cached
  - v-phase venv active

Usage:
  python smoke_26b_qlora.py --steps 5 --seq-len 512 --lora-rank 16
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from unfuse_gemma4 import unfuse_gemma4_model


def check_vram_or_abort(min_free_gb: float = 22.0):
    free_b, total_b = torch.cuda.mem_get_info(0)
    free_gb, total_gb = free_b / 1e9, total_b / 1e9
    print(f"  GPU free VRAM: {free_gb:.1f} / {total_gb:.1f} GB")
    if free_gb < min_free_gb:
        print(f"  ABORT: need ≥{min_free_gb} GB free for this smoke. "
              "Stop llama-server with `pkill -f llama-server` before running.")
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--skip-vram-check", action="store_true")
    args = parser.parse_args()

    print(f"=== M6 SMOKE: {args.model} ===")
    print(f"  steps={args.steps} seq_len={args.seq_len} lora_rank={args.lora_rank}")

    if not args.skip_vram_check:
        print("\n[0/6] Checking VRAM availability...")
        check_vram_or_abort(min_free_gb=22.0)

    # ── Step 1: Load bf16 to CPU ──────────────────────────────────────
    print("\n[1/6] Loading model in bf16 to CPU...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="cpu",
        low_cpu_mem_usage=True,
    )
    print(f"  loaded in {time.time()-t0:.0f}s")

    # ── Step 2: Unfuse experts ────────────────────────────────────────
    print("\n[2/6] Unfusing 3D fused experts to nn.Linear...")
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextExperts
    n_fused_before = sum(1 for m in model.modules() if isinstance(m, Gemma4TextExperts))
    t1 = time.time()
    unfuse_gemma4_model(model, free_after=True)
    n_fused_after = sum(1 for m in model.modules() if isinstance(m, Gemma4TextExperts))
    print(f"  unfused {n_fused_before} → {n_fused_after} fused layers in {time.time()-t1:.0f}s")
    assert n_fused_after == 0, "unfuse incomplete"

    # ── Step 3+4: Replace nn.Linear with Linear4bit AND move to GPU per-module ──
    # transformers' replace_with_bnb_linear creates Linear4bit on meta device
    # without copying weights, expecting from_pretrained checkpoint load. We
    # need a post-hoc conversion: copy weight to GPU bf16, wrap in Params4bit,
    # let bnb quantize on .cuda(), replace in parent.
    print("\n[3/6] Replacing nn.Linear → Linear4bit and moving to GPU per-module...")
    import torch.nn as nn
    import bitsandbytes as bnb
    # Quantize EVERYTHING except the heads/embeddings (need fp16 for output precision).
    # Vision tower IS quantized — frozen anyway, saves VRAM.
    EXCLUDE_SUBSTRINGS = ("lm_head", "embed_tokens")
    targets = []
    for name, mod in model.named_modules():
        if type(mod) is nn.Linear:
            if any(ex in name for ex in EXCLUDE_SUBSTRINGS):
                continue
            targets.append(name)
    print(f"  {len(targets)} Linear modules to convert+quantize")

    t2 = time.time()
    converted = 0
    for name in targets:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        old: nn.Linear = getattr(parent, attr)
        new = bnb.nn.Linear4bit(
            old.in_features,
            old.out_features,
            bias=(old.bias is not None),
            compute_dtype=torch.bfloat16,
            quant_type="nf4",
            compress_statistics=True,
        )
        new.weight = bnb.nn.Params4bit(
            old.weight.data,
            requires_grad=False,
            quant_type="nf4",
        )
        if old.bias is not None:
            new.bias = nn.Parameter(old.bias.data, requires_grad=False)
        new = new.to("cuda")  # triggers quantization
        setattr(parent, attr, new)
        del old
        converted += 1
        if converted % 200 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            free_b, _ = torch.cuda.mem_get_info(0)
            print(f"  {converted}/{len(targets)} converted; GPU free: {free_b/1e9:.1f} GB")
    gc.collect()
    torch.cuda.empty_cache()

    # Move the not-converted modules (norm, lm_head, embed_tokens, vision_tower) to GPU as bf16
    print("  moving non-quantized modules (norm/lm_head/embed) to GPU bf16...")
    for name, mod in list(model.named_modules()):
        # Skip parents of already-converted Linear4bit modules
        if any(isinstance(c, bnb.nn.Linear4bit) for c in mod.children()):
            continue
        if any(p.device.type == "cpu" for p in mod.parameters(recurse=False)) or \
           any(b.device.type == "cpu" for b in mod.buffers(recurse=False)):
            mod.to("cuda")

    print(f"  converted+moved in {time.time()-t2:.0f}s")
    print(f"  VRAM after quantize: {torch.cuda.memory_allocated()//1024//1024} MiB")

    # ── Step 5: Add LoRA via PEFT ─────────────────────────────────────
    print("\n[5/6] Configuring LoRA (attention only; experts not targeted)...")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Step 6: Run N training steps on synthetic data ────────────────
    print(f"\n[6/6] Running {args.steps} training steps...")
    texts = [
        "INDRA scores phosphorylation evidence with belief calibration. "
        "Statements with curated database support tend to be more reliable.",
        "Activation of STAT3 by IL10 has multiple supporting publications. "
        "The effect is dose-dependent and tissue-specific.",
        "The MAPK pathway involves MEK and ERK phosphorylation cascades. "
        "Negative regulation by phosphatases provides feedback control.",
        "Loss of function mutations in TP53 affect cell cycle regulation. "
        "Tumor suppressor activity is required for genome stability.",
    ]
    batch = tok(texts, padding=True, truncation=True, max_length=args.seq_len, return_tensors="pt").to("cuda")
    labels = batch["input_ids"].clone()
    labels[labels == tok.pad_token_id] = -100

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-4)

    losses = []
    torch.cuda.reset_peak_memory_stats()
    t4 = time.time()
    for step in range(args.steps):
        out = model(**batch, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(out.loss.item())
        peak_mb = torch.cuda.max_memory_allocated() // 1024 // 1024
        elapsed = time.time() - t4
        print(f"  step {step+1}/{args.steps}: loss={out.loss.item():.4f} "
              f"peak_VRAM={peak_mb} MiB elapsed={elapsed:.1f}s")

    print(f"\n=== M6 RESULT ===")
    print(f"  loss trajectory: {[f'{l:.4f}' for l in losses]}")
    print(f"  peak VRAM during training: {torch.cuda.max_memory_allocated()//1024//1024} MiB")
    print(f"  total wall time: {time.time()-t0:.0f}s")
    if losses[-1] < losses[0]:
        print("  PASS: loss decreased over training steps")
    else:
        print("  WARN: loss did not decrease; investigate before V-phase commits")


if __name__ == "__main__":
    main()
