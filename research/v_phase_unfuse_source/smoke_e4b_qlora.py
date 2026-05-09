"""M6': dense Gemma 4 E4B QLoRA smoke.

E4B is dense — no unfuse needed. Standard transformers + bnb 4-bit + PEFT.

Usage:
  python smoke_e4b_qlora.py --steps 5 --seq-len 512 --lora-rank 16
"""

import argparse
import sys
import time

import torch


def check_vram_or_abort(min_free_gb: float = 6.0):
    free_b, total_b = torch.cuda.mem_get_info(0)
    free_gb, total_gb = free_b / 1e9, total_b / 1e9
    print(f"  GPU free VRAM: {free_gb:.1f} / {total_gb:.1f} GB")
    if free_gb < min_free_gb:
        print(f"  ABORT: need ≥{min_free_gb} GB free; stop llama-server first.")
        sys.exit(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-4-E4B-it")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--skip-vram-check", action="store_true")
    args = parser.parse_args()

    print(f"=== E4B SMOKE: {args.model} ===")
    print(f"  steps={args.steps} seq_len={args.seq_len} lora_rank={args.lora_rank}")

    if not args.skip_vram_check:
        print("\n[0/4] VRAM check...")
        check_vram_or_abort(min_free_gb=6.0)

    print("\n[1/4] Loading model with bnb 4-bit + GPU placement...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print(f"  loaded in {time.time()-t0:.0f}s")
    print(f"  VRAM after load: {torch.cuda.memory_allocated()//1024//1024} MiB")

    print("\n[2/4] Adding LoRA (attention modules)...")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    # Target only the language_model's attention projections (plain Linear4bit).
    # vision_tower and audio_tower wrap their q/k/v/o in Gemma4ClippableLinear which
    # PEFT can't handle. Regex pattern restricts to language_model layers only.
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=r"^model\.language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$",
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"\n[3/4] Running {args.steps} training steps...")
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
    t1 = time.time()
    for step in range(args.steps):
        out = model(**batch, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(out.loss.item())
        peak_mb = torch.cuda.max_memory_allocated() // 1024 // 1024
        elapsed = time.time() - t1
        print(f"  step {step+1}/{args.steps}: loss={out.loss.item():.4f} "
              f"peak_VRAM={peak_mb} MiB elapsed={elapsed:.1f}s", flush=True)

    print(f"\n[4/4] Result")
    print(f"  loss trajectory: {[f'{l:.4f}' for l in losses]}")
    print(f"  peak VRAM during training: {torch.cuda.max_memory_allocated()//1024//1024} MiB")
    print(f"  total wall time: {time.time()-t0:.0f}s")
    if losses[-1] < losses[0]:
        print("  PASS: loss decreased")
    else:
        print("  WARN: loss did not decrease")


if __name__ == "__main__":
    main()
