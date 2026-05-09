"""V9 — QLoRA fine-tune Gemma 4 E4B-it on relation_axis labels (V8b).

Run on the ROCm host (100.97.101.59) with `~/venvs/v-phase/bin/python`.

Pipeline:
  1. Load gemma-4-E4B-it with bnb 4-bit (NF4 + double-quant + bf16 compute).
  2. Apply LoRA on `model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj`
     — pattern from M6' smoke. Vision/audio towers are skipped.
  3. Stream V8b chat-format JSONL, apply chat template, mask non-completion
     tokens so loss is only on the assistant's `{answer, rationale}` JSON.
  4. Train with AdamW + linear warmup; checkpoint at end + every save_steps.
  5. Save adapter (PEFT-format) and trainer state.

V9.1 additions (2026-05-07, before V9 re-run + V11-V15):
  - Class-balanced loss weighting (inverse-frequency, capped).
  - Three-layer telemetry: TensorBoard + metrics.jsonl + predictions.jsonl.
  - run.json snapshot at start, training_report.md at end.

Defaults are tuned for a single Radeon RX 7900 XTX (~25 GB VRAM) with
~22 GB free; lower seq_len / batch / grad_accum if VRAM is tighter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import torch
import torch.nn.functional as F


RELATION_AXIS_CLASSES = [
    "direct_sign_match",
    "direct_sign_mismatch",
    "direct_axis_mismatch",
    "no_relation",
]
CLASS_TO_IDX = {c: i for i, c in enumerate(RELATION_AXIS_CLASSES)}

# Same regex V10 uses to parse the model's generation.
_CLASS_RE = re.compile(
    r'"answer"\s*:\s*"(direct_sign_match|direct_sign_mismatch|'
    r'direct_axis_mismatch|no_relation)"'
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E4B-it")
    ap.add_argument("--train-jsonl",
                     default="data/v_phase/train/v8b_relation_axis.jsonl")
    ap.add_argument("--val-jsonl",
                     default="data/v_phase/val/v8b_relation_axis.jsonl")
    ap.add_argument("--out-dir", default="data/v_phase/lora/relation_axis_v9")
    ap.add_argument("--seq-len", type=int, default=3072)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--log-steps", type=int, default=10)
    ap.add_argument("--max-train-records", type=int, default=0,
                     help="cap training records (0 = use all)")
    ap.add_argument("--smoke-steps", type=int, default=0,
                     help="if >0, train this many steps and exit (smoke check)")
    ap.add_argument("--min-free-gb", type=float, default=8.0,
                     help="abort if free VRAM below this (GB)")
    ap.add_argument("--seed", type=int, default=42)
    # Class-balanced loss weighting: ON by default; pass --no-class-balance-weights to disable.
    ap.add_argument(
        "--class-balance-weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply inverse-frequency per-class weights to the loss (default ON)",
    )
    ap.add_argument(
        "--class-weight-cap", type=float, default=5.0,
        help="cap any single class weight at this value to prevent rare-class blow-ups",
    )
    # Eval-loss objective: by default, mirror the training objective (weighted CE when
    # class-balanced training is on). Pass --no-eval-weighted to force unweighted eval.
    ap.add_argument(
        "--eval-weighted",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="compute eval_loss with the same weighted CE used in training (default ON)",
    )
    # Prediction-trace telemetry: how many val records to greedy-decode at every eval.
    ap.add_argument("--pred-trace-n", type=int, default=20,
                     help="num val records to decode at each eval for predictions.jsonl")
    ap.add_argument("--pred-trace-max-new-tokens", type=int, default=64,
                     help="max_new_tokens for predictions.jsonl decoding")
    return ap.parse_args()


def check_vram(min_free_gb: float = 8.0) -> None:
    free_b, total_b = torch.cuda.mem_get_info(0)
    free_gb, total_gb = free_b / 1e9, total_b / 1e9
    print(f"  GPU free VRAM: {free_gb:.1f} / {total_gb:.1f} GB")
    if free_gb < min_free_gb:
        print(f"  ABORT: need >={min_free_gb} GB free; stop other GPU services first.")
        sys.exit(2)


def gpu_mem_gb() -> tuple[float, float]:
    free_b, total_b = torch.cuda.mem_get_info(0)
    return free_b / 1e9, (total_b - free_b) / 1e9


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def encode_record(rec: dict, tok, seq_len: int) -> dict | None:
    """Tokenize a chat record. Returns dict with input_ids, attention_mask,
    labels (final assistant turn only — earlier tokens are masked to -100),
    and class_idx (carried through so the loss layer can pick a per-record
    weight). Returns None if the encoded length exceeds `seq_len` even after
    the final assistant is included.
    """
    messages = rec["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        return None

    # Render the full conversation including the final assistant turn.
    # apply_chat_template(tokenize=True) returns a BatchEncoding; pull
    # `input_ids` to get the flat token list.
    full_enc = tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
    )
    full_ids = full_enc.input_ids if hasattr(full_enc, "input_ids") else list(full_enc)
    # Render up to (but excluding) the final assistant turn. Use
    # add_generation_prompt=True so the boundary token (e.g. `<start_of_turn>model`)
    # is on the prefix side and the assistant tokens are exactly what we
    # want to compute loss on.
    prefix_enc = tok.apply_chat_template(
        messages[:-1], tokenize=True, add_generation_prompt=True,
    )
    prefix_ids = prefix_enc.input_ids if hasattr(prefix_enc, "input_ids") else list(prefix_enc)

    if len(full_ids) > seq_len:
        return None
    if len(prefix_ids) >= len(full_ids):
        return None

    labels = [-100] * len(prefix_ids) + list(full_ids[len(prefix_ids):])
    # Pad to seq_len with attention 0.
    pad_id = tok.pad_token_id or tok.eos_token_id
    n_pad = seq_len - len(full_ids)
    input_ids = list(full_ids) + [pad_id] * n_pad
    attention_mask = [1] * len(full_ids) + [0] * n_pad
    labels = labels + [-100] * n_pad

    answer = (rec.get("metadata") or {}).get("answer")
    class_idx = CLASS_TO_IDX.get(answer, -1)  # -1 = unknown class -> weight 1.0

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "class_idx": class_idx,
    }


def collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
        "class_idx": torch.tensor([b["class_idx"] for b in batch], dtype=torch.long),
    }


def encode_dataset(records: list[dict], tok, seq_len: int) -> list[dict]:
    encoded = []
    n_dropped = 0
    for r in records:
        e = encode_record(r, tok, seq_len)
        if e is None:
            n_dropped += 1
            continue
        encoded.append(e)
    if n_dropped:
        print(f"  dropped {n_dropped}/{len(records)} records exceeding seq_len={seq_len}")
    return encoded


def compute_class_weights(
    train_recs: list[dict], cap: float = 5.0
) -> tuple[torch.Tensor, dict[str, int]]:
    """Inverse-frequency class weights: w_c = N / (K * count_c), capped at `cap`.
    Returns (weights tensor [K], counts dict). Classes absent from training
    data get weight 1.0.
    """
    counts: Counter = Counter()
    for r in train_recs:
        ans = (r.get("metadata") or {}).get("answer")
        if ans in CLASS_TO_IDX:
            counts[ans] += 1
    K = len(RELATION_AXIS_CLASSES)
    N = sum(counts.values())
    weights = torch.ones(K, dtype=torch.float32)
    if N > 0:
        for c, idx in CLASS_TO_IDX.items():
            n_c = counts.get(c, 0)
            if n_c > 0:
                w = N / (K * n_c)
                weights[idx] = float(min(w, cap))
            else:
                weights[idx] = float(cap)  # absent -> cap (unobserved is rare)
    return weights, dict(counts)


def file_meta(path: Path) -> dict:
    """Return {path, size, sha256} for a file (cheap streaming hash)."""
    h = hashlib.sha256()
    sz = 0
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            sz += len(chunk)
    return {"path": str(path), "bytes": sz, "sha256": h.hexdigest()}


def _git_state() -> dict:
    """Best-effort {sha, dirty} for the current working tree. Returns
    {"sha": None, "dirty": None, "error": "..."} when not in a git repo or
    when git is unavailable, so the run.json snapshot still serializes.
    """
    import subprocess
    try:
        # Round-2 fix: 5-second timeout so a slow/hung filesystem can't
        # block training startup indefinitely.
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode().strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL, timeout=5,
        ).decode()
        return {"sha": sha, "dirty": bool(status.strip())}
    except Exception as e:  # FileNotFoundError, CalledProcessError, TimeoutExpired, etc.
        return {"sha": None, "dirty": None, "error": str(e)}


def _env_snapshot() -> dict:
    """Snapshot ROCm/HF/tokenizer env vars relevant for reproducibility."""
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "PYTORCH_HIP_ALLOC_CONF",
        "PYTORCH_ALLOC_CONF",
        "HSA_OVERRIDE_GFX_VERSION",
        "HF_HOME",
        "TRANSFORMERS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
    )
    return {k: os.environ.get(k) for k in keys}


def parse_answer(text: str) -> str | None:
    """Pull the answer class out of the model's generation. Same logic V10
    uses; tolerant to JSON not parsing cleanly."""
    m = _CLASS_RE.search(text)
    if m:
        return m.group(1)
    for c in RELATION_AXIS_CLASSES:
        if c in text:
            return c
    return None


def per_class_metrics(eval_records: list[dict]) -> dict:
    """Compute per-class precision/recall/F1 + micro/macro from a list of
    {gold_class, parsed_pred} dicts. Records with parsed_pred=None count
    against recall but are excluded from precision denominators.
    """
    gold_counts: Counter = Counter()
    pred_counts: Counter = Counter()
    tp: Counter = Counter()
    n_total = 0
    n_correct = 0
    n_unparseable = 0
    for r in eval_records:
        gold = r["gold_class"]
        pred = r["parsed_pred"]
        gold_counts[gold] += 1
        n_total += 1
        if pred is None:
            n_unparseable += 1
            continue
        pred_counts[pred] += 1
        if pred == gold:
            tp[gold] += 1
            n_correct += 1
    micro = n_correct / n_total if n_total else 0.0
    per_class = {}
    macro_recall_sum = 0.0
    macro_recall_n = 0
    for c in RELATION_AXIS_CLASSES:
        s = gold_counts[c]
        p = pred_counts[c]
        recall = (tp[c] / s) if s else None
        prec = (tp[c] / p) if p else None
        f1 = (
            2 * prec * recall / (prec + recall)
            if (prec and recall and (prec + recall) > 0)
            else None
        )
        per_class[c] = {
            "support": s,
            "predicted": p,
            "precision": prec,
            "recall": recall,
            "f1": f1,
        }
        if recall is not None:
            macro_recall_sum += recall
            macro_recall_n += 1
    macro_recall = macro_recall_sum / macro_recall_n if macro_recall_n else 0.0
    return {
        "micro_acc": micro,
        "macro_recall": macro_recall,
        "n_total": n_total,
        "n_unparseable": n_unparseable,
        "per_class": per_class,
    }


def sparkline(values: list[float], width: int = 60) -> str:
    """Tiny ASCII sparkline. Returns empty string for empty input."""
    if not values:
        return ""
    bars = " ▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    rng = hi - lo if hi > lo else 1.0
    if len(values) > width:
        # Downsample by averaging into `width` buckets.
        out = []
        n = len(values)
        for i in range(width):
            a = int(i * n / width)
            b = int((i + 1) * n / width)
            seg = values[a:b] if b > a else values[a:a + 1]
            out.append(sum(seg) / len(seg))
        values = out
    chars = []
    for v in values:
        idx = int(round((v - lo) / rng * (len(bars) - 1)))
        idx = max(0, min(len(bars) - 1, idx))
        chars.append(bars[idx])
    return "".join(chars)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    print(f"=== V9 RELATION_AXIS LORA TRAIN ===")
    print(f"  model={args.model}")
    print(f"  train={args.train_jsonl} val={args.val_jsonl}")
    print(f"  seq_len={args.seq_len} lora_rank={args.lora_rank}")
    print(f"  batch={args.batch_size} grad_accum={args.grad_accum} lr={args.lr}")
    print(f"  epochs={args.epochs} warmup={args.warmup_steps}")
    print(f"  class_balance_weights={args.class_balance_weights} cap={args.class_weight_cap}")

    check_vram(min_free_gb=args.min_free_gb)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/5] Loading model + tokenizer...")
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
    print(f"  loaded in {time.time()-t0:.0f}s; VRAM allocated: {torch.cuda.memory_allocated()//1024//1024} MiB")

    print("\n[2/5] Adding LoRA adapter (attention modules)...")
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    # use_reentrant=False is required for PEFT + gradient checkpointing to
    # actually free activation memory between checkpoints (otherwise the
    # whole forward graph stays resident).
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # Round-2 fix: prepare_model_for_kbit_training internally re-enables
    # gradient checkpointing with default (use_reentrant=True) kwargs,
    # clobbering the use_reentrant=False set just above. Pass the kwargs
    # through so PEFT actually frees activation memory between checkpoints.
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
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

    print("\n[3/5] Loading + encoding data...")
    train_recs = load_jsonl(Path(args.train_jsonl))
    val_recs = load_jsonl(Path(args.val_jsonl))
    if args.max_train_records > 0:
        train_recs = train_recs[:args.max_train_records]
    print(f"  train: {len(train_recs)} | val: {len(val_recs)}")

    # Class weights from the *capped* training records (so smoke runs use
    # the smoke-set distribution, not the full corpus).
    class_weights_t, train_class_counts = compute_class_weights(
        train_recs, cap=args.class_weight_cap,
    )
    print(f"  train class counts: {dict(train_class_counts)}")
    print(f"  per-class weights (cap={args.class_weight_cap}):")
    for c in RELATION_AXIS_CLASSES:
        idx = CLASS_TO_IDX[c]
        n_c = train_class_counts.get(c, 0)
        print(f"    {c}: count={n_c} weight={class_weights_t[idx].item():.4f}")
    class_weights_dev = class_weights_t.to("cuda")

    train_enc = encode_dataset(train_recs, tok, args.seq_len)
    val_enc = encode_dataset(val_recs, tok, args.seq_len)
    print(f"  encoded train: {len(train_enc)} | val: {len(val_enc)}")
    if not train_enc:
        print("  ABORT: no training records survived encoding")
        return 2

    # Pre-sample the prediction-trace val records ONCE so the trace is
    # comparable across eval steps. Keep both the encoded form (for any
    # future use) and the original record (to extract gold_class + chat).
    n_trace = min(args.pred_trace_n, len(val_recs))
    if n_trace > 0:
        rng = random.Random(args.seed)
        trace_indices = rng.sample(range(len(val_recs)), n_trace)
    else:
        trace_indices = []
    trace_records = [val_recs[i] for i in trace_indices]
    print(f"  prediction-trace records: {len(trace_records)} (seed={args.seed})")

    print("\n[4/5] Setting up training loop + telemetry...")
    from torch.utils.data import DataLoader
    train_loader = DataLoader(
        train_enc, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_enc, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate, num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    total_micro_steps = (len(train_enc) * args.epochs) // args.batch_size
    total_optim_steps = total_micro_steps // args.grad_accum
    if args.smoke_steps > 0:
        total_optim_steps = args.smoke_steps
    print(f"  optim steps: {total_optim_steps} (micro: {total_optim_steps * args.grad_accum})")

    def lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return max(0.0, (total_optim_steps - step) / max(1, total_optim_steps - args.warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ---------- Telemetry layer ----------
    # TensorBoard is optional — if `tensorboard` isn't installed (or
    # `torch.utils.tensorboard` import fails for any other reason), fall
    # back to JSONL-only telemetry. The `writer` shim below no-ops every
    # SummaryWriter method so call sites don't have to branch.
    tb_dir = out_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"  TensorBoard logs -> {tb_dir}")
    except Exception as _tb_err:
        print(f"  WARN: TensorBoard unavailable ({_tb_err}); JSONL-only telemetry")
        class _NoopWriter:
            def add_scalar(self, *a, **kw): pass
            def add_text(self, *a, **kw): pass
            def flush(self, *a, **kw): pass
            def close(self, *a, **kw): pass
        writer = _NoopWriter()

    metrics_path = out_dir / "metrics.jsonl"
    # buffering=1 -> line buffering; we also explicitly flush after each write.
    metrics_f = metrics_path.open("w", buffering=1)

    pred_trace_path = out_dir / "predictions.jsonl"
    pred_trace_f = pred_trace_path.open("w", buffering=1)

    def log_metric_event(event: dict) -> None:
        event = {**event, "wall_time": time.time()}
        metrics_f.write(json.dumps(event) + "\n")
        metrics_f.flush()

    # Run config snapshot.
    train_path = Path(args.train_jsonl)
    val_path = Path(args.val_jsonl)
    free_gb, used_gb = gpu_mem_gb()
    try:
        import transformers as _tf  # noqa: F401
        tf_ver = _tf.__version__
    except Exception:
        tf_ver = None
    try:
        import peft as _peft  # noqa: F401
        peft_ver = _peft.__version__
    except Exception:
        peft_ver = None
    try:
        import bitsandbytes as _bnb  # noqa: F401
        bnb_ver = _bnb.__version__
    except Exception:
        bnb_ver = None
    run_config = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "model": args.model,
        "train_file": file_meta(train_path) if train_path.exists() else {"path": str(train_path)},
        "val_file": file_meta(val_path) if val_path.exists() else {"path": str(val_path)},
        "train_class_counts": dict(train_class_counts),
        "class_weights": {c: class_weights_t[CLASS_TO_IDX[c]].item() for c in RELATION_AXIS_CLASSES},
        "class_balance_weights_enabled": bool(args.class_balance_weights),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": tf_ver,
            "peft": peft_ver,
            "bitsandbytes": bnb_ver,
        },
        "platform": platform.platform(),
        "gpu": {
            "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "free_gb_at_start": free_gb,
            "used_gb_at_start": used_gb,
        },
        "total_optim_steps_planned": total_optim_steps,
        "n_train_encoded": len(train_enc),
        "n_val_encoded": len(val_enc),
        "trace_indices": trace_indices,
        "git": _git_state(),
        "env_vars": _env_snapshot(),
    }
    with (out_dir / "run.json").open("w") as f:
        json.dump(run_config, f, indent=2)
    print(f"  wrote {out_dir/'run.json'}")
    log_metric_event({"type": "start", "step": 0, **{
        "n_train": len(train_enc),
        "n_val": len(val_enc),
        "total_optim_steps_planned": total_optim_steps,
    }})

    # ---------- Eval loop (loss) ----------
    # Use the SAME weighted-CE objective as training (when --eval-weighted is ON,
    # which is the default) so checkpoint selection optimizes the same loss the
    # training step descends. With --no-class-balance-weights, both branches fall
    # through to plain unweighted HF CE.
    use_weighted_eval = bool(args.eval_weighted and args.class_balance_weights)

    @torch.no_grad()
    def eval_loss() -> float:
        # Gemma4Config (multimodal) doesn't expose `use_cache` at the top
        # level; only the text sub-config does. Tolerate missing attribute.
        prev_use_cache = getattr(model.config, "use_cache", None)
        # Disable gradient checkpointing during eval — it's a no-op without
        # backward, but HF clamps use_cache=False as a side effect.
        model.gradient_checkpointing_disable()
        if prev_use_cache is not None:
            model.config.use_cache = True
        model.eval()
        try:
            if use_weighted_eval:
                weighted_sum = 0.0
                weight_mass_sum = 0.0
                n = 0
                for batch in val_loader:
                    class_idx_cpu = batch["class_idx"]
                    batch_dev = {
                        k: v.to("cuda") for k, v in batch.items()
                        if k in ("input_ids", "attention_mask", "labels")
                    }
                    class_idx_dev = class_idx_cpu.to("cuda")
                    out = model(**batch_dev)
                    logits = out.logits
                    labels = batch_dev["labels"]
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = labels[:, 1:].contiguous()
                    B, T1, V = shift_logits.shape
                    flat_logits = shift_logits.view(-1, V)
                    flat_labels = shift_labels.view(-1)
                    tok_loss = F.cross_entropy(
                        flat_logits, flat_labels, ignore_index=-100, reduction="none",
                    ).view(B, T1)
                    safe_idx = class_idx_dev.clamp(min=0)
                    rec_w = class_weights_dev[safe_idx]
                    rec_w = torch.where(
                        class_idx_dev >= 0,
                        rec_w,
                        torch.ones_like(rec_w),
                    )
                    valid = (shift_labels != -100).to(tok_loss.dtype)
                    weighted = tok_loss * rec_w.unsqueeze(1) * valid
                    weight_mass = (rec_w.unsqueeze(1) * valid).sum()
                    weighted_sum += float(weighted.sum().item())
                    weight_mass_sum += float(weight_mass.item())
                    n += 1
                    if n >= 32:
                        break
                return weighted_sum / max(1.0, weight_mass_sum)
            else:
                total_loss = 0.0
                n = 0
                for batch in val_loader:
                    batch_dev = {
                        k: v.to("cuda") for k, v in batch.items()
                        if k in ("input_ids", "attention_mask", "labels")
                    }
                    out = model(**batch_dev)
                    total_loss += out.loss.item()
                    n += 1
                    if n >= 32:  # cap eval batches for speed
                        break
                return total_loss / max(1, n)
        finally:
            model.train()
            if prev_use_cache is not None:
                model.config.use_cache = prev_use_cache
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

    # ---------- Prediction-trace eval ----------
    @torch.no_grad()
    def eval_pred_trace(step: int) -> dict:
        """Greedy-decode the pre-sampled trace records and emit per-record
        lines to predictions.jsonl. Returns aggregated per-class metrics so
        callers can also log them to TB and metrics.jsonl.
        """
        if not trace_records:
            return {"micro_acc": 0.0, "macro_recall": 0.0, "per_class": {}, "n_total": 0, "n_unparseable": 0}
        # Without KV cache, generate() re-runs the full forward pass on the
        # whole prefix per token (O(prefix^2)). gradient_checkpointing_enable
        # forces use_cache=False, so we must explicitly disable checkpointing
        # and re-enable use_cache for decoding speed. Wrapped in try/finally so
        # an exception cannot leave the model in eval / no-checkpointing state.
        # Gemma4Config (multimodal) doesn't expose `use_cache` at the top
        # level; only the text sub-config does. Tolerate missing attribute.
        prev_use_cache = getattr(model.config, "use_cache", None)
        model.gradient_checkpointing_disable()
        if prev_use_cache is not None:
            model.config.use_cache = True
        model.eval()
        results = []
        try:
            for rec in trace_records:
                messages = rec["messages"]
                # Strip the assistant turn so the model has to produce it.
                chat_prefix = (
                    messages[:-1] if messages and messages[-1]["role"] == "assistant" else messages
                )
                enc = tok.apply_chat_template(
                    chat_prefix, tokenize=True, add_generation_prompt=True,
                    return_tensors="pt",
                )
                if hasattr(enc, "input_ids"):
                    input_ids = enc.input_ids.to("cuda")
                    attention_mask = enc.attention_mask.to("cuda") if hasattr(enc, "attention_mask") else None
                else:
                    input_ids = enc.to("cuda")
                    attention_mask = None
                prefix_len = input_ids.shape[1]
                gen_kwargs = dict(
                    max_new_tokens=args.pred_trace_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    use_cache=True,
                )
                if attention_mask is not None:
                    gen_kwargs["attention_mask"] = attention_mask
                gen = model.generate(input_ids, **gen_kwargs)
                new_tokens = gen[0, prefix_len:]
                text = tok.decode(new_tokens, skip_special_tokens=True)
                parsed = parse_answer(text)
                gold = (rec.get("metadata") or {}).get("answer")
                record_id = (rec.get("metadata") or {}).get("source_hash")
                entry = {
                    "step": step,
                    "record_id": record_id,
                    "gold_class": gold,
                    "generated_text": text,
                    "parsed_pred": parsed,
                    "correct": (parsed == gold) if (parsed is not None and gold is not None) else False,
                }
                results.append(entry)
                pred_trace_f.write(json.dumps(entry) + "\n")
            pred_trace_f.flush()
        finally:
            model.train()
            if prev_use_cache is not None:
                model.config.use_cache = prev_use_cache
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        # Only score records with a known gold class (all val records should
        # have one, but guard).
        scoreable = [r for r in results if r["gold_class"] in CLASS_TO_IDX]
        agg = per_class_metrics(scoreable) if scoreable else {
            "micro_acc": 0.0, "macro_recall": 0.0, "per_class": {}, "n_total": 0, "n_unparseable": 0,
        }
        return agg

    print("\n[5/5] Training...")
    model.train()
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()
    optim_step = 0
    micro_step = 0
    accum_loss = 0.0
    accum_count = 0
    log_buffer: list[float] = []  # all train-loss values per micro-step (for sparkline)
    eval_history: list[dict] = []
    saved_checkpoints: list[dict] = []
    best_eval_loss = float("inf")
    best_ckpt_path: str | None = None

    def step_done():
        nonlocal optim_step, accum_loss, accum_count
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        train_loss_avg = accum_loss / max(1, accum_count)
        log_buffer.append(train_loss_avg)
        accum_loss = 0.0
        accum_count = 0
        optim_step += 1
        if optim_step % args.log_steps == 0:
            recent = log_buffer[-args.log_steps:]
            recent_loss = sum(recent) / len(recent)
            peak_mb = torch.cuda.max_memory_allocated() // 1024 // 1024
            free_gb_now, used_gb_now = gpu_mem_gb()
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"  step {optim_step}/{total_optim_steps}: "
                f"loss={recent_loss:.4f} "
                f"peak_VRAM={peak_mb}MiB lr={lr_now:.2e} "
                f"elapsed={time.time()-t_start:.0f}s",
                flush=True,
            )
            # TB scalars
            writer.add_scalar("train/loss", recent_loss, optim_step)
            writer.add_scalar("train/lr", lr_now, optim_step)
            writer.add_scalar("train/peak_vram_mib", peak_mb, optim_step)
            writer.add_scalar("gpu/free_gb", free_gb_now, optim_step)
            writer.add_scalar("gpu/used_gb", used_gb_now, optim_step)
            writer.flush()
            # JSONL metrics
            log_metric_event({
                "type": "train",
                "step": optim_step,
                "loss": recent_loss,
                "lr": lr_now,
                "peak_vram_mib": peak_mb,
                "free_gb": free_gb_now,
                "used_gb": used_gb_now,
            })

    for epoch in range(args.epochs):
        for batch in train_loader:
            class_idx_cpu = batch.pop("class_idx")
            batch = {k: v.to("cuda") for k, v in batch.items()}
            class_idx_dev = class_idx_cpu.to("cuda")
            out = model(**batch)
            # Compute weighted CE manually so we can apply per-record class
            # weights. Standard SFT shift: predict token t from logits at t-1.
            if args.class_balance_weights:
                logits = out.logits  # (B, T, V)
                labels = batch["labels"]  # (B, T)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                B, T1, V = shift_logits.shape
                flat_logits = shift_logits.view(-1, V)
                flat_labels = shift_labels.view(-1)
                # Per-token CE without reduction (ignore_index handles -100 mask).
                tok_loss = F.cross_entropy(
                    flat_logits, flat_labels, ignore_index=-100, reduction="none",
                ).view(B, T1)
                # Per-record weight, broadcast to per-token via mask of valid
                # (non-ignored) positions.
                # Records with class_idx == -1 (unknown class) get weight 1.0.
                safe_idx = class_idx_dev.clamp(min=0)
                rec_w = class_weights_dev[safe_idx]  # (B,)
                rec_w = torch.where(
                    class_idx_dev >= 0,
                    rec_w,
                    torch.ones_like(rec_w),
                )
                valid = (shift_labels != -100).to(tok_loss.dtype)  # (B, T1)
                weighted = tok_loss * rec_w.unsqueeze(1) * valid
                # Standard weighted-mean denominator: sum(w * mask), not unweighted token count.
                # With rec_w == 1.0 everywhere, weight_mass == valid.sum() so this matches the
                # unweighted CE; with class-balanced weights, the per-batch loss scale is bounded
                # by the per-token CE itself instead of fluctuating with class composition.
                weight_mass = (rec_w.unsqueeze(1) * valid).sum().clamp(min=1.0)
                loss_unscaled = weighted.sum() / weight_mass
            else:
                loss_unscaled = out.loss
            loss = loss_unscaled / args.grad_accum
            loss.backward()
            accum_loss += loss_unscaled.item()
            accum_count += 1
            micro_step += 1
            if micro_step % args.grad_accum == 0:
                step_done()
                if args.smoke_steps > 0 and optim_step >= args.smoke_steps:
                    break
                if optim_step % args.eval_steps == 0:
                    el = eval_loss()
                    print(f"  >> eval_loss={el:.4f}", flush=True)
                    writer.add_scalar("eval/loss", el, optim_step)
                    # Greedy-decode trace + per-class metrics.
                    trace_agg = eval_pred_trace(optim_step)
                    writer.add_scalar("eval/micro_acc", trace_agg["micro_acc"], optim_step)
                    writer.add_scalar("eval/macro_recall", trace_agg["macro_recall"], optim_step)
                    pc_dict = {}
                    for c in RELATION_AXIS_CLASSES:
                        d = trace_agg["per_class"].get(c, {})
                        for k in ("precision", "recall", "f1"):
                            v = d.get(k)
                            if v is not None:
                                writer.add_scalar(f"eval/{c}/{k}", v, optim_step)
                            pc_dict.setdefault(c, {})[k] = v
                        pc_dict[c]["support"] = d.get("support", 0)
                        pc_dict[c]["predicted"] = d.get("predicted", 0)
                    writer.flush()
                    print(
                        f"  >> trace: micro={trace_agg['micro_acc']:.3f} "
                        f"macro_R={trace_agg['macro_recall']:.3f} "
                        f"unparseable={trace_agg['n_unparseable']}/{trace_agg['n_total']}",
                        flush=True,
                    )
                    eval_event = {
                        "type": "eval",
                        "step": optim_step,
                        "eval_loss": el,
                        "micro_acc": trace_agg["micro_acc"],
                        "macro_recall": trace_agg["macro_recall"],
                        "n_unparseable": trace_agg["n_unparseable"],
                        "n_total": trace_agg["n_total"],
                        "per_class": pc_dict,
                    }
                    log_metric_event(eval_event)
                    eval_history.append(eval_event)
                    if el < best_eval_loss:
                        best_eval_loss = el
                        ckpt = out_dir / "best"
                        model.save_pretrained(ckpt)
                        best_ckpt_path = str(ckpt)
                        print(f"  >> saved best to {ckpt}")
                if optim_step % args.save_steps == 0:
                    ckpt = out_dir / f"step_{optim_step}"
                    model.save_pretrained(ckpt)
                    saved_checkpoints.append({
                        "step": optim_step,
                        "path": str(ckpt),
                    })
        if args.smoke_steps > 0 and optim_step >= args.smoke_steps:
            break

    # Final save
    final_ckpt = out_dir / "final"
    model.save_pretrained(final_ckpt)
    saved_checkpoints.append({"step": optim_step, "path": str(final_ckpt)})
    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  final adapter -> {final_ckpt}")
    if best_ckpt_path is not None:
        print(f"  best adapter  -> {best_ckpt_path} (eval_loss={best_eval_loss:.4f})")
    else:
        print(f"  best adapter  -> (none — no eval ran)")

    log_metric_event({
        "type": "end",
        "step": optim_step,
        "elapsed_seconds": elapsed,
        "best_eval_loss": best_eval_loss if best_eval_loss != float("inf") else None,
        "best_ckpt": best_ckpt_path,
    })

    # ---------- Final report ----------
    train_losses = list(log_buffer)
    eval_losses = [e["eval_loss"] for e in eval_history]
    eval_micro = [e["micro_acc"] for e in eval_history]
    last = eval_history[-1] if eval_history else None
    report_lines: list[str] = []
    report_lines.append(f"# V9 training report — {out_dir.name}\n")
    report_lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}\n")
    report_lines.append(f"Elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)\n")
    report_lines.append(f"Optim steps: {optim_step} / planned {total_optim_steps}\n")
    report_lines.append("\n## Run config\n\n")
    report_lines.append(f"- model: `{args.model}`\n")
    report_lines.append(f"- train: `{args.train_jsonl}`  ({run_config['train_file'].get('bytes', '?')} bytes)\n")
    report_lines.append(f"- val: `{args.val_jsonl}`  ({run_config['val_file'].get('bytes', '?')} bytes)\n")
    report_lines.append(f"- class_balance_weights: {args.class_balance_weights}  (cap={args.class_weight_cap})\n")
    report_lines.append("- class weights:\n")
    for c in RELATION_AXIS_CLASSES:
        idx = CLASS_TO_IDX[c]
        report_lines.append(
            f"    - `{c}`: count={train_class_counts.get(c, 0)} "
            f"weight={class_weights_t[idx].item():.4f}\n"
        )
    report_lines.append("\n## Final eval\n\n")
    if last is not None:
        report_lines.append(f"- step: {last['step']}\n")
        report_lines.append(f"- eval_loss: {last['eval_loss']:.4f}\n")
        report_lines.append(f"- micro_acc: {last['micro_acc']:.3f}\n")
        report_lines.append(f"- macro_recall: {last['macro_recall']:.3f}\n")
        report_lines.append(f"- unparseable: {last['n_unparseable']}/{last['n_total']}\n")
        report_lines.append("\n### Per-class\n\n")
        report_lines.append("| Class | Support | Predicted | Precision | Recall | F1 |\n")
        report_lines.append("|---|---|---|---|---|---|\n")
        for c in RELATION_AXIS_CLASSES:
            d = last["per_class"].get(c, {})
            def fmt(v):
                return f"{v:.3f}" if isinstance(v, (int, float)) else "-"
            report_lines.append(
                f"| {c} | {d.get('support', 0)} | {d.get('predicted', 0)} | "
                f"{fmt(d.get('precision'))} | {fmt(d.get('recall'))} | {fmt(d.get('f1'))} |\n"
            )
    else:
        report_lines.append("(no eval ran — likely a smoke test with eval_steps > smoke_steps)\n")

    report_lines.append("\n## Loss curves (ASCII sparkline)\n\n")
    if train_losses:
        report_lines.append(f"train (per optim step, n={len(train_losses)}):\n\n")
        report_lines.append(f"    `{sparkline(train_losses)}`  "
                            f"(min={min(train_losses):.3f} max={max(train_losses):.3f} "
                            f"last={train_losses[-1]:.3f})\n")
    if eval_losses:
        report_lines.append(f"\neval_loss (n={len(eval_losses)}):\n\n")
        report_lines.append(f"    `{sparkline(eval_losses)}`  "
                            f"(min={min(eval_losses):.3f} max={max(eval_losses):.3f} "
                            f"last={eval_losses[-1]:.3f})\n")
    if eval_micro:
        report_lines.append(f"\neval/micro_acc (n={len(eval_micro)}):\n\n")
        report_lines.append(f"    `{sparkline(eval_micro)}`  "
                            f"(min={min(eval_micro):.3f} max={max(eval_micro):.3f} "
                            f"last={eval_micro[-1]:.3f})\n")

    report_lines.append("\n## Checkpoints\n\n")
    if best_ckpt_path is not None:
        report_lines.append(f"- best: `{best_ckpt_path}` (eval_loss={best_eval_loss:.4f})\n")
    for ck in saved_checkpoints:
        report_lines.append(f"- step {ck['step']}: `{ck['path']}`\n")

    report_lines.append("\n## Artifacts\n\n")
    report_lines.append(f"- TensorBoard: `{tb_dir}` (`tensorboard --logdir {tb_dir}`)\n")
    report_lines.append(f"- Metrics JSONL: `{metrics_path}`\n")
    report_lines.append(f"- Predictions trace: `{pred_trace_path}`\n")
    report_lines.append(f"- Run config: `{out_dir/'run.json'}`\n")

    with (out_dir / "training_report.md").open("w") as f:
        f.writelines(report_lines)
    print(f"  wrote {out_dir/'training_report.md'}")

    writer.close()
    metrics_f.close()
    pred_trace_f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
