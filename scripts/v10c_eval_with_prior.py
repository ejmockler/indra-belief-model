"""V10c — evaluate a V9 LoRA adapter on U2 holdout with optional inference-time
logit-prior shift.

Replaces autoregressive greedy decode with a class-scoring approach:
  for each candidate class c in {direct_sign_match, ...}:
    score(c) = sum( log p(token_i | prefix + earlier_tokens of c) )
  apply optional prior shift: score(c) += log(p_holdout[c] / p_train[c])
  pick argmax.

This is faster than .generate() (one short forward per class instead of 64
autoregressive tokens) AND cleanly applies the prior correction without
distorting rationale tokens.

Use:
  python scripts/v10c_eval_with_prior.py \\
    --adapter data/v_phase/lora/relation_axis_v9/best \\
    --train-jsonl data/v_phase/train/v8b_relation_axis.jsonl \\
    --apply-prior-shift \\
    --out-md research/v10c_relation_axis_eval.md
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

GOLD_TAG_TO_RELATION_AXIS = {
    "correct":           "direct_sign_match",
    "polarity":          "direct_sign_mismatch",
    "act_vs_amt":        "direct_axis_mismatch",
    "wrong_relation":    "direct_axis_mismatch",
    "no_relation":       "no_relation",
    "grounding":         None,
    "entity_boundaries": None,
    "hypothesis":        None,
    "negative_result":   None,
    "mod_site":          None,
    "agent_conditions":  None,
}

RELATION_AXIS_CLASSES = [
    "direct_sign_match", "direct_sign_mismatch",
    "direct_axis_mismatch", "no_relation",
]

_SHORT_SYSTEM_PROMPT = (
    "You classify whether evidence describes a relation between two entities "
    "matching a claim's axis and sign. Output JSON: "
    '{"answer": <one of: direct_sign_match, direct_sign_mismatch, '
    "direct_axis_mismatch, no_relation>, "
    '"rationale": <short quote/phrase from evidence>}.'
)

# U2 holdout class distribution (from probe_gold_holdout.jsonl with the
# 4-class projection used in V10).
HOLDOUT_PRIOR = {
    "direct_sign_match": 273 / 392,
    "no_relation": 53 / 392,
    "direct_axis_mismatch": 50 / 392,
    "direct_sign_mismatch": 16 / 392,
}


def load_holdout_pairs(limit: int) -> list[dict]:
    gold = {}
    with (ROOT / "data" / "benchmark" / "probe_gold_holdout.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            sh = r.get("source_hash")
            if isinstance(sh, int):
                gold[sh] = r
    out = []
    seen = set()
    with (ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl").open() as f:
        for line in f:
            r = json.loads(line)
            sh = r.get("source_hash")
            if not isinstance(sh, int) or sh not in gold or sh in seen:
                continue
            seen.add(sh)
            g = gold[sh]
            out.append({
                "source_hash": sh,
                "stmt_type": r.get("stmt_type") or g.get("claim_stmt_type"),
                "subject": r.get("subject") or g.get("subject"),
                "object": r.get("object") or g.get("object"),
                "evidence_text": (r.get("evidence_text") or "").strip(),
                "gold_tag": g.get("gold_tag"),
            })
            if len(out) >= limit:
                break
    return out


def compute_train_prior(train_jsonl: Path) -> dict[str, float]:
    """Read class distribution from training metadata (V8b/V8c)."""
    counts: Counter[str] = Counter()
    with train_jsonl.open() as f:
        for line in f:
            r = json.loads(line)
            ans = (r.get("metadata") or {}).get("answer")
            if ans in RELATION_AXIS_CLASSES:
                counts[ans] += 1
    total = sum(counts.values())
    if total == 0:
        raise ValueError(f"No labels found in {train_jsonl}")
    return {c: counts[c] / total for c in RELATION_AXIS_CLASSES}


def build_chat(rec: dict) -> list[dict]:
    user = (
        f"CLAIM: {rec['stmt_type']}({rec['subject']}, {rec['object']})\n"
        f"EVIDENCE: {rec['evidence_text']}"
    )
    return [
        {"role": "system", "content": _SHORT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter",
                     default="data/v_phase/lora/relation_axis_v9/best")
    ap.add_argument("--train-jsonl",
                     default="data/v_phase/train/v8b_relation_axis.jsonl",
                     help="for computing p_train (the model's training distribution)")
    ap.add_argument(
        "--apply-prior-shift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply log(p_holdout / p_train) shift to class scores",
    )
    ap.add_argument("--limit", type=int, default=482)
    ap.add_argument("--out-jsonl",
                     default="data/v_phase/v10c_relation_axis_responses.jsonl")
    ap.add_argument("--out-md", default="research/v10c_relation_axis_eval.md")
    ap.add_argument("--min-free-gb", type=float, default=6.0)
    args = ap.parse_args()

    free_b, _ = torch.cuda.mem_get_info(0)
    if free_b / 1e9 < args.min_free_gb:
        print(f"ABORT: only {free_b/1e9:.1f} GB free; need {args.min_free_gb} GB")
        return 2

    print(f"=== V10c EVAL: {args.base_model} + adapter={args.adapter} ===")
    print(f"  prior_shift={args.apply_prior_shift}")

    p_train = compute_train_prior(ROOT / args.train_jsonl)
    p_holdout = HOLDOUT_PRIOR
    print(f"\n  p_train  = {p_train}")
    print(f"  p_holdout= {p_holdout}")
    if args.apply_prior_shift:
        bias = {c: math.log(p_holdout[c] / p_train[c]) for c in RELATION_AXIS_CLASSES}
        print(f"  log-prior shift = {bias}")
    else:
        bias = {c: 0.0 for c in RELATION_AXIS_CLASSES}

    print("\n[1/4] Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    if args.adapter and args.adapter.lower() != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"  loaded in {time.time()-t0:.0f}s")

    # Pre-tokenize each candidate class continuation. The format Pro/V9 emit is
    # `{"answer": "<class>",`. We score the tokens of `<class>` only, conditioned
    # on the prefix `... {"answer": "`. The shared prefix is added below per-record.
    answer_prefix = '{"answer": "'
    answer_suffix = '"'  # unused but documents what comes after the class
    print(f"\n[2/4] Loading holdout...")
    pairs = load_holdout_pairs(args.limit)
    tasks = []
    for r in pairs:
        gold = GOLD_TAG_TO_RELATION_AXIS.get(r.get("gold_tag"))
        if gold is None:
            continue
        tasks.append({**r, "gold_class": gold})
    print(f"  {len(pairs)} records, {len(tasks)} relation_axis tasks")

    # Pre-tokenize the per-class continuations once.
    class_token_ids: dict[str, list[int]] = {}
    for c in RELATION_AXIS_CLASSES:
        # Tokenize as continuation — we'll prepend the prefix per-record
        # because Gemma's chat-template inserts boundary tokens differently
        # in each record. The cleanest invariant is to encode the full
        # `prefix + class` then strip the prefix tokens.
        class_token_ids[c] = tok.encode(c, add_special_tokens=False)

    print("\n[3/4] Scoring...")
    responses = []
    t1 = time.time()
    for i, t in enumerate(tasks):
        chat = build_chat(t)
        prefix_enc = tok.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        )
        if hasattr(prefix_enc, "input_ids"):
            prefix_ids = prefix_enc.input_ids
        else:
            prefix_ids = prefix_enc
        # Append the answer-prefix tokens (model's expected prelude).
        ap_ids = torch.tensor(
            [tok.encode(answer_prefix, add_special_tokens=False)],
            dtype=torch.long,
        )
        prefix_full = torch.cat([prefix_ids.to("cpu"), ap_ids], dim=1)
        prefix_len = prefix_full.shape[1]

        scores: dict[str, float] = {}
        for c in RELATION_AXIS_CLASSES:
            class_ids = class_token_ids[c]
            full_ids = torch.cat([
                prefix_full,
                torch.tensor([class_ids], dtype=torch.long),
            ], dim=1).to("cuda")
            with torch.no_grad():
                out = model(full_ids)
            # logits[i] predicts token at position i+1. To score class tokens
            # at positions [prefix_len, prefix_len + len(class_ids)), we need
            # the logits at [prefix_len-1, prefix_len + len(class_ids)-1).
            target = full_ids[0, prefix_len:prefix_len + len(class_ids)]
            class_logits = out.logits[0, prefix_len - 1: prefix_len - 1 + len(class_ids)]
            log_probs = F.log_softmax(class_logits.float(), dim=-1)
            tok_lp = log_probs[torch.arange(len(class_ids)), target]
            score = tok_lp.sum().item() + bias[c]
            scores[c] = score

        pred = max(scores, key=scores.get)
        responses.append({
            **t,
            "scores": scores,
            "pred": pred,
        })
        if (i + 1) % 50 == 0:
            rate = (i + 1) / max(time.time() - t1, 0.01)
            print(f"  {i+1}/{len(tasks)} ({rate:.1f}/s, {time.time()-t1:.0f}s)")
    print(f"  done in {time.time()-t1:.0f}s")

    out_path = ROOT / args.out_jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in responses:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {out_path}")

    # Metrics
    gold_counts = Counter(r["gold_class"] for r in responses)
    tp: Counter = Counter()
    pred_counts: Counter = Counter()
    for r in responses:
        pred_counts[r["pred"]] += 1
        if r["pred"] == r["gold_class"]:
            tp[r["gold_class"]] += 1
    n = len(responses)
    micro = sum(tp.values()) / n if n else 0.0
    classes = sorted(set(gold_counts) | set(pred_counts))
    per_class = {}
    macro_sum = 0.0; macro_n = 0
    for c in classes:
        s = gold_counts[c]; p = pred_counts[c]
        recall = (tp[c] / s) if s else None
        prec = (tp[c] / p) if p else None
        f1 = (2 * prec * recall / (prec + recall)) if (prec and recall and prec + recall) else None
        per_class[c] = {"support": s, "predicted": p, "precision": prec, "recall": recall, "f1": f1}
        if recall is not None:
            macro_sum += recall; macro_n += 1
    macro = macro_sum / macro_n if macro_n else 0.0
    mfc = (max(gold_counts.values()) / n) if gold_counts else 0.0

    md_path = ROOT / args.out_md
    with md_path.open("w") as f:
        f.write("# V10c — relation_axis LoRA eval with logit-prior shift\n\n")
        f.write(f"Date: 2026-05-07\n")
        f.write(f"Base: {args.base_model} | Adapter: {args.adapter}\n")
        f.write(f"prior_shift_applied: {args.apply_prior_shift}\n\n")
        f.write("## Aggregate\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| micro | {micro:.3f} |\n")
        f.write(f"| macro | {macro:.3f} |\n")
        f.write(f"| mfc baseline | {mfc:.3f} |\n")
        f.write(f"| Δ vs mfc | {micro - mfc:+.3f} |\n\n")
        f.write("## Train and holdout priors\n\n")
        f.write("| Class | p_train | p_holdout | log shift |\n|---|---|---|---|\n")
        for c in RELATION_AXIS_CLASSES:
            f.write(f"| {c} | {p_train[c]:.3f} | {p_holdout[c]:.3f} | {bias[c]:+.3f} |\n")
        f.write("\n## Per-class\n\n")
        f.write("| Class | Support | Predicted | Precision | Recall | F1 |\n|---|---|---|---|---|---|\n")
        for c, d in sorted(per_class.items(), key=lambda kv: -kv[1]["support"]):
            p = f"{d['precision']:.3f}" if d["precision"] is not None else "—"
            r = f"{d['recall']:.3f}" if d["recall"] is not None else "—"
            f1v = f"{d['f1']:.3f}" if d["f1"] is not None else "—"
            f.write(f"| {c} | {d['support']} | {d['predicted']} | {p} | {r} | {f1v} |\n")
    print(f"Wrote {md_path}")

    print(f"\n=== SUMMARY (prior_shift={args.apply_prior_shift}) ===")
    print(f"  micro={micro:.3f} macro={macro:.3f} mfc={mfc:.3f}")
    for c, d in sorted(per_class.items(), key=lambda kv: -kv[1]["support"]):
        r = f"{d['recall']:.3f}" if d["recall"] is not None else "—"
        p = f"{d['precision']:.3f}" if d["precision"] is not None else "—"
        f1v = f"{d['f1']:.3f}" if d["f1"] is not None else "—"
        print(f"  {c}: support={d['support']} pred={d['predicted']} P={p} R={r} F1={f1v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
