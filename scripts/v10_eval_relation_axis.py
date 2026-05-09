"""V10 — evaluate the V9 LoRA adapter on the U2 holdout for relation_axis.

Mirror of V6g evaluation but with the LoRA-adapted local Gemma 4 E4B in
place of the Gemini Pro curator. Uses the same:
  - holdout join (probe_gold_holdout.jsonl × holdout_v15_sample.jsonl)
  - GOLD_TAG_TO_PROBE_CLASS mapping (4 effective classes for relation_axis)
  - short system prompt and user-message format that V8b training used.

Output:
  research/v10_relation_axis_eval.md   — per-class P/R/F1 + composite gate
  data/v_phase/v10_relation_axis_responses.jsonl — raw responses
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


GOLD_TAG_TO_RELATION_AXIS = {
    "correct":           "direct_sign_match",
    "polarity":          "direct_sign_mismatch",
    "act_vs_amt":        "direct_axis_mismatch",
    "wrong_relation":    "direct_axis_mismatch",
    "no_relation":       "no_relation",
    # The rest map to None (skipped from accuracy calc)
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


def load_holdout_pairs() -> list[dict]:
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
    return out


def build_chat(rec: dict) -> list[dict]:
    user = (
        f"CLAIM: {rec['stmt_type']}({rec['subject']}, {rec['object']})\n"
        f"EVIDENCE: {rec['evidence_text']}"
    )
    return [
        {"role": "system", "content": _SHORT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


_CLASS_RE = re.compile(
    r'"answer"\s*:\s*"(direct_sign_match|direct_sign_mismatch|'
    r'direct_axis_mismatch|no_relation)"'
)


def parse_answer(text: str) -> str | None:
    """Pull the answer class out of the model's generation. Robust to JSON
    not parsing cleanly — falls back to regex on the closed set."""
    m = _CLASS_RE.search(text)
    if m:
        return m.group(1)
    # Last-resort: see if any class name appears verbatim as a substring
    for c in RELATION_AXIS_CLASSES:
        if c in text:
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="google/gemma-4-E4B-it")
    ap.add_argument("--adapter", default="data/v_phase/lora/relation_axis_v9/best",
                     help="LoRA adapter path (set to 'none' to evaluate base only)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=482)
    ap.add_argument("--out-jsonl",
                     default="data/v_phase/v10_relation_axis_responses.jsonl")
    ap.add_argument("--out-md", default="research/v10_relation_axis_eval.md")
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    args = ap.parse_args()

    free_b, _ = torch.cuda.mem_get_info(0)
    if free_b / 1e9 < args.min_free_gb:
        print(f"ABORT: only {free_b/1e9:.1f} GB free; need {args.min_free_gb} GB")
        return 2

    print(f"=== V10 EVAL: {args.base_model} + adapter={args.adapter} ===")

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
    if args.adapter and args.adapter != "none":
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    print(f"  loaded in {time.time()-t0:.0f}s")

    print("\n[2/4] Loading holdout...")
    pairs = load_holdout_pairs()[:args.limit]
    tasks = []
    for r in pairs:
        gold = GOLD_TAG_TO_RELATION_AXIS.get(r.get("gold_tag"), None)
        if gold is None:
            continue
        tasks.append({**r, "gold_class": gold})
    print(f"  {len(pairs)} records, {len(tasks)} relation_axis tasks with gold")

    print("\n[3/4] Generating...")
    out_path = ROOT / args.out_jsonl
    out_path.parent.mkdir(parents=True, exist_ok=True)
    responses: list[dict] = []
    t1 = time.time()
    for i, t in enumerate(tasks):
        chat = build_chat(t)
        enc_full = tok.apply_chat_template(
            chat, tokenize=True, add_generation_prompt=True,
            return_tensors="pt",
        )
        # apply_chat_template returns either a BatchEncoding (transformers>=4.45)
        # or a Tensor — handle both.
        if hasattr(enc_full, "input_ids"):
            input_ids = enc_full.input_ids.to("cuda")
            attention_mask = enc_full.attention_mask.to("cuda") if hasattr(enc_full, "attention_mask") else None
        else:
            input_ids = enc_full.to("cuda")
            attention_mask = None
        prefix_len = input_ids.shape[1]
        gen_kwargs = dict(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        with torch.no_grad():
            gen = model.generate(input_ids, **gen_kwargs)
        new_tokens = gen[0, prefix_len:]
        text = tok.decode(new_tokens, skip_special_tokens=True)
        pred = parse_answer(text)
        responses.append({
            **t,
            "raw": text,
            "pred": pred,
        })
        if (i + 1) % 50 == 0:
            rate = (i + 1) / max(time.time() - t1, 0.01)
            print(f"  {i+1}/{len(tasks)} ({rate:.1f}/s, {time.time()-t1:.0f}s)")
    print(f"  done in {time.time()-t1:.0f}s")

    with out_path.open("w") as f:
        for r in responses:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {out_path}")

    # Compute metrics
    gold_counts: Counter = Counter()
    pred_counts: Counter = Counter()
    tp: Counter = Counter()
    n_err = 0
    for r in responses:
        gold = r["gold_class"]
        pred = r["pred"]
        gold_counts[gold] += 1
        if pred is None:
            n_err += 1
            continue
        pred_counts[pred] += 1
        if pred == gold:
            tp[gold] += 1

    n = len(responses)
    micro = sum(tp.values()) / n if n else 0.0
    classes = sorted(set(gold_counts) | set(pred_counts))
    per_class = {}
    macro_sum = 0.0
    macro_n = 0
    for c in classes:
        s = gold_counts[c]; p = pred_counts[c]
        recall = (tp[c] / s) if s else None
        prec = (tp[c] / p) if p else None
        f1 = (2 * prec * recall / (prec + recall)) if (prec and recall and prec + recall) else None
        per_class[c] = {"support": s, "pred": p, "precision": prec, "recall": recall, "f1": f1}
        if recall is not None:
            macro_sum += recall
            macro_n += 1
    macro = macro_sum / macro_n if macro_n else 0.0
    mfc = (max(gold_counts.values()) / n) if gold_counts else 0.0

    md_path = ROOT / args.out_md
    with md_path.open("w") as f:
        f.write("# V10 — relation_axis LoRA evaluation on U2 holdout\n\n")
        f.write(f"Date: 2026-05-07\n")
        f.write(f"Base: {args.base_model} | Adapter: {args.adapter}\n")
        f.write(f"Records: {len(pairs)} | tasks: {len(tasks)}\n\n")
        f.write(f"## Aggregate\n\n")
        f.write(f"| Metric | Value |\n|---|---|\n")
        f.write(f"| micro | {micro:.3f} |\n")
        f.write(f"| macro | {macro:.3f} |\n")
        f.write(f"| mfc baseline | {mfc:.3f} |\n")
        f.write(f"| Δ vs mfc | {micro - mfc:+.3f} |\n")
        f.write(f"| errors (unparseable) | {n_err} |\n\n")
        f.write(f"## Per-class\n\n")
        f.write("| Class | Support | Predicted | Precision | Recall | F1 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for c, d in sorted(per_class.items(), key=lambda kv: -kv[1]["support"]):
            p = f"{d['precision']:.3f}" if d["precision"] is not None else "—"
            r = f"{d['recall']:.3f}" if d["recall"] is not None else "—"
            f1v = f"{d['f1']:.3f}" if d["f1"] is not None else "—"
            f.write(f"| {c} | {d['support']} | {d['pred']} | {p} | {r} | {f1v} |\n")
    print(f"Wrote {md_path}")

    print(f"\n=== SUMMARY ===")
    print(f"  micro={micro:.3f} macro={macro:.3f} mfc={mfc:.3f} err={n_err}")
    for c, d in sorted(per_class.items(), key=lambda kv: -kv[1]["support"]):
        r = f"{d['recall']:.3f}" if d["recall"] is not None else "—"
        p = f"{d['precision']:.3f}" if d["precision"] is not None else "—"
        f1v = f"{d['f1']:.3f}" if d["f1"] is not None else "—"
        print(f"  {c}: support={d['support']} pred={d['pred']} P={p} R={r} F1={f1v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
