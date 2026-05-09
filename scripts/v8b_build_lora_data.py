"""V8b — convert V8a Pro-curated labels to LoRA training format.

Builds train/val split (90/10) of chat-formatted records for the
relation_axis probe using the curator system prompt as context. The
trained LoRA will produce {answer, rationale} JSON when given a
CLAIM/EVIDENCE user message — matching the curator and production probe
call shape so production can swap in the LoRA-tuned model directly.

Output:
  data/v_phase/train/v8b_relation_axis.jsonl
  data/v_phase/val/v8b_relation_axis.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.v_phase.curator_prompts import (
    CURATOR_FEW_SHOTS,
    CURATOR_SYSTEM_PROMPTS,
)


_SHORT_SYSTEM_PROMPT = (
    "You classify whether evidence describes a relation between two entities "
    "matching a claim's axis and sign. Output JSON: "
    '{"answer": <one of: direct_sign_match, direct_sign_mismatch, '
    "direct_axis_mismatch, no_relation>, "
    '"rationale": <short quote/phrase from evidence>}.'
)


def make_chat(rec: dict, *, include_few_shots: bool = False, short_system: bool = True) -> dict:
    """Build a chat-format training example for relation_axis.

    Default (include_few_shots=False, short_system=True) emits a minimal
    3-turn chat:
        [system (~70 tokens), user (~100), assistant (~50)] = ~250 tokens
    suitable for LoRA SFT on a single 24GB GPU. The supervised examples
    teach the task; the full 7KB curator system prompt is overkill at
    training time (logits+CE on 256K-vocab × 2K-seq peaks at ~22GB VRAM
    on E4B). The short system stays consistent at inference.

    Pass `include_few_shots=True` to emit the full 25-shot chat for
    in-context-learning baselines (incompatible with short_system).
    Pass `short_system=False` to use the full curator system prompt.
    """
    system = _SHORT_SYSTEM_PROMPT if short_system else CURATOR_SYSTEM_PROMPTS["relation_axis"]
    user = (
        f"CLAIM: {rec['stmt_type']}({rec['subject']}, {rec['object']})\n"
        f"EVIDENCE: {rec['evidence_text']}"
    )
    assistant_answer = json.dumps({
        "answer": rec["answer"],
        "rationale": rec["rationale"],
    }, separators=(",", ": "))

    messages = [{"role": "system", "content": system}]
    if include_few_shots:
        for q, a in CURATOR_FEW_SHOTS["relation_axis"]:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": user})
    messages.append({"role": "assistant", "content": assistant_answer})

    return {
        "messages": messages,
        "completion": assistant_answer,
        "metadata": {
            "source_hash": rec["source_hash"],
            "stmt_type": rec["stmt_type"],
            "axis": rec["axis"],
            "answer": rec["answer"],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path",
                     default="data/v_phase/v8a_relation_axis_labels.jsonl")
    ap.add_argument("--out-train",
                     default="data/v_phase/train/v8b_relation_axis.jsonl")
    ap.add_argument("--out-val",
                     default="data/v_phase/val/v8b_relation_axis.jsonl")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = ROOT / args.in_path
    out_train = ROOT / args.out_train
    out_val = ROOT / args.out_val
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)

    # Load + filter
    kept: list[dict] = []
    n_total = 0
    n_dropped = 0
    drop_reasons: Counter[str] = Counter()
    with in_path.open() as f:
        for line in f:
            r = json.loads(line)
            n_total += 1
            if not r.get("kept"):
                n_dropped += 1
                drop_reasons[r.get("filter_reason") or "no_kept"] += 1
                continue
            if not r.get("answer"):
                n_dropped += 1
                drop_reasons["no_answer"] += 1
                continue
            kept.append(r)

    print(f"Loaded {n_total} records; {len(kept)} kept; {n_dropped} dropped")
    print(f"  drop reasons: {dict(drop_reasons)}")

    # Stratified split by class so train/val have similar class balance
    by_class: dict[str, list[dict]] = {}
    for r in kept:
        by_class.setdefault(r["answer"], []).append(r)

    rng = random.Random(args.seed)
    train: list[dict] = []
    val: list[dict] = []
    for cls, items in by_class.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * args.val_frac))
        val.extend(items[:n_val])
        train.extend(items[n_val:])
    rng.shuffle(train)
    rng.shuffle(val)

    print(f"Split: {len(train)} train / {len(val)} val")
    print(f"  train class dist: {Counter(r['answer'] for r in train)}")
    print(f"  val   class dist: {Counter(r['answer'] for r in val)}")

    n_written = 0
    with out_train.open("w") as f:
        for r in train:
            f.write(json.dumps(make_chat(r)) + "\n")
            n_written += 1
    print(f"Wrote {out_train} ({n_written} records)")

    n_written = 0
    with out_val.open("w") as f:
        for r in val:
            f.write(json.dumps(make_chat(r)) + "\n")
            n_written += 1
    print(f"Wrote {out_val} ({n_written} records)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
