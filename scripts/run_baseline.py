"""Run v5_no_grounding baseline on v7 holdout — for apples-to-apples comparison.

Uses the exact v5_no_grounding design from EvidenceScorer:
- 8 contrastive pairs (no act_vs_amt, hypothesis, discourse, metadata layers)
- No claim enrichment
- No entity alias context
- Same system prompt as v5
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "belief_benchmark"))

from cliquefinder.stats.evidence_scorer import (
    CONTRASTIVE_EXAMPLES,
    SYSTEM_PROMPT,
    _render_example,
    _extract_verdict,
    _verdict_to_score,
)
from model_client import ModelClient


def score_record(client, subject, stmt_type, obj, evidence_text, max_tokens=4000):
    claim = f"{subject} [{stmt_type}] {obj}"
    messages = []
    for ex in CONTRASTIVE_EXAMPLES:
        u, a = _render_example(ex)
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    messages.append({
        "role": "user",
        "content": f'CLAIM: {claim}\nEVIDENCE: "{evidence_text}"',
    })

    try:
        response = client.call(
            system=SYSTEM_PROMPT,
            messages=messages,
            max_tokens=max_tokens,
        )
        verdict, confidence = _extract_verdict(response.raw_text)
        return {
            "score": _verdict_to_score(verdict, confidence),
            "verdict": verdict,
            "confidence": confidence,
            "raw_text": response.raw_text,
            "tokens": response.tokens,
        }
    except Exception as e:
        return {
            "score": 0.5,
            "verdict": None,
            "confidence": None,
            "raw_text": f"error: {e}",
            "tokens": 0,
        }


def main():
    holdout_path = ROOT / "data" / "benchmark" / "holdout_v7.jsonl"
    output_path = ROOT / "data" / "benchmark" / "results" / "v5_baseline_on_v7_holdout.jsonl"

    with open(holdout_path) as f:
        holdout = [json.loads(line) for line in f]

    print(f"v5_no_grounding baseline: {len(holdout)} records")
    print(f"  Examples: {len(CONTRASTIVE_EXAMPLES)} ({len(CONTRASTIVE_EXAMPLES)//2} pairs)")
    print(f"  System prompt length: {len(SYSTEM_PROMPT)}")

    client = ModelClient("gemma-moe")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_fh = open(output_path, "w")

    correct = 0
    parsed = 0
    t_start = time.time()

    for i, record in enumerate(holdout):
        result = score_record(
            client=client,
            subject=record["subject"],
            stmt_type=record["stmt_type"],
            obj=record["object"],
            evidence_text=record.get("evidence_text", ""),
            max_tokens=4000,
        )
        gt = (record["tag"] == "correct")
        llm = (result["verdict"] == "correct") if result["verdict"] else None
        if llm is not None:
            parsed += 1
            if llm == gt:
                correct += 1

        result.update({
            "source_hash": record["source_hash"],
            "tag": record["tag"],
            "subject": record["subject"],
            "stmt_type": record["stmt_type"],
            "object": record["object"],
        })
        r_save = {k: v for k, v in result.items() if k != "raw_text"}
        r_save["raw_text_preview"] = result.get("raw_text", "")[:500]
        out_fh.write(json.dumps(r_save) + "\n")
        out_fh.flush()

        elapsed = time.time() - t_start
        acc = correct / parsed * 100 if parsed > 0 else 0
        mark = "✓" if (llm == gt) else ("✗" if llm is not None else "?")
        print(f"  [{i+1:3d}/{len(holdout)}] {mark} {record['subject']:>10s} [{record['stmt_type']:>15s}] {record['object']:10s} "
              f"→ {result['verdict'] or 'PARSE_FAIL':>9s} acc={acc:.1f}%")

    out_fh.close()
    print(f"\nv5_baseline on v7 holdout: {correct}/{parsed} = {correct/parsed*100:.1f}%")


if __name__ == "__main__":
    main()
