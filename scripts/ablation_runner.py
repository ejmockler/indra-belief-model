"""Run v5 with ablations to isolate what actually matters.

Conditions:
A. v5_full: grounding context + all 9 few-shot examples (incl. 2 inversion)
B. v5_no_grounding: no grounding info, just few-shot examples
C. v5_randomized_grounding: shuffled grounding info (tests if LLM ignores it)
D. v4_replay: original v4 prompt (no grounding, different examples)

If A >> B, grounding helps.
If A ≈ C, LLM is ignoring grounding (just reading text).
If A > C and A > B, grounding genuinely contributes.
If A ≈ B ≈ C, grounding is answer-leakage and LLM follows it blindly.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from experiments.belief_benchmark.model_client import (
    ModelClient, extract_verdict, verdict_to_score,
)
from experiments.belief_benchmark.llm_scorer_v5 import (
    CONTRASTIVE_EXAMPLES_V5, SYSTEM_PROMPT_V5, compute_grounding, render_example,
)

BENCHMARK_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmark"


SYSTEM_PROMPT_NO_GROUNDING = """\
You judge whether a biomedical text-mining extraction is correct. You'll see
the claim + evidence sentence. Use contrastive examples to learn the judgment
standard. Output JSON:
{"verdict": "correct" or "incorrect", "confidence": "high" | "medium" | "low"}\
"""


def render_example_no_grounding(ex):
    user = (
        f"CLAIM: {ex['claim']}\n"
        f'EVIDENCE: "{ex["evidence"]}"'
    )
    assistant = (
        f"Reason: {ex['reason']}\n"
        f'{{"verdict": "{ex["verdict"]}", "confidence": "{ex["confidence"]}"}}'
    )
    return user, assistant


def run_condition(
    records: list[dict],
    condition: str,
    output_path: Path,
    model_name: str = "gemma-moe",
    seed: int = 42,
) -> None:
    """Run one ablation condition."""
    client = ModelClient(model_name)
    rng = random.Random(seed)

    # Pre-compute all groundings (for shuffling)
    groundings = [compute_grounding(r) for r in records]

    if condition == "randomized_grounding":
        # Shuffle groundings relative to records
        indices = list(range(len(groundings)))
        rng.shuffle(indices)
        shuffled = [groundings[i] for i in indices]
        groundings = shuffled

    # Skip already scored
    already = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                already.add(json.loads(line)["source_hash"])

    remaining_idx = [i for i, r in enumerate(records) if r["source_hash"] not in already]
    print(f"[{condition}] Scoring {len(remaining_idx)}/{len(records)} records...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(output_path, "a")

    try:
        for pos, i in enumerate(remaining_idx):
            rec = records[i]
            if not rec.get("evidence_text"):
                continue

            # Build messages based on condition
            if condition == "no_grounding":
                messages = []
                for ex in CONTRASTIVE_EXAMPLES_V5:
                    user, assistant = render_example_no_grounding(ex)
                    messages.append({"role": "user", "content": user})
                    messages.append({"role": "assistant", "content": assistant})
                claim = f"{rec['subject']} [{rec['stmt_type']}] {rec['object']}"
                messages.append({
                    "role": "user",
                    "content": f'CLAIM: {claim}\nEVIDENCE: "{rec["evidence_text"]}"',
                })
                system = SYSTEM_PROMPT_NO_GROUNDING
            else:
                # full or randomized_grounding
                grounding = groundings[i]
                messages = []
                for ex in CONTRASTIVE_EXAMPLES_V5:
                    user, assistant = render_example(ex)
                    messages.append({"role": "user", "content": user})
                    messages.append({"role": "assistant", "content": assistant})
                claim = f"{rec['subject']} [{rec['stmt_type']}] {rec['object']}"
                messages.append({
                    "role": "user",
                    "content": (
                        f"CLAIM: {claim}\n"
                        f'EVIDENCE: "{rec["evidence_text"]}"\n'
                        f"GROUNDING:\n"
                        f"  subject: {grounding['subject']}\n"
                        f"  object: {grounding['object']}"
                    ),
                })
                system = SYSTEM_PROMPT_V5

            try:
                response = client.call(system=system, messages=messages, max_tokens=1500)
                verdict, confidence, _ = extract_verdict(response.raw_text)
                score = verdict_to_score(verdict, confidence)
            except Exception as e:
                response = None
                verdict = confidence = None
                score = 0.5

            out_f.write(json.dumps({
                "source_hash": rec["source_hash"],
                "pa_hash": rec["pa_hash"],
                "tag": rec["tag"],
                "score": score,
                "verdict": verdict,
                "confidence": confidence,
                "raw_text": response.raw_text if response else "",
                "tokens": response.tokens if response else 0,
                "grounding": groundings[i] if condition != "no_grounding" else None,
                "condition": condition,
            }) + "\n")
            out_f.flush()

            if (pos + 1) % 10 == 0:
                print(f"[{condition}]  {pos + 1}/{len(remaining_idx)}")
    finally:
        out_f.close()


def main():
    import sys
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["full", "no_grounding", "randomized_grounding"], required=True)
    parser.add_argument("--eval-set", default="holdout")  # "holdout" or "v4"
    parser.add_argument("--model", default="gemma-moe")
    args = parser.parse_args()

    eval_files = {
        "holdout": BENCHMARK_DIR / "holdout_set.jsonl",
        "v4": BENCHMARK_DIR / "eval_set_v4.jsonl",
    }
    input_path = eval_files[args.eval_set]
    output_path = BENCHMARK_DIR / "results" / f"v5_{args.condition}_{args.eval_set}.jsonl"

    records = []
    with open(input_path) as f:
        for line in f:
            records.append(json.loads(line))

    run_condition(records, args.condition, output_path, model_name=args.model)


if __name__ == "__main__":
    main()
