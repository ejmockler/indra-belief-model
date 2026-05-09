"""Run the unified-call scorer against the FN or FP calibration set.

Mirrors calibrate_dec.py but routes through the unified single-call path.
Per-pattern accuracy + overall, with sample failures listed for triage
into D3 curriculum iteration.

Usage:
  scripts/calibrate_unified.py fn       # FN calibration (target = correct on the gold)
  scripts/calibrate_unified.py fp       # FP calibration
  scripts/calibrate_unified.py fn --limit 20

The pass criterion uses `gold_tag`: a record passes if the unified
verdict matches gold_tag's mapped truth (correct → "correct",
no_relation/wrong_relation → "incorrect"). This sidesteps the dec-
specific `expected_post_fix_verdict` field, which is grounded in the
multi-call architecture's failure modes.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.data.corpus import CorpusIndex
from indra_belief.model_client import ModelClient
from indra_belief.scorers.scorer import score_evidence

HOLDOUT = ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl"
CAL_FN = ROOT / "data" / "benchmark" / "calibration_dec_fn.jsonl"
CAL_FP = ROOT / "data" / "benchmark" / "calibration_dec_fp.jsonl"


def _gold_to_target_verdict(gold_tag: str) -> str:
    """Map gold_tag to unified-call verdict vocabulary.

    gold_tag uses {correct, no_relation, wrong_relation, ...}
    unified verdict uses {correct, incorrect, abstain}
    """
    if gold_tag == "correct":
        return "correct"
    return "incorrect"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["fn", "fp"],
                        help="Which calibration set to run")
    parser.add_argument("--model", default="gemma-remote")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default=None,
                        help="Save raw results to JSONL")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("calibrate_unified")

    cal_path = CAL_FN if args.which == "fn" else CAL_FP
    cal_records = [json.loads(l) for l in open(cal_path)]
    if args.limit:
        cal_records = cal_records[:args.limit]
    log.info("loaded %d calibration records from %s", len(cal_records), cal_path)

    index = CorpusIndex()
    records = index.build_records(str(HOLDOUT))
    by_hash = {r.source_hash: r for r in records}
    log.info("indexed %d holdout records", len(records))

    client = ModelClient(args.model)

    pattern_results: dict[str, list[bool]] = defaultdict(list)
    pattern_fail_examples: dict[str, list[dict]] = defaultdict(list)
    pattern_pass_examples: dict[str, list[dict]] = defaultdict(list)
    raw: list[dict] = []

    t0 = time.time()
    for i, cal in enumerate(cal_records, 1):
        h = cal["source_hash"]
        rec = by_hash.get(h)
        if rec is None:
            log.warning("source_hash %s not in holdout; skipping", h)
            continue

        try:
            result = score_evidence(
                rec.statement, rec.evidence, client,
                use_unified=True,
            )
        except Exception as e:
            log.warning("score_evidence failed on %s: %s", h, e)
            continue

        actual = result.get("verdict")
        target = _gold_to_target_verdict(cal["gold_tag"])
        passed = (actual == target)
        pattern_results[cal["pattern"]].append(passed)

        summary = {
            "source_hash": h,
            "subject": cal["subject"],
            "object": cal["object"],
            "stmt_type": cal["stmt_type"],
            "gold_tag": cal["gold_tag"],
            "target": target,
            "actual": actual,
            "confidence": result.get("confidence"),
            "reasons": result.get("reasons", []),
            "tokens": result.get("tokens", 0),
            "rationale": result.get("rationale", ""),
        }
        if passed:
            pattern_pass_examples[cal["pattern"]].append(summary)
        else:
            pattern_fail_examples[cal["pattern"]].append(summary)
        raw.append({**summary, "pattern": cal["pattern"]})

        if i % 5 == 0 or i == len(cal_records):
            elapsed = time.time() - t0
            print(f"  progress {i}/{len(cal_records)} "
                  f"({elapsed:.0f}s, {elapsed/i:.1f}s/rec)", flush=True)

    elapsed = time.time() - t0

    # --- Report ---
    print(f"\n=== {args.which.upper()} calibration UNIFIED "
          f"(model={args.model}) ===\n")

    total_pass = total_n = 0
    for pattern in sorted(pattern_results):
        results = pattern_results[pattern]
        n = len(results)
        p = sum(results)
        total_pass += p
        total_n += n
        pct = 100 * p / n if n else 0
        print(f"  {pattern:35s} {p:3d}/{n:3d}  ({pct:5.1f}%)")
    overall = 100 * total_pass / total_n if total_n else 0
    print(f"\n  {'OVERALL':35s} {total_pass:3d}/{total_n:3d}  ({overall:5.1f}%)")
    print(f"  wall_clock                          {elapsed:.0f}s "
          f"({elapsed/max(total_n,1):.1f}s/record)")

    # Show fail examples for D3 triage
    print("\n--- Sample failures (first 3 per pattern) ---")
    for pattern in sorted(pattern_fail_examples):
        fails = pattern_fail_examples[pattern]
        if not fails:
            continue
        print(f"\n  [{pattern}]  ({len(fails)} failures)")
        for ex in fails[:3]:
            print(f"    {ex['stmt_type']:18s} {ex['subject']:>10s}→{ex['object']:<10s} "
                  f"target={ex['target']:<9s} actual={ex['actual']:<9s} "
                  f"reasons={ex['reasons']}")

    if args.output:
        with open(args.output, "w") as f:
            for r in raw:
                f.write(json.dumps(r) + "\n")
        print(f"\nRaw results saved to {args.output}")


if __name__ == "__main__":
    main()
