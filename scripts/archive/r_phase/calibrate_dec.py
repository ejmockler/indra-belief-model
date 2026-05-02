"""Run a dec-pipeline calibration against the FN or FP calibration set.

Joins source_hash from a calibration JSONL to the source holdout file,
re-scores each record with the current dec pipeline, and reports
per-pattern accuracy + overall.

Usage:
  scripts/calibrate_dec.py fn                    # runs FN calibration
  scripts/calibrate_dec.py fp                    # runs FP calibration
  scripts/calibrate_dec.py fn --model gemma-remote

The calibration set's `expected_post_fix_verdict` field is the
ground-truth target: a record passes if the dec verdict matches that.
For FN-cal, expected = "correct" (the dec path should now recover what
the pre-fix architecture missed). For FP-cal, expected matches the
curator-aligned verdict.

Per-pattern reporting lets each fix declare which patterns it should lift
without dropping `negative_test_should_pass` records.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.data.corpus import CorpusIndex
from indra_belief.model_client import ModelClient
from indra_belief.scorers.scorer import score_evidence

HOLDOUT = ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl"
CAL_FN = ROOT / "data" / "benchmark" / "calibration_dec_fn.jsonl"
CAL_FP = ROOT / "data" / "benchmark" / "calibration_dec_fp.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("which", choices=["fn", "fp"],
                        help="Which calibration set to run")
    parser.add_argument("--model", default="gemma-remote")
    parser.add_argument("--max-tokens", type=int, default=12000)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("calibrate_dec")

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
    pattern_pass_examples: dict[str, list[dict]] = defaultdict(list)
    pattern_fail_examples: dict[str, list[dict]] = defaultdict(list)

    for i, cal in enumerate(cal_records, 1):
        h = cal["source_hash"]
        rec = by_hash.get(h)
        if rec is None:
            log.warning("source_hash %s not in holdout; skipping", h)
            continue

        result = score_evidence(
            rec.statement, rec.evidence, client,
            max_tokens=args.max_tokens,
            use_decomposed=True,
        )
        actual = result.get("verdict")
        expected = cal["expected_post_fix_verdict"]
        passed = (actual == expected)
        pattern_results[cal["pattern"]].append(passed)
        record_summary = {
            "source_hash": h,
            "subject": cal["subject"],
            "object": cal["object"],
            "stmt_type": cal["stmt_type"],
            "expected": expected,
            "actual": actual,
            "tier": result.get("tier"),
            "reasons": result.get("reasons", []),
        }
        if passed:
            pattern_pass_examples[cal["pattern"]].append(record_summary)
        else:
            pattern_fail_examples[cal["pattern"]].append(record_summary)

        if i % 5 == 0 or i == len(cal_records):
            log.info("progress: %d/%d", i, len(cal_records))

    # --- Report ---
    print(f"\n=== {args.which.upper()} calibration "
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

    # Show 1-2 fail examples per pattern (cap noise)
    print("\n--- Sample failures (first 2 per pattern) ---")
    for pattern in sorted(pattern_fail_examples):
        fails = pattern_fail_examples[pattern]
        if not fails:
            continue
        print(f"\n  [{pattern}]")
        for ex in fails[:2]:
            print(f"    {ex['stmt_type']} {ex['subject']}→{ex['object']}: "
                  f"expected={ex['expected']!r} actual={ex['actual']!r} "
                  f"tier={ex['tier']!r} reasons={ex['reasons']}")


if __name__ == "__main__":
    main()
