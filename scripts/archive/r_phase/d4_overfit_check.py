"""D4 overfit check: run scorer on a held-back sample drawn from
holdout-minus-calibration, compare accuracy to the calibration baseline.

Pass criterion: held-back accuracy tracks the calibration baseline within
±5pp; outside that, the curriculum has overfit toward calibration.

Default architecture: decomposed (production). Pass --use-monolithic to
ablate against the monolithic path.

Usage:
  scripts/d4_overfit_check.py
  scripts/d4_overfit_check.py --limit 10
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

HELD_BACK = ROOT / "data" / "benchmark" / "holdout_d4_held_back.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gemma-remote")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--input", default=str(HELD_BACK))
    parser.add_argument("--output",
                        default=str(ROOT / "data" / "results"
                                    / "d4_held_back_result.jsonl"))
    parser.add_argument("--use-monolithic", action="store_true", default=False,
                        help="Bypass the decomposed pipeline (ablation only; "
                             "decomposed is the production default).")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume: skip records already in --output, append rather than overwrite")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("d4_overfit_check")

    # Load held-back records (gold_tag is null in this file because it
    # came from the holdout, not the calibration set; we use record.tag
    # via CorpusIndex which carries gold from the corpus index).
    index = CorpusIndex()
    records = index.build_records(args.input)
    if args.limit:
        records = records[:args.limit]
    log.info("loaded %d held-back records", len(records))

    client = ModelClient(args.model)
    results: list[dict] = []
    by_stmt: dict[str, list[bool]] = defaultdict(list)

    # Per-record checkpoint: open output file in line-buffered append mode and
    # flush after each write. A long run that crashes mid-flight leaves a
    # usable partial result on disk; --resume can pick up where it left off.
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: load already-scored source_hashes and replay them into stats so
    # the progress meter and final accuracy match the cumulative run.
    already_scored: set[int] = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                already_scored.add(r["source_hash"])
                results.append(r)
                by_stmt[r["stmt_type"]].append(r["actual"] == r["target"])
        log.info("resume: %d records already scored — skipping", len(already_scored))

    out_mode = "a" if args.resume else "w"
    out_f = open(output_path, out_mode, buffering=1)

    t0 = time.time()
    new_scored = 0  # records actually processed in THIS run (for rate)
    for i, rec in enumerate(records, 1):
        if rec.source_hash in already_scored:
            continue
        new_scored += 1
        # Map gold to verdict-vocab: holdout's tag is "correct" or anything else
        target = "correct" if rec.tag == "correct" else "incorrect"
        try:
            result = score_evidence(
                rec.statement, rec.evidence, client,
                use_decomposed=not args.use_monolithic,
            )
        except Exception as e:
            log.warning("score failed on %s: %s", rec.source_hash, e)
            continue
        actual = result.get("verdict")
        passed = (actual == target)
        by_stmt[rec.stmt_type].append(passed)
        record = {
            "source_hash": rec.source_hash,
            "subject": rec.subject,
            "stmt_type": rec.stmt_type,
            "object": rec.object,
            "tag": rec.tag,
            "target": target,
            "actual": actual,
            "confidence": result.get("confidence"),
            "reasons": result.get("reasons", []),
            "tier": result.get("tier"),
            "tokens": result.get("tokens", 0),
            "call_log": result.get("call_log", []),
        }
        results.append(record)
        out_f.write(json.dumps(record) + "\n")
        out_f.flush()
        if i % 5 == 0 or i == len(records):
            elapsed = time.time() - t0
            n_pass = sum(1 for r in results if r.get("actual") == r.get("target"))
            n_total = len(results)
            rate = elapsed / max(new_scored, 1)
            print(f"  progress {i}/{len(records)} acc={100*n_pass/max(n_total,1):.1f}% "
                  f"({elapsed:.0f}s, {rate:.1f}s/rec this run)", flush=True)
    out_f.close()

    elapsed = time.time() - t0

    n_total = len(results)
    n_pass = sum(1 for r in results if r["actual"] == r["target"])
    overall = 100 * n_pass / max(n_total, 1)

    print()
    print(f"=== D4 held-back overfit check (model={args.model}) ===")
    print()
    print(f"  OVERALL                          {n_pass:>3}/{n_total:<3} ({overall:>5.1f}%)")
    print(f"  wall_clock                       {elapsed:.0f}s ({elapsed/max(n_total,1):.1f}s/rec)")
    print()
    print("  Per-stmt-type:")
    for st in sorted(by_stmt):
        passes = by_stmt[st]
        n = len(passes)
        p = sum(passes)
        print(f"    {st:<20s} {p}/{n} ({100*p/n:.1f}%)")

    # Drift check vs combined cal baseline (D1+D2 = 84.8%, iter-1 expected ~85-88%)
    print()
    print("  G_D drift check (overfit guard):")
    print("    Combined cal baseline (D1+D2): 84.8%")
    print(f"    Held-back overall:             {overall:.1f}%")
    drift = overall - 84.8
    print(f"    Drift:                         {drift:+.1f}pp")
    if abs(drift) <= 5:
        print(f"    VERDICT: PASS (within ±5pp; curriculum generalizes)")
    else:
        print(f"    VERDICT: FAIL (drift > 5pp; curriculum may have overfit)")

    print(f"\nRaw results saved to {output_path}")


if __name__ == "__main__":
    main()
