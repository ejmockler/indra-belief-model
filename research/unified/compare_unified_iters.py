"""Compare two unified-call calibration runs (e.g., D2 baseline vs D3
iter-1) and produce a flip-table per pattern.

Usage:
  scripts/compare_unified_iters.py \\
      --baseline data/results/unified_cal_fn.jsonl \\
      --candidate data/results/unified_cal_fn_v2.jsonl

Reports overall accuracy, per-pattern accuracy delta, and lists records
that flipped (pass→fail or fail→pass) so iter decisions are traceable.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def _load(path: Path) -> dict[int, dict]:
    """Index a calibration result file by source_hash."""
    out: dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            out[r["source_hash"]] = r
    return out


def _pass(r: dict) -> bool:
    return r["actual"] == r["target"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--label-baseline", default="baseline")
    p.add_argument("--label-candidate", default="candidate")
    args = p.parse_args()

    base = _load(Path(args.baseline))
    cand = _load(Path(args.candidate))

    common = set(base.keys()) & set(cand.keys())
    only_base = set(base.keys()) - set(cand.keys())
    only_cand = set(cand.keys()) - set(base.keys())

    print(f"# {args.label_baseline} = {args.baseline}")
    print(f"# {args.label_candidate} = {args.candidate}")
    print(f"common: {len(common)} | only-{args.label_baseline}: "
          f"{len(only_base)} | only-{args.label_candidate}: {len(only_cand)}")
    print()

    # Per-pattern table
    by_pattern: dict[str, dict] = defaultdict(
        lambda: {"both_pass": 0, "both_fail": 0,
                 "improved": 0, "regressed": 0,
                 "improved_records": [], "regressed_records": []})

    base_total_pass = 0
    cand_total_pass = 0
    for h in common:
        b = base[h]
        c = cand[h]
        pat = c.get("pattern", "?")
        bp = _pass(b)
        cp = _pass(c)
        if bp:
            base_total_pass += 1
        if cp:
            cand_total_pass += 1
        bucket = by_pattern[pat]
        if bp and cp:
            bucket["both_pass"] += 1
        elif (not bp) and (not cp):
            bucket["both_fail"] += 1
        elif (not bp) and cp:
            bucket["improved"] += 1
            bucket["improved_records"].append(c)
        elif bp and (not cp):
            bucket["regressed"] += 1
            bucket["regressed_records"].append(c)

    print(f"{'pattern':<35} {args.label_baseline:>12} "
          f"{args.label_candidate:>12} {'Δ':>6} {'pass→fail':>10} {'fail→pass':>10}")
    for pat in sorted(by_pattern):
        b = by_pattern[pat]
        n = b["both_pass"] + b["both_fail"] + b["improved"] + b["regressed"]
        bp = b["both_pass"] + b["regressed"]
        cp = b["both_pass"] + b["improved"]
        delta = cp - bp
        sign = "+" if delta > 0 else ""
        print(f"  {pat:<33} {bp:>3}/{n:<3} ({100*bp/n:>5.1f}%) "
              f"{cp:>3}/{n:<3} ({100*cp/n:>5.1f}%) "
              f"{sign}{delta:>4} {b['regressed']:>10} {b['improved']:>10}")

    n = len(common)
    delta = cand_total_pass - base_total_pass
    sign = "+" if delta > 0 else ""
    print(f"\n  {'OVERALL':<33} {base_total_pass:>3}/{n:<3} "
          f"({100*base_total_pass/n:>5.1f}%) "
          f"{cand_total_pass:>3}/{n:<3} ({100*cand_total_pass/n:>5.1f}%) "
          f"{sign}{delta:>4}")

    # Detail: regressions (concerning)
    print("\n--- REGRESSIONS (was passing, now failing) ---")
    any_reg = False
    for pat in sorted(by_pattern):
        recs = by_pattern[pat]["regressed_records"]
        if not recs:
            continue
        any_reg = True
        print(f"\n  [{pat}]")
        for r in recs:
            print(f"    {r['stmt_type']:18s} {r['subject']}→{r['object']}: "
                  f"target={r['target']} actual={r['actual']} "
                  f"reasons={r['reasons']}")
    if not any_reg:
        print("  (none)")

    # Detail: improvements
    print("\n--- IMPROVEMENTS (was failing, now passing) ---")
    any_imp = False
    for pat in sorted(by_pattern):
        recs = by_pattern[pat]["improved_records"]
        if not recs:
            continue
        any_imp = True
        print(f"\n  [{pat}]")
        for r in recs:
            print(f"    {r['stmt_type']:18s} {r['subject']}→{r['object']}: "
                  f"target={r['target']} actual={r['actual']} "
                  f"reasons={r['reasons']}")
    if not any_imp:
        print("  (none)")


if __name__ == "__main__":
    main()
