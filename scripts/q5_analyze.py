"""Q5: analyze 10-record probe → GO/NO-GO for the 501-record holdout.

Reads data/results/q5_probe.jsonl and reports against Q5 ship gates:
  - median per-record ≤ 60s
  - p99 per-record ≤ 180s
  - 0 timeout incidents on healthy endpoint
  - median parse_evidence input ≤ 2500 tokens
  - parse_evidence median out ≤ 600 tokens
  - accuracy on 10 records compared to ground truth
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "results" / "q5_probe.jsonl"


def _percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    if p <= 0:
        return s[0]
    if p >= 100:
        return s[-1]
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def main():
    if not INPUT.exists():
        print(f"ERROR: {INPUT} missing", file=sys.stderr)
        sys.exit(1)
    rows = [json.loads(l) for l in open(INPUT) if l.strip()]
    n = len(rows)
    if n == 0:
        print("no records scored — abort", file=sys.stderr)
        sys.exit(1)

    print(f"\n=== Q5 probe analysis (n={n}) ===\n", file=sys.stderr)

    # accuracy
    decided = [r for r in rows if r.get("actual") in ("correct", "incorrect")]
    correct = sum(1 for r in decided if r["actual"] == r["target"])
    abstain = sum(1 for r in rows if r.get("actual") == "abstain")
    print(f"verdicts: {correct}/{len(decided)} correct, "
          f"{abstain} abstain, {n - len(decided) - abstain} other",
          file=sys.stderr)

    # per-record wall time (sum across calls)
    record_walls = []
    pe_durs, pe_ins, pe_outs = [], [], []
    vg_durs = []
    timeouts = 0
    finish = Counter()
    for r in rows:
        log = r.get("call_log") or []
        rec_total = 0.0
        for entry in log:
            d = float(entry.get("duration_s") or 0)
            rec_total += d
            finish[entry.get("finish_reason") or "?"] += 1
            if entry.get("error") == "TimeoutError":
                timeouts += 1
            if entry.get("kind") == "parse_evidence":
                pe_durs.append(d)
                pe_ins.append(int(entry.get("prompt_tokens") or 0))
                pe_outs.append(int(entry.get("out_tokens") or 0))
            elif entry.get("kind") == "verify_grounding":
                vg_durs.append(d)
        record_walls.append(rec_total)

    print("\nPer-record wall (sum across calls):", file=sys.stderr)
    print(f"  median: {median(record_walls):.1f}s "
          f"p99: {_percentile(record_walls, 99):.1f}s "
          f"max: {max(record_walls):.1f}s", file=sys.stderr)

    print("\nparse_evidence:", file=sys.stderr)
    if pe_durs:
        print(f"  duration: median {median(pe_durs):.1f}s "
              f"p99 {_percentile(pe_durs, 99):.1f}s "
              f"max {max(pe_durs):.1f}s", file=sys.stderr)
        print(f"  in_tokens: median {int(median(pe_ins))} "
              f"p99 {int(_percentile(pe_ins, 99))} "
              f"max {max(pe_ins)}", file=sys.stderr)
        print(f"  out_tokens: median {int(median(pe_outs))} "
              f"p99 {int(_percentile(pe_outs, 99))} "
              f"max {max(pe_outs)}", file=sys.stderr)

    print("\nverify_grounding:", file=sys.stderr)
    if vg_durs:
        print(f"  duration: median {median(vg_durs):.2f}s "
              f"p99 {_percentile(vg_durs, 99):.2f}s "
              f"n_calls {len(vg_durs)}", file=sys.stderr)

    print(f"\ntimeouts: {timeouts}", file=sys.stderr)
    print(f"finish_reasons: {dict(finish)}", file=sys.stderr)

    # Gates
    gates = {
        "median per-record ≤ 60s":
            median(record_walls) <= 60,
        "p99 per-record ≤ 180s":
            _percentile(record_walls, 99) <= 180,
        "0 timeouts": timeouts == 0,
        "median pe.in ≤ 2500":
            (median(pe_ins) if pe_ins else 0) <= 2500,
        "median pe.out ≤ 600":
            (median(pe_outs) if pe_outs else 0) <= 600,
    }
    print("\nGates:", file=sys.stderr)
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}", file=sys.stderr)

    # Holdout projection
    hours = (median(record_walls) * 501) / 3600
    print(f"\nProjected 501-record run: ~{hours:.1f} hours single-worker",
          file=sys.stderr)

    overall = all(gates.values())
    print(f"\nVERDICT: {'GO — proceed to holdout' if overall else 'NO-GO — iterate'}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
