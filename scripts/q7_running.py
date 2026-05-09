"""Q7 running-error-profile snapshot — quick analysis of dec_q_phase.jsonl
as it grows. Run after every progress batch to track the trajectory.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

INPUT = Path("data/results/dec_q_phase.jsonl")
M12 = Path("data/results/dec_e3_v18_mphase.jsonl")


def main() -> None:
    rows = [json.loads(l) for l in open(INPUT) if l.strip()]
    n = len(rows)

    # Cumulative accuracy
    decided = [r for r in rows if r.get("actual") in ("correct", "incorrect")]
    correct = [r for r in decided if r["actual"] == r["target"]]
    errors = [r for r in decided if r["actual"] != r["target"]]
    acc = len(correct) / max(len(decided), 1)

    # M12 reference at same n
    m12 = [json.loads(l) for l in open(M12) if l.strip()][:n]
    m12_dec = [r for r in m12 if r.get("actual") in ("correct", "incorrect")]
    m12_correct = [r for r in m12_dec if r["actual"] == r["target"]]
    m12_acc = len(m12_correct) / max(len(m12_dec), 1)

    # Recent batch (last 30)
    last30 = rows[-30:]
    last30_dec = [r for r in last30 if r.get("actual") in ("correct", "incorrect")]
    last30_correct = sum(1 for r in last30_dec if r["actual"] == r["target"])
    last30_acc = last30_correct / max(len(last30_dec), 1) if last30_dec else 0

    print(f"\n=== n={n} | cum acc {acc:.3f} | last30 {last30_acc:.3f} | M12 ref {m12_acc:.3f} | Δ vs M12 {(acc-m12_acc)*100:+.1f}pp ===")

    # FN/FP split
    fp = [r for r in errors if r["actual"] == "correct" and r["target"] == "incorrect"]
    fn = [r for r in errors if r["actual"] == "incorrect" and r["target"] == "correct"]
    print(f"errors: FN={len(fn)} FP={len(fp)} (total {len(errors)}/{len(decided)})")

    # By stmt_type
    err_by_type = Counter(r["stmt_type"] for r in errors)
    n_by_type = Counter(r["stmt_type"] for r in rows)
    print("\nstmt_type errors:")
    for st, ne in err_by_type.most_common(8):
        nt = n_by_type[st]
        print(f"  {st:<22} {ne:>3}/{nt:<3} ({ne/nt:.0%})")

    # FN reason histogram
    fn_reasons = Counter()
    for r in fn:
        for c in (r.get("reasons") or []):
            fn_reasons[c] += 1
    print("\nFN reasons:")
    for c, n_ in fn_reasons.most_common(8):
        print(f"  {c:<30} {n_}")

    # FP reason histogram
    fp_reasons = Counter()
    for r in fp:
        for c in (r.get("reasons") or []):
            fp_reasons[c] += 1
    if fp_reasons:
        print("\nFP reasons:")
        for c, n_ in fp_reasons.most_common(5):
            print(f"  {c:<30} {n_}")

    # Newest FN of each (first time the error code appears in the recent batch)
    recent_fn = [r for r in rows[-30:]
                 if r.get("actual") == "incorrect" and r["target"] == "correct"]
    if recent_fn:
        print("\nrecent FN (last 30):")
        for r in recent_fn:
            print(f"  {r['subject']:<15} -[{r['stmt_type']:>16}]-> {r['object']:<15}  "
                  f"reasons={(r.get('reasons') or [None])[:1]}")


if __name__ == "__main__":
    main()
