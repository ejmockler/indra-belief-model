"""Q7 deep error profile — flip table vs M12 + reason-code stratification."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "data" / "results" / "dec_q_phase.jsonl"
M = ROOT / "data" / "results" / "dec_e3_v18_mphase.jsonl"


def _index(rows):
    return {r["source_hash"]: r for r in rows}


def main() -> None:
    q_rows = [json.loads(l) for l in open(Q) if l.strip()]
    m_rows = [json.loads(l) for l in open(M) if l.strip()]
    q_idx = _index(q_rows)
    m_idx = _index(m_rows)
    common = sorted(set(q_idx) & set(m_idx))

    # Flip table
    flips = {("correct", "correct"): [], ("correct", "incorrect"): [],
             ("incorrect", "correct"): [], ("incorrect", "incorrect"): [],
             ("other", "other"): []}
    target = {}
    for h in common:
        q = q_idx[h]
        m = m_idx[h]
        target[h] = q.get("target")
        qa = q.get("actual") if q.get("actual") in ("correct", "incorrect") else "other"
        ma = m.get("actual") if m.get("actual") in ("correct", "incorrect") else "other"
        if (ma, qa) not in flips:
            flips[(ma, qa)] = []
        flips[(ma, qa)].append(h)

    print(f"=== Q vs M12 flip table (n={len(common)} common records) ===\n")
    print(f"{'M12 \\ Q':<15}{'correct':>15}{'incorrect':>15}{'other':>10}")
    for m_v in ("correct", "incorrect", "other"):
        row = [f"{m_v:<15}"]
        for q_v in ("correct", "incorrect", "other"):
            n = len(flips.get((m_v, q_v), []))
            row.append(f"{n:>15}" if q_v != "other" else f"{n:>10}")
        print("".join(row))

    # Stratify the off-diagonal cells by gold target
    print("\n=== Q regression cells (M12 right, Q wrong against gold) ===\n")
    # M12 right + Q wrong = regressions
    regressions = []
    for h in common:
        q = q_idx[h]; m = m_idx[h]
        tgt = target[h]
        m_right = m.get("actual") == tgt
        q_right = q.get("actual") == tgt
        if m_right and not q_right and q.get("actual") in ("correct", "incorrect"):
            regressions.append(h)
    gains = []
    for h in common:
        q = q_idx[h]; m = m_idx[h]
        tgt = target[h]
        m_right = m.get("actual") == tgt
        q_right = q.get("actual") == tgt
        if q_right and not m_right and m.get("actual") in ("correct", "incorrect"):
            gains.append(h)

    print(f"Regressions (M12 right → Q wrong): {len(regressions)}")
    print(f"Gains (M12 wrong → Q right):       {len(gains)}")
    print(f"Net delta:                          {len(gains) - len(regressions):+d}")

    # Stratify regressions by stmt_type and Q-side reason
    print("\nRegressions by stmt_type:")
    by_type = Counter(q_idx[h]["stmt_type"] for h in regressions)
    for st, n in by_type.most_common(10):
        print(f"  {st:<22} {n}")

    print("\nRegressions by Q reason code:")
    by_reason = Counter()
    for h in regressions:
        for c in q_idx[h].get("reasons") or ():
            by_reason[c] += 1
    for c, n in by_reason.most_common(10):
        print(f"  {c:<30} {n}")

    print("\nGains by stmt_type:")
    by_type = Counter(q_idx[h]["stmt_type"] for h in gains)
    for st, n in by_type.most_common(10):
        print(f"  {st:<22} {n}")

    # Sample regressions with text patterns
    print("\n=== Sample regressions (first 12) — what M12 got, Q dropped ===\n")
    for h in regressions[:12]:
        q = q_idx[h]; m = m_idx[h]
        print(f"  {q['subject']:<18}-[{q['stmt_type']:>17}]->{q['object']:<18}  "
              f"target={q['target']:<10}  M12={m['actual']:<10}  Q={q['actual']:<10}  "
              f"Q.reasons={q.get('reasons') or []}")

    print("\n=== Sample gains (first 12) — what Q recovered ===\n")
    for h in gains[:12]:
        q = q_idx[h]; m = m_idx[h]
        print(f"  {q['subject']:<18}-[{q['stmt_type']:>17}]->{q['object']:<18}  "
              f"target={q['target']:<10}  M12={m['actual']:<10}  Q={q['actual']:<10}  "
              f"Q.reasons={q.get('reasons') or []}")

    # absent_relationship deep-dive
    print("\n=== absent_relationship FN regressions (parser missed binding) ===\n")
    abs_rel = [h for h in regressions
               if "absent_relationship" in (q_idx[h].get("reasons") or ())]
    print(f"Count: {len(abs_rel)}")
    type_breakdown = Counter(q_idx[h]["stmt_type"] for h in abs_rel)
    print("By stmt_type:")
    for st, n in type_breakdown.most_common():
        print(f"  {st:<22} {n}")

    # hedging_hypothesis FN regressions (presupposition mistakes)
    print("\n=== hedging_hypothesis FN regressions (presupposition class) ===\n")
    hh = [h for h in regressions
          if "hedging_hypothesis" in (q_idx[h].get("reasons") or ())]
    print(f"Count: {len(hh)}")
    for h in hh[:8]:
        q = q_idx[h]
        print(f"  {q['subject']:<18}-[{q['stmt_type']:>17}]->{q['object']:<18}")

    # Sign mismatch regressions
    print("\n=== sign_mismatch FN/FP regressions ===\n")
    sm = [h for h in regressions
          if "sign_mismatch" in (q_idx[h].get("reasons") or ())]
    print(f"Count: {len(sm)}")
    for h in sm[:6]:
        q = q_idx[h]
        print(f"  {q['subject']:<18}-[{q['stmt_type']:>17}]->{q['object']:<18}  "
              f"Q={q['actual']}  target={q['target']}")


if __name__ == "__main__":
    main()
