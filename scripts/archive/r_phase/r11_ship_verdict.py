"""R11 ship verdict — three-way comparison of R-phase vs M12 vs Q-phase.

Inputs:
  data/results/dec_r_phase.jsonl    — R-phase
  data/results/dec_e3_v18_mphase.jsonl — M12 baseline
  data/results/dec_q_phase.jsonl    — Q-phase prior

Outputs:
  printed report — accuracy, flip table, regression+gain breakdowns,
                   substrate-fallback rescue rate, escalation telemetry,
                   ship/iterate/revert verdict.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
R = ROOT / "data" / "results" / "dec_r_phase.jsonl"
M = ROOT / "data" / "results" / "dec_e3_v18_mphase.jsonl"
Q = ROOT / "data" / "results" / "dec_q_phase.jsonl"


def _load(path: Path) -> dict:
    if not path.exists():
        return {}
    return {r["source_hash"]: r for r in (json.loads(l)
                                           for l in open(path) if l.strip())}


def _percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def _accuracy(rows):
    decided = [r for r in rows
               if r.get("actual") in ("correct", "incorrect")]
    correct = sum(1 for r in decided if r["actual"] == r["target"])
    return correct, len(decided)


def _flip_table(a_idx, b_idx, common):
    """3-way classifier of A vs B verdict×target → cell counts."""
    cells = defaultdict(list)
    for h in common:
        a = a_idx[h].get("actual") if a_idx[h].get("actual") in ("correct", "incorrect") else "abstain"
        b = b_idx[h].get("actual") if b_idx[h].get("actual") in ("correct", "incorrect") else "abstain"
        cells[(a, b)].append(h)
    return cells


def main() -> None:
    r = _load(R)
    m = _load(M)
    q = _load(Q)

    if not r:
        print(f"ERROR: {R} not found or empty")
        sys.exit(1)

    common = sorted(set(r) & set(m) & set(q))
    print(f"=== R11 SHIP VERDICT — n={len(common)} common records ===\n")

    # === ACCURACY ===
    R_correct, R_decided = _accuracy([r[h] for h in common])
    M_correct, M_decided = _accuracy([m[h] for h in common])
    Q_correct, Q_decided = _accuracy([q[h] for h in common])
    R_acc = R_correct / max(R_decided, 1)
    M_acc = M_correct / max(M_decided, 1)
    Q_acc = Q_correct / max(Q_decided, 1)

    print(f"ACCURACY:")
    print(f"  R-phase: {R_correct}/{R_decided} = {R_acc:.4f}  ({R_acc*100:.2f}%)")
    print(f"  M12:     {M_correct}/{M_decided} = {M_acc:.4f}  ({M_acc*100:.2f}%)")
    print(f"  Q-phase: {Q_correct}/{Q_decided} = {Q_acc:.4f}  ({Q_acc*100:.2f}%)")
    print(f"\n  Δ R vs M12: {(R_acc - M_acc) * 100:+.2f}pp")
    print(f"  Δ R vs Q:   {(R_acc - Q_acc) * 100:+.2f}pp")

    # === FLIP TABLES ===
    print(f"\n=== R vs M12 flip table ===\n")
    cells = _flip_table(r, m, common)
    print(f"{'M12 / R':<12}{'correct':>12}{'incorrect':>12}{'abstain':>12}")
    for mv in ("correct", "incorrect", "abstain"):
        row = [f"{mv:<12}"]
        for rv in ("correct", "incorrect", "abstain"):
            n = len(cells.get((rv, mv), []))
            row.append(f"{n:>12}")
        print("".join(row))

    print(f"\n=== R vs Q-phase flip table ===\n")
    cells_q = _flip_table(r, q, common)
    print(f"{'Q / R':<12}{'correct':>12}{'incorrect':>12}{'abstain':>12}")
    for qv in ("correct", "incorrect", "abstain"):
        row = [f"{qv:<12}"]
        for rv in ("correct", "incorrect", "abstain"):
            n = len(cells_q.get((rv, qv), []))
            row.append(f"{n:>12}")
        print("".join(row))

    # === REGRESSIONS + GAINS vs M12 ===
    print(f"\n=== R-phase vs M12 — regressions and gains ===\n")
    regressions_vs_m = []
    gains_vs_m = []
    for h in common:
        ra, ma = r[h].get("actual"), m[h].get("actual")
        tgt = r[h]["target"]
        if ra not in ("correct", "incorrect") or ma not in ("correct", "incorrect"):
            continue
        if ma == tgt and ra != tgt:
            regressions_vs_m.append(h)
        if ra == tgt and ma != tgt:
            gains_vs_m.append(h)
    print(f"  Regressions (M12 right, R wrong): {len(regressions_vs_m)}")
    print(f"  Gains       (M12 wrong, R right): {len(gains_vs_m)}")
    print(f"  Net delta:                          {len(gains_vs_m) - len(regressions_vs_m):+d}")

    # By stmt_type
    by_type_reg = Counter(r[h]["stmt_type"] for h in regressions_vs_m)
    by_type_gain = Counter(r[h]["stmt_type"] for h in gains_vs_m)
    print(f"\n  Regressions by stmt_type:")
    for st, n in by_type_reg.most_common(10):
        print(f"    {st:<22} {n}")
    print(f"\n  Gains by stmt_type:")
    for st, n in by_type_gain.most_common(10):
        print(f"    {st:<22} {n}")

    # === SUBSTRATE-FALLBACK RESCUE COUNT ===
    print(f"\n=== Substrate fallback (regex_substrate_match) ===\n")
    substrate_matches = [h for h in common
                         if "regex_substrate_match" in (r[h].get("reasons") or [])]
    print(f"  Records with substrate-fallback rescue: {len(substrate_matches)}")
    sm_correct = sum(1 for h in substrate_matches if r[h]["actual"] == r[h]["target"])
    print(f"  Of which correct vs gold: {sm_correct}/{len(substrate_matches)}")

    # === ESCALATION TELEMETRY ===
    print(f"\n=== Escalation flagged (R6) ===\n")
    escalated = [h for h in common if r[h].get("escalation_triggered")]
    print(f"  Total flagged: {len(escalated)}/{len(common)} "
          f"({100*len(escalated)/max(len(common),1):.1f}%)")
    by_reason = Counter(r[h].get("escalation_reason") for h in escalated)
    for reason, n in by_reason.most_common():
        print(f"    {reason}: {n}")

    # Of escalated records, how often did R get the right answer?
    if escalated:
        esc_correct = sum(1 for h in escalated
                          if r[h].get("actual") == r[h]["target"])
        print(f"\n  Of escalated records, correct: {esc_correct}/{len(escalated)} "
              f"({100*esc_correct/len(escalated):.1f}%)")

    # === LATENCY ===
    print(f"\n=== Latency (per-record from call_log) ===\n")
    walls = []
    for h in common:
        log = r[h].get("call_log") or []
        wall = sum(float(c.get("duration_s") or 0) for c in log)
        if wall > 0:
            walls.append(wall)
    if walls:
        print(f"  median: {median(walls):.1f}s")
        print(f"  p99:    {_percentile(walls, 99):.1f}s")
        print(f"  max:    {max(walls):.1f}s")

    # === FAILURE-CLASS BREAKDOWN ===
    print(f"\n=== R-phase failure reasons (full corpus) ===\n")
    r_failures = [h for h in common
                  if r[h].get("actual") not in (None, r[h]["target"])
                  or r[h].get("actual") is None]
    reason_count = Counter()
    for h in r_failures:
        for c in (r[h].get("reasons") or []):
            reason_count[c] += 1
    for reason, n in reason_count.most_common(15):
        print(f"  {reason:<32} {n}")

    # === SHIP GATES ===
    print(f"\n\n=== R11 SHIP GATE ===\n")
    gates = {
        "R beats Q-phase by ≥0.5pp": (R_acc - Q_acc) * 100 >= 0.5,
        "R within 0.5pp of M12 (≥)": (R_acc - M_acc) * 100 >= -0.5,
        "Net delta vs M12 not worse than -3 records":
            (len(gains_vs_m) - len(regressions_vs_m)) >= -3,
        "Median wall ≤ 25s":
            (median(walls) if walls else 0) <= 25,
        "p99 wall ≤ 90s (no widespread timeouts)":
            (_percentile(walls, 99) if walls else 0) <= 90,
        "Substrate fallback rescues ≥ 8 records":
            len(substrate_matches) >= 8,
    }
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    pass_count = sum(1 for v in gates.values() if v)
    total = len(gates)
    print(f"\n  Score: {pass_count}/{total} gates passing")
    if pass_count == total:
        print(f"\nVERDICT: SHIP — push to origin/main, update memory.")
    elif pass_count >= total - 1:
        print(f"\nVERDICT: SHIP-WITH-RESERVATIONS — write up the failed gate "
              f"as known limitation; ship if not silent FP.")
    elif (R_acc - Q_acc) * 100 < 0:
        print(f"\nVERDICT: REVERT — R-phase regresses Q-phase; fall back to M12 baseline.")
    else:
        print(f"\nVERDICT: ITERATE — address failed gates before ship.")


if __name__ == "__main__":
    main()
