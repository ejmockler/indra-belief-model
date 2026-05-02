"""S11 ship verdict — four-way comparison S vs R vs M12 vs Q-phase.

Inputs:
  data/results/dec_s_phase.jsonl       — S-phase
  data/results/dec_r_phase.jsonl       — R-phase
  data/results/dec_e3_v18_mphase.jsonl — M12 baseline
  data/results/dec_q_phase.jsonl       — Q-phase prior

Outputs:
  printed report — accuracy four-way, flip tables, regression+gain
                   breakdowns, substrate-fast-path coverage,
                   ship/iterate/revert verdict.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
S = ROOT / "data" / "results" / "dec_s_phase.jsonl"
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
    cells = defaultdict(list)
    for h in common:
        a = (a_idx[h].get("actual")
             if a_idx[h].get("actual") in ("correct", "incorrect")
             else "abstain")
        b = (b_idx[h].get("actual")
             if b_idx[h].get("actual") in ("correct", "incorrect")
             else "abstain")
        cells[(a, b)].append(h)
    return cells


def main() -> None:
    s = _load(S)
    r = _load(R)
    m = _load(M)
    q = _load(Q)

    if not s:
        print(f"ERROR: {S} not found or empty")
        sys.exit(1)

    common = sorted(set(s) & set(r) & set(m) & set(q))
    print(f"=== S11 SHIP VERDICT — n={len(common)} common records ===\n")

    # === ACCURACY ===
    S_correct, S_decided = _accuracy([s[h] for h in common])
    R_correct, R_decided = _accuracy([r[h] for h in common])
    M_correct, M_decided = _accuracy([m[h] for h in common])
    Q_correct, Q_decided = _accuracy([q[h] for h in common])
    S_acc = S_correct / max(S_decided, 1)
    R_acc = R_correct / max(R_decided, 1)
    M_acc = M_correct / max(M_decided, 1)
    Q_acc = Q_correct / max(Q_decided, 1)

    print(f"ACCURACY:")
    print(f"  S-phase: {S_correct}/{S_decided} = {S_acc:.4f}  "
          f"({S_acc*100:.2f}%)")
    print(f"  R-phase: {R_correct}/{R_decided} = {R_acc:.4f}  "
          f"({R_acc*100:.2f}%)")
    print(f"  M12:     {M_correct}/{M_decided} = {M_acc:.4f}  "
          f"({M_acc*100:.2f}%)")
    print(f"  Q-phase: {Q_correct}/{Q_decided} = {Q_acc:.4f}  "
          f"({Q_acc*100:.2f}%)")
    print(f"\n  Δ S vs R: {(S_acc - R_acc) * 100:+.2f}pp")
    print(f"  Δ S vs M: {(S_acc - M_acc) * 100:+.2f}pp")
    print(f"  Δ S vs Q: {(S_acc - Q_acc) * 100:+.2f}pp")

    # === FLIP TABLES ===
    print(f"\n=== S vs R-phase flip table ===\n")
    cells = _flip_table(s, r, common)
    print(f"{'R / S':<12}{'correct':>12}{'incorrect':>12}{'abstain':>12}")
    for rv in ("correct", "incorrect", "abstain"):
        row = [f"{rv:<12}"]
        for sv in ("correct", "incorrect", "abstain"):
            n = len(cells.get((sv, rv), []))
            row.append(f"{n:>12}")
        print("".join(row))

    print(f"\n=== S vs M12 flip table ===\n")
    cells_m = _flip_table(s, m, common)
    print(f"{'M / S':<12}{'correct':>12}{'incorrect':>12}{'abstain':>12}")
    for mv in ("correct", "incorrect", "abstain"):
        row = [f"{mv:<12}"]
        for sv in ("correct", "incorrect", "abstain"):
            n = len(cells_m.get((sv, mv), []))
            row.append(f"{n:>12}")
        print("".join(row))

    # === REGRESSIONS + GAINS vs M12 ===
    print(f"\n=== S-phase vs M12 — regressions and gains ===\n")
    regressions_vs_m, gains_vs_m = [], []
    for h in common:
        sa, ma = s[h].get("actual"), m[h].get("actual")
        tgt = s[h]["target"]
        if sa not in ("correct", "incorrect"):
            continue
        if ma not in ("correct", "incorrect"):
            continue
        if ma == tgt and sa != tgt:
            regressions_vs_m.append(h)
        if sa == tgt and ma != tgt:
            gains_vs_m.append(h)
    print(f"  Regressions (M12 right, S wrong): {len(regressions_vs_m)}")
    print(f"  Gains       (M12 wrong, S right): {len(gains_vs_m)}")
    print(f"  Net delta:                          "
          f"{len(gains_vs_m) - len(regressions_vs_m):+d}")

    # === REGRESSIONS + GAINS vs R ===
    print(f"\n=== S-phase vs R-phase — regressions and gains ===\n")
    regressions_vs_r, gains_vs_r = [], []
    for h in common:
        sa, ra = s[h].get("actual"), r[h].get("actual")
        tgt = s[h]["target"]
        if sa not in ("correct", "incorrect"):
            continue
        if ra not in ("correct", "incorrect"):
            continue
        if ra == tgt and sa != tgt:
            regressions_vs_r.append(h)
        if sa == tgt and ra != tgt:
            gains_vs_r.append(h)
    print(f"  Regressions (R right, S wrong): {len(regressions_vs_r)}")
    print(f"  Gains       (R wrong, S right): {len(gains_vs_r)}")
    print(f"  Net delta:                        "
          f"{len(gains_vs_r) - len(regressions_vs_r):+d}")

    # === SUBSTRATE FAST-PATH COVERAGE ===
    print(f"\n=== Substrate fast-path (n_probe_llm_calls per record) ===\n")
    zero_llm = sum(1 for h in common
                   if s[h].get("n_probe_llm_calls") == 0)
    by_n = Counter(s[h].get("n_probe_llm_calls", 0) for h in common)
    print(f"  Records with zero LLM probe calls: "
          f"{zero_llm}/{len(common)} "
          f"({100 * zero_llm / max(len(common), 1):.1f}%)")
    for n, cnt in sorted(by_n.items()):
        print(f"    {n} LLM calls: {cnt}")

    # === ABSTAIN COUNTS ===
    print(f"\n=== Abstain rates ===")
    s_abstain = sum(1 for h in common if s[h].get("actual") == "abstain")
    r_abstain = sum(1 for h in common if r[h].get("actual") == "abstain")
    m_abstain = sum(1 for h in common if m[h].get("actual") == "abstain")
    print(f"  S-phase abstains: {s_abstain}/{len(common)} "
          f"({100 * s_abstain / max(len(common),1):.1f}%)")
    print(f"  R-phase abstains: {r_abstain}/{len(common)} "
          f"({100 * r_abstain / max(len(common),1):.1f}%)")
    print(f"  M12     abstains: {m_abstain}/{len(common)} "
          f"({100 * m_abstain / max(len(common),1):.1f}%)")

    # === LATENCY ===
    print(f"\n=== Latency (per-record, S-phase) ===")
    walls = []
    for h in common:
        log = s[h].get("call_log") or []
        wall = sum(float(c.get("duration_s") or 0) for c in log)
        if wall > 0:
            walls.append(wall)
    if walls:
        print(f"  median: {median(walls):.1f}s")
        print(f"  p99:    {_percentile(walls, 99):.1f}s")
        print(f"  max:    {max(walls):.1f}s")

    # === REASON BREAKDOWN ===
    print(f"\n=== S-phase reason codes (full corpus) ===")
    reason_count: Counter[str] = Counter()
    for h in common:
        for c in (s[h].get("reasons") or []):
            reason_count[c] += 1
    for reason, n in reason_count.most_common(15):
        print(f"  {reason:<32} {n}")

    # === SHIP GATES ===
    print(f"\n\n=== S11 SHIP GATE ===\n")
    gates = {
        "S beats R-phase by ≥ 1pp": (S_acc - R_acc) * 100 >= 1.0,
        "S within 1pp of M12": (S_acc - M_acc) * 100 >= -1.0,
        "Net delta vs M12 ≥ -3 records":
            (len(gains_vs_m) - len(regressions_vs_m)) >= -3,
        "Median wall ≤ 25s":
            (median(walls) if walls else 0) <= 25,
        "p99 wall ≤ 90s":
            (_percentile(walls, 99) if walls else 0) <= 90,
        "Substrate fast-path resolves ≥ 50% records":
            zero_llm / max(len(common), 1) >= 0.5,
    }
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")

    pass_count = sum(1 for v in gates.values() if v)
    total = len(gates)
    print(f"\n  Score: {pass_count}/{total} gates passing")
    if pass_count == total:
        print(f"\nVERDICT: SHIP — push to origin/main, update memory.")
    elif pass_count >= total - 1:
        print(f"\nVERDICT: SHIP-WITH-RESERVATIONS — write up the failed "
              f"gate as known limitation; ship if not silent FP.")
    elif (S_acc - R_acc) * 100 < -1.0:
        print(f"\nVERDICT: REVERT — S regresses R-phase; "
              f"fall back to R-phase or M12.")
    else:
        print(f"\nVERDICT: ITERATE — address failed gates before ship.")


if __name__ == "__main__":
    main()
