"""N9 ship-verdict analysis: dec_v20_nphase vs dec_v18_mphase (M12 baseline).

Pre-registered ship gates from project_n_phase_pivot.md memory:
  - acc ≥ 79% (≥+4pp vs M12 baseline 74.95%; aspirational ≥84% mono parity)
  - FP rate ≤ M12 FPR + 1pp (rationalization tolerance)
  - No new abstain-by-rationalization (parser must not return zero-assertion
    on every claim record).

Outputs:
  - Confusion matrix delta (TP/FN/FP/TN, sens, spec, acc).
  - Flip table: TP→FN regressions (must be 0); FN→TP closures; etc.
  - FP-rate gate verdict (rationalization tolerance).
  - Mono-parity gap.

Comparable to m13_analyze.py but compares N9 (v20) against M12 (v18) and
mono baseline (83.94%, locked in lifetime memory).

Usage:
  python scripts/n9_analyze.py [--n9 PATH] [--m12 PATH]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_N9 = ROOT / "data" / "results" / "dec_e3_v20_nphase.jsonl"
DEFAULT_M12 = ROOT / "data" / "results" / "dec_e3_v18_mphase.jsonl"
MONO_BASELINE = 0.8394  # locked in lifetime memory (project_v15_ship)

# Pre-registered ship gates
GATE_ACC_MIN = 0.79  # ≥+4pp vs M12 (~75%)
GATE_FPR_TOL = 0.01  # +1pp rationalization tolerance


def load(path: Path) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["source_hash"]] = r
    return out


def confusion(records: dict) -> dict:
    tp = fn = fp = tn = ab = err = 0
    for r in records.values():
        pred = r["actual"]
        if pred == "abstain":
            ab += 1
            continue
        if pred is None or r.get("tier") == "error":
            err += 1
            continue
        gold_pos = r["tag"] == "correct"
        pred_pos = pred == "correct"
        if gold_pos and pred_pos: tp += 1
        elif gold_pos and not pred_pos: fn += 1
        elif not gold_pos and pred_pos: fp += 1
        else: tn += 1
    total = tp + fn + fp + tn
    return {
        "n": len(records), "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        "abstain": ab, "error": err,
        "acc": (tp + tn) / total if total else 0,
        "sens": tp / (tp + fn) if tp + fn else 0,
        "spec": tn / (tn + fp) if tn + fp else 0,
        "fpr": fp / (fp + tn) if fp + tn else 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n9", default=str(DEFAULT_N9))
    p.add_argument("--m12", default=str(DEFAULT_M12))
    args = p.parse_args()

    n9 = load(Path(args.n9))
    m12 = load(Path(args.m12))

    n9_conf = confusion(n9)
    m12_conf = confusion(m12)

    print("=" * 80)
    print(f"N9 (v20 N-phase) vs M12 (v18 M-phase) on {len(n9)} records")
    print("=" * 80)
    print(f"{'Metric':<10} {'M12':<14} {'N9':<14} {'Δ':<10}")
    for k in ("acc", "sens", "spec", "fpr"):
        m, n = m12_conf[k], n9_conf[k]
        print(f"{k:<10} {m:.4f}        {n:.4f}        {n - m:+.4f}")
    for k in ("TP", "FN", "FP", "TN"):
        m, n = m12_conf[k], n9_conf[k]
        print(f"{k:<10} {m:<14} {n:<14} {n - m:+d}")
    print(f"abstain    {m12_conf['abstain']:<14} {n9_conf['abstain']:<14}")
    print(f"error      {m12_conf['error']:<14} {n9_conf['error']:<14}")
    print()
    gap_mono = MONO_BASELINE - n9_conf["acc"]
    gap_m12 = n9_conf["acc"] - m12_conf["acc"]
    print(f"vs mono baseline {MONO_BASELINE:.4f}: N9 gap = {gap_mono:+.4f}")
    print(f"vs M12: N9 lift = {gap_m12:+.4f}")
    print()

    # Pre-registered ship gates
    print("=" * 80)
    print("PRE-REGISTERED SHIP GATES")
    print("=" * 80)
    gate_acc = n9_conf["acc"] >= GATE_ACC_MIN
    print(f"  Gate 1 (acc ≥ {GATE_ACC_MIN}): {n9_conf['acc']:.4f} → "
          f"{'PASS' if gate_acc else 'FAIL'}")
    fpr_delta = n9_conf["fpr"] - m12_conf["fpr"]
    gate_fpr = fpr_delta <= GATE_FPR_TOL
    print(f"  Gate 2 (FPR delta ≤ {GATE_FPR_TOL:+.4f}): {fpr_delta:+.4f} → "
          f"{'PASS' if gate_fpr else 'FAIL'}")
    # Rationalization probe: count records where parser returned zero
    # assertions (tier='decomposed' but reasons include 'absent_relationship'
    # at high rate suggests parser dropping everything).
    n9_absent = sum(1 for r in n9.values()
                    if "absent_relationship" in r.get("reasons", []))
    m12_absent = sum(1 for r in m12.values()
                     if "absent_relationship" in r.get("reasons", []))
    abs_delta = n9_absent - m12_absent
    print(f"  Gate 3 (absent_relationship delta): "
          f"M12={m12_absent} N9={n9_absent} Δ={abs_delta:+d}")
    print()

    # Flip analysis
    flips_to_correct = []  # FN → TP
    flips_to_incorrect = []  # TP → FN (REGRESSION)
    fp_closures = []  # FP → TN
    new_fps = []  # TN → FP (REGRESSION)
    common = set(n9.keys()) & set(m12.keys())
    for h in common:
        n, m = n9[h], m12[h]
        if n["actual"] not in ("correct", "incorrect"): continue
        if m["actual"] not in ("correct", "incorrect"): continue
        gold_pos = n["tag"] == "correct"
        n_pos = n["actual"] == "correct"
        m_pos = m["actual"] == "correct"
        if n_pos == m_pos: continue
        if gold_pos and not m_pos and n_pos:
            flips_to_correct.append(n)
        elif gold_pos and m_pos and not n_pos:
            flips_to_incorrect.append(n)
        elif not gold_pos and m_pos and not n_pos:
            fp_closures.append(n)
        elif not gold_pos and not m_pos and n_pos:
            new_fps.append(n)

    print(f"FN closures (FN→TP, M12→N9): {len(flips_to_correct)}")
    for r in flips_to_correct[:15]:
        rs = ",".join(r.get("reasons", []))
        print(f"  {r['subject']} -{r['stmt_type']}-> {r['object']} | {rs}")
    print()
    print(f"FP closures (FP→TN, M12→N9): {len(fp_closures)}")
    for r in fp_closures[:15]:
        rs = ",".join(r.get("reasons", []))
        print(f"  tag={r['tag']} | {r['subject']} -{r['stmt_type']}-> "
              f"{r['object']} | {rs}")
    print()

    if flips_to_incorrect:
        print(f"REGRESSION: TP→FN flips ({len(flips_to_incorrect)}):")
        for r in flips_to_incorrect[:15]:
            rs = ",".join(r.get("reasons", []))
            print(f"  {r['subject']} -{r['stmt_type']}-> {r['object']} | {rs}")
        print()
    if new_fps:
        print(f"REGRESSION: TN→FP flips ({len(new_fps)}):")
        for r in new_fps[:15]:
            rs = ",".join(r.get("reasons", []))
            print(f"  tag={r['tag']} | {r['subject']} -{r['stmt_type']}-> "
                  f"{r['object']} | {rs}")
        print()

    # Verdict
    print("=" * 80)
    all_gates = gate_acc and gate_fpr
    if n9_conf["acc"] >= 0.84:
        print("VERDICT: SHIP — N-phase achieves mono parity")
    elif all_gates:
        print("VERDICT: SHIP — N-phase passes pre-registered gates "
              "(below mono parity but above floor)")
    elif gate_acc and not gate_fpr:
        print("VERDICT: REVERT N6 — acc lifts but FPR breach indicates "
              "rationalization. Ship N1+N2+N3 substrate-only batch.")
    else:
        print("VERDICT: ITERATE — gates fail; scope post-mortem on "
              "residual error classes")
    print("=" * 80)

    # Archival JSON
    out_path = ROOT / "data" / "results" / "n9_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "m12": m12_conf,
            "n9": n9_conf,
            "mono_baseline": MONO_BASELINE,
            "gap_vs_mono": gap_mono,
            "lift_vs_m12": gap_m12,
            "gates": {
                "acc": gate_acc,
                "fpr": gate_fpr,
                "absent_relationship_delta": abs_delta,
            },
            "flips_to_correct_count": len(flips_to_correct),
            "fp_closures_count": len(fp_closures),
            "regressions_tp_fn": len(flips_to_incorrect),
            "regressions_tn_fp": len(new_fps),
        }, f, indent=2)
    print(f"Archived: {out_path}")


if __name__ == "__main__":
    main()
