"""M13 analysis: M12 holdout vs L8 baseline.

Computes:
  - Confusion matrix delta (TP/FN/FP/TN, sens, spec, acc).
  - FN class distribution change (A-I taxonomy from L9).
  - M-phase reason-code attribution (regex_substrate_match,
    cascade_terminal_match, chain_match counts).
  - Per-record flip table: records where dec changed verdict.

Outputs both a printable summary and a JSON dump for archival.

Usage:
  python scripts/m13_analyze.py [--m12 PATH] [--l8 PATH]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_M12 = ROOT / "data" / "results" / "dec_e3_v18_mphase.jsonl"
DEFAULT_L8 = ROOT / "data" / "results" / "dec_e3_v17_jklphase.jsonl"
MONO_BASELINE = 0.8394  # locked in lifetime memory

M_PHASE_REASONS = {
    "regex_substrate_match",   # M3
    "cascade_terminal_match",  # M7
    "chain_match",             # M8
}


def load(path: Path) -> dict:
    """Return {source_hash: record_dict}."""
    out = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            out[r["source_hash"]] = r
    return out


def confusion(records: dict) -> dict:
    tp = fn = fp = tn = ab = 0
    for r in records.values():
        gold_pos = r["tag"] == "correct"
        pred = r["actual"]
        if pred == "abstain":
            ab += 1
            continue
        pred_pos = pred == "correct"
        if gold_pos and pred_pos: tp += 1
        elif gold_pos and not pred_pos: fn += 1
        elif not gold_pos and pred_pos: fp += 1
        else: tn += 1
    total = tp + fn + fp + tn
    return {
        "n": len(records), "TP": tp, "FN": fn, "FP": fp, "TN": tn, "abstain": ab,
        "acc": (tp + tn) / total if total else 0,
        "sens": tp / (tp + fn) if tp + fn else 0,
        "spec": tn / (tn + fp) if tn + fp else 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m12", default=str(DEFAULT_M12))
    p.add_argument("--l8", default=str(DEFAULT_L8))
    args = p.parse_args()

    m12 = load(Path(args.m12))
    l8 = load(Path(args.l8))

    # Confusion comparison
    m12_conf = confusion(m12)
    l8_conf = confusion(l8)

    print("=" * 80)
    print(f"M-phase holdout (M12) vs J+K+L baseline (L8) on {l8_conf['n']} records")
    print("=" * 80)
    print(f"{'Metric':<15} {'L8':<12} {'M12':<12} {'Δ':<10}")
    for k in ("acc", "sens", "spec"):
        l8v, m12v = l8_conf[k], m12_conf[k]
        delta = m12v - l8v
        print(f"{k:<15} {l8v:.4f}     {m12v:.4f}     {delta:+.4f}")
    print(f"{'TP':<15} {l8_conf['TP']:<12} {m12_conf['TP']:<12} "
          f"{m12_conf['TP'] - l8_conf['TP']:+d}")
    print(f"{'FN':<15} {l8_conf['FN']:<12} {m12_conf['FN']:<12} "
          f"{m12_conf['FN'] - l8_conf['FN']:+d}")
    print(f"{'FP':<15} {l8_conf['FP']:<12} {m12_conf['FP']:<12} "
          f"{m12_conf['FP'] - l8_conf['FP']:+d}")
    print(f"{'TN':<15} {l8_conf['TN']:<12} {m12_conf['TN']:<12} "
          f"{m12_conf['TN'] - l8_conf['TN']:+d}")
    print()
    gap = MONO_BASELINE - m12_conf["acc"]
    print(f"vs mono baseline {MONO_BASELINE:.4f}: M12 gap = {gap:+.4f}pp")
    print()

    # Reason-code attribution
    m_phase_attribution = Counter()
    for r in m12.values():
        for reason in r.get("reasons", []):
            if reason in M_PHASE_REASONS:
                m_phase_attribution[reason] += 1
    print("M-phase reason attribution:")
    for r, n in m_phase_attribution.most_common():
        print(f"  {r}: {n}")
    print()

    # Flip analysis
    flips_to_correct = []  # FN → TP
    flips_to_incorrect = []  # TP → FN (regression!)
    fp_closures = []  # FP → TN
    new_fps = []  # TN → FP (regression!)
    for h, m in m12.items():
        if h not in l8:
            continue
        l = l8[h]
        gold_pos = m["tag"] == "correct"
        l_pos = l["actual"] == "correct"
        m_pos = m["actual"] == "correct"
        if l_pos == m_pos:
            continue
        if gold_pos and not l_pos and m_pos:
            flips_to_correct.append(m)
        elif gold_pos and l_pos and not m_pos:
            flips_to_incorrect.append(m)
        elif not gold_pos and l_pos and not m_pos:
            fp_closures.append(m)
        elif not gold_pos and not l_pos and m_pos:
            new_fps.append(m)

    print(f"FN closures (FN→TP): {len(flips_to_correct)}")
    for r in flips_to_correct[:20]:
        reasons = ",".join(r.get("reasons", []))
        print(f"  {r['subject']} -{r['stmt_type']}-> {r['object']} | reasons={reasons}")
    print()
    print(f"FP closures (FP→TN): {len(fp_closures)}")
    for r in fp_closures[:20]:
        reasons = ",".join(r.get("reasons", []))
        print(f"  tag={r['tag']} | {r['subject']} -{r['stmt_type']}-> {r['object']} | reasons={reasons}")
    print()

    if flips_to_incorrect:
        print(f"REGRESSION: TP→FN flips ({len(flips_to_incorrect)}):")
        for r in flips_to_incorrect[:20]:
            reasons = ",".join(r.get("reasons", []))
            print(f"  {r['subject']} -{r['stmt_type']}-> {r['object']} | reasons={reasons}")
        print()
    if new_fps:
        print(f"REGRESSION: TN→FP flips ({len(new_fps)}):")
        for r in new_fps[:20]:
            reasons = ",".join(r.get("reasons", []))
            print(f"  tag={r['tag']} | {r['subject']} -{r['stmt_type']}-> {r['object']} | reasons={reasons}")

    # Verdict
    print()
    print("=" * 80)
    if m12_conf["acc"] >= 0.84:
        print("VERDICT: SHIP — M-phase achieves parity with mono baseline")
    elif m12_conf["acc"] >= 0.80:
        print("VERDICT: ITERATE — M-phase narrows gap; M' scope on residuals")
    else:
        print("VERDICT: SCALE-WALL — M-phase insufficient; consider model upgrade")
    print("=" * 80)

    # Archival JSON
    out_path = ROOT / "data" / "results" / "m13_analysis.json"
    with open(out_path, "w") as f:
        json.dump({
            "l8": l8_conf,
            "m12": m12_conf,
            "mono_baseline": MONO_BASELINE,
            "m_phase_attribution": dict(m_phase_attribution),
            "flips_to_correct_count": len(flips_to_correct),
            "fp_closures_count": len(fp_closures),
            "regressions_tp_fn": len(flips_to_incorrect),
            "regressions_tn_fp": len(new_fps),
        }, f, indent=2)
    print(f"Archived: {out_path}")


if __name__ == "__main__":
    main()
