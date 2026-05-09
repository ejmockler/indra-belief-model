"""Ceiling analysis v2 — reads parse_evidence v2 calibration output.

v2 uses set-containment matching and load-bearing joint accuracy
(match_found + axis + sign + negation all true simultaneously on the
same record). Grounding still reads the original format.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


BASELINE_ACCURACY = 0.828


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _p_evidence_v2(rows: list[dict]) -> tuple[float, dict]:
    if not rows:
        return 0.0, {}
    n = len(rows)
    # Joint: parsed + found_matching_assertion (or absent path satisfied)
    # + axis + sign + negation
    def joint_ok(r: dict) -> bool:
        res = r.get("result", {})
        if not res.get("parsed"):
            return False
        if not res.get("primary_axis"):
            return False
        if not res.get("primary_sign"):
            return False
        if not res.get("primary_negation"):
            return False
        # Perturbation is load-bearing under two-step inversion — the
        # adjudicator flips sign based on it, so getting it wrong at parse
        # time is equivalent to a sign error at adjudication.
        if not res.get("primary_perturbation", True):
            return False
        if res.get("found_matching_assertion"):
            return True
        if r.get("expected", {}).get("primary_axis") == "absent":
            return True
        return False

    joint = sum(1 for r in rows if joint_ok(r))
    per_field = {}
    for k in ("parsed", "found_matching_assertion", "primary_axis",
              "primary_sign", "primary_negation", "primary_perturbation",
              "primary_site"):
        hits = sum(1 for r in rows if r.get("result", {}).get(k))
        per_field[k] = hits / n
    return joint / n, per_field


def _p_grounding_boundary(rows: list[dict]) -> tuple[float, float, int]:
    """Returns (overall, boundary, n). Boundary = present vs not_present."""
    if not rows:
        return 0.0, 0.0, 0
    n = len(rows)
    correct = sum(1 for r in rows if r["actual_status"] == r["expected_status"])
    present = {"mentioned", "equivalent"}
    b_correct = sum(
        1 for r in rows
        if ("present" if r["actual_status"] in present else r["actual_status"])
        == ("present" if r["expected_status"] in present
            else r["expected_status"])
    )
    return correct / n, b_correct / n, n


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parse-evidence",
                   default=str(ROOT / "data" / "results"
                               / "parse_evidence_calibration_v2.jsonl"))
    p.add_argument("--grounding",
                   default=str(ROOT / "data" / "results"
                               / "grounding_calibration.jsonl"))
    p.add_argument("--baseline", type=float, default=BASELINE_ACCURACY)
    args = p.parse_args()

    print("=" * 70)
    print("Decomposed pipeline ceiling analysis v2 (#23, iterated)")
    print("=" * 70)

    pe_rows = _load(Path(args.parse_evidence))
    gr_rows = _load(Path(args.grounding))
    print(f"\nparse_evidence v2 records: {len(pe_rows)}")
    print(f"grounding records:         {len(gr_rows)}")

    p_claim = 1.0
    p_adjudicate = 1.0
    p_evidence, per_field = _p_evidence_v2(pe_rows)
    p_grounding_overall, p_grounding_boundary, gn = _p_grounding_boundary(gr_rows)

    print("\n--- Per-step reliabilities ---")
    print(f"  p_parse_claim       = {p_claim:.3f}  (deterministic)")
    if pe_rows:
        print(f"  p_parse_evidence    = {p_evidence:.3f}  "
              f"(joint: match + axis + sign + negation all true)")
        for k, v in per_field.items():
            print(f"                        {k:<28}= {v:.3f}")
    else:
        print("  p_parse_evidence    = <not measured>")
    if gr_rows:
        print(f"  p_grounding         = {p_grounding_overall:.3f} overall, "
              f"{p_grounding_boundary:.3f} present/not_present boundary")
    else:
        print("  p_grounding         = <not measured>")
    print(f"  p_adjudicate        = {p_adjudicate:.3f}  (deterministic)")

    if not pe_rows or not gr_rows:
        print("\nStatus: partial measurement — cannot compute ceiling")
        return

    ceiling = p_claim * p_evidence * p_grounding_boundary * p_adjudicate
    print(f"\n--- Ceiling ---")
    print(f"  ceiling (optimistic) = {ceiling:.3f}")
    print(f"  baseline (v16)       = {args.baseline:.3f}")
    print(f"  delta vs baseline    = {ceiling - args.baseline:+.3f}")
    if ceiling >= args.baseline:
        print("\nVERDICT: PASS — ceiling ≥ baseline. Proceed to #24.")
        print("  Caveat: optimistic upper bound; real end-to-end may be below.")
    else:
        print("\nVERDICT: FAIL — ceiling < baseline. Abort or iterate further.")


if __name__ == "__main__":
    main()
