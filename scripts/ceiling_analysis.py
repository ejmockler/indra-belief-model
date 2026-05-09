"""End-to-end reliability ceiling analysis (task #23).

Reads calibration outputs for each sub-call and computes the upper-bound
accuracy of the decomposed pipeline assuming INDEPENDENT errors:

    ceiling = p_parse_claim × p_parse_evidence × p_grounding × p_adjudicate

parse_claim and adjudicate are pure deterministic functions (p=1.0).
parse_evidence and verify_grounding are LLM sub-calls with measured
reliabilities from their calibration runs.

If the ceiling falls below the shipped monolithic baseline (82.8% on
v15_sample), the decomposed pipeline cannot, even in its best case,
out-perform the baseline — compose-cannot-beat-components. Abort or
redesign before paying integration cost.

This analysis is optimistic: real errors may be correlated (e.g., a
model bad at axis is usually also bad at sign on the same record).
The real end-to-end accuracy will typically be BELOW this ceiling, not
above it. The gate is: ceiling ≥ baseline is necessary but not
sufficient. Ceiling < baseline is a hard no-go signal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


BASELINE_ACCURACY = 0.828   # v16_sample on v15_holdout, monolithic scorer


def _per_field_rates(rows: list[dict]) -> dict[str, float]:
    """From parse_evidence calibration rows, compute per-field hit rate."""
    if not rows:
        return {}
    fields = rows[0]["result"].keys()
    rates = {}
    for f in fields:
        hits = sum(1 for r in rows if r["result"].get(f))
        rates[f] = hits / len(rows)
    return rates


def _grounding_accuracy(rows: list[dict]) -> dict[str, float]:
    """From grounding calibration rows, compute overall + boundary accuracy."""
    if not rows:
        return {}
    n = len(rows)
    correct = sum(1 for r in rows if r["actual_status"] == r["expected_status"])
    # Boundary accuracy: present vs not_present (which is what the
    # grounding_gap adjudicator check actually uses).
    present_set = {"mentioned", "equivalent"}

    def _binarize(s: str) -> str:
        return "present" if s in present_set else s

    boundary_correct = sum(
        1 for r in rows
        if _binarize(r["actual_status"]) == _binarize(r["expected_status"])
    )
    return {
        "n": n,
        "overall": correct / n,
        "boundary_present_vs_not_present": boundary_correct / n,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parse-evidence",
                   default=str(ROOT / "data" / "results"
                               / "parse_evidence_calibration.jsonl"))
    p.add_argument("--grounding",
                   default=str(ROOT / "data" / "results"
                               / "grounding_calibration.jsonl"))
    p.add_argument("--baseline", type=float, default=BASELINE_ACCURACY)
    args = p.parse_args()

    print("=" * 70)
    print("Decomposed pipeline ceiling analysis (#23)")
    print("=" * 70)

    pe_path = Path(args.parse_evidence)
    gr_path = Path(args.grounding)

    pe_rows = []
    gr_rows = []
    if pe_path.exists():
        with open(pe_path) as f:
            pe_rows = [json.loads(l) for l in f if l.strip()]
    if gr_path.exists():
        with open(gr_path) as f:
            gr_rows = [json.loads(l) for l in f if l.strip()]

    print(f"\nparse_evidence records: {len(pe_rows)}")
    print(f"grounding records:      {len(gr_rows)}")

    # p_parse_claim — deterministic from INDRA statement type
    p_claim = 1.0

    # p_parse_evidence — the driver fields affecting adjudication are
    # primary_axis and primary_sign (axis/sign_mismatch reason codes) and
    # primary_negation (contradicted reason code). Compute RECORD-LEVEL
    # joint accuracy (all load-bearing fields correct on the same record)
    # rather than a per-field product, because errors are correlated within
    # records (a model that gets axis wrong often gets sign wrong too).
    # Per-field product assumes independence and is too pessimistic.
    pe_rates = _per_field_rates(pe_rows)
    if pe_rows:
        load_bearing = ("parsed", "primary_axis", "primary_sign",
                        "primary_negation")
        joint_correct = sum(
            1 for r in pe_rows
            if all(r["result"].get(f) for f in load_bearing)
        )
        p_evidence = joint_correct / len(pe_rows)
        # Also compute the per-field product for comparison (pessimistic).
        p_evidence_indep = 1.0
        for f in load_bearing:
            p_evidence_indep *= pe_rates.get(f, 0.0)
    else:
        p_evidence = None
        p_evidence_indep = None

    # p_grounding — overall accuracy AND present/not_present boundary.
    gr_stats = _grounding_accuracy(gr_rows)
    p_grounding_overall = gr_stats.get("overall")
    p_grounding_boundary = gr_stats.get("boundary_present_vs_not_present")

    # p_adjudicate — deterministic rubric over typed inputs
    p_adjudicate = 1.0

    print("\n--- Per-step reliabilities ---")
    print(f"  p_parse_claim       = {p_claim:.3f}  (deterministic)")
    if pe_rows:
        print(f"  p_parse_evidence    = {p_evidence:.3f}  "
              f"(record-level joint on load-bearing fields)")
        print(f"                        per-field rates: "
              + ", ".join(f"{f[8:] if f.startswith('primary_') else f}="
                          f"{pe_rates.get(f, 0):.2f}"
                          for f in ("parsed", "primary_axis",
                                    "primary_sign", "primary_negation")))
        print(f"                        (per-field product "
              f"lower-bound: {p_evidence_indep:.3f})")
    else:
        print("  p_parse_evidence    = <not measured>")
    if p_grounding_overall is not None:
        print(f"  p_grounding         = {p_grounding_overall:.3f} overall "
              f"({p_grounding_boundary:.3f} present/not_present boundary)")
    else:
        print("  p_grounding         = <not measured>")
    print(f"  p_adjudicate        = {p_adjudicate:.3f}  (deterministic)")

    if p_evidence is None or p_grounding_overall is None:
        print("\nStatus: partial measurement — cannot compute ceiling")
        return

    # Ceiling (product of independent terms)
    ceiling = p_claim * p_evidence * p_grounding_boundary * p_adjudicate
    print(f"\n--- Ceiling ---")
    print(f"  ceiling (optimistic) = {ceiling:.3f}")
    print(f"  baseline (v16)       = {args.baseline:.3f}")
    print(f"  delta vs baseline    = {ceiling - args.baseline:+.3f}")
    if ceiling >= args.baseline:
        print("\nVERDICT: PASS — ceiling ≥ baseline. Proceed to pipeline "
              "wiring (#24).")
        print("  Caveat: this is an optimistic upper bound assuming")
        print("  independent errors. Real end-to-end accuracy may be")
        print("  below this ceiling. Calibrate against the actual")
        print("  pipeline once wired (#27).")
    else:
        print("\nVERDICT: FAIL — ceiling < baseline.")
        print("  The decomposed pipeline cannot out-perform the")
        print("  monolithic scorer even in its best case. Options:")
        print("  - Iterate parse_evidence prompt/fewshot to raise p_evidence")
        print("  - Iterate grounding prompt/fewshot to raise p_grounding")
        print("  - Redesign (e.g., split hedging into its own sub-call)")
        print("  - Abort the decomposition and focus on targeted")
        print("    monolithic fixes instead.")


if __name__ == "__main__":
    main()
