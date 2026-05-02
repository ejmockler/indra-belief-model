"""Calibration runner for parse_evidence.

Runs parse_evidence against a labeled JSONL eval set; computes per-field
accuracy (axis, sign, negation, n_assertions, agent_set, target_set) and
an overall pass rate. This is the #18 isolation calibration — feeds
p_evidence into the ceiling analysis (#23).

Per-field criterion: ≥85%. Sub-criterion: each per-axis accuracy ≥80%.

Usage:
    .venv/bin/python3 scripts/calibrate_parse_evidence.py \\
        --model gemma-remote \\
        --input data/benchmark/calibration_parse_evidence.jsonl \\
        --output data/results/parse_evidence_calibration.jsonl

The input JSONL format:
    {"id": str, "evidence": str, "expected": {EvidenceCommitment dict}, "note": str}
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Enable module imports before any project imports.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.model_client import ModelClient                      # noqa: E402
from indra_belief.scorers.commitments import EvidenceCommitment         # noqa: E402
from indra_belief.scorers.parse_evidence import parse_evidence          # noqa: E402


def _load_labeled(path: Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _compare_commitment(
    expected: dict, actual: EvidenceCommitment | None,
) -> dict:
    """Return a per-field comparison dict.

    Fields compared:
      - parsed:            commitment successfully produced (not None)
      - n_assertions:      number of assertions matches expected
      - primary_axis:      first assertion's axis matches
      - primary_sign:      first assertion's sign matches
      - primary_negation:  first assertion's negation matches
      - primary_site:      first assertion's site matches (lax — None matches any)
      - agents_set:        set(agent_candidates) == set(expected)
      - targets_set:       set(target_candidates) == set(expected)
    """
    out = {
        "parsed": actual is not None,
        "n_assertions": False,
        "primary_axis": False,
        "primary_sign": False,
        "primary_negation": False,
        "primary_site": False,
        "agents_set": False,
        "targets_set": False,
    }
    if actual is None:
        return out

    exp_assertions = expected.get("assertions", [])
    out["n_assertions"] = len(actual.assertions) == len(exp_assertions)

    if actual.assertions and exp_assertions:
        primary_exp = exp_assertions[0]
        primary_got = actual.assertions[0]
        out["primary_axis"] = primary_got.axis == primary_exp.get("axis")
        out["primary_sign"] = primary_got.sign == primary_exp.get("sign")
        out["primary_negation"] = primary_got.negation == bool(
            primary_exp.get("negation", False)
        )
        # Site: if expected has no site, treat any model-provided site as OK.
        exp_site = primary_exp.get("site")
        if exp_site is None:
            out["primary_site"] = True
        else:
            out["primary_site"] = primary_got.site == exp_site

    elif not actual.assertions and not exp_assertions:
        # Empty == empty — vacuously correct for the primary fields.
        for k in ("primary_axis", "primary_sign", "primary_negation",
                  "primary_site"):
            out[k] = True

    exp_agents = set(expected.get("agent_candidates", []))
    exp_targets = set(expected.get("target_candidates", []))
    out["agents_set"] = set(actual.agent_candidates) == exp_agents
    out["targets_set"] = set(actual.target_candidates) == exp_targets

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma-remote")
    p.add_argument("--input",
                   default=str(ROOT / "data" / "benchmark"
                               / "calibration_parse_evidence.jsonl"))
    p.add_argument("--output",
                   default=str(ROOT / "data" / "results"
                               / "parse_evidence_calibration.jsonl"))
    p.add_argument("--max-tokens", type=int, default=4000)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("calibrate")

    records = _load_labeled(Path(args.input))
    log.info("loaded %d labeled records", len(records))

    client = ModelClient(args.model)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    totals: Counter = Counter()
    per_axis_correct: dict[str, Counter] = defaultdict(Counter)
    rows: list[dict] = []
    with open(out_path, "w") as out_fh:
        for i, rec in enumerate(records, 1):
            rec_id = rec.get("id", f"rec_{i}")
            evidence = rec["evidence"]
            expected = rec["expected"]

            commitment = parse_evidence(evidence, client,
                                        max_tokens=args.max_tokens)
            result = _compare_commitment(expected, commitment)
            exp_axis = (
                (expected.get("assertions") or [{}])[0].get("axis", "unknown")
            )
            for k, v in result.items():
                totals[k] += int(bool(v))
            per_axis_correct[exp_axis]["total"] += 1
            if result["primary_axis"]:
                per_axis_correct[exp_axis]["axis_correct"] += 1
            if result["primary_sign"]:
                per_axis_correct[exp_axis]["sign_correct"] += 1

            row = {
                "id": rec_id,
                "evidence": evidence,
                "expected": expected,
                "actual": (
                    None if commitment is None
                    else {
                        "agent_candidates": list(commitment.agent_candidates),
                        "target_candidates": list(commitment.target_candidates),
                        "bystanders": list(commitment.bystanders),
                        "assertions": [
                            {"axis": a.axis, "sign": a.sign,
                             "negation": a.negation, "site": a.site,
                             "location_from": a.location_from,
                             "location_to": a.location_to,
                             "agents": list(a.agents),
                             "targets": list(a.targets)}
                            for a in commitment.assertions
                        ],
                    }
                ),
                "result": result,
            }
            rows.append(row)
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()

            status = "OK" if all(result.values()) else " ".join(
                k for k, v in result.items() if not v
            )
            log.info("[%d/%d] %s %s", i, len(records), rec_id, status)

    # --- Report ---
    n = len(records)
    print("\n" + "=" * 70)
    print(f"parse_evidence calibration: {n} records, model={args.model}")
    print("=" * 70)
    print("\n--- Per-field accuracy ---")
    fields = ["parsed", "n_assertions", "primary_axis", "primary_sign",
              "primary_negation", "primary_site", "agents_set", "targets_set"]
    for k in fields:
        acc = totals[k] / n if n else 0.0
        mark = "✓" if acc >= 0.85 else "✗"
        print(f"  {mark} {k:<18} {totals[k]:>3}/{n} = {acc:.1%}")

    print("\n--- Per-axis accuracy ---")
    for axis, counts in sorted(per_axis_correct.items()):
        t = counts["total"]
        if t == 0:
            continue
        axis_acc = counts["axis_correct"] / t
        sign_acc = counts["sign_correct"] / t
        print(f"  {axis:<15} n={t:<3} axis={axis_acc:.1%} sign={sign_acc:.1%}")

    # Overall pass = ALL fields above 85%
    passed = all(totals[k] / n >= 0.85 for k in fields if n > 0)
    print("\nVERDICT:", "PASS" if passed else "FAIL (below 85% threshold)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
