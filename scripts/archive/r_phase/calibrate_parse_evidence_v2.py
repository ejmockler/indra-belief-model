"""Calibration runner v2 for parse_evidence.

Measures what the adjudicator actually USES downstream:

  For a record with expected primary_agents and primary_targets, does the
  commitment contain an assertion whose agents ⊇ expected.primary_agents
  AND targets ⊇ expected.primary_targets (set containment, not strict
  equality) AND whose (axis, sign, negation) match?

This is a more faithful calibration than strict per-field equality: the
adjudicator tolerates superfluous entities and extra assertions — it only
cares that the claim's binding is represented by SOME assertion with the
right (axis, sign, negation).

Label format (jsonl):
    {
        "id": str,
        "evidence": str,
        "expected": {
            "primary_agents": [str, ...],
            "primary_targets": [str, ...],
            "primary_axis": <axis>,
            "primary_sign": <sign>,
            "primary_negation": bool,   # optional, default false
            "primary_site": str|null,   # optional
            "location_from": str|null,  # optional
            "location_to": str|null,    # optional
            "n_assertions_at_least": int,   # optional
            "n_assertions_at_most":  int,   # optional
            "note_allow_unclear": bool      # optional — absent-relationship
        },
        "note": str
    }
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.model_client import (                                 # noqa: E402
    ModelClient, concurrency_hint,
)
from indra_belief.scorers.commitments import EvidenceCommitment         # noqa: E402
from indra_belief.scorers.parse_evidence import parse_evidence          # noqa: E402


def _find_matching_assertion(
    commitment: EvidenceCommitment, expected_agents: set[str],
    expected_targets: set[str],
):
    """Find the first assertion whose agents ⊇ expected_agents and targets
    ⊇ expected_targets (set containment). Returns None if no match."""
    for a in commitment.assertions:
        if expected_agents.issubset(set(a.agents)) and \
                expected_targets.issubset(set(a.targets)):
            return a
    return None


def _evaluate(expected: dict, actual: EvidenceCommitment | None) -> dict:
    """Compute per-criterion pass/fail against the v2 label schema."""
    out = {
        "parsed": actual is not None,
        "n_assertions_bounds": False,
        "found_matching_assertion": False,
        "primary_axis": False,
        "primary_sign": False,
        "primary_negation": False,
        "primary_perturbation": False,
        "primary_site": False,
        "location_from": False,
        "location_to": False,
    }
    if actual is None:
        return out

    # --- n_assertions bounds ---
    n = len(actual.assertions)
    lo = expected.get("n_assertions_at_least")
    hi = expected.get("n_assertions_at_most")
    bounds_ok = True
    if lo is not None and n < lo:
        bounds_ok = False
    if hi is not None and n > hi:
        bounds_ok = False
    out["n_assertions_bounds"] = bounds_ok

    # --- absent-relationship path ---
    exp_axis = expected.get("primary_axis")
    _absent_keys = ("found_matching_assertion", "primary_axis",
                    "primary_sign", "primary_negation",
                    "primary_perturbation", "primary_site",
                    "location_from", "location_to")
    if exp_axis == "absent":
        if n == 0:
            for k in _absent_keys:
                out[k] = True
        elif expected.get("note_allow_unclear") and all(
                a.axis in ("unclear", "absent") for a in actual.assertions):
            for k in _absent_keys:
                out[k] = True
        return out

    # --- Locate the matching assertion via set-containment ---
    exp_agents = set(expected.get("primary_agents", []))
    exp_targets = set(expected.get("primary_targets", []))
    matched = _find_matching_assertion(actual, exp_agents, exp_targets)
    out["found_matching_assertion"] = matched is not None
    if matched is None:
        return out

    # --- Compare the primary fields on the matched assertion ---
    out["primary_axis"] = matched.axis == exp_axis
    out["primary_sign"] = matched.sign == expected.get("primary_sign")
    out["primary_negation"] = matched.negation == bool(
        expected.get("primary_negation", False)
    )
    # perturbation: if label specifies, check it; otherwise default "none"
    exp_perturb = expected.get("primary_perturbation", "none")
    out["primary_perturbation"] = matched.perturbation == exp_perturb
    exp_site = expected.get("primary_site")
    if exp_site is None:
        out["primary_site"] = True  # site-agnostic expectation
    else:
        got = matched.site or ""
        out["primary_site"] = exp_site in got or got == exp_site

    exp_lf = expected.get("location_from")
    if exp_lf is None:
        out["location_from"] = True
    else:
        out["location_from"] = matched.location_from == exp_lf

    exp_lt = expected.get("location_to")
    if exp_lt is None:
        out["location_to"] = True
    else:
        out["location_to"] = matched.location_to == exp_lt

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma-remote")
    p.add_argument("--input",
                   default=str(ROOT / "data" / "benchmark"
                               / "calibration_parse_evidence_v2.jsonl"))
    p.add_argument("--output",
                   default=str(ROOT / "data" / "results"
                               / "parse_evidence_calibration_v2.jsonl"))
    p.add_argument("--max-tokens", type=int, default=6000)
    p.add_argument("--workers", default="auto",
                   help="Concurrent worker count. 'auto' uses the model's "
                        "concurrency_hint (1 for local, 8 for Google). "
                        "Pass an integer to override.")
    args = p.parse_args()

    if args.workers == "auto":
        workers = concurrency_hint(args.model)
    else:
        workers = max(1, int(args.workers))

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("calibrate_v2")

    records = []
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    log.info("loaded %d labeled records (workers=%d)", len(records), workers)

    client = ModelClient(args.model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    totals: Counter = Counter()
    per_axis: dict[str, Counter] = defaultdict(Counter)
    rows: list[dict | None] = [None] * len(records)
    write_lock = threading.Lock()

    def _serialize(rec_id: str, evidence: str, expected: dict,
                   commitment: EvidenceCommitment | None,
                   result: dict) -> dict:
        return {
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
                         "negation": a.negation,
                         "perturbation": a.perturbation,
                         "site": a.site,
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

    def _process(i_rec: tuple[int, dict]) -> tuple[int, dict, str]:
        i, rec = i_rec
        rec_id = rec.get("id", f"rec_{i+1}")
        evidence = rec["evidence"]
        expected = rec["expected"]
        try:
            commitment = parse_evidence(evidence, client,
                                        max_tokens=args.max_tokens)
        except Exception as e:
            log.warning("[%d] %s raised %s — counted as parse failure",
                        i + 1, rec_id, type(e).__name__)
            commitment = None
        result = _evaluate(expected, commitment)
        row = _serialize(rec_id, evidence, expected, commitment, result)
        return i, row, rec_id

    completed = 0
    with open(out_path, "w") as out_fh, \
            ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, (i, r)): i
                   for i, r in enumerate(records)}
        for fut in as_completed(futures):
            i, row, rec_id = fut.result()
            rows[i] = row
            result = row["result"]
            exp_ax = row["expected"].get("primary_axis", "unknown")
            with write_lock:
                for k, v in result.items():
                    totals[k] += int(bool(v))
                per_axis[exp_ax]["total"] += 1
                if result.get("primary_axis"):
                    per_axis[exp_ax]["axis_correct"] += 1
                if result.get("primary_sign"):
                    per_axis[exp_ax]["sign_correct"] += 1
                if result.get("found_matching_assertion"):
                    per_axis[exp_ax]["match_found"] += 1
                out_fh.write(json.dumps(row) + "\n")
                out_fh.flush()
                completed += 1
            status = "OK" if all(result.values()) else " ".join(
                k for k, v in result.items() if not v
            )
            log.info("[%d/%d] %s %s", completed, len(records), rec_id, status)

    # Drop any None placeholders if a future never completed (defensive).
    rows = [r for r in rows if r is not None]

    n = len(records)
    print("\n" + "=" * 70)
    print(f"parse_evidence calibration v2: {n} records, model={args.model}")
    print("=" * 70)
    print("\n--- Per-criterion pass rate ---")
    criteria = ["parsed", "found_matching_assertion", "n_assertions_bounds",
                "primary_axis", "primary_sign", "primary_negation",
                "primary_perturbation", "primary_site",
                "location_from", "location_to"]
    for k in criteria:
        acc = totals[k] / n if n else 0.0
        mark = "✓" if acc >= 0.85 else "✗"
        print(f"  {mark} {k:<28} {totals[k]:>3}/{n} = {acc:.1%}")

    # Joint: load-bearing fields (axis, sign, negation) on the matched
    # assertion all correct simultaneously.
    joint = sum(
        1 for r in rows
        if r["result"].get("found_matching_assertion")
        and r["result"].get("primary_axis")
        and r["result"].get("primary_sign")
        and r["result"].get("primary_negation")
    )
    # For absent-relationship cases, the absent-path assertion counts as joint.
    absent_joint = sum(
        1 for r in rows
        if r["expected"].get("primary_axis") == "absent"
        and r["result"].get("primary_axis")
    )
    total_joint = joint + sum(
        1 for r in rows
        if r["expected"].get("primary_axis") == "absent"
        and r["result"].get("primary_axis")
        and not r["result"].get("found_matching_assertion")
    )
    # Simpler: joint = match-found + axis + sign + negation all true OR absent-path satisfied
    joint_v2 = sum(
        1 for r in rows
        if (r["result"].get("primary_axis") and r["result"].get("primary_sign")
            and r["result"].get("primary_negation")
            and r["result"].get("primary_perturbation")
            and (r["result"].get("found_matching_assertion")
                 or r["expected"].get("primary_axis") == "absent"))
    )
    print(f"\n  JOINT load-bearing (match+axis+sign+neg): "
          f"{joint_v2}/{n} = {joint_v2/n:.1%}")

    print("\n--- Per-axis accuracy ---")
    for axis in sorted(per_axis):
        counts = per_axis[axis]
        t = counts["total"]
        if t == 0:
            continue
        a = counts["axis_correct"] / t
        s = counts["sign_correct"] / t
        m = counts["match_found"] / t
        print(f"  {axis:<15} n={t:<3} match={m:.1%} axis={a:.1%} sign={s:.1%}")

    p_evidence = joint_v2 / n if n else 0
    print(f"\np_evidence (joint load-bearing) = {p_evidence:.3f}")
    passed = p_evidence >= 0.85
    print("VERDICT:", "PASS (≥85%)" if passed else "FAIL (<85%)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
