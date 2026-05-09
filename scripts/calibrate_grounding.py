"""Calibration runner for the grounding sub-verifier.

Runs verify_grounding against a labeled JSONL eval set; computes a
confusion matrix over {mentioned, equivalent, not_present, uncertain} and
reports overall accuracy. This is the #20 isolation calibration — feeds
p_grounding into the ceiling analysis (#23).

Criterion: overall ≥80% (the monolithic flagged-tier baseline). mentioned
vs equivalent confusion is expected at ~10%; the critical boundaries are
(mentioned|equivalent) vs not_present (grounding gate correctness).

Usage:
    .venv/bin/python3 scripts/calibrate_grounding.py \\
        --model gemma-remote \\
        --input data/benchmark/calibration_grounding.jsonl \\
        --output data/results/grounding_calibration.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.model_client import ModelClient                       # noqa: E402
from indra_belief.scorers.grounding import verify_grounding             # noqa: E402


@dataclass
class _FakeEntity:
    """Duck-typed GroundedEntity stand-in — avoids instantiating the real
    GroundedEntity (which would call Gilda). The grounding sub-verifier
    only reads attributes."""
    name: str
    raw_text: str | None = None
    canonical: str | None = None
    db: str | None = None
    db_id: str | None = None
    aliases: list = field(default_factory=list)
    all_names: list = field(default_factory=list)
    is_family: bool = False
    family_members: list = field(default_factory=list)
    description: str = ""
    is_pseudogene: bool = False
    verification_status: str | None = None
    verification_note: str = ""
    gilda_score: float | None = None
    is_low_confidence: bool = False
    is_known_alias: bool = True
    competing_candidates: list = field(default_factory=list)
    text_top_name: str | None = None


def _load_labeled(path: Path) -> list[dict]:
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _build_entity(spec: dict) -> _FakeEntity:
    return _FakeEntity(
        name=spec["name"],
        db=spec.get("db"),
        db_id=spec.get("db_id"),
        aliases=list(spec.get("aliases", [])),
        is_family=bool(spec.get("is_family", False)),
        family_members=list(spec.get("family_members", [])),
        is_pseudogene=bool(spec.get("is_pseudogene", False)),
        gilda_score=spec.get("gilda_score"),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="gemma-remote")
    p.add_argument("--input",
                   default=str(ROOT / "data" / "benchmark"
                               / "calibration_grounding.jsonl"))
    p.add_argument("--output",
                   default=str(ROOT / "data" / "results"
                               / "grounding_calibration.jsonl"))
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("calibrate_grounding")

    records = _load_labeled(Path(args.input))
    log.info("loaded %d labeled records", len(records))

    client = ModelClient(args.model)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Confusion: confusion[expected][actual] = count
    confusion: dict[str, Counter] = defaultdict(Counter)
    total = 0
    correct = 0

    with open(out_path, "w") as out_fh:
        for i, rec in enumerate(records, 1):
            rec_id = rec.get("id", f"rec_{i}")
            entity = _build_entity(rec["claim_entity"])
            evidence = rec["evidence"]
            expected = rec["expected_status"]

            verdict = verify_grounding(entity, evidence, client)
            actual = verdict.status

            total += 1
            if actual == expected:
                correct += 1
            confusion[expected][actual] += 1

            row = {
                "id": rec_id,
                "claim_entity": rec["claim_entity"],
                "evidence": evidence,
                "expected_status": expected,
                "actual_status": actual,
                "rationale": verdict.rationale,
                "match": actual == expected,
            }
            out_fh.write(json.dumps(row) + "\n")
            out_fh.flush()
            mark = "✓" if actual == expected else "✗"
            log.info("[%d/%d] %s %s expected=%s actual=%s",
                     i, len(records), mark, rec_id, expected, actual)

    # --- Report ---
    n = total
    print("\n" + "=" * 70)
    print(f"grounding calibration: {n} records, model={args.model}")
    print("=" * 70)
    print(f"\nOverall accuracy: {correct}/{n} = {correct/n:.1%}")

    print("\n--- Confusion matrix (rows=expected, cols=actual) ---")
    statuses = ("mentioned", "equivalent", "not_present", "uncertain")
    header = "  " + " ".join(f"{s:>12}" for s in statuses)
    print(f"  {'expected \\ actual':<14}" + header)
    for exp in statuses:
        row = f"  {exp:<14}"
        for got in statuses:
            cnt = confusion.get(exp, {}).get(got, 0)
            row += f" {cnt:>12}"
        print(row)

    # Critical boundary: (mentioned|equivalent) vs not_present.
    # A present entity classified as not_present = hard grounding_gap FP downstream.
    # not_present → present = hard FN (legitimate reject bypassed).
    present = {"mentioned", "equivalent"}
    present_to_absent = sum(
        confusion.get(e, {}).get("not_present", 0) for e in present
    )
    absent_to_present = sum(
        confusion.get("not_present", {}).get(g, 0) for g in present
    )
    print("\n--- Critical boundary errors ---")
    print(f"  present → not_present (would cause grounding_gap FP): {present_to_absent}")
    print(f"  not_present → present (would bypass grounding reject): {absent_to_present}")

    passed = (correct / n) >= 0.80 if n else False
    print("\nVERDICT:", "PASS" if passed else "FAIL (below 80% threshold)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
