"""Check for data contamination between few-shot examples and holdout set.

Run this BEFORE any evaluation to ensure no example leaks into the holdout.
Checks both example_bank.json and CONTRASTIVE_EXAMPLES from evidence_scorer.py.

Usage:
    PYTHONPATH=src python scripts/check_contamination.py [--holdout PATH]

Exit code 0 = clean, 1 = contaminated.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_examples() -> list[dict]:
    """Load all few-shot examples from both sources."""
    from indra_belief.scorers._prompts import CONTRASTIVE_EXAMPLES

    bank_path = ROOT / "src" / "indra_belief" / "data" / "example_bank.json"
    examples = []

    # Base contrastive examples
    for ex in CONTRASTIVE_EXAMPLES:
        examples.append({
            "source": "CONTRASTIVE_EXAMPLES",
            "claim": ex["claim"],
            "evidence": ex["evidence"],
        })

    # Example bank
    if bank_path.exists():
        with open(bank_path) as f:
            bank = json.load(f)
        for key, pair in bank.items():
            for ex in pair:
                examples.append({
                    "source": f"example_bank:{key}",
                    "claim": ex["claim"],
                    "evidence": ex["evidence"],
                })

    return examples


def parse_claim(claim: str) -> tuple[str, str, str]:
    """Extract (subject, stmt_type, object) from 'SUBJ [TYPE] OBJ'."""
    parts = claim.replace("[", "|").replace("]", "|").split("|")
    subj = parts[0].strip()
    stype = parts[1].strip() if len(parts) > 1 else ""
    obj = parts[2].strip() if len(parts) > 2 else ""
    return subj, stype, obj


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Check example/holdout contamination")
    parser.add_argument(
        "--holdout",
        default=str(ROOT / "data" / "benchmark" / "holdout_large.jsonl"),
    )
    args = parser.parse_args()

    # Load holdout
    holdout_pairs = set()
    holdout_evidence = set()
    with open(args.holdout) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                h = json.loads(line)
            except json.JSONDecodeError:
                continue
            subj = h.get("subject", "") or ""
            obj = h.get("object", "") or ""
            ev = h.get("evidence_text", "") or ""
            holdout_pairs.add((subj, obj))
            if ev:
                holdout_evidence.add(ev[:80])

    print(f"Holdout: {len(holdout_pairs)} unique (subj, obj) pairs, "
          f"{len(holdout_evidence)} unique evidence texts")

    # Load examples
    examples = load_examples()
    print(f"Examples: {len(examples)} total")

    # Check contamination
    contaminated = []
    for ex in examples:
        subj, stype, obj = parse_claim(ex["claim"])
        ev_key = ex["evidence"][:80]

        pair_hit = (subj, obj) in holdout_pairs
        ev_hit = ev_key in holdout_evidence

        if pair_hit or ev_hit:
            reasons = []
            if pair_hit:
                reasons.append("pair")
            if ev_hit:
                reasons.append("evidence")
            contaminated.append({
                **ex,
                "subject": subj,
                "object": obj,
                "reasons": reasons,
            })

    if not contaminated:
        print("\nCLEAN — no contamination detected.")
        return 0

    print(f"\nCONTAMINATED — {len(contaminated)} examples overlap with holdout:\n")
    for c in contaminated:
        print(f"  {c['source']}")
        print(f"    claim: {c['claim']}")
        print(f"    match: {', '.join(c['reasons'])}")
        print()

    return 1


if __name__ == "__main__":
    sys.exit(main())
