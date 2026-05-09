"""Stage dec-disagreement records for gold-label audit.

Reads one or more dual_run JSONL outputs, extracts records where the
decomposed verdict disagrees with the gold polarity (after collapsing
gold reasons to a binary correct/incorrect), and writes a JSONL where
each line carries everything a human auditor or a structured LLM
classifier needs to make a (a) dec-wrong / (b) gold-loose / (c)
genuinely-ambiguous call.

The disagreement pool is the audit input for #54: estimating the
gold-noise rate. Without an estimate, we don't know what dec accuracy
actually MEANS — a 90% dec accuracy on a 10%-noisy gold caps real
accuracy at ~95-100% which we'd never measure. The audit unblocks
realistic gate-setting on #27 (full holdout n=501).

Usage:
    .venv/bin/python scripts/gold_audit_prep.py \\
        --inputs data/results/dual_run_small.jsonl \\
        --output data/benchmark/gold_audit_pool.jsonl

Each output record:
    {
      "rid": <int>,                 # original index
      "stmt_type": str,
      "subject": str, "object": str,
      "evidence_text": str,
      "gold_reason": str,           # raw gold tag (correct, no_relation, etc.)
      "gold_polarity": str,         # collapsed: correct | incorrect
      "dec_verdict": str,           # what dec said
      "dec_reasons": [str, ...],    # structured attribution
      "dec_rationale": str,
      "mono_verdict": str,
      "mono_label": str,            # whether mono matched gold
      "audit_classification": null  # to be filled by rater: dec_wrong, gold_loose, ambiguous
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _gold_polarity(gold: str) -> str:
    return "correct" if gold == "correct" else "incorrect"


def _load(paths: list[Path]) -> list[tuple[Path, list[dict]]]:
    out = []
    for p in paths:
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        out.append((p, rows))
    return out


def _extract_disagreements(rows: list[dict], source: str) -> list[dict]:
    """Yield audit records for entries where dec verdict != gold polarity.
    Includes both loud disagreements (FN/FP) and abstentions on records
    where mono got it right — both are signal for the audit."""
    audits = []
    for i, r in enumerate(rows, start=1):
        gp = _gold_polarity(r["gold"])
        dec_v = r["decomposed"]["verdict"]
        if dec_v == gp:
            continue
        audits.append({
            "source": source,
            "rid": i,
            "stmt_type": r["stmt_type"],
            "subject": r["subject"],
            "object": r["object"],
            "evidence_text": r["evidence_text"],
            "gold_reason": r["gold"],
            "gold_polarity": gp,
            "dec_verdict": dec_v,
            "dec_confidence": r["decomposed"].get("confidence"),
            "dec_reasons": r["decomposed"].get("reasons", []),
            "dec_rationale": r["decomposed"].get("rationale", ""),
            "mono_verdict": r["monolithic"]["verdict"],
            # Whether mono matched gold polarity — useful prior for the rater.
            "mono_matched_gold": r["monolithic"]["verdict"] == gp,
            "audit_classification": None,
        })
    return audits


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="One or more dual_run JSONL outputs to pool.")
    p.add_argument("--output", required=True,
                   help="Audit JSONL pool (one record per disagreement).")
    args = p.parse_args()

    sources = _load([Path(p) for p in args.inputs])
    audits: list[dict] = []
    for path, rows in sources:
        disagreements = _extract_disagreements(rows, source=path.name)
        audits.extend(disagreements)
        print(f"  {path.name}: {len(disagreements)} disagreements / {len(rows)} records")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for a in audits:
            fh.write(json.dumps(a) + "\n")

    print(f"\nWrote {len(audits)} audit records to {out}")
    print("\nNext steps:")
    print("  1. Each record has audit_classification=null. A human rater "
          "(or structured LLM classifier) sets it to one of:")
    print("       'dec_wrong'   — dec missed something the gold curator "
          "got right")
    print("       'gold_loose'  — gold over-attributes; dec's reasoning "
          "is sound")
    print("       'ambiguous'   — evidence genuinely supports either read")
    print("  2. Aggregate the rates → if gold_loose > 10%, the n=501 "
          "holdout's true accuracy ceiling is below the raw measurement.")


if __name__ == "__main__":
    main()
