"""Stratified error-class breakdown for dual_run results.

Reads a dual_run JSONL (from `dual_run.py`) and reports per-path accuracy
bucketed by the holdout `tag` field. The tags are not binary labels —
they carry error-class information (polarity, grounding, no_relation,
negative_result, act_vs_amt, …). Tag-level breakdown is how we'll
attribute F5 directional targets (#27 prereg):

    F5: directional improvement vs monolithic v16 on
      - polarity FNs
      - family/grounding FPs
      - explicit-negation FNs

This script is deliberately a thin view over the already-produced JSONL,
not a second-pass LLM call. Re-run against the same JSONL to regenerate
the report without re-scoring.

Usage:
    .venv/bin/python scripts/dual_run_stratify.py \\
        data/results/dual_run_small.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _verdict_ok(gold: str | None, verdict: str | None) -> bool | None:
    """Mirror dual_run._verdict_ok so stratification uses the same rule."""
    if verdict in (None, "abstain") or gold is None:
        return None
    return (verdict.lower() == "correct") == (gold.lower() == "correct")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input", nargs="?",
                   default="data/results/dual_run_small.jsonl")
    args = p.parse_args()

    path = Path(args.input)
    if not path.exists():
        sys.exit(f"not found: {path}")

    rows = [json.loads(line) for line in path.open() if line.strip()]
    print(f"Loaded {len(rows)} records from {path}")

    # Tag-level accuracy breakdown
    per_tag_counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        tag = (r["gold"] or "?").lower()
        per_tag_counts[tag]["n"] += 1
        mono_ok = _verdict_ok(r["gold"], r["monolithic"]["verdict"])
        dec_ok = _verdict_ok(r["gold"], r["decomposed"]["verdict"])
        if mono_ok is True:
            per_tag_counts[tag]["mono_right"] += 1
        elif mono_ok is False:
            per_tag_counts[tag]["mono_wrong"] += 1
        else:
            per_tag_counts[tag]["mono_undecided"] += 1
        if dec_ok is True:
            per_tag_counts[tag]["dec_right"] += 1
        elif dec_ok is False:
            per_tag_counts[tag]["dec_wrong"] += 1
        else:
            per_tag_counts[tag]["dec_undecided"] += 1

    print("\n" + "=" * 72)
    print("Per-tag accuracy (monolithic vs decomposed)")
    print("=" * 72)
    print(f"{'tag':<20} {'n':>3}  {'mono':>12}  {'dec':>12}")
    for tag, c in sorted(per_tag_counts.items(),
                         key=lambda kv: -kv[1]["n"]):
        n = c["n"]
        mr = c["mono_right"]
        dr = c["dec_right"]
        mono_acc = f"{mr}/{n} ({mr/n:.0%})" if n else "-"
        dec_acc = f"{dr}/{n} ({dr/n:.0%})" if n else "-"
        print(f"{tag:<20} {n:>3}  {mono_acc:>12}  {dec_acc:>12}")

    # Decomposed reason codes grouped by tag — which reason fires on each
    # error class? Load-bearing for #28 (stratified error analysis).
    tag_reason_counts: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        tag = (r["gold"] or "?").lower()
        for rc in r["decomposed"].get("reasons") or []:
            tag_reason_counts[tag][rc] += 1

    print("\n" + "=" * 72)
    print("Decomposed reasons by tag")
    print("=" * 72)
    for tag, rcs in sorted(tag_reason_counts.items(), key=lambda kv: kv[0]):
        if not rcs:
            continue
        top = ", ".join(f"{r}={c}" for r, c in rcs.most_common())
        print(f"{tag:<20}  {top}")

    # Disagreement sampler — first five records where decomposed flips
    # monolithic's verdict (in either direction). Useful eyeball check.
    print("\n" + "=" * 72)
    print("First disagreements (sample)")
    print("=" * 72)
    shown = 0
    for r in rows:
        mv = (r["monolithic"]["verdict"] or "").lower()
        dv = (r["decomposed"]["verdict"] or "").lower()
        if mv and dv and mv != dv:
            print(f"\n[{r['stmt_type']}] gold={r['gold']}")
            print(f"  evidence: {r['evidence_text'][:110]}")
            print(f"  mono={mv}/{r['monolithic'].get('confidence')}, "
                  f"dec={dv}/{r['decomposed'].get('confidence')} "
                  f"reasons={r['decomposed'].get('reasons')}")
            shown += 1
            if shown >= 5:
                break


if __name__ == "__main__":
    main()
