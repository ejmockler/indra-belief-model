"""V6b — sanity check on holdout-overlap fixtures.

Runs all clean LFs on the first 100 records of `data/benchmark/holdout_v15_sample.jsonl`
and reports:
  (a) per-LF firing rate (n_fires / n_records)
  (b) per-class vote distribution per probe

Usage:
    .venv/bin/python scripts/v6b_sanity_check.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from indra_belief.v_phase.clean_lfs import (
    LF_INDEX_CLEAN,
    all_clean_lf_votes,
    all_clean_grounding_lf_votes,
)
from indra_belief.v_phase.substrate_lfs import (
    ABSTAIN,
    GROUNDING_CLASSES,
    RELATION_AXIS_CLASSES,
    ROLE_CLASSES,
    SCOPE_CLASSES,
)


PROBE_CLASS_MAP = {
    "relation_axis":  RELATION_AXIS_CLASSES,
    "subject_role":   ROLE_CLASSES,
    "object_role":    ROLE_CLASSES,
    "scope":          SCOPE_CLASSES,
    "verify_grounding": GROUNDING_CLASSES,
}


def main(n: int = 100) -> None:
    fpath = ROOT / "data" / "benchmark" / "holdout_v15_sample.jsonl"
    fires_per_lf: dict[str, int] = defaultdict(int)
    vote_dist: dict[tuple[str, int], int] = defaultdict(int)
    probe_votes_seen: dict[str, int] = defaultdict(int)
    total = 0

    with fpath.open() as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            rec = json.loads(line)
            total += 1
            ev = {
                "text": rec.get("evidence_text", ""),
                "source_api": rec.get("source_api"),
                "pmid": rec.get("pmid"),
            }
            stmt_votes = all_clean_lf_votes(rec, ev)
            for kind, name, fn, kwargs in LF_INDEX_CLEAN:
                if kind == "verify_grounding":
                    continue
                vote = stmt_votes.get(name, ABSTAIN)
                if vote != ABSTAIN:
                    fires_per_lf[name] += 1
                    vote_dist[(kind, vote)] += 1
                    probe_votes_seen[kind] += 1

            # Grounding LFs run per-entity (subject + object).
            for ent_role in ("subject", "object"):
                ename = rec.get(ent_role)
                if not ename:
                    continue
                ent = {"name": ename, "canonical": ename}
                gvotes = all_clean_grounding_lf_votes(ent, ev)
                for lf_name, vote in gvotes.items():
                    keyed = f"{lf_name}::{ent_role}"
                    if vote != ABSTAIN:
                        fires_per_lf[keyed] += 1
                        vote_dist[("verify_grounding", vote)] += 1
                        probe_votes_seen["verify_grounding"] += 1

    print(f"# V6b clean-LF sanity check (n={total} records)\n")

    # (a) per-LF firing rate
    print("## Per-LF firing rate (clean LFs, n=100 records)\n")
    print("| LF | fires | rate |")
    print("|---|---|---|")
    # Order: by LF_INDEX_CLEAN sequence
    seen_names: list[str] = []
    for kind, name, fn, kwargs in LF_INDEX_CLEAN:
        if kind == "verify_grounding":
            for ent_role in ("subject", "object"):
                keyed = f"{name}::{ent_role}"
                fires = fires_per_lf.get(keyed, 0)
                print(f"| {keyed} | {fires} | {fires/total:.0%} |")
                seen_names.append(keyed)
        else:
            fires = fires_per_lf.get(name, 0)
            print(f"| {name} | {fires} | {fires/total:.0%} |")
            seen_names.append(name)

    # (b) per-class vote distribution per probe
    print("\n## Per-class vote distribution per probe\n")
    for probe, classes in PROBE_CLASS_MAP.items():
        idx_to_name = {v: k for k, v in classes.items()}
        total_votes = probe_votes_seen.get(probe, 0)
        print(f"### {probe} (total non-ABSTAIN votes: {total_votes})\n")
        if total_votes == 0:
            print("  (no votes)\n")
            continue
        print("| class | count | share |")
        print("|---|---|---|")
        for idx in sorted(idx_to_name):
            count = vote_dist.get((probe, idx), 0)
            share = count / total_votes if total_votes else 0
            print(f"| {idx_to_name[idx]} ({idx}) | {count} | {share:.0%} |")
        print()


if __name__ == "__main__":
    main()
