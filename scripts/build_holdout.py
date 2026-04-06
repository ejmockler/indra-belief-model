"""Reproducibly build holdout.jsonl — the fresh 200-record evaluation set.

Exclusions:
- All source_hashes in old holdout_set.jsonl (v5/v6 evaluation set)
- All source_hashes in eval_set_v4.jsonl (v4 evaluation)
- All source_hashes in fewshot_pool_v4.jsonl AND fewshot_pool.jsonl (prompt pools)
- All source_hashes used as v6/v7 contrastive examples
- All agent-pair (subject, object) tuples in v6/v7 examples OR legacy prompt pools
- Placeholder "?" entities (unevaluable)

Stratification: 100 correct + 100 incorrect, tags proportional to natural frequency.
Seed: 42.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "benchmark"

# Explicit set of source_hashes used as contrastive examples in v6/v7 prompts
V6_V7_EXAMPLE_HASHES = {
    7573779735080809356,   # TGFB1->ADAM17 correct
    -239508306376493971,   # TGFB1->ADAM17 act_vs_amt
    -1303766619543169753,  # CCR7->AKT correct (v6 only)
    -2130696303216728252,  # CCR7->AKT hypothesis (v6 only)
    8359432524362575925,   # TNFSF10->CASP3 correct (v6 only, replaced in v7)
    -113992818617033052,   # TNFSF10->CASP8 no_relation
    1095129517427259895,   # p14_3_3->CDC25C hypothesis (v6 only)
    7619869527954625866,   # p14_3_3->CDC25C correct (v6 only)
    6749626743890915148,   # IFNA->NFkappaB correct
    2253277188767597204,   # MYB->PPID correct (v7)
    6038249661651140566,   # MYB->PPID hypothesis (v7)
    -8116562012361219275,  # HIF1A->TP53 correct (v7)
    8353214881988006642,   # HIF1A->TP53 incorrect (v7)
    9071342424668569671,   # ROBO1->SRC correct (v7)
    5688396891926661437,   # Proteasome->ESR1 act_vs_amt (v7)
}

# All entity pairs used in any v6/v7 example
EXAMPLE_PAIRS = {
    ("Actin", "CDK9"), ("AKT", "CASP3"),
    ("TGFB1", "ADAM17"),
    ("AGER", "MMP2"), ("TP53", "MDM2"),
    ("CCR7", "AKT"), ("MYB", "PPID"),
    ("IFNA", "NFkappaB"), ("TNFSF10", "CASP8"), ("TNFSF10", "CASP3"),
    ("AURKB", "ATXN10"), ("MTOR", "RPS6KB1"), ("P70S6K", "RPS6"),
    ("p14_3_3", "CDC25C"),
    ("HIF1A", "TP53"), ("ROBO1", "SRC"), ("Proteasome", "ESR1"),
}


def load_hashes(path: Path) -> set[int]:
    if not path.exists():
        return set()
    hashes = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            hashes.add(r["source_hash"])
    return hashes


def load_pairs(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    pairs = set()
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            pairs.add((r["subject"], r["object"]))
    return pairs


def main(seed: int = 42, n_correct: int = 100, n_incorrect: int = 100):
    random.seed(seed)

    # Load all exclusion sets
    excluded_hashes = set()
    excluded_hashes |= load_hashes(DATA / "holdout_set.jsonl")      # old holdout
    excluded_hashes |= load_hashes(DATA / "eval_set_v4.jsonl")       # v4 eval
    excluded_hashes |= load_hashes(DATA / "fewshot_pool_v4.jsonl")   # v4 pool
    excluded_hashes |= load_hashes(DATA / "fewshot_pool.jsonl")      # legacy pool
    excluded_hashes |= V6_V7_EXAMPLE_HASHES

    excluded_pairs = set()
    excluded_pairs |= EXAMPLE_PAIRS
    excluded_pairs |= load_pairs(DATA / "fewshot_pool_v4.jsonl")
    excluded_pairs |= load_pairs(DATA / "fewshot_pool.jsonl")

    print(f"Exclusions:")
    print(f"  source_hashes: {len(excluded_hashes)}")
    print(f"  entity pairs: {len(excluded_pairs)}")

    # Load benchmark + apply exclusions
    available = []
    with open(DATA / "belief_benchmark.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if r["source_hash"] in excluded_hashes:
                continue
            if (r["subject"], r["object"]) in excluded_pairs:
                continue
            if not r.get("evidence_text"):
                continue
            if r["subject"] == "?" and r["object"] == "?":
                continue  # unevaluable
            available.append(r)

    print(f"Available after exclusions: {len(available)}")

    # Deduplicate by source_hash
    seen = set()
    dedup = []
    for r in available:
        if r["source_hash"] in seen:
            continue
        seen.add(r["source_hash"])
        dedup.append(r)

    correct = [r for r in dedup if r["tag"] == "correct"]
    incorrect = [r for r in dedup if r["tag"] != "correct"]
    print(f"After dedup: {len(correct)} correct, {len(incorrect)} incorrect")

    random.shuffle(correct)
    random.shuffle(incorrect)
    sample = correct[:n_correct] + incorrect[:n_incorrect]
    random.shuffle(sample)

    print(f"\nHoldout: {len(sample)} records")
    tags = Counter(r["tag"] for r in sample)
    for t, c in tags.most_common():
        print(f"  {t}: {c}")

    # Write
    out = DATA / "holdout.jsonl"
    with open(out, "w") as f:
        for r in sample:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
