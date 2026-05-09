"""V8c — stratified resample of V8a labels to match U2 holdout class distribution.

V8b preserved the V8a corpus distribution (41% direct_sign_match, 39%
no_relation, 12% direct_axis_mismatch, 8% direct_sign_mismatch). The U2
relation_axis holdout is heavily skewed toward direct_sign_match (70%),
with only 14% no_relation, 13% direct_axis_mismatch, 4%
direct_sign_mismatch. V9' LoRA tuned on the corpus distribution
under-predicted direct_sign_match and over-predicted no_relation,
losing to both Pro curator (78.8% micro) and the most-frequent-class
trivial baseline (69.6%) on the holdout (V9' scored 54.3% micro / 51.5%
macro).

V8c re-derives train/val from V8a kept records using stratified
sampling without replacement so that the **train class distribution
matches the U2 holdout** (within rounding). Val is held out per-class
first so it is not contaminated by re-sampling.

Output:
  data/v_phase/train/v8c_relation_axis.jsonl
  data/v_phase/val/v8c_relation_axis.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


_SHORT_SYSTEM_PROMPT = (
    "You classify whether evidence describes a relation between two entities "
    "matching a claim's axis and sign. Output JSON: "
    '{"answer": <one of: direct_sign_match, direct_sign_mismatch, '
    "direct_axis_mismatch, no_relation>, "
    '"rationale": <short quote/phrase from evidence>}.'
)


# U2 holdout class proportions (relation_axis tasks, n=392).
# Raw values used to sum to 1.01 (0.70+0.14+0.13+0.04 = 1.01); they are
# normalized at module load so quotas always sum to exactly target_n.
_HOLDOUT_PROPORTIONS_RAW = {
    "direct_sign_match": 0.70,
    "no_relation": 0.14,
    "direct_axis_mismatch": 0.13,
    "direct_sign_mismatch": 0.04,
}
_HOLDOUT_RAW_SUM = sum(_HOLDOUT_PROPORTIONS_RAW.values())
HOLDOUT_PROPORTIONS = {
    cls: v / _HOLDOUT_RAW_SUM for cls, v in _HOLDOUT_PROPORTIONS_RAW.items()
}
assert abs(sum(HOLDOUT_PROPORTIONS.values()) - 1.0) < 1e-9, (
    "HOLDOUT_PROPORTIONS must normalize to 1.0"
)

# V8a corpus proportions (kept records, n=2956). Approximate.
_CORPUS_PROPORTIONS_RAW = {
    "direct_sign_match": 0.41,
    "no_relation": 0.39,
    "direct_axis_mismatch": 0.12,
    "direct_sign_mismatch": 0.08,
}
_CORPUS_RAW_SUM = sum(_CORPUS_PROPORTIONS_RAW.values())
CORPUS_PROPORTIONS = {
    cls: v / _CORPUS_RAW_SUM for cls, v in _CORPUS_PROPORTIONS_RAW.items()
}
assert abs(sum(CORPUS_PROPORTIONS.values()) - 1.0) < 1e-9, (
    "CORPUS_PROPORTIONS must normalize to 1.0"
)


# U2 holdout sources used for the train/val ∩ holdout assertion.
U2_HOLDOUT_SOURCES = (
    "data/benchmark/probe_gold_holdout.jsonl",
    "data/benchmark/holdout_v15_sample.jsonl",
)


def make_chat(rec: dict) -> dict:
    """Same chat format V8b uses (short system, no in-context few-shots)."""
    system = _SHORT_SYSTEM_PROMPT
    user = (
        f"CLAIM: {rec['stmt_type']}({rec['subject']}, {rec['object']})\n"
        f"EVIDENCE: {rec['evidence_text']}"
    )
    assistant_answer = json.dumps(
        {"answer": rec["answer"], "rationale": rec["rationale"]},
        separators=(",", ": "),
    )

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant_answer},
    ]
    return {
        "messages": messages,
        "completion": assistant_answer,
        "metadata": {
            "source_hash": rec["source_hash"],
            "stmt_type": rec["stmt_type"],
            "axis": rec["axis"],
            "answer": rec["answer"],
        },
    }


def compute_quotas(target_n: int, proportions: dict[str, float]) -> dict[str, int]:
    """Largest-remainder allocation so quotas sum exactly to target_n.

    Proportions are assumed to sum to 1.0 (normalized at module load); a
    final assertion enforces the post-allocation invariant in case caller
    passes a custom dict.
    """
    raw = {cls: target_n * p for cls, p in proportions.items()}
    floor = {cls: int(v) for cls, v in raw.items()}
    remainder = target_n - sum(floor.values())
    # distribute leftover slots to classes with the largest fractional parts
    fracs = sorted(
        ((cls, raw[cls] - floor[cls]) for cls in proportions),
        key=lambda kv: kv[1],
        reverse=True,
    )
    for i in range(remainder):
        floor[fracs[i % len(fracs)][0]] += 1
    assert sum(floor.values()) == target_n, (
        f"compute_quotas invariant: sum {sum(floor.values())} != {target_n}"
    )
    return floor


def feasible_target_n(
    pool_sizes: dict[str, int], proportions: dict[str, float]
) -> int:
    """Largest target_n such that ceil(target_n*p[c]) <= pool[c] for every class.

    Equivalent to floor(min(pool[c] / p[c])); since the largest-remainder
    quota for a class is ceil(target_n * p), using floor of the per-class
    bound ensures every class quota is satisfiable from its pool.
    """
    finite = [pool_sizes[c] / p for c, p in proportions.items() if p > 0]
    if not finite:
        return 0
    return int(math.floor(min(finite)))


def load_u2_holdout_hashes(root: Path, sources: tuple[str, ...]) -> set[int]:
    """Read source_hash field from each U2 holdout source and union.

    Round-2 fix: HARD-FAIL on missing files or empty result. The previous
    WARN-and-continue path silently degraded the contamination assertion
    to a vacuous one when a path was renamed.
    """
    hashes: set[int] = set()
    for rel in sources:
        path = root / rel
        if not path.exists():
            print(
                f"FATAL: U2 holdout source missing: {path}. "
                "Cannot validate train/val ∩ holdout = ∅ without it.",
                file=sys.stderr,
            )
            sys.exit(2)
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if "source_hash" in r:
                    hashes.add(r["source_hash"])
    if not hashes:
        print(
            "FATAL: U2 holdout sources loaded but yielded zero source_hash "
            "values. Refusing to ship a vacuous contamination check.",
            file=sys.stderr,
        )
        sys.exit(2)
    return hashes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in", dest="in_path",
        default="data/v_phase/v8a_relation_axis_labels.jsonl",
    )
    ap.add_argument(
        "--out-train",
        default="data/v_phase/train/v8c_relation_axis.jsonl",
    )
    ap.add_argument(
        "--out-val",
        default="data/v_phase/val/v8c_relation_axis.jsonl",
    )
    ap.add_argument(
        "--match",
        choices=["holdout", "corpus"],
        default="holdout",
        help="Target class distribution: U2 holdout proportions or V8a corpus.",
    )
    ap.add_argument(
        "--target-n-train",
        type=int,
        default=2000,
        help="Total train records desired; per-class quotas derive proportionally.",
    )
    ap.add_argument(
        "--val-per-class",
        type=int,
        default=50,
        help="Class-balanced val held out before stratified train sample.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    in_path = ROOT / args.in_path
    out_train = ROOT / args.out_train
    out_val = ROOT / args.out_val
    out_train.parent.mkdir(parents=True, exist_ok=True)
    out_val.parent.mkdir(parents=True, exist_ok=True)

    proportions = HOLDOUT_PROPORTIONS if args.match == "holdout" else CORPUS_PROPORTIONS

    # Load + filter
    kept: list[dict] = []
    n_total = 0
    n_dropped = 0
    drop_reasons: Counter[str] = Counter()
    with in_path.open() as f:
        for line in f:
            r = json.loads(line)
            n_total += 1
            if not r.get("kept"):
                n_dropped += 1
                drop_reasons[r.get("filter_reason") or "no_kept"] += 1
                continue
            if not r.get("answer"):
                n_dropped += 1
                drop_reasons["no_answer"] += 1
                continue
            kept.append(r)

    print(f"Loaded {n_total} records; {len(kept)} kept; {n_dropped} dropped")
    print(f"  drop reasons: {dict(drop_reasons)}")

    # ---- Bug #2 fix: GroupShuffleSplit by source_hash ----
    # Duplicate source_hashes carry meaningful variation (≥201 of 281
    # duplicate groups have different (subject, object, stmt_type) claims
    # against the same evidence). To avoid train/val leakage we keep all
    # records of a given source_hash together — they go entirely to train
    # or entirely to val. Within each class we partition source_hashes,
    # not records.
    #
    # Multi-class hashes (44 of 2607, ≈95 records) span multiple answer
    # classes within a single group. They are dropped entirely from V8c:
    # keeping them would either leak across train/val (if split by class)
    # or distort the per-class distribution (if assigned by primary class,
    # since secondary records still count toward their actual `answer`).
    # The 3.2% record loss is the cost of clean group-integrity + clean
    # per-class quotas.
    rng = random.Random(args.seed)

    hash_records: dict[int, list[dict]] = {}
    for r in kept:
        hash_records.setdefault(r["source_hash"], []).append(r)

    n_multi = 0
    n_multi_records = 0
    single_class_records: list[dict] = []
    for h, recs in hash_records.items():
        answers = {r["answer"] for r in recs}
        if len(answers) > 1:
            n_multi += 1
            n_multi_records += len(recs)
        else:
            single_class_records.extend(recs)

    print(
        f"  group-integrity: dropped {n_multi} multi-class hashes "
        f"({n_multi_records} records); {len(single_class_records)} "
        f"single-class records remain"
    )

    # Re-bucket by class using only single-class hashes; class is now the
    # unique answer of every record in the group.
    by_class: dict[str, list[dict]] = {}
    for r in single_class_records:
        by_class.setdefault(r["answer"], []).append(r)

    # Group records back by hash within each class to enable group-aware
    # shuffling (treat each hash as an atomic unit).
    hashes_by_class: dict[str, list[int]] = {}
    class_hash_records: dict[str, dict[int, list[dict]]] = {}
    for cls, recs in by_class.items():
        hash_to_records: dict[int, list[dict]] = {}
        for r in recs:
            hash_to_records.setdefault(r["source_hash"], []).append(r)
        hashes_by_class[cls] = list(hash_to_records.keys())
        class_hash_records[cls] = hash_to_records

    available_records = {cls: len(recs) for cls, recs in by_class.items()}
    print(f"  unique source_hashes (single-class): "
          f"{sum(len(v) for v in hashes_by_class.values())}")
    print(f"  records per class: {available_records}")

    # ---- Step 1: hold out class-balanced val by source_hash group ----
    # We take val_per_class *records* per class (counted record-wise), but
    # always taking whole hash-groups to preserve group integrity. Within
    # each class all records of a chosen hash go to val; remaining hashes
    # form the train pool.
    val: list[dict] = []
    val_hashes_by_class: dict[str, set[int]] = {}
    pool_hashes: dict[str, list[int]] = {}
    for cls, hashes in hashes_by_class.items():
        shuffled = list(hashes)
        rng.shuffle(shuffled)
        val_hash_list: list[int] = []
        val_record_count = 0
        for h in shuffled:
            recs = class_hash_records[cls][h]
            if val_record_count >= args.val_per_class:
                break
            if val_record_count + len(recs) > args.val_per_class:
                # adding this hash would overshoot val target; skip THIS group
                # but keep searching for smaller groups that still fit. Round-2
                # fix: was `break`, which silently aborted on the first oversize
                # group and left singleton hashes untouched.
                continue
            val.extend(recs)
            val_record_count += len(recs)
            val_hash_list.append(h)
        val_hashes_by_class[cls] = set(val_hash_list)
        pool_hashes[cls] = [h for h in shuffled if h not in set(val_hash_list)]

    pool_record_counts = {
        cls: sum(len(class_hash_records[cls][h]) for h in hashes)
        for cls, hashes in pool_hashes.items()
    }
    print(f"  pool record counts (after val held out): {pool_record_counts}")

    # ---- Step 3 prep: scale target_n_train down if any class would be short ----
    # Bug #3 fix: instead of silently capping, scale target_n down to the
    # largest size that hits the target distribution exactly.
    requested_n = args.target_n_train
    feasible_n = feasible_target_n(pool_record_counts, proportions)
    if feasible_n < requested_n:
        print(
            f"  WARN: requested target_n_train={requested_n} infeasible — "
            f"scaling down to feasible_n={feasible_n} "
            "(largest size where every class quota fits its pool)"
        )
        target_n = feasible_n
    else:
        target_n = requested_n

    # ---- Step 2 (post-scale): derive per-class train quotas ----
    quotas = compute_quotas(target_n, proportions)
    print(f"  target distribution: {args.match} → quotas: {quotas} (sum={sum(quotas.values())})")

    # Bug #3 hard-fail: any zero quota means the proportion rounds away.
    zero_classes = [cls for cls, q in quotas.items() if q == 0]
    if zero_classes:
        print(
            f"FATAL: per-class quota is 0 for {zero_classes}. "
            "Increase --target-n-train or adjust proportions.",
            file=sys.stderr,
        )
        return 2

    # ---- Step 3: stratified sample by source_hash without replacement ----
    # We greedily take whole hash-groups until the per-class record quota
    # is met (or we exhaust the pool — which after feasibility-scaling
    # means we got within group-rounding of the quota).
    train: list[dict] = []
    train_taken_records: dict[str, int] = {}
    train_hashes_by_class: dict[str, set[int]] = {}
    shortfalls: dict[str, int] = {}
    for cls, quota in quotas.items():
        avail = pool_hashes.get(cls, [])
        taken_records = 0
        taken_hashes: list[int] = []
        for h in avail:
            recs = class_hash_records[cls][h]
            if taken_records + len(recs) > quota:
                # adding this hash would overshoot the quota; skip THIS group
                # but keep searching for smaller groups that still fit. Round-2
                # fix: was `break`.
                continue
            train.extend(recs)
            taken_records += len(recs)
            taken_hashes.append(h)
        train_taken_records[cls] = taken_records
        train_hashes_by_class[cls] = set(taken_hashes)
        if taken_records < quota:
            shortfalls[cls] = quota - taken_records

    rng.shuffle(train)
    rng.shuffle(val)

    print(f"Split: {len(train)} train / {len(val)} val")
    train_dist = Counter(r["answer"] for r in train)
    val_dist = Counter(r["answer"] for r in val)
    print(f"  train class dist: {dict(train_dist)}")
    if len(train):
        print(
            "  train class ratios: "
            + ", ".join(
                f"{cls}={train_dist[cls] / len(train) * 100:.1f}%"
                for cls in proportions
            )
        )
    print(f"  val   class dist: {dict(val_dist)}")
    if shortfalls:
        # Group-integrity rounding can leave a 1-2 record gap per class even
        # after feasibility scaling; this is expected and small.
        print(f"  per-class shortfalls (group-rounding only): {shortfalls}")
    else:
        print("  no per-class shortfalls")

    # ---- Bug #4 fix: assertions ----
    train_hashes = {r["source_hash"] for r in train}
    val_hashes = {r["source_hash"] for r in val}

    overlap_tv = train_hashes & val_hashes
    if overlap_tv:
        sample = sorted(overlap_tv)[:10]
        raise AssertionError(
            f"train/val source_hash overlap detected: {len(overlap_tv)} hashes "
            f"(first 10: {sample})"
        )

    u2_hashes = load_u2_holdout_hashes(ROOT, U2_HOLDOUT_SOURCES)
    print(f"  U2 holdout hash count: {len(u2_hashes)}")
    overlap_u2 = (train_hashes | val_hashes) & u2_hashes
    if overlap_u2:
        sample = sorted(overlap_u2)[:10]
        raise AssertionError(
            f"V8c train/val ∩ U2 holdout overlap: {len(overlap_u2)} hashes "
            f"(first 10: {sample})"
        )

    print(
        "  assertions: train∩val empty: True, "
        f"(train∪val)∩U2 empty: True"
    )

    n_written = 0
    with out_train.open("w") as f:
        for r in train:
            f.write(json.dumps(make_chat(r)) + "\n")
            n_written += 1
    print(f"Wrote {out_train} ({n_written} records)")

    n_written = 0
    with out_val.open("w") as f:
        for r in val:
            f.write(json.dumps(make_chat(r)) + "\n")
            n_written += 1
    print(f"Wrote {out_val} ({n_written} records)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
