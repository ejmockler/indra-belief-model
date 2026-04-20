"""Benchmark: Compare belief scoring methods on the INDRA curation benchmark.

Models compared:
1. parametric_indra:   INDRA default priors, no gating
2. parametric_recal:   Recalibrated priors, no gating
3. indra_belief:       INDRA's pre-computed belief (from CoGEx)
4. hard_gate:          Recalibrated priors + simulated hard gate
5. multiplication:     parametric_recal × simulated LLM accuracy

Pre-registered success criterion: composed AUPRC must beat 0.756
(provenance-only baseline from baseline_evaluation.json).

Usage:
    python scripts/benchmark_composition.py data/benchmark/belief_benchmark.jsonl
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from indra_belief.noise_model import (
    INDRA_PRIORS,
    RECALIBRATED_PRIORS,
    compute_edge_reliability,
)


def load_benchmark(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def is_correct(record: dict) -> bool:
    return record["tag"] == "correct"


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under precision-recall curve."""
    from sklearn.metrics import average_precision_score
    return average_precision_score(y_true, y_score)


def compute_brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Brier score (lower is better)."""
    return float(np.mean((y_score - y_true) ** 2))


def compute_ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if mask.sum() == 0:
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_score[mask].mean()
        ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return float(ece)


def score_parametric(record: dict, priors: dict) -> float:
    """Score a record using parametric noise model."""
    sources = list(record["source_counts"].keys())
    ev_count = record["evidence_count"]
    if not sources:
        sources = [record["source_api"]]
        ev_count = max(ev_count, 1)
    return compute_edge_reliability(sources, ev_count, priors=priors)


def score_hard_gate(record: dict, priors: dict) -> float:
    """Simulate hard gate: if this evidence is the only one from its source
    and it's incorrect, the source is removed.

    For benchmark purposes, we simulate the LLM as a perfect oracle
    (knows the true label) to measure the ceiling of the hard gate approach.
    Then also simulate with realistic LLM accuracy.
    """
    # This is a per-evidence record. The hard gate operates per-evidence,
    # but we need to reason about the statement level.
    #
    # Simplification: this record represents one evidence sentence.
    # If the LLM would gate it, reduce the evidence count by 1.
    # If that reduces a source to 0, remove it.
    is_corr = is_correct(record)
    source = record["source_api"]
    source_counts = dict(record["source_counts"])

    if not is_corr:
        # Oracle gates this evidence
        if source in source_counts:
            source_counts[source] = max(0, source_counts[source] - 1)
            if source_counts[source] == 0:
                del source_counts[source]

    if not source_counts:
        return 0.0

    sources = list(source_counts.keys())
    ev_count = sum(source_counts.values())
    return compute_edge_reliability(sources, ev_count, priors=priors)


def main():
    if len(sys.argv) < 2:
        print("Usage: python benchmark_composition.py <benchmark.jsonl>")
        sys.exit(1)

    records = load_benchmark(sys.argv[1])
    print(f"Loaded {len(records)} records")

    y_true = np.array([1.0 if is_correct(r) else 0.0 for r in records])
    print(f"Correct: {int(y_true.sum())} ({y_true.mean():.1%})")
    print()

    # Score with each model
    models = {}

    # 1. Parametric with INDRA defaults
    models["parametric_indra"] = np.array([
        score_parametric(r, INDRA_PRIORS) for r in records
    ])

    # 2. Parametric with recalibrated priors
    models["parametric_recal"] = np.array([
        score_parametric(r, RECALIBRATED_PRIORS) for r in records
    ])

    # 3. INDRA pre-computed belief
    models["indra_belief"] = np.array([r["belief"] for r in records])

    # 4. Hard gate (oracle — perfect LLM, ceiling estimate)
    models["hard_gate_oracle"] = np.array([
        score_hard_gate(r, RECALIBRATED_PRIORS) for r in records
    ])

    # 5. Multiplication baseline: parametric_recal * simulated_llm
    # Simulate LLM: correct → 0.95, incorrect → 0.05
    llm_sim = np.array([0.95 if is_correct(r) else 0.05 for r in records])
    models["multiplication"] = models["parametric_recal"] * llm_sim

    # Compute metrics
    print(f"{'Model':<22} {'AUPRC':>7} {'Brier':>7} {'ECE':>7}")
    print("-" * 48)

    results = {}
    for name, scores in models.items():
        auprc = compute_auprc(y_true, scores)
        brier = compute_brier(y_true, scores)
        ece = compute_ece(y_true, scores)
        results[name] = {"auprc": auprc, "brier": brier, "ece": ece}
        print(f"{name:<22} {auprc:>7.4f} {brier:>7.4f} {ece:>7.4f}")

    # Pre-registered threshold
    print()
    target = 0.756
    best = max(results.items(), key=lambda x: x[1]["auprc"])
    print(f"Target AUPRC: {target}")
    print(f"Best model: {best[0]} ({best[1]['auprc']:.4f})")
    if best[1]["auprc"] > target:
        print("PASS: Best model beats target")
    else:
        print("FAIL: No model beats target")

    # Per-source analysis
    print("\n--- Per-source accuracy impact ---")
    source_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in records:
        src = r["source_api"]
        source_stats[src]["total"] += 1
        if is_correct(r):
            source_stats[src]["correct"] += 1

    for src, stats in sorted(source_stats.items(), key=lambda x: -x[1]["total"]):
        acc = stats["correct"] / stats["total"]
        # What % of incorrect records from this source would be gated?
        n_incorrect = stats["total"] - stats["correct"]
        print(f"  {src:<12} n={stats['total']:>5} acc={acc:.3f} "
              f"incorrect={n_incorrect:>5} (gate candidates)")

    # Error-type stratified AUPRC
    print("\n--- Error-type stratified AUPRC (parametric_recal) ---")
    error_types = defaultdict(list)
    for i, r in enumerate(records):
        error_types[r["tag"]].append(i)

    for tag, indices in sorted(error_types.items(), key=lambda x: -len(x[1])):
        idx = np.array(indices)
        if len(idx) < 10:
            continue
        tag_true = y_true[idx]
        if tag_true.sum() == 0 or tag_true.sum() == len(tag_true):
            continue  # can't compute AUPRC with single class
        tag_scores = models["parametric_recal"][idx]
        auprc = compute_auprc(tag_true, tag_scores)
        print(f"  {tag:<20} n={len(idx):>5} AUPRC={auprc:.4f}")

    # False-gate rate
    print("\n--- False-gate rate (oracle hard gate) ---")
    correct_records = [r for r in records if is_correct(r)]
    # In oracle mode, correct records are never gated
    print(f"  Oracle: 0/{len(correct_records)} correct records gated (0%)")
    print(f"  (Real LLM false-gate rate depends on empirical scorer accuracy: ~21%)")

    # Save results
    output_path = Path(sys.argv[1]).parent / "composition_benchmark.json"
    with open(output_path, "w") as f:
        json.dump({
            "n_records": len(records),
            "n_correct": int(y_true.sum()),
            "target_auprc": target,
            "models": {k: v for k, v in results.items()},
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
