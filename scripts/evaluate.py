"""Evaluate belief scorers against the INDRA benchmark corpus.

Metrics:
  - AUPRC (area under precision-recall curve) — primary metric, matches paper
  - Calibration: reliability diagram (predicted probability vs observed frequency)
  - Per-error-type detection: which error categories does each scorer catch?
  - Per-source accuracy: breakdown by extraction system

Usage:
    python -m experiments.belief_benchmark.evaluate
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from experiments.belief_benchmark.scorers import (
    BeliefScorer,
    INDRABeliefScorer,
    INDRASimpleScorer,
    ProvenanceScorer,
    ScoredRecord,
    load_benchmark,
    score_dataset,
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmark" / "results"


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under precision-recall curve (sklearn-style).

    Uses right-endpoint Riemann sum: sum((recall[k] - recall[k-1]) * precision[k]).
    This is the formal definition of average precision, which matches
    sklearn.metrics.average_precision_score.

    Note: trapezoidal integration under-reports AUPRC when there are ties,
    because it treats the curve as starting at (recall=1/n_pos, precision=1)
    rather than (recall=0, precision=1).
    """
    order = np.argsort(-y_score)
    y_true = y_true[order]

    tp = np.cumsum(y_true)
    fp = np.cumsum(1 - y_true)
    precision = tp / (tp + fp)
    recall = tp / tp[-1] if tp[-1] > 0 else tp

    # Right-endpoint: sum of (delta_recall) * precision
    recall_prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - recall_prev) * precision))


def calibration_bins(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> list[dict]:
    """Compute calibration bins: predicted probability vs observed frequency."""
    bins = []
    edges = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_score >= lo) & (y_score < hi)
        if not mask.any():
            continue
        bins.append({
            "bin_center": float((lo + hi) / 2),
            "mean_predicted": float(y_score[mask].mean()),
            "observed_frequency": float(y_true[mask].mean()),
            "count": int(mask.sum()),
        })
    return bins


def per_error_type_analysis(
    scored: list[ScoredRecord],
    scorer_name: str,
    threshold: float = 0.5,
) -> dict[str, dict]:
    """For each error type, compute how often the scorer flags it (score < threshold)."""
    by_tag: dict[str, list[float]] = defaultdict(list)
    for r in scored:
        by_tag[r.tag].append(r.scores[scorer_name])

    results = {}
    for tag, scores in sorted(by_tag.items()):
        scores_arr = np.array(scores)
        results[tag] = {
            "count": len(scores),
            "mean_score": float(scores_arr.mean()),
            "median_score": float(np.median(scores_arr)),
            "flagged_rate": float((scores_arr < threshold).mean()),
        }
    return results


def per_source_analysis(
    scored: list[ScoredRecord],
    scorer_name: str,
) -> dict[str, dict]:
    """Breakdown by extraction source."""
    by_source: dict[str, dict] = defaultdict(lambda: {"correct": [], "incorrect": []})
    for r in scored:
        key = "correct" if r.is_correct else "incorrect"
        src = r.metadata.get("source_api", "unknown")
        by_source[src][key].append(r.scores[scorer_name])

    results = {}
    for src, groups in sorted(by_source.items()):
        n_correct = len(groups["correct"])
        n_incorrect = len(groups["incorrect"])
        total = n_correct + n_incorrect
        if total == 0:
            continue

        all_scores = groups["correct"] + groups["incorrect"]
        all_labels = [1] * n_correct + [0] * n_incorrect
        auprc = compute_auprc(np.array(all_labels), np.array(all_scores))

        results[src] = {
            "n_correct": n_correct,
            "n_incorrect": n_incorrect,
            "accuracy": n_correct / total,
            "auprc": auprc,
            "mean_score_correct": float(np.mean(groups["correct"])) if groups["correct"] else None,
            "mean_score_incorrect": float(np.mean(groups["incorrect"])) if groups["incorrect"] else None,
        }
    return results


def run_evaluation(scored: list[ScoredRecord], scorer_names: list[str]) -> dict:
    """Run full evaluation suite."""
    y_true = np.array([r.is_correct for r in scored], dtype=float)

    results = {"n_records": len(scored), "scorers": {}}

    for name in scorer_names:
        y_score = np.array([r.scores[name] for r in scored])

        results["scorers"][name] = {
            "auprc": compute_auprc(y_true, y_score),
            "mean_score": float(y_score.mean()),
            "calibration": calibration_bins(y_true, y_score),
            "per_error_type": per_error_type_analysis(scored, name),
            "per_source": per_source_analysis(scored, name),
        }

    return results


def print_results(results: dict):
    """Pretty-print evaluation results."""
    print(f"\n{'='*70}")
    print(f"BELIEF BENCHMARK EVALUATION — {results['n_records']} records")
    print(f"{'='*70}")

    # AUPRC comparison
    print(f"\n--- AUPRC (primary metric) ---")
    for name, data in results["scorers"].items():
        print(f"  {name:20s}  AUPRC = {data['auprc']:.4f}")

    # Calibration
    for name, data in results["scorers"].items():
        print(f"\n--- Calibration: {name} ---")
        print(f"  {'Bin':>8s}  {'Predicted':>10s}  {'Observed':>10s}  {'Count':>6s}")
        for b in data["calibration"]:
            print(f"  {b['bin_center']:8.2f}  {b['mean_predicted']:10.3f}  "
                  f"{b['observed_frequency']:10.3f}  {b['count']:6d}")

    # Per-error-type
    for name, data in results["scorers"].items():
        print(f"\n--- Error detection: {name} ---")
        print(f"  {'Tag':>20s}  {'Count':>6s}  {'Mean score':>10s}  {'Median':>8s}")
        for tag, info in sorted(data["per_error_type"].items(),
                                 key=lambda x: x[1]["mean_score"]):
            print(f"  {tag:>20s}  {info['count']:6d}  {info['mean_score']:10.3f}  "
                  f"{info['median_score']:8.3f}")

    # Per-source
    for name, data in results["scorers"].items():
        print(f"\n--- Per-source: {name} ---")
        print(f"  {'Source':>25s}  {'Acc':>6s}  {'AUPRC':>7s}  "
              f"{'Correct':>8s}  {'Incorrect':>10s}")
        for src, info in sorted(data["per_source"].items(),
                                 key=lambda x: -x[1]["auprc"]):
            mc = f"{info['mean_score_correct']:.3f}" if info['mean_score_correct'] else "   -  "
            mi = f"{info['mean_score_incorrect']:.3f}" if info['mean_score_incorrect'] else "   -  "
            print(f"  {src:>25s}  {info['accuracy']:5.1%}  {info['auprc']:7.4f}  "
                  f"{mc:>8s}  {mi:>10s}")


def main():
    print("Loading benchmark dataset...")
    records = load_benchmark()
    print(f"  {len(records)} records loaded")

    print("Initializing scorers...")
    scorers: list[BeliefScorer] = [
        ProvenanceScorer(),
        INDRABeliefScorer(),
        INDRASimpleScorer(),
    ]
    scorer_names = [s.name for s in scorers]
    print(f"  Scorers: {scorer_names}")

    print("Scoring...")
    scored = score_dataset(records, scorers)

    print("Evaluating...")
    results = run_evaluation(scored, scorer_names)

    print_results(results)

    # Save raw results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "baseline_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {output_path}")


if __name__ == "__main__":
    main()
