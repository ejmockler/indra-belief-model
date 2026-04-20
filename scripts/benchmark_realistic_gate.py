"""Benchmark: realistic LLM false-gate rate and hard-gate composition.

Simulates the empirically-measured LLM scorer accuracy profile (78.7%
overall, with per-statement-type variation) to measure:
  1. False-gate rate: correct evidence wrongly rejected
  2. False-pass rate: incorrect evidence wrongly accepted
  3. AUPRC/Brier/ECE of composed hard-gate scoring vs baselines

Compares 4 models:
  - parametric_recal:    Recalibrated priors, no gating
  - hard_gate_oracle:    Perfect LLM gating (ceiling)
  - hard_gate_realistic: Simulated LLM gating at empirical accuracy
  - indra_belief:        INDRA pre-computed belief

Usage:
    python scripts/benchmark_realistic_gate.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score

from indra_belief.noise_model import (
    RECALIBRATED_PRIORS,
    compute_edge_reliability_from_counts,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BENCHMARK_PATH = (
    "/Users/noot/Documents/biomolecular-clique-finding"
    "/data/benchmark/belief_benchmark.jsonl"
)

SEED = 20260408

# Per-statement-type LLM accuracy measured on a 3,754-record half-corpus
# holdout. Used as simulation input — NOT a live scoring table.
#
# Provenance: overall 78.7% on holdout_large.jsonl with the gemma-4-26b
# scorer configuration current as of commit 41aa44b (gilda confidence
# threshold, pseudogene detection, full provenance). See the `Design
# decisions we already paid for` table in README for measurement context.
# Update BOTH the numbers and this provenance note when re-measuring.
EMPIRICAL_TYPE_ACCURACY: dict[str, float] = {
    "Dephosphorylation": 0.583,
    "Phosphorylation": 0.80,
    "Autophosphorylation": 0.80,  # group with phosphorylation
    "IncreaseAmount": 0.82,
    "DecreaseAmount": 0.82,
    "Activation": 0.78,
    "Inhibition": 0.78,
    "Complex": 0.75,
}
EMPIRICAL_OVERALL_ACCURACY = 0.787


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_benchmark(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def is_correct(record: dict) -> bool:
    return record["tag"] == "correct"


def compute_auprc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))


def compute_brier(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(np.mean((y_score - y_true) ** 2))


def compute_ece(y_true: np.ndarray, y_score: np.ndarray, n_bins: int = 10) -> float:
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


def get_type_accuracy(stmt_type: str) -> float:
    """Return empirical accuracy for a statement type, falling back to overall."""
    return EMPIRICAL_TYPE_ACCURACY.get(stmt_type, EMPIRICAL_OVERALL_ACCURACY)


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------

def simulate_empirical_verdicts(
    records: list[dict], rng: np.random.Generator
) -> np.ndarray:
    """Simulate per-record LLM verdicts at the empirical accuracy profile.

    Returns array of booleans: True = LLM says "correct", False = "incorrect".

    For each record:
      - With prob = type_accuracy, verdict matches truth
      - With prob = 1 - type_accuracy, verdict is wrong
    """
    verdicts = np.empty(len(records), dtype=bool)
    for i, r in enumerate(records):
        acc = get_type_accuracy(r["stmt_type"])
        truth = is_correct(r)
        coin = rng.random()
        if coin < acc:
            verdicts[i] = truth        # correct verdict
        else:
            verdicts[i] = not truth     # wrong verdict
    return verdicts


def score_hard_gate(
    record: dict, llm_says_correct: bool, priors: dict
) -> float:
    """Compute belief with hard-gate applied to this evidence.

    If LLM says "incorrect": reduce source_counts for this source by 1.
    If LLM says "correct": keep original counts.
    """
    source_counts = dict(record["source_counts"])

    if not llm_says_correct:
        source = record["source_api"]
        if source in source_counts:
            source_counts[source] = max(0, source_counts[source] - 1)
            if source_counts[source] == 0:
                del source_counts[source]

    if not source_counts:
        return 0.0

    return compute_edge_reliability_from_counts(source_counts, priors)


def score_hard_gate_oracle(record: dict, priors: dict) -> float:
    """Oracle hard gate: gate iff the record is truly incorrect."""
    return score_hard_gate(record, is_correct(record), priors)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metrics_table(models: dict[str, np.ndarray], y_true: np.ndarray) -> dict:
    """Print and return metrics for all models."""
    print(f"{'Model':<24} {'AUPRC':>7} {'Brier':>7} {'ECE':>7}")
    print("-" * 50)

    results = {}
    for name, scores in models.items():
        auprc = compute_auprc(y_true, scores)
        brier = compute_brier(y_true, scores)
        ece = compute_ece(y_true, scores)
        results[name] = {"auprc": auprc, "brier": brier, "ece": ece}
        print(f"{name:<24} {auprc:>7.4f} {brier:>7.4f} {ece:>7.4f}")

    return results


def print_false_gate_analysis(
    records: list[dict],
    verdicts: np.ndarray,
    y_true: np.ndarray,
):
    """Print false-gate and false-pass rates, stratified."""
    correct_mask = y_true.astype(bool)
    incorrect_mask = ~correct_mask

    # Overall rates
    n_correct = correct_mask.sum()
    n_incorrect = incorrect_mask.sum()

    # False gate: correct record where LLM says "incorrect"
    false_gates = correct_mask & ~verdicts
    false_gate_rate = false_gates.sum() / n_correct

    # False pass: incorrect record where LLM says "correct"
    false_passes = incorrect_mask & verdicts
    false_pass_rate = false_passes.sum() / n_incorrect

    print("=== False-gate / False-pass analysis ===")
    print()
    print(f"Overall false-gate rate:  {false_gates.sum():>5}/{n_correct} "
          f"= {false_gate_rate:.1%} of correct records wrongly rejected")
    print(f"Overall false-pass rate:  {false_passes.sum():>5}/{n_incorrect} "
          f"= {false_pass_rate:.1%} of incorrect records wrongly accepted")
    print()

    # Stratify by source_api
    print("--- Stratified by source_api ---")
    print(f"  {'source':<12} {'n':>5} {'false_gate':>12} {'false_pass':>12}")
    print(f"  {'':<12} {'':>5} {'(of correct)':>12} {'(of incorrect)':>12}")
    print("  " + "-" * 47)

    source_stats = defaultdict(lambda: {
        "n": 0, "n_correct": 0, "n_incorrect": 0,
        "false_gate": 0, "false_pass": 0,
    })
    for i, r in enumerate(records):
        src = r["source_api"]
        source_stats[src]["n"] += 1
        if correct_mask[i]:
            source_stats[src]["n_correct"] += 1
            if not verdicts[i]:
                source_stats[src]["false_gate"] += 1
        else:
            source_stats[src]["n_incorrect"] += 1
            if verdicts[i]:
                source_stats[src]["false_pass"] += 1

    for src, s in sorted(source_stats.items(), key=lambda x: -x[1]["n"]):
        fg = f"{s['false_gate']}/{s['n_correct']}" if s["n_correct"] > 0 else "n/a"
        fp = f"{s['false_pass']}/{s['n_incorrect']}" if s["n_incorrect"] > 0 else "n/a"
        fg_pct = f" ({s['false_gate']/s['n_correct']:.0%})" if s["n_correct"] > 0 else ""
        fp_pct = f" ({s['false_pass']/s['n_incorrect']:.0%})" if s["n_incorrect"] > 0 else ""
        print(f"  {src:<12} {s['n']:>5} {fg+fg_pct:>12} {fp+fp_pct:>12}")
    print()

    # Stratify by tag (error type) -- only for incorrect records
    print("--- Stratified by tag (error types, false-pass rate) ---")
    print(f"  {'tag':<22} {'n_incorr':>8} {'false_pass':>12} {'rate':>6}")
    print("  " + "-" * 52)

    tag_stats = defaultdict(lambda: {"n": 0, "false_pass": 0})
    for i, r in enumerate(records):
        if r["tag"] == "correct":
            continue
        tag_stats[r["tag"]]["n"] += 1
        if verdicts[i]:  # LLM wrongly says correct
            tag_stats[r["tag"]]["false_pass"] += 1

    for tag, s in sorted(tag_stats.items(), key=lambda x: -x[1]["n"]):
        rate = s["false_pass"] / s["n"] if s["n"] > 0 else 0
        print(f"  {tag:<22} {s['n']:>8} "
              f"{s['false_pass']:>5}/{s['n']:<5} {rate:>6.1%}")
    print()

    # Stratify by stmt_type for false-gate
    print("--- Stratified by stmt_type (false-gate rate on correct records) ---")
    print(f"  {'stmt_type':<22} {'n_corr':>6} {'false_gate':>12} {'rate':>6} {'acc':>7}")
    print("  " + "-" * 58)

    type_stats = defaultdict(lambda: {"n_correct": 0, "false_gate": 0})
    for i, r in enumerate(records):
        if not correct_mask[i]:
            continue
        type_stats[r["stmt_type"]]["n_correct"] += 1
        if not verdicts[i]:
            type_stats[r["stmt_type"]]["false_gate"] += 1

    for stype, s in sorted(type_stats.items(), key=lambda x: -x[1]["n_correct"]):
        rate = s["false_gate"] / s["n_correct"] if s["n_correct"] > 0 else 0
        acc = get_type_accuracy(stype)
        print(f"  {stype:<22} {s['n_correct']:>6} "
              f"{s['false_gate']:>5}/{s['n_correct']:<5} {rate:>6.1%} {acc:>7.1%}")

    return {
        "overall_false_gate_rate": float(false_gate_rate),
        "overall_false_pass_rate": float(false_pass_rate),
        "n_false_gates": int(false_gates.sum()),
        "n_false_passes": int(false_passes.sum()),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    records = load_benchmark(BENCHMARK_PATH)
    print(f"Loaded {len(records)} records from benchmark")

    y_true = np.array([1.0 if is_correct(r) else 0.0 for r in records])
    n_correct = int(y_true.sum())
    n_incorrect = len(records) - n_correct
    print(f"Correct: {n_correct} ({n_correct/len(records):.1%}), "
          f"Incorrect: {n_incorrect} ({n_incorrect/len(records):.1%})")
    print()

    # Simulate LLM verdicts at the empirical accuracy profile
    rng = np.random.default_rng(SEED)
    verdicts = simulate_empirical_verdicts(records, rng)
    print(f"Simulated LLM verdicts: {verdicts.sum()} 'correct', "
          f"{(~verdicts).sum()} 'incorrect'")
    print()

    # Score all 4 models
    models: dict[str, np.ndarray] = {}

    # 1. Parametric recalibrated (no gating)
    models["parametric_recal"] = np.array([
        compute_edge_reliability_from_counts(r["source_counts"], RECALIBRATED_PRIORS)
        for r in records
    ])

    # 2. Hard gate oracle (perfect LLM)
    models["hard_gate_oracle"] = np.array([
        score_hard_gate_oracle(r, RECALIBRATED_PRIORS)
        for r in records
    ])

    # 3. Hard gate realistic (simulated LLM at empirical accuracy)
    models["hard_gate_realistic"] = np.array([
        score_hard_gate(r, bool(verdicts[i]), RECALIBRATED_PRIORS)
        for i, r in enumerate(records)
    ])

    # 4. INDRA pre-computed belief
    models["indra_belief"] = np.array([r["belief"] for r in records])

    # Print metrics
    print("=== Model comparison ===")
    print()
    metrics = print_metrics_table(models, y_true)
    print()

    # Delta analysis
    oracle_auprc = metrics["hard_gate_oracle"]["auprc"]
    realistic_auprc = metrics["hard_gate_realistic"]["auprc"]
    recal_auprc = metrics["parametric_recal"]["auprc"]
    print(f"Oracle ceiling lift over recal:    "
          f"+{oracle_auprc - recal_auprc:.4f} AUPRC")
    print(f"Realistic LLM lift over recal:     "
          f"{realistic_auprc - recal_auprc:+.4f} AUPRC")
    print(f"Realistic captures "
          f"{(realistic_auprc - recal_auprc) / (oracle_auprc - recal_auprc) * 100:.1f}% "
          f"of oracle ceiling" if oracle_auprc != recal_auprc else "")
    print()

    # False-gate analysis
    fg_results = print_false_gate_analysis(records, verdicts, y_true)

    # Save results
    output_path = Path(__file__).parent / "realistic_gate_results.json"
    output = {
        "n_records": len(records),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "seed": SEED,
        "empirical_overall_accuracy": EMPIRICAL_OVERALL_ACCURACY,
        "empirical_type_accuracy": EMPIRICAL_TYPE_ACCURACY,
        "models": metrics,
        "false_gate_analysis": fg_results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
