"""Composed belief scorer: parametric noise model + LLM hard gate.

Bridges LLM verdict output with the parametric noise model. The LLM
acts as a binary filter: each evidence sentence is either included
(counts in the noise model) or excluded (removed). Sources with all
evidence excluded are dropped entirely.

Usage:
    scorer = ComposedBeliefScorer(priors=RECALIBRATED_PRIORS)
    result = scorer.score_edge(evidence_with_verdicts)
    print(result.belief, result.parametric_only, result.n_gated)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from indra_belief.noise_model import (
    INDRA_PRIORS,
    RECALIBRATED_PRIORS,
    GatedBeliefResult,
    compute_gated_belief,
    compute_edge_reliability,
    compute_edge_reliability_with_contradiction,
)


# Verdict categories that pass the hard gate
_PASS_VERDICTS = frozenset({"correct"})

# Verdict categories that are gated out
_GATE_VERDICTS = frozenset({"incorrect"})

# Everything else (None, "ambiguous", parse failures) → gated out (conservative)


@dataclass(frozen=True)
class EvidenceRecord:
    """Single evidence sentence with source metadata and optional LLM verdict."""
    source_api: str
    verdict: str | None = None          # "correct", "incorrect", or None
    confidence: str | None = None       # "high", "medium", "low", or None
    regulation_type: str | None = None  # for contradiction handling
    stmt_hash: int | None = None        # for deduplication / audit trail


@dataclass(frozen=True)
class ComposedScore:
    """Result of composed belief scoring."""
    belief: float                   # final composed belief
    parametric_only: float          # belief without any gating
    n_total: int                    # total evidence sentences
    n_surviving: int                # evidence that passed the gate
    n_gated: int                    # evidence removed by LLM
    gated_result: GatedBeliefResult # full per-source breakdown
    has_llm_scores: bool            # whether any LLM verdicts were present


class ComposedBeliefScorer:
    """Composes parametric noise model with LLM hard-gate evidence filtering.

    The LLM verdict determines whether each evidence sentence counts:
    - verdict="correct" → included (evidence counts in noise model)
    - verdict="incorrect" → gated out
    - verdict=None (unscored) → included (graceful degradation)
    - verdict=anything else → gated out (conservative default)

    Args:
        priors: Source error priors. Default: INDRA_PRIORS.
            Use RECALIBRATED_PRIORS for benchmark-calibrated values.
        gate_unscored: If True, evidence without LLM verdicts is gated out.
            Default False (unscored evidence passes — pure parametric fallback).
    """

    def __init__(
        self,
        priors: dict[str, tuple[float, float]] | None = None,
        gate_unscored: bool = False,
    ):
        self.priors = priors or INDRA_PRIORS
        self.gate_unscored = gate_unscored

    def _should_include(self, record: EvidenceRecord) -> bool:
        """Determine if an evidence record passes the hard gate."""
        if record.verdict is None:
            # No LLM score available — include by default (graceful degradation)
            return not self.gate_unscored
        return record.verdict.lower() in _PASS_VERDICTS

    def score_edge(self, evidence: list[EvidenceRecord]) -> ComposedScore:
        """Score an edge using composed parametric + LLM belief.

        Args:
            evidence: List of evidence records for a single edge.

        Returns:
            ComposedScore with belief, parametric-only, and breakdown.
        """
        if not evidence:
            return ComposedScore(
                belief=0.0, parametric_only=0.0,
                n_total=0, n_surviving=0, n_gated=0,
                gated_result=GatedBeliefResult(
                    belief=0.0, parametric_only=0.0,
                    n_total_evidence=0, n_surviving_evidence=0, n_gated=0,
                    per_source=[],
                ),
                has_llm_scores=False,
            )

        has_llm = any(r.verdict is not None for r in evidence)

        # Build evidence dicts for the gated belief computation
        gated_evidence = [
            {
                "source_api": r.source_api,
                "included": self._should_include(r),
            }
            for r in evidence
        ]

        result = compute_gated_belief(gated_evidence, priors=self.priors)

        return ComposedScore(
            belief=result.belief,
            parametric_only=result.parametric_only,
            n_total=result.n_total_evidence,
            n_surviving=result.n_surviving_evidence,
            n_gated=result.n_gated,
            gated_result=result,
            has_llm_scores=has_llm,
        )

    def score_edge_with_contradiction(
        self,
        evidence: list[EvidenceRecord],
    ) -> tuple[ComposedScore, str, bool]:
        """Score an edge with contradiction penalty.

        Groups evidence by regulation_type, scores each direction,
        then applies: belief = belief_dominant * (1 - belief_opposing).

        Returns:
            (ComposedScore, dominant_direction, is_contradictory)
        """
        if not evidence:
            empty = self.score_edge([])
            return empty, "unknown", False

        # Group by regulation_type
        by_direction: dict[str, list[EvidenceRecord]] = {}
        for r in evidence:
            d = r.regulation_type or "unknown"
            by_direction.setdefault(d, []).append(r)

        # Score each direction independently
        dir_scores: dict[str, ComposedScore] = {}
        for direction, dir_evidence in by_direction.items():
            dir_scores[direction] = self.score_edge(dir_evidence)

        if not dir_scores:
            empty = self.score_edge([])
            return empty, "unknown", False

        # Find dominant direction
        dominant_dir = max(dir_scores, key=lambda d: dir_scores[d].belief)
        dominant = dir_scores[dominant_dir]

        # Check for contradiction
        opposing = {
            d: s for d, s in dir_scores.items()
            if d != dominant_dir and d != "unknown"
        }
        if opposing:
            opposing_belief = max(s.belief for s in opposing.values())
            penalized_belief = dominant.belief * (1.0 - opposing_belief)

            # Build a combined score
            all_evidence = [r for records in by_direction.values() for r in records]
            combined = self.score_edge(all_evidence)

            return ComposedScore(
                belief=penalized_belief,
                parametric_only=combined.parametric_only,
                n_total=combined.n_total,
                n_surviving=combined.n_surviving,
                n_gated=combined.n_gated,
                gated_result=combined.gated_result,
                has_llm_scores=combined.has_llm_scores,
            ), dominant_dir, True

        return dominant, dominant_dir, False
