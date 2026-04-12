"""Tests for parametric belief scoring (noise model)."""
import pytest
from indra_belief.noise_model import (
    compute_edge_reliability,
    compute_edge_reliability_with_contradiction,
    compute_gated_belief,
    INDRA_PRIORS,
)


class TestComputeEdgeReliability:
    """Tests for the additive INDRA noise model."""

    def test_single_reach(self):
        # Additive: 1 - (syst + rand^1) = 1 - (0.05 + 0.30) = 0.65
        b = compute_edge_reliability(["reach"], 1)
        assert b == pytest.approx(0.65, abs=0.001)

    def test_reach_multiple_evidence(self):
        b1 = compute_edge_reliability(["reach"], 1)
        b5 = compute_edge_reliability(["reach"], 5)
        assert b5 > b1

    def test_cross_source_corroboration(self):
        b_single = compute_edge_reliability(["reach"], 2)
        b_cross = compute_edge_reliability(["reach", "signor"], 2)
        assert b_cross > b_single

    def test_curated_higher_than_nlp(self):
        b_nlp = compute_edge_reliability(["reach"], 1)
        b_curated = compute_edge_reliability(["signor"], 1)
        assert b_curated > b_nlp

    def test_signor_single(self):
        # 1 - (0.01 + 0.049) = 0.941
        b = compute_edge_reliability(["signor"], 1)
        assert b == pytest.approx(0.941, abs=0.001)

    def test_empty_sources(self):
        assert compute_edge_reliability([], 0) == 0.0
        assert compute_edge_reliability([], 5) == 0.0

    def test_unknown_source_uses_default(self):
        b = compute_edge_reliability(["unknown_source"], 1)
        assert 0.0 < b < 1.0

    def test_reliability_bounded(self):
        b = compute_edge_reliability(["reach"], 100)
        assert 0.0 < b <= 1.0

    def test_additive_formula_matches_indra(self):
        """Verify we match INDRA SimpleScorer: syst + prod(rand), NOT syst + (1-syst)*rand^n."""
        rand, syst = INDRA_PRIORS["reach"]
        expected = 1.0 - (syst + rand ** 3)  # additive, 3 evidence
        actual = compute_edge_reliability(["reach"], 3)
        assert actual == pytest.approx(expected, abs=1e-10)


class TestEdgeReliabilityWithContradiction:
    def test_single_direction(self):
        edges = [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 1}]
        b, d, c = compute_edge_reliability_with_contradiction(edges)
        assert b > 0.0
        assert d == "activation"
        assert c is False

    def test_contradictory_penalizes(self):
        clean = [{"regulation_type": "activation", "sources": ["reach"], "evidence_count": 2}]
        contra = [
            {"regulation_type": "activation", "sources": ["reach"], "evidence_count": 2},
            {"regulation_type": "repression", "sources": ["reach"], "evidence_count": 1},
        ]
        b_clean, _, _ = compute_edge_reliability_with_contradiction(clean)
        b_contra, _, c = compute_edge_reliability_with_contradiction(contra)
        assert c is True
        assert b_contra < b_clean

    def test_dominant_direction(self):
        edges = [
            {"regulation_type": "activation", "sources": ["reach"], "evidence_count": 5},
            {"regulation_type": "repression", "sources": ["reach"], "evidence_count": 1},
        ]
        _, d, c = compute_edge_reliability_with_contradiction(edges)
        assert d == "activation"
        assert c is True

    def test_empty_edges(self):
        b, d, c = compute_edge_reliability_with_contradiction([])
        assert b == 0.0
        assert d == "unknown"
        assert c is False


class TestGatedBelief:
    """Tests for LLM-gated belief computation."""

    def test_all_included(self):
        """No gating — gated belief equals parametric."""
        evidence = [
            {"source_api": "reach", "included": True},
            {"source_api": "reach", "included": True},
        ]
        result = compute_gated_belief(evidence)
        assert result.belief == pytest.approx(result.parametric_only)
        assert result.n_gated == 0
        assert result.n_surviving_evidence == 2

    def test_partial_gating_reduces_belief(self):
        """Gating out some evidence reduces belief."""
        evidence = [
            {"source_api": "reach", "included": True},
            {"source_api": "reach", "included": True},
            {"source_api": "reach", "included": False},
        ]
        result = compute_gated_belief(evidence)
        assert result.belief < result.parametric_only
        assert result.n_gated == 1
        assert result.n_surviving_evidence == 2

    def test_all_gated_single_source_returns_zero(self):
        """All evidence from only source gated → source removed → belief = 0."""
        evidence = [
            {"source_api": "reach", "included": False},
            {"source_api": "reach", "included": False},
        ]
        result = compute_gated_belief(evidence)
        assert result.belief == 0.0
        assert result.n_gated == 2
        assert result.n_surviving_evidence == 0

    def test_source_removal_when_all_gated(self):
        """Source with all evidence gated is removed; other sources still count."""
        evidence = [
            {"source_api": "reach", "included": False},
            {"source_api": "reach", "included": False},
            {"source_api": "signor", "included": True},
        ]
        result = compute_gated_belief(evidence)
        # Only signor survives: 1 - (0.01 + 0.049) = 0.941
        assert result.belief == pytest.approx(0.941, abs=0.001)
        assert result.n_gated == 2
        assert result.n_surviving_evidence == 1

    def test_no_invalid_probability_from_gating(self):
        """Gating must never produce probabilities > 1 or < 0.

        Under the additive formula, syst + rand_j with rand_j=1.0
        would give syst + 1.0 > 1.0. Our source-removal approach
        avoids this entirely.
        """
        evidence = [
            {"source_api": "reach", "included": False},
        ]
        result = compute_gated_belief(evidence)
        assert 0.0 <= result.belief <= 1.0

    def test_empty_evidence(self):
        result = compute_gated_belief([])
        assert result.belief == 0.0
        assert result.parametric_only == 0.0

    def test_default_included_is_true(self):
        """Evidence without 'included' key defaults to included."""
        evidence = [{"source_api": "reach"}]
        result = compute_gated_belief(evidence)
        assert result.belief == pytest.approx(0.65, abs=0.001)
        assert result.n_surviving_evidence == 1

    def test_per_source_breakdown(self):
        evidence = [
            {"source_api": "reach", "included": True},
            {"source_api": "reach", "included": False},
            {"source_api": "signor", "included": True},
        ]
        result = compute_gated_belief(evidence)
        assert len(result.per_source) == 2
        reach_bd = [s for s in result.per_source if s.source == "reach"][0]
        assert reach_bd.n_total == 2
        assert reach_bd.n_surviving == 1
        signor_bd = [s for s in result.per_source if s.source == "signor"][0]
        assert signor_bd.n_total == 1
        assert signor_bd.n_surviving == 1

    def test_mixed_sources_gating(self):
        """Complex case: multiple sources, mixed gating."""
        evidence = [
            {"source_api": "reach", "included": True},
            {"source_api": "reach", "included": True},
            {"source_api": "reach", "included": False},
            {"source_api": "sparser", "included": False},
            {"source_api": "sparser", "included": False},
            {"source_api": "signor", "included": True},
        ]
        result = compute_gated_belief(evidence)
        # sparser fully gated → removed
        # reach: 2 surviving, signor: 1 surviving
        # P(wrong) = (0.05 + 0.30^2) * (0.01 + 0.049^1) = 0.14 * 0.059 = 0.00826
        # belief ≈ 0.992
        assert result.n_gated == 3
        assert result.n_surviving_evidence == 3
        assert result.belief > 0.99

    def test_belief_bounded_zero_to_one(self):
        """Belief is always in [0, 1] regardless of input."""
        for src in ["reach", "sparser", "signor", "trips", "rlimsp"]:
            for included in [True, False]:
                evidence = [{"source_api": src, "included": included}]
                result = compute_gated_belief(evidence)
                assert 0.0 <= result.belief <= 1.0
                assert 0.0 <= result.parametric_only <= 1.0
