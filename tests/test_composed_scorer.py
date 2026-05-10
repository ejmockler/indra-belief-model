"""Tests for the composed belief scorer (hard gate + parametric)."""
import pytest
from indra_belief.composed_scorer import (
    ComposedBeliefScorer,
    EvidenceRecord,
)
from indra_belief.noise_model import INDRA_PRIORS, RECALIBRATED_PRIORS


@pytest.fixture
def scorer():
    return ComposedBeliefScorer(priors=INDRA_PRIORS)


@pytest.fixture
def recal_scorer():
    return ComposedBeliefScorer(priors=RECALIBRATED_PRIORS)


class TestHardGate:
    """Core hard-gate behavior."""

    def test_correct_passes(self, scorer):
        evidence = [EvidenceRecord(source_api="reach", verdict="correct", confidence="high")]
        result = scorer.score_edge(evidence)
        assert result.n_surviving == 1
        assert result.n_gated == 0
        assert result.belief > 0.0

    def test_incorrect_gated(self, scorer):
        evidence = [EvidenceRecord(source_api="reach", verdict="incorrect", confidence="high")]
        result = scorer.score_edge(evidence)
        assert result.n_surviving == 0
        assert result.n_gated == 1
        assert result.belief == 0.0

    def test_none_verdict_passes_by_default(self, scorer):
        """Unscored evidence passes — graceful degradation to pure parametric."""
        evidence = [EvidenceRecord(source_api="reach", verdict=None)]
        result = scorer.score_edge(evidence)
        assert result.n_surviving == 1
        assert result.has_llm_scores is False
        assert result.belief == pytest.approx(0.65, abs=0.001)

    def test_none_verdict_gated_when_configured(self):
        scorer = ComposedBeliefScorer(gate_unscored=True)
        evidence = [EvidenceRecord(source_api="reach", verdict=None)]
        result = scorer.score_edge(evidence)
        assert result.n_surviving == 0
        assert result.n_gated == 1

    def test_ambiguous_gated(self, scorer):
        """Ambiguous/unknown verdicts are conservatively gated out."""
        evidence = [EvidenceRecord(source_api="reach", verdict="ambiguous")]
        result = scorer.score_edge(evidence)
        assert result.n_surviving == 0
        assert result.n_gated == 1

    def test_abstain_verdict_passes_like_none(self, scorer):
        """Gate-#26 fix: "abstain" from the decomposed path means "no usable
        judgment" — same semantics as verdict=None. Gating it like
        "incorrect" would systematically deflate belief scores on the
        decomposed path and make the dual-run comparison uninterpretable."""
        evidence = [EvidenceRecord(source_api="reach", verdict="abstain")]
        result = scorer.score_edge(evidence)
        assert result.n_surviving == 1
        assert result.n_gated == 0

    def test_abstain_verdict_gated_when_configured(self):
        """When gate_unscored=True, abstain follows the same path as None —
        excluded from the surviving set."""
        scorer = ComposedBeliefScorer(gate_unscored=True)
        evidence = [EvidenceRecord(source_api="reach", verdict="abstain")]
        result = scorer.score_edge(evidence)
        assert result.n_surviving == 0
        assert result.n_gated == 1

    def test_mixed_verdicts(self, scorer):
        evidence = [
            EvidenceRecord(source_api="reach", verdict="correct", confidence="high"),
            EvidenceRecord(source_api="reach", verdict="incorrect", confidence="high"),
            EvidenceRecord(source_api="reach", verdict="correct", confidence="medium"),
        ]
        result = scorer.score_edge(evidence)
        assert result.n_surviving == 2
        assert result.n_gated == 1
        assert result.belief < result.parametric_only  # gating reduces belief


class TestGracefulDegradation:
    """No LLM scores = pure parametric."""

    def test_all_unscored_equals_parametric(self, scorer):
        evidence = [
            EvidenceRecord(source_api="reach"),
            EvidenceRecord(source_api="reach"),
            EvidenceRecord(source_api="signor"),
        ]
        result = scorer.score_edge(evidence)
        assert result.has_llm_scores is False
        assert result.belief == pytest.approx(result.parametric_only)

    def test_empty_evidence(self, scorer):
        result = scorer.score_edge([])
        assert result.belief == 0.0
        assert result.n_total == 0


class TestSourceRemoval:
    """Source removal when all evidence gated."""

    def test_single_source_all_gated(self, scorer):
        evidence = [
            EvidenceRecord(source_api="reach", verdict="incorrect"),
            EvidenceRecord(source_api="reach", verdict="incorrect"),
        ]
        result = scorer.score_edge(evidence)
        assert result.belief == 0.0

    def test_one_source_survives(self, scorer):
        evidence = [
            EvidenceRecord(source_api="reach", verdict="incorrect"),
            EvidenceRecord(source_api="signor", verdict="correct"),
        ]
        result = scorer.score_edge(evidence)
        # Only signor survives: belief ≈ 0.941
        assert result.belief == pytest.approx(0.941, abs=0.001)


class TestContradiction:
    """Contradiction penalty with gating."""

    def test_no_contradiction(self, scorer):
        evidence = [
            EvidenceRecord(source_api="reach", verdict="correct", regulation_type="activation"),
        ]
        score, direction, contradictory = scorer.score_edge_with_contradiction(evidence)
        assert direction == "activation"
        assert contradictory is False

    def test_contradiction_penalizes(self, scorer):
        evidence = [
            EvidenceRecord(source_api="reach", verdict="correct", regulation_type="activation"),
            EvidenceRecord(source_api="reach", verdict="correct", regulation_type="activation"),
            EvidenceRecord(source_api="reach", verdict="correct", regulation_type="repression"),
        ]
        score, direction, contradictory = scorer.score_edge_with_contradiction(evidence)
        assert contradictory is True
        assert direction == "activation"
        # Penalized belief < unpenalized
        unpenalized = scorer.score_edge(evidence)
        assert score.belief < unpenalized.belief

    def test_gating_deflates_contradiction(self, scorer):
        """If opposing evidence is gated, contradiction penalty shrinks."""
        evidence_with_opposing = [
            EvidenceRecord(source_api="reach", verdict="correct", regulation_type="activation"),
            EvidenceRecord(source_api="reach", verdict="correct", regulation_type="activation"),
            EvidenceRecord(source_api="reach", verdict="correct", regulation_type="repression"),
        ]
        evidence_opposing_gated = [
            EvidenceRecord(source_api="reach", verdict="correct", regulation_type="activation"),
            EvidenceRecord(source_api="reach", verdict="correct", regulation_type="activation"),
            EvidenceRecord(source_api="reach", verdict="incorrect", regulation_type="repression"),
        ]
        score_with, _, _ = scorer.score_edge_with_contradiction(evidence_with_opposing)
        score_gated, _, contra = scorer.score_edge_with_contradiction(evidence_opposing_gated)

        if contra:
            # Opposing evidence was gated → opposing belief is 0 → no penalty
            assert score_gated.belief >= score_with.belief
        else:
            # Opposing source fully removed → no contradiction detected
            assert score_gated.belief > 0


class TestRecalibratedPriors:
    """Composed scorer with recalibrated priors."""

    def test_reach_lower_with_recalibrated(self, scorer, recal_scorer):
        evidence = [EvidenceRecord(source_api="reach", verdict="correct")]
        default_score = scorer.score_edge(evidence)
        recal_score = recal_scorer.score_edge(evidence)
        assert recal_score.belief < default_score.belief

    def test_trips_higher_with_recalibrated(self, scorer, recal_scorer):
        evidence = [EvidenceRecord(source_api="trips", verdict="correct")]
        default_score = scorer.score_edge(evidence)
        recal_score = recal_scorer.score_edge(evidence)
        assert recal_score.belief > default_score.belief


class TestDecomposition:
    """Verify results are decomposable."""

    def test_per_source_present(self, scorer):
        evidence = [
            EvidenceRecord(source_api="reach", verdict="correct"),
            EvidenceRecord(source_api="signor", verdict="correct"),
        ]
        result = scorer.score_edge(evidence)
        sources = {s.source for s in result.gated_result.per_source}
        assert "reach" in sources
        assert "signor" in sources

    def test_parametric_only_ignores_gating(self, scorer):
        evidence = [
            EvidenceRecord(source_api="reach", verdict="incorrect"),
            EvidenceRecord(source_api="reach", verdict="incorrect"),
        ]
        result = scorer.score_edge(evidence)
        assert result.parametric_only > 0.0  # would be > 0 without gating
        assert result.belief == 0.0  # but gating kills it
