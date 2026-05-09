"""P5.5 perturbation-coverage guardrail.

For each parse_evidence fewshot demonstrating perturbation framing
(#8 LOF knockdown, #9 LOF inhibitor, #10 GOF overexpression, #10b LOF
abolished, #10c LOF blocked phos), verify that the M9 substrate detector
returns the matching class on the fewshot's evidence text. If it
doesn't, P6 cannot drop the parser's `perturbation` field — the
substrate would lose signal the parser was catching.

Anchors are entity-scoped: _detect_perturbation_for needs the entity
name + alias set. Tests pass a minimal alias frozenset (just `{name}`)
which mirrors the simplest production case where Gilda/FPLX expansion
hasn't fired.
"""
from __future__ import annotations

import pytest

from indra_belief.scorers.context_builder import _detect_perturbation_for


class TestFewshot8_LofKnockdown:
    """#8: 'TP53 knockdown increased MDM2 levels' → LOF on TP53."""
    text = "TP53 knockdown increased MDM2 protein levels."

    def test_subject_lof_detected(self):
        assert _detect_perturbation_for(
            self.text, "TP53", frozenset({"TP53"})
        ) == "loss_of_function"

    def test_object_no_perturbation(self):
        # MDM2 isn't perturbed in this sentence — it's the readout.
        assert _detect_perturbation_for(
            self.text, "MDM2", frozenset({"MDM2"})
        ) is None


class TestFewshot9_LofInhibitor:
    """#9: 'MEK inhibitor U0126 reduced phosphorylation of ERK1/2' → LOF on MEK."""
    text = (
        "Treatment with the MEK inhibitor U0126 reduced phosphorylation of "
        "ERK1/2 in HeLa cells."
    )

    def test_subject_lof_detected(self):
        assert _detect_perturbation_for(
            self.text, "MEK", frozenset({"MEK"})
        ) == "loss_of_function"

    def test_object_no_perturbation(self):
        assert _detect_perturbation_for(
            self.text, "ERK1/2", frozenset({"ERK1/2"})
        ) is None


class TestFewshot10_GofOverexpression:
    """#10: 'overexpression of FOXO3 ... decreased BCL2' → GOF on FOXO3."""
    text = (
        "Stable overexpression of FOXO3 in HEK293 cells decreased BCL2 "
        "protein abundance."
    )

    def test_subject_gof_detected(self):
        assert _detect_perturbation_for(
            self.text, "FOXO3", frozenset({"FOXO3"})
        ) == "gain_of_function"

    def test_object_no_perturbation(self):
        assert _detect_perturbation_for(
            self.text, "BCL2", frozenset({"BCL2"})
        ) is None


class TestFewshot10b_LofAbolished:
    """#10b: 'Knockdown of PROTKIN8 abolished ... TARGET6' → LOF on PROTKIN8."""
    text = (
        "Knockdown of PROTKIN8 abolished stimulus-induced increases in "
        "TARGET6 expression in HEK293 cells."
    )

    def test_subject_lof_detected(self):
        assert _detect_perturbation_for(
            self.text, "PROTKIN8", frozenset({"PROTKIN8"})
        ) == "loss_of_function"


class TestFewshot10c_LofBlockedPhos:
    """#10c: 'PROTKIN9 inhibitor blocked ... phos of TARGET8' → LOF on PROTKIN9."""
    text = (
        "Pretreatment with PROTKIN9 inhibitor blocked agonist-induced "
        "phosphorylation of TARGET8 in primary cells."
    )

    def test_subject_lof_detected(self):
        assert _detect_perturbation_for(
            self.text, "PROTKIN9", frozenset({"PROTKIN9"})
        ) == "loss_of_function"


class TestParentheticalAlias:
    """The HDAC parenthetical case ('histone deacetylase (HDAC) inhibitors')
    is a known M9 carve-out — the inner parenthetical also matches as an
    LOF anchor for HDAC. Worth pinning so future regex tightening doesn't
    silently break it."""

    def test_parenthetical_inner_anchored(self):
        text = "Histone deacetylase (HDAC) inhibitors induced AR acetylation."
        assert _detect_perturbation_for(
            text, "HDAC", frozenset({"HDAC"})
        ) == "loss_of_function"

    def test_outer_form_also_anchored(self):
        text = "Histone deacetylase (HDAC) inhibitors induced AR acetylation."
        # The outer form "histone deacetylase ... inhibitors" anchors via
        # the {name}\s+inhibitor pattern allowing parenthetical insertion.
        assert _detect_perturbation_for(
            text, "histone deacetylase", frozenset({"histone deacetylase"})
        ) == "loss_of_function"


class TestNegativePerturbation:
    """Sentences without perturbation framing must return None — the
    detector must NOT over-fire on plain regulator language."""

    def test_direct_phosphorylation_no_perturbation(self):
        text = "AKT1 phosphorylates GSK3B at serine 9."
        assert _detect_perturbation_for(
            text, "AKT1", frozenset({"AKT1"})
        ) is None

    def test_unrelated_inhibitor_mention(self):
        # An inhibitor mention that is NOT anchored to AKT1 must not
        # mark AKT1 as LOF.
        text = "MEK inhibitor U0126 reduced AKT1 phosphorylation."
        assert _detect_perturbation_for(
            text, "AKT1", frozenset({"AKT1"})
        ) is None
