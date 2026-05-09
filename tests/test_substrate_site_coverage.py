"""P5.5 site-coverage guardrail.

Substrate-primary doctrine (P-phase): perturbation/sites/perturbation_class
should be detected by deterministic regex/Gilda BEFORE the parser is asked
to extract them. These tests assert the L5 site detector recognizes the
spelled-out residue variants the parser used to catch via natural-language
understanding.

If a future regex tightening drops one of these forms, the schema
inversion (P6) will silently lose site signal — that's why these tests
are coverage guardrails, not just feature tests.
"""
from __future__ import annotations

import pytest

from indra_belief.scorers.context_builder import _detect_sites


class TestSpelledOutSerine:
    """serine variants: full word, abbreviation, with hyphen or space."""

    def test_full_word_with_space(self):
        assert "S299" in _detect_sites("phosphorylation at serine 299")

    def test_full_word_capitalized(self):
        assert "S299" in _detect_sites("Phosphorylation at Serine 299")

    def test_abbrev_with_hyphen(self):
        assert "S299" in _detect_sites("phosphorylated on Ser-299")

    def test_abbrev_with_space(self):
        assert "S299" in _detect_sites("phosphorylated on Ser 299")

    def test_letter_with_hyphen(self):
        assert "S299" in _detect_sites("phosphorylation at S-299")


class TestSpelledOutThreonine:
    def test_full_word_with_space(self):
        assert "T461" in _detect_sites("phosphorylates threonine 461")

    def test_abbrev_with_hyphen(self):
        assert "T461" in _detect_sites("phosphorylated on Thr-461")

    def test_letter_with_hyphen(self):
        assert "T461" in _detect_sites("phosphorylation at T-461")


class TestSpelledOutTyrosine:
    def test_full_word_with_space(self):
        assert "Y732" in _detect_sites("phosphorylates tyrosine 732")

    def test_abbrev_with_hyphen(self):
        assert "Y732" in _detect_sites("phosphorylated on Tyr-732")

    def test_letter_with_hyphen(self):
        assert "Y732" in _detect_sites("phosphorylation at Y-732")


class TestMixedFormsInOneSentence:
    """Compound sentences mixing several forms — common in figure legends."""

    def test_serine_and_threonine_mixed(self):
        sites = _detect_sites(
            "X phosphorylates Y at Ser-299 and Thr-461."
        )
        assert "S299" in sites
        assert "T461" in sites

    def test_three_forms_in_one_sentence(self):
        sites = _detect_sites(
            "Phosphorylation at serine 102, Thr-461, and Y-732 was observed."
        )
        assert "S102" in sites
        assert "T461" in sites
        assert "Y732" in sites


class TestRegressionPreservedFromL5:
    """The hyphen extension must NOT break the prior space-form coverage."""

    def test_serine_space(self):
        assert "S102" in _detect_sites("Phosphorylation at serine 102 was observed.")

    def test_threonine_space(self):
        assert "T461" in _detect_sites("X phosphorylates Y at threonine 461.")

    def test_letter_form_preserved(self):
        sites = _detect_sites("X phosphorylates Y at S152, S156 and S163.")
        assert {"S152", "S156", "S163"}.issubset(sites)

    def test_letter_dash_form_preserved(self):
        sites = _detect_sites("Phosphorylated at S-102 and T-461.")
        assert "S102" in sites
        assert "T461" in sites


class TestNegativeCases:
    """Pre-existing denylist + anchor logic must still suppress false fires."""

    def test_figure_callout_suppressed(self):
        """Fig. 4 should NOT produce S4 or T4."""
        sites = _detect_sites(
            "As shown in Fig. 4, phosphorylation at S102 was observed."
        )
        assert "S102" in sites

    def test_s100_protein_family_suppressed(self):
        """S100 is a protein family, not a site."""
        sites = _detect_sites("S100 protein expression was elevated.")
        assert "S100" not in sites
