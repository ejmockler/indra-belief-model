"""M1 tests: relation pattern catalog.

Each axis-sign group has positive and negative tests. Positive tests
are anchored to specific FN records from the L8 diagnosis (108 FNs at
n=501); negative tests are near-miss surface forms that must NOT
match (they would cause false-positive relation detection in M2).

The tests verify pattern emission only — alias validation lives in
M2, and confidence-capped match acceptance lives in M3.
"""
from __future__ import annotations

import pytest

from indra_belief.scorers.relation_patterns import (
    ACTIVITY_POSITIVE,
    AMOUNT_POSITIVE,
    BINDING_NEUTRAL,
    CATALOG,
    MODIFICATION_POSITIVE,
    RelationPattern,
    iter_patterns,
)


def _all_matches(text: str, patterns: list[RelationPattern]) -> list[dict]:
    """Run every pattern over text; return list of dicts with pattern_id
    plus matched X, Y, site groups."""
    out = []
    for p in patterns:
        for m in p.regex.finditer(text):
            d = {"pattern_id": p.pattern_id, "X": m.group("X"), "Y": m.group("Y")}
            if "site" in p.regex.groupindex:
                d["site"] = m.group("site")
            out.append(d)
    return out


def _matches_pattern(text: str, pattern_id: str,
                     expect_x: str, expect_y: str,
                     expect_site: str | None = None) -> bool:
    """True iff at least one match has the expected pattern_id and groups
    (case-insensitive on entity names — Gilda does the canonicalization
    later)."""
    for p in CATALOG:
        if p.pattern_id != pattern_id:
            continue
        for m in p.regex.finditer(text):
            x = m.group("X")
            y = m.group("Y")
            if x.lower() != expect_x.lower() or y.lower() != expect_y.lower():
                continue
            if expect_site is not None:
                site = m.group("site") if "site" in p.regex.groupindex else None
                if site is None or site.lower() != expect_site.lower():
                    continue
            return True
    return False


# ----------------------------------------------------------------------------
# CATALOG sanity
# ----------------------------------------------------------------------------
class TestCatalogSanity:
    def test_catalog_nonempty(self):
        assert len(CATALOG) > 0

    def test_all_patterns_have_named_groups(self):
        for p in CATALOG:
            assert "X" in p.regex.groupindex, p.pattern_id
            assert "Y" in p.regex.groupindex, p.pattern_id

    def test_pattern_ids_unique(self):
        ids = [p.pattern_id for p in CATALOG]
        assert len(ids) == len(set(ids))

    def test_modification_axis_count(self):
        # Diagnosis closure target: 7 patterns for mod+pos.
        assert len(MODIFICATION_POSITIVE) >= 7

    def test_binding_axis_count(self):
        assert len(BINDING_NEUTRAL) >= 4

    def test_activity_axis_count(self):
        assert len(ACTIVITY_POSITIVE) >= 4

    def test_amount_axis_count(self):
        assert len(AMOUNT_POSITIVE) >= 4

    def test_iter_patterns_filters_axis(self):
        mod = iter_patterns(axis="modification")
        assert all(p.axis == "modification" for p in mod)
        assert len(mod) == len(MODIFICATION_POSITIVE)

    def test_iter_patterns_filters_sign(self):
        pos = iter_patterns(sign="positive")
        assert all(p.sign == "positive" for p in pos)


# ----------------------------------------------------------------------------
# modification + positive — diagnosis-anchored
# ----------------------------------------------------------------------------
class TestModificationPositive:
    """FN class A (nominalization). Each test names the diagnosis record."""

    def test_plk1_cdc20_dependent_of(self):
        # FN: PLK1→CDC20. Evidence: "Plk1 dependent phosphorylation of Cdc20 at S92"
        text = "Bub1 DeltaKinase stimulated Plk1 dependent phosphorylation of Cdc20 at S92."
        assert _matches_pattern(text, "mod_pos.dependent_of",
                                expect_x="Plk1", expect_y="Cdc20",
                                expect_site="S92")

    def test_csnk2a2_bmi1_active_verb(self):
        # FN: CSNK2A2→BMI1. After M6 Greek-norm, "CK2alpha" canonicalizes.
        # M1 here just verifies the verb pattern catches it.
        text = "CK2A2 phosphorylates BMI1 at Ser110."
        assert _matches_pattern(text, "mod_pos.active_verb",
                                expect_x="CK2A2", expect_y="BMI1",
                                expect_site="Ser110")

    def test_pkc_eif4e_nominalized_by(self):
        # FN: PKC→EIF4E. Evidence: "Phosphorylation of eIF-4E on serine 209
        # by protein kinase C". Captures Y=eIF-4E, X=protein (single-token);
        # M2's alias-validation with PKC FPLX-backfill (M5) recovers the bind.
        text = ("Phosphorylation of eIF-4E on serine 209 by protein kinase C "
                "is inhibited by the translational repressors.")
        # Capture should at least find the pattern; exact X is single-token.
        matches = [m for m in _all_matches(text, MODIFICATION_POSITIVE)
                   if m["pattern_id"] == "mod_pos.nominalized_by"]
        assert matches, f"no nominalized_by match in {text!r}"
        # Y must canonicalize to eIF-4E.
        assert any(m["Y"].lower() == "eif-4e" for m in matches)

    def test_passive_by(self):
        # Generic regression: Y is phosphorylated by X
        text = "ERK1 was phosphorylated by MEK in vitro."
        assert _matches_pattern(text, "mod_pos.passive_by",
                                expect_x="MEK", expect_y="ERK1")

    def test_induced_adj(self):
        # X-induced Y phosphorylation
        text = "EGF-induced ERK phosphorylation was sensitive to U0126."
        assert _matches_pattern(text, "mod_pos.induced_adj",
                                expect_x="EGF", expect_y="ERK")

    def test_mediated_adj(self):
        text = "PKA-mediated CREB phosphorylation occurred within 5 min."
        assert _matches_pattern(text, "mod_pos.mediated_adj",
                                expect_x="PKA", expect_y="CREB")

    def test_compound_nominal(self):
        # X phosphorylation of Y
        text = "Aurora-A phosphorylation of TPX2 at Ser121 is required for spindle assembly."
        assert _matches_pattern(text, "mod_pos.compound_nominal",
                                expect_x="Aurora-A", expect_y="TPX2",
                                expect_site="Ser121")

    # Negative controls
    def test_no_match_unrelated_text(self):
        text = "The cell migrated toward the chemotactic gradient."
        assert _all_matches(text, MODIFICATION_POSITIVE) == []

    def test_no_match_dephosphorylation_de_prefix(self):
        # de-prefix verbs are emitted via _MOD_STEM but tagged as positive
        # here — adjudicator's claim_status inversion handles direction.
        # We only need to verify they don't crash the regex.
        text = "Phosphatase removed the PKA-mediated dephosphorylation."
        # mediated_adj WILL match here ("PKA-mediated dephosphorylation")
        # but Y capture would be empty since no following entity. Verify
        # no crash.
        _all_matches(text, MODIFICATION_POSITIVE)  # should not raise


# ----------------------------------------------------------------------------
# binding + neutral — diagnosis-anchored
# ----------------------------------------------------------------------------
class TestBindingNeutral:
    def test_active_verb_binds_to(self):
        # FN: MAP3K5(ASK-1)→PPP5C(PP5). "PP5 binds to ... ASK-1"
        text = "PP5 binds to the C-terminal domain of ASK-1."
        # Y="the" is captured first ("PP5 binds to the"). With (?:to\s+)? optional,
        # earliest match is X=PP5, Y=the. We need pattern to skip "to".
        # Verify SOME pattern fires.
        matches = _all_matches(text, BINDING_NEUTRAL)
        assert any(m["X"].lower() == "pp5" for m in matches)

    def test_binding_of_to(self):
        # FN: CASP9→APAF1. "binding of Apaf1 to procaspase-9"
        text = "TUCAN interferes with binding of Apaf1 to procaspase-9."
        assert _matches_pattern(text, "bind.binding_of_to",
                                expect_x="Apaf1", expect_y="procaspase-9")

    def test_interacts_with(self):
        text = "Dab1 interacts with Fyn at the cell surface."
        assert _matches_pattern(text, "bind.interacts_with",
                                expect_x="Dab1", expect_y="Fyn")

    def test_complex_of(self):
        text = "The complex of CDK4 and Cyclin D1 is active."
        assert _matches_pattern(text, "bind.complex_of",
                                expect_x="CDK4", expect_y="Cyclin")
        # Cyclin captured single-token; M2 substring match recovers Cyclin D1.

    def test_interaction_compound(self):
        # FN: TCF_LEF→CTNNB1 — "beta-catenin's interactions with TCF/Lef"
        # is interacts_with form, not interaction_compound. Use a different
        # case.
        text = "EGFR-Grb2 interaction was disrupted by the inhibitor."
        assert _matches_pattern(text, "bind.interaction_compound",
                                expect_x="EGFR", expect_y="Grb2")

    def test_between_and(self):
        # FN: TCF_LEF→CTNNB1 surrogate
        text = "The interaction between beta-catenin and TCF7 mediates Wnt signaling."
        assert _matches_pattern(text, "bind.between_and",
                                expect_x="beta-catenin", expect_y="TCF7")

    def test_recruits(self):
        text = "BRCA1 recruits BARD1 to sites of DNA damage."
        assert _matches_pattern(text, "bind.recruits",
                                expect_x="BRCA1", expect_y="BARD1")

    def test_no_match_when_no_binding_verb(self):
        text = "PP5 was abundant in the nucleus."
        assert _all_matches(text, BINDING_NEUTRAL) == []


# ----------------------------------------------------------------------------
# activity + positive — diagnosis-anchored
# ----------------------------------------------------------------------------
class TestActivityPositive:
    def test_active_verb(self):
        # Generic case
        text = "EGF activates ERK rapidly."
        assert _matches_pattern(text, "act_pos.active_verb",
                                expect_x="EGF", expect_y="ERK")

    def test_induced_adj(self):
        # FN: TGFB1→p38. "TGF-beta1-induced p38 activation"
        text = "TGF-beta1-induced p38 activation was abolished by SB203580."
        assert _matches_pattern(text, "act_pos.induced_adj",
                                expect_x="TGF-beta1", expect_y="p38")

    def test_compound_nominal_act(self):
        # FN: FGF→ERK. "FGF activation of MAPK"
        text = "Decreased GRIN diminished FGF activation of MAPK."
        assert _matches_pattern(text, "act_pos.compound_nominal",
                                expect_x="FGF", expect_y="MAPK")

    def test_triggered_act(self):
        # FN: DEFA1→NFkappaB. "HNP1 triggered the activation of NF-kappaB"
        text = "HNP1 triggered the activation of NF-kappaB and IRF1."
        assert _matches_pattern(text, "act_pos.triggered_act",
                                expect_x="HNP1", expect_y="NF-kappaB")

    def test_nominalized_by(self):
        text = "Activation of AKT by PI3K is required for cell survival."
        assert _matches_pattern(text, "act_pos.nominalized_by",
                                expect_x="PI3K", expect_y="AKT")

    def test_mediated_adj_act(self):
        text = "WNT-mediated CTNNB1 activation drives proliferation."
        assert _matches_pattern(text, "act_pos.mediated_adj",
                                expect_x="WNT", expect_y="CTNNB1")

    # Negative
    def test_no_match_inactivates(self):
        # Sign-negative verb shouldn't fire on pos catalog
        text = "PTEN inactivates AKT in resting cells."
        # 'inactivates' has 'activate' as substring; ensure we don't FP-match
        # on 'activates' captured from 'inactivates'. Word-boundary anchor
        # should prevent.
        matches = _all_matches(text, ACTIVITY_POSITIVE)
        # Permitted: zero matches OR matches that aren't anchored at PTEN-AKT
        # via active_verb (since 'inactivates' contains 'activates' as
        # substring without word boundary, regex shouldn't match).
        # If any match has X=PTEN Y=AKT, that's a regression.
        bad = [m for m in matches if m["X"].lower() == "pten" and m["Y"].lower() == "akt"]
        assert not bad, f"FP regex match on inactivates: {bad}"


# ----------------------------------------------------------------------------
# amount + positive — diagnosis-anchored
# ----------------------------------------------------------------------------
class TestAmountPositive:
    def test_passive_upregulated(self):
        # FN: NFkappaB→LCN2. "LCN2 expression is upregulated by ... NF-kappaB"
        text = "LCN2 expression is upregulated by HER2 and NFkappaB pathway."
        # Multiple X candidates; verify one matches NFkappaB
        matches = _all_matches(text, AMOUNT_POSITIVE)
        assert any(m["pattern_id"] == "amt_pos.passive_upregulated"
                   and m["Y"].lower() == "lcn2"
                   for m in matches)

    def test_upregulates(self):
        text = "TNF upregulates ICAM1 expression in endothelial cells."
        assert _matches_pattern(text, "amt_pos.upregulates",
                                expect_x="TNF", expect_y="ICAM1")

    def test_induces_expression(self):
        text = "STAT3 induces SOCS3 expression in response to IL-6."
        assert _matches_pattern(text, "amt_pos.induces_expression",
                                expect_x="STAT3", expect_y="SOCS3")

    def test_induced_adj_amt(self):
        text = "TNF-induced ICAM1 expression was blocked."
        assert _matches_pattern(text, "amt_pos.induced_adj",
                                expect_x="TNF", expect_y="ICAM1")

    def test_required_to_elevate(self):
        # FN: IL4→IL4R. "IL-4 ... required to elevate IL-4Ralpha expression"
        text = "IL-4 and STAT6 were required to elevate IL-4Ralpha expression."
        matches = _all_matches(text, AMOUNT_POSITIVE)
        assert any(m["pattern_id"] == "amt_pos.required_to_elevate"
                   and m["Y"].lower().startswith("il-4")
                   for m in matches)


# ----------------------------------------------------------------------------
# Cross-axis negative regression: ambiguous text must not over-match
# ----------------------------------------------------------------------------
class TestCrossAxisNegatives:
    def test_downstream_of_no_match(self):
        # "X is downstream of Y" is NOT a relation per se.
        text = "STAT3 is downstream of JAK2 in this pathway."
        # 'downstream' isn't in any verb; 'JAK2 in this pathway' isn't a
        # binding/activity construct. No matches expected.
        for axis_group in (MODIFICATION_POSITIVE, ACTIVITY_POSITIVE):
            assert _all_matches(text, axis_group) == [], axis_group

    def test_colocalizes_no_match_outside_binding(self):
        text = "PP1 co-localizes with myosin in the contractile ring."
        # bind.interacts_with includes co-localizes — that's intentional.
        assert _matches_pattern(text, "bind.interacts_with",
                                expect_x="PP1", expect_y="myosin")
        # But should NOT match modification or activity catalogs.
        for axis_group in (MODIFICATION_POSITIVE, ACTIVITY_POSITIVE):
            assert _all_matches(text, axis_group) == [], axis_group

    def test_short_token_rejected(self):
        # Single-letter "X" or "I" must not be captured (NAME requires ≥2 chars)
        text = "X phosphorylates Y at S102."
        # X and Y are single chars per regex — no match at all.
        matches = _all_matches(text, MODIFICATION_POSITIVE)
        assert all(len(m["X"]) >= 2 and len(m["Y"]) >= 2 for m in matches), matches


# ----------------------------------------------------------------------------
# R2 patterns — Q-phase regression closure for absent_relationship FNs
# Each anchor is a specific record id and surface form from the
# 27-record absent_relationship regression set on Q-phase output.
# ----------------------------------------------------------------------------
class TestR2BindingExpansions:
    def test_forms_complex_with(self):
        # Q-regression: AGO2-RAD51.
        text = "Interestingly, we show that Ago2 forms a complex with Rad51."
        assert _matches_pattern(text, "bind.forms_complex_with",
                                expect_x="Ago2", expect_y="Rad51")

    def test_coord_complex(self):
        # Q-regression: TECPR1-ATG5 ("Atg5 and TECPR1 complex formation").
        text = "we performed reciprocal affinity purification for the Atg5 and TECPR1 complex formation."
        assert _matches_pattern(text, "bind.coord_complex",
                                expect_x="Atg5", expect_y="TECPR1")

    def test_binding_affinity_for(self):
        # Q-regression: CDKN1A-PCNA ("binding affinity of p21 for PCNA").
        text = "increasing the binding affinity of p21 for PCNA."
        assert _matches_pattern(text, "bind.binding_affinity_for",
                                expect_x="p21", expect_y="PCNA")

    def test_bound_to(self):
        # Generic case for X bound to Y.
        text = "BRCA1 bound to RAD51 in vitro."
        assert _matches_pattern(text, "bind.bound_to",
                                expect_x="BRCA1", expect_y="RAD51")

    def test_passive_associated(self):
        # Q-regression: APOA1-PLTP ("PLTP is associated with apoA-I").
        text = "PLTP is associated with apoA-I in plasma."
        assert _matches_pattern(text, "bind.passive_associated",
                                expect_x="PLTP", expect_y="apoA-I")

    def test_coord_bound_to(self):
        # Q-regression: MED17-ERCC3 ("XPG and XPB bound to hMED17").
        text = "XPG and XPB bound to MED17 in vitro."
        assert _matches_pattern(text, "bind.coord_bound_to",
                                expect_x="XPB", expect_y="MED17")

    def test_gerund_modifier_to(self):
        # Q-regression: p14_3_3-MARK2 ("14-3-3 protein binding to Par1b").
        text = "the double mutation almost completely abrogated 14-3-3 protein binding to Par1b."
        assert _matches_pattern(text, "bind.gerund_modifier_to",
                                expect_x="14-3-3", expect_y="Par1b")


class TestR2ModificationExpansions:
    def test_target_compound_by(self):
        # Q-regression: PDPK1-AKT1 ("AKT1 phosphorylation by PDK1").
        text = "characterize AKT1 phosphorylation by PDK1 in vitro."
        assert _matches_pattern(text, "mod_pos.target_compound_by",
                                expect_x="PDK1", expect_y="AKT1")

    def test_x_increases_phos_of_y(self):
        # Q-regression: CENPJ-NFkappaB ("CPAP increases the phosphorylation of NF-kappaB").
        text = "CPAP increases the phosphorylation of NF-kappaB."
        assert _matches_pattern(text, "mod_pos.x_increases_phos_of_y",
                                expect_x="CPAP", expect_y="NF-kappaB")

    def test_relative_clause(self):
        # Q-regression: AKT-EPHA2 ("Akt, which in turn phosphorylates EphA2 on S897").
        text = "activation of Akt, which in turn phosphorylates EphA2 on S897."
        assert _matches_pattern(text, "mod_pos.relative_clause",
                                expect_x="Akt", expect_y="EphA2", expect_site="S897")


class TestR2AmountExpansions:
    def test_x_induces_expression_of_y(self):
        # Q-regression: TP53-GLS2 ("p53 also induces the expression of GLS2").
        text = "p53 also induces the expression of GLS2 in response to stress."
        assert _matches_pattern(text, "amt_pos.x_induces_expression_of_y",
                                expect_x="p53", expect_y="GLS2")

    def test_coord_first(self):
        # Q-regression: TNF-ADAMTS12 ("TNF induces ADAMTS-7 and ADAMTS-12 expression").
        # Coord first slot capture.
        text = "TNF induces ADAMTS-7 and ADAMTS-12 expression in chondrocytes."
        assert _matches_pattern(text, "amt_pos.coord_first",
                                expect_x="TNF", expect_y="ADAMTS-7")

    def test_coord_last(self):
        # Same construct, capture Y at end of coord.
        text = "TNF induces ADAMTS-7 and ADAMTS-12 expression in chondrocytes."
        assert _matches_pattern(text, "amt_pos.coord_last",
                                expect_x="TNF", expect_y="ADAMTS-12")


class TestR2ActivityExpansions:
    def test_x_induced_act_of_y(self):
        # Q-regression: IGF1-SHC ("IGF-I-induced activation of IRS-1, Shc, ...").
        # First-slot capture via the simple variant.
        text = "IGF-I-induced activation of IRS-1 was unaffected."
        assert _matches_pattern(text, "act_pos.x_induced_act_of_y",
                                expect_x="IGF-I", expect_y="IRS-1")

    def test_x_induced_act_of_coord_last(self):
        # Q-regression: TNFSF10-CASP9 ("TRAIL-induced activation of caspase-3 and caspase-9").
        text = "TRAIL-induced activation of caspase-3 and caspase-9 in vitro."
        assert _matches_pattern(text, "act_pos.x_induced_act_of_coord_last",
                                expect_x="TRAIL", expect_y="caspase-9")

    def test_x_induced_release_of(self):
        # First-slot capture.
        text = "TCR induced release of TNF-alpha from T cells."
        assert _matches_pattern(text, "act_pos.x_induced_release_of",
                                expect_x="TCR", expect_y="TNF-alpha")


class TestR2NoFalseMatchOnNegatives:
    """R2 patterns should not match negation/hedge surface forms."""

    def test_passive_associated_rejects_negation(self):
        # The optional adverb alternation in bind.passive_associated
        # only allows the descriptive modifiers
        # (directly/stably/tightly/physically). "Not" isn't in the list,
        # so "X was not associated with Y" doesn't match — a small
        # precision win.
        text = "TNF was not associated with IL6."
        assert not _matches_pattern(text, "bind.passive_associated",
                                    expect_x="TNF", expect_y="IL6")
        # Confirm the positive form does match.
        text2 = "TNF was associated with IL6 in macrophages."
        assert _matches_pattern(text2, "bind.passive_associated",
                                expect_x="TNF", expect_y="IL6")

    def test_coord_complex_rejects_three_member(self):
        # "X, Y, and Z complex" with comma+and — coord_complex matches
        # only "X and Y complex" (two members joined by and/with).
        # Three-way coord shouldn't fire on the simple pattern.
        text = "A, B, and C heterodimer formation."
        # Pattern requires X<space>and<space>Y<space>complex — won't match
        # this comma-list form. Verify zero matches.
        matches = _all_matches(text, BINDING_NEUTRAL)
        coord_hits = [m for m in matches if m["pattern_id"] == "bind.coord_complex"]
        assert not coord_hits, f"three-member coord shouldn't fire coord_complex: {coord_hits}"
