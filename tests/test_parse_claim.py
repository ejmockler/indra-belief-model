"""Contract tests for parse_claim: deterministic INDRA Statement → ClaimCommitment.

Every INDRA statement type must project to the correct (axis, sign) pair.
100% reliability here is non-negotiable — parse_claim is the rosetta stone
against which evidence parses are adjudicated. Any drift here propagates
to every downstream sub-call.
"""
import pytest

pytest.importorskip("indra", reason="indra statement library required for parse_claim")

from indra.statements import (  # noqa: E402
    Acetylation,
    Activation,
    Agent,
    Autophosphorylation,
    Complex,
    Conversion,
    Deacetylation,
    DecreaseAmount,
    Demethylation,
    Dephosphorylation,
    Deubiquitination,
    Gap,
    Gef,
    GtpActivation,
    IncreaseAmount,
    Inhibition,
    Methylation,
    Phosphorylation,
    Transphosphorylation,
    Translocation,
    Ubiquitination,
)

from indra_belief.scorers.parse_claim import parse_claim, _is_mirna_name


# ---------------------------------------------------------------------------
# Full INDRA-type coverage: every supported statement type → correct axis/sign
# ---------------------------------------------------------------------------

# Each row: (constructor, args, expected_axis, expected_sign).
# Uses minimal Agent objects — the name is what matters for the commitment.
A = Agent("A")
B = Agent("B")

INDRA_COVERAGE = [
    (lambda: Activation(A, B),         "Activation",          "activity",     "positive"),
    (lambda: Inhibition(A, B),         "Inhibition",          "activity",     "negative"),
    (lambda: IncreaseAmount(A, B),     "IncreaseAmount",      "amount",       "positive"),
    (lambda: DecreaseAmount(A, B),     "DecreaseAmount",      "amount",       "negative"),
    (lambda: Complex([A, B]),          "Complex",             "binding",      "neutral"),
    (lambda: Phosphorylation(A, B),    "Phosphorylation",     "modification", "positive"),
    (lambda: Dephosphorylation(A, B),  "Dephosphorylation",   "modification", "negative"),
    (lambda: Autophosphorylation(A),   "Autophosphorylation", "modification", "positive"),
    (lambda: Transphosphorylation(A),  "Transphosphorylation","modification", "positive"),
    (lambda: Ubiquitination(A, B),     "Ubiquitination",      "modification", "positive"),
    (lambda: Deubiquitination(A, B),   "Deubiquitination",    "modification", "negative"),
    (lambda: Acetylation(A, B),        "Acetylation",         "modification", "positive"),
    (lambda: Deacetylation(A, B),      "Deacetylation",       "modification", "negative"),
    (lambda: Methylation(A, B),        "Methylation",         "modification", "positive"),
    (lambda: Demethylation(A, B),      "Demethylation",       "modification", "negative"),
    (lambda: Translocation(A, "cytoplasm", "nucleus"),
                                       "Translocation",       "localization", "neutral"),
    (lambda: Conversion(A, [B], [Agent("C")]),
                                       "Conversion",          "conversion",   "neutral"),
    (lambda: GtpActivation(A, B),      "GtpActivation",       "gtp_state",    "positive"),
    (lambda: Gef(A, B),                "Gef",                 "gtp_state",    "positive"),
    (lambda: Gap(A, B),                "Gap",                 "gtp_state",    "negative"),
]


@pytest.mark.parametrize("factory,stmt_type,axis,sign", INDRA_COVERAGE,
                         ids=[row[1] for row in INDRA_COVERAGE])
def test_every_indra_type_projects_correctly(factory, stmt_type, axis, sign):
    stmt = factory()
    c = parse_claim(stmt)
    assert c.stmt_type == stmt_type, f"stmt_type: {c.stmt_type} != {stmt_type}"
    assert c.axis == axis, f"{stmt_type}: axis {c.axis} != {axis}"
    assert c.sign == sign, f"{stmt_type}: sign {c.sign} != {sign}"


# ---------------------------------------------------------------------------
# Sign preservation across sign-paired INDRA types
# ---------------------------------------------------------------------------

class TestSignPreservation:
    def test_phos_and_dephos_differ_only_in_sign(self):
        phos = parse_claim(Phosphorylation(A, B))
        dephos = parse_claim(Dephosphorylation(A, B))
        assert phos.axis == dephos.axis
        assert phos.sign != dephos.sign

    def test_act_and_inh_differ_only_in_sign(self):
        act = parse_claim(Activation(A, B))
        inh = parse_claim(Inhibition(A, B))
        assert act.axis == inh.axis
        assert act.sign != inh.sign

    def test_inc_and_dec_amount_differ_only_in_sign(self):
        inc = parse_claim(IncreaseAmount(A, B))
        dec = parse_claim(DecreaseAmount(A, B))
        assert inc.axis == dec.axis
        assert inc.sign != dec.sign

    def test_ubiq_and_deubiq_differ_only_in_sign(self):
        u = parse_claim(Ubiquitination(A, B))
        du = parse_claim(Deubiquitination(A, B))
        assert u.axis == du.axis
        assert u.sign != du.sign


# ---------------------------------------------------------------------------
# Entity handling: subject/objects for different arities
# ---------------------------------------------------------------------------

class TestEntities:
    def test_binary_subject_and_object(self):
        c = parse_claim(Phosphorylation(Agent("MAPK1"), Agent("BRAF")))
        assert c.subject == "MAPK1"
        assert c.objects == ("BRAF",)

    def test_complex_n_ary(self):
        c = parse_claim(Complex([Agent("P"), Agent("Q"), Agent("R")]))
        assert c.subject == "P"
        assert c.objects == ("Q", "R")

    def test_self_modification_subject_equals_object(self):
        c = parse_claim(Autophosphorylation(Agent("BRAF")))
        assert c.subject == "BRAF"
        assert c.objects == ("BRAF",)

    def test_missing_subject_renders_question_mark(self):
        c = parse_claim(Complex([]))
        assert c.subject == "?"


# ---------------------------------------------------------------------------
# Modification site is preserved
# ---------------------------------------------------------------------------

class TestSite:
    def test_residue_and_position_combine_into_site(self):
        c = parse_claim(Phosphorylation(A, B, "S", "299"))
        assert c.site == "S299"

    def test_residue_only(self):
        c = parse_claim(Phosphorylation(A, B, residue="T"))
        assert c.site == "T"

    def test_position_only(self):
        c = parse_claim(Phosphorylation(A, B, position="308"))
        assert c.site == "308"

    def test_no_site(self):
        c = parse_claim(Phosphorylation(A, B))
        assert c.site is None


# ---------------------------------------------------------------------------
# Translocation endpoints are preserved
# ---------------------------------------------------------------------------

class TestTranslocation:
    def test_endpoints_populate(self):
        c = parse_claim(Translocation(Agent("NFkB"), "cytoplasm", "nucleus"))
        assert c.axis == "localization"
        assert c.sign == "neutral"
        assert c.location_from == "cytoplasm"
        assert c.location_to == "nucleus"

    def test_non_translocation_has_no_endpoints(self):
        c = parse_claim(Phosphorylation(A, B))
        assert c.location_from is None
        assert c.location_to is None


# ---------------------------------------------------------------------------
# miRNA subject rule — the one identity-modulated mapping
# ---------------------------------------------------------------------------

class TestMiRNARule:
    @pytest.mark.parametrize("name,expected", [
        ("MIR101",   True),
        ("MIR-21",   True),
        ("MIRLET7A", True),
        ("mir21",    True),          # case-insensitive
        ("let-7a",   True),
        ("LET-7",    True),
        ("TP53",     False),
        ("MDM2",     False),
        ("AKT1",     False),
        ("",         False),
    ])
    def test_mirna_name_recognition(self, name, expected):
        assert _is_mirna_name(name) is expected

    def test_mirna_inhibition_projects_to_amount_axis(self):
        """Inhibition by a miRNA subject commits to (amount, negative), not (activity, negative)."""
        c = parse_claim(Inhibition(Agent("MIR101"), Agent("TARGET")))
        assert c.subject_is_mirna is True
        assert c.axis == "amount"
        assert c.sign == "negative"
        assert c.stmt_type == "Inhibition"  # the statement type is preserved

    def test_protein_inhibition_stays_on_activity_axis(self):
        c = parse_claim(Inhibition(Agent("TP53"), Agent("MDM2")))
        assert c.subject_is_mirna is False
        assert c.axis == "activity"
        assert c.sign == "negative"

    def test_mirna_activation_does_NOT_override(self):
        """The rule is ONLY for Inhibition — other types of miRNA-subject
        statements project normally."""
        c = parse_claim(Activation(Agent("MIR101"), Agent("TARGET")))
        assert c.subject_is_mirna is True
        assert c.axis == "activity"   # NOT overridden
        assert c.sign == "positive"

    def test_let_7_variant_recognized(self):
        c = parse_claim(Inhibition(Agent("let-7a"), Agent("TARGET")))
        assert c.subject_is_mirna is True
        assert c.axis == "amount"


# ---------------------------------------------------------------------------
# Every produced commitment must satisfy the semantic (axis, sign) pairing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("factory", [row[0] for row in INDRA_COVERAGE],
                         ids=[row[1] for row in INDRA_COVERAGE])
def test_produced_commitment_validates(factory):
    """parse_claim must never produce an invalid (axis, sign) pairing."""
    # If the commitment were invalid, ClaimCommitment.__post_init__ would raise.
    # This test exists to make that expectation explicit for every type.
    c = parse_claim(factory())
    assert c is not None
