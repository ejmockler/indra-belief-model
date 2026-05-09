"""M-phase: catalog of canonical relation surface forms.

Each RelationPattern captures one surface variant of a canonical
(axis, sign) tuple. The catalog is consulted by context_builder
during EvidenceContext construction; matches are alias-validated
(agent_text and target_text must be present in ctx.aliases) before
inclusion in ctx.detected_relations. The adjudicator (M3) consumes
ctx.detected_relations as a substrate-fallback when the parser
yields no matching assertion.

Confidence ceiling: substrate matches resolve to confidence=medium
in the adjudicator. Parser-confirmed evidence still wins at high.

Why this catalog at 27B scale: parse_evidence has high recall on ONE
canonical surface form per relation but degrades on alternates under
attention pressure. The catalog makes the alternate forms first-class
detection targets without spending parser attention budget.

Each pattern is justified by at least one diagnosis FN it must close;
each pattern has positive and negative regression tests in
tests/test_relation_patterns.py.

Per-axis catalogs are exported as module constants so the builder
can iterate them without re-introspecting via reflection. Adding a
new axis means adding a new constant + extending CATALOG below.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


# Permissive entity-name token. Captures alphanumerics with internal
# hyphens, slashes, dots, or trailing digits — covers HGNC (PLK1), FPLX
# (NFkappaB), Greek-replaced (p38), hyphenated aliases (HNP-1, eIF-4E),
# and numeric-led tokens (14-3-3, 25-OH-D3). Greek letters are
# intentionally excluded — M6 normalizes Greek↔Latin BEFORE catalog
# lookup, so the regex never sees raw Greek glyphs.
# The trailing `\w` ensures we match at least 2 chars (rejects single-
# letter captures like "a", "I" that would over-match). Validation in
# context_builder._bind_to_claim_canonical filters numeric-only captures
# (e.g., "14") since they don't bind to any claim alias.
_NAME = r"[A-Za-z\d][\w/.-]*\w"

# Modification site forms.
_SITE = (
    r"(?:[STYsty]-?\d+"            # S102, T-461, y732
    r"|[Ss]er-?\d+|[Tt]hr-?\d+|[Yy]r?-?\d+|[Tt]yr-?\d+"  # Ser123, Thr-461
    r"|serine\s+\d+|threonine\s+\d+|tyrosine\s+\d+)"
)

Axis = Literal["modification", "activity", "amount", "binding", "translocation"]
Sign = Literal["positive", "negative", "neutral"]


@dataclass(frozen=True)
class RelationPattern:
    """One canonical surface form for a (axis, sign) tuple.

    Fields:
      pattern_id: stable identifier 'axis_sign.label', for tests + telemetry
      axis: which canonical axis this maps to
      sign: which canonical sign (positive/negative/neutral)
      regex: compiled regex with named groups X, Y, optional site
      surface_form_label: human-readable example for docs/debugging

    The regex MUST have named groups 'X' (agent surface form) and 'Y'
    (target surface form); 'site' is optional and only present for
    modification patterns that capture residue.
    """
    pattern_id: str
    axis: Axis
    sign: Sign
    regex: re.Pattern
    surface_form_label: str


# Modification-verb backbone: phosphorylate, acetylate, methylate, ubiquitinate,
# sumoylate, hydroxylate, glycosylate, palmitoylate, farnesylate, prenylate,
# nitrosylate, polyubiquitinate AND their de- counterparts. The catalog does
# not distinguish which modification — claim.stmt_type already routes that.
_MOD_STEM = (
    r"(?:phosphorylat|acetylat|methylat|ubiquitinat|sumoylat|hydroxylat|"
    r"glycosylat|palmitoylat|farnesylat|prenylat|nitrosylat|polyubiquitinat|"
    r"deacetylat|demethylat|deubiquitinat|dephosphorylat)"
)
_MOD_VERB_FORM = rf"{_MOD_STEM}(?:es|ed|ing|e)?"          # phosphorylates/-ed/-ing
_MOD_NOUN_FORM = rf"{_MOD_STEM}ion"                         # phosphorylation

_ACT_STEM = r"(?:activat|stimulat|induc|trigger|elicit|provok|initiat|promot)"
_ACT_VERB_FORM = rf"{_ACT_STEM}(?:es|ed|ing|e)?"
_ACT_NOUN_FORM = r"(?:activation|stimulation|induction|promotion)"

_AMT_UP_STEM = r"(?:upregulat|up-regulat|increas|enhanc|elevat|augment|amplif|boost)"
_AMT_UP_VERB_FORM = rf"{_AMT_UP_STEM}(?:es|ed|ing|e)?"
_AMT_UP_NOUN_FORM = r"(?:upregulation|up-regulation|increase|enhancement|elevation)"

_AMT_DOWN_STEM = r"(?:downregulat|down-regulat|decreas|reduc|diminish|suppress|attenuat|inhibit)"
_AMT_DOWN_VERB_FORM = rf"{_AMT_DOWN_STEM}(?:es|ed|ing|e)?"


# ============================================================================
# modification + positive
# Diagnosis FNs targeted: PKC→EIF4E, PLK1→CDC20, CSNK2A2→BMI1, ERK→MAPK1,
# MEK→MAPK1 (also F class), TYK2→IL13RA1, and others under nominalization.
# ============================================================================
MODIFICATION_POSITIVE: list[RelationPattern] = [
    RelationPattern(
        pattern_id="mod_pos.active_verb",
        axis="modification", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+(?:kinase[s]?\s+)?{_MOD_VERB_FORM}\s+(?P<Y>{_NAME})"
            rf"(?:\s+(?:on|at)\s+(?P<site>{_SITE}))?",
            re.IGNORECASE,
        ),
        surface_form_label="X (kinase) phosphorylates Y (on Sxx)",
    ),
    RelationPattern(
        pattern_id="mod_pos.passive_by",
        axis="modification", sign="positive",
        regex=re.compile(
            rf"\b(?P<Y>{_NAME})\s+(?:was|is|were|are|gets?|got|been)\s+"
            rf"(?:{_MOD_STEM}ed)\s+by\s+(?P<X>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="Y is phosphorylated by X",
    ),
    RelationPattern(
        pattern_id="mod_pos.nominalized_by",
        axis="modification", sign="positive",
        regex=re.compile(
            rf"\b{_MOD_NOUN_FORM}\s+of\s+(?P<Y>{_NAME})"
            rf"(?:\s+(?:on|at)\s+(?P<site>{_SITE}))?\s+by\s+(?P<X>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="phosphorylation of Y (on Sxx) by X",
    ),
    RelationPattern(
        pattern_id="mod_pos.induced_adj",
        axis="modification", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})-induced\s+(?P<Y>{_NAME})\s+{_MOD_NOUN_FORM}",
            re.IGNORECASE,
        ),
        surface_form_label="X-induced Y phosphorylation",
    ),
    RelationPattern(
        pattern_id="mod_pos.mediated_adj",
        axis="modification", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})-mediated\s+(?P<Y>{_NAME})\s+{_MOD_NOUN_FORM}",
            re.IGNORECASE,
        ),
        surface_form_label="X-mediated Y phosphorylation",
    ),
    RelationPattern(
        pattern_id="mod_pos.dependent_of",
        axis="modification", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})[ -]dependent\s+{_MOD_NOUN_FORM}\s+of\s+(?P<Y>{_NAME})"
            rf"(?:\s+(?:on|at)\s+(?P<site>{_SITE}))?",
            re.IGNORECASE,
        ),
        surface_form_label="X-dependent phosphorylation of Y",
    ),
    RelationPattern(
        pattern_id="mod_pos.compound_nominal",
        axis="modification", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+{_MOD_NOUN_FORM}\s+of\s+(?P<Y>{_NAME})"
            rf"(?:\s+(?:on|at)\s+(?P<site>{_SITE}))?",
            re.IGNORECASE,
        ),
        surface_form_label="X phosphorylation of Y",
    ),
    # R2: "Y phosphorylation by X" — different word order from
    # mod_pos.nominalized_by ("phosphorylation of Y by X") and
    # mod_pos.compound_nominal ("X phosphorylation of Y").
    # Q-phase regression: PDPK1-AKT1 ("AKT1 phosphorylation by PDK1").
    RelationPattern(
        pattern_id="mod_pos.target_compound_by",
        axis="modification", sign="positive",
        regex=re.compile(
            rf"\b(?P<Y>{_NAME})\s+{_MOD_NOUN_FORM}\s+by\s+(?P<X>{_NAME})"
            rf"(?:\s+(?:on|at)\s+(?P<site>{_SITE}))?",
            re.IGNORECASE,
        ),
        surface_form_label="Y phosphorylation by X",
    ),
    # R2: "X increases/induces (the) phosphorylation of Y".
    # Q-phase regression: CENPJ-NFkappaB ("CPAP increases the
    # phosphorylation of NF-kappaB"). Permissive on optional adverb
    # between subject and verb to handle "X also increases", "X further
    # induces", etc.
    RelationPattern(
        pattern_id="mod_pos.x_increases_phos_of_y",
        axis="modification", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})"
            rf"(?:\s+(?:also|then|further|subsequently|directly|"
            rf"here|moreover|in\s+turn))?"
            rf"\s+(?:increases?|enhances?|elevates?|induces?|"
            rf"promotes?|drives?|stimulates?)\s+"
            rf"(?:the\s+)?{_MOD_NOUN_FORM}\s+of\s+(?P<Y>{_NAME})"
            rf"(?:\s+(?:on|at)\s+(?P<site>{_SITE}))?",
            re.IGNORECASE,
        ),
        surface_form_label="X increases the phosphorylation of Y",
    ),
    # R2: relative-clause variant — "X, which (in turn|then|...)
    # phosphorylates Y on Sxx". The relative clause inserts a comma +
    # "which" between subject and verb. Without this pattern, the
    # active-verb pattern fails because it requires direct \s+ between X
    # and the verb.
    # Q-phase regression: AKT-EPHA2 ("Akt, which in turn phosphorylates
    # EphA2 on S897").
    RelationPattern(
        pattern_id="mod_pos.relative_clause",
        axis="modification", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME}),?\s+which\s+"
            rf"(?:in\s+turn\s+|then\s+|subsequently\s+|directly\s+)?"
            rf"{_MOD_VERB_FORM}\s+(?P<Y>{_NAME})"
            rf"(?:\s+(?:on|at)\s+(?P<site>{_SITE}))?",
            re.IGNORECASE,
        ),
        surface_form_label="X, which (in turn) phosphorylates Y",
    ),
]


# ============================================================================
# binding + neutral (Complex)
# Diagnosis FNs: MAP3K5(ASK-1)↔PPP5C(PP5), CASP9↔APAF1, p14_3_3↔CDC25C,
# TCF_LEF↔CTNNB1, GPCR↔CCL3, TRIM14↔WRNIP1.
# ============================================================================
BINDING_NEUTRAL: list[RelationPattern] = [
    RelationPattern(
        pattern_id="bind.active_verb",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+binds?\s+(?:to\s+)?(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X binds (to) Y",
    ),
    RelationPattern(
        pattern_id="bind.binding_of_to",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\bbinding\s+of\s+(?P<X>{_NAME})\s+to\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="binding of X to Y",
    ),
    RelationPattern(
        pattern_id="bind.interacts_with",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+(?:interact[s]?|associates?|coimmunoprecipitates?|"
            rf"co-immunoprecipitates?|co-localizes?)\s+with\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X interacts/associates with Y",
    ),
    RelationPattern(
        pattern_id="bind.complex_of",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\bcomplex\s+(?:of|between)\s+(?P<X>{_NAME})\s+and\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="complex of/between X and Y",
    ),
    RelationPattern(
        pattern_id="bind.interaction_compound",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})[/\-](?P<Y>{_NAME})\s+(?:interaction|complex|binding|heterodimer)",
            re.IGNORECASE,
        ),
        surface_form_label="X-Y interaction/complex",
    ),
    RelationPattern(
        pattern_id="bind.between_and",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\binteraction\s+between\s+(?P<X>{_NAME})\s+and\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="interaction between X and Y",
    ),
    RelationPattern(
        pattern_id="bind.recruits",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+(?:recruits?|tethers?|anchors?)\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X recruits Y",
    ),
    # M11 gap-fix: gerund form "X binding to Y" (compound nominal).
    # Diagnosis: p14_3_3↔CDC25C — "14-3-3 binding to Cdc25C".
    RelationPattern(
        pattern_id="bind.gerund_to",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+binding\s+to\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X binding to Y",
    ),
    # M11 gap-fix: possessive form "X's interaction(s) with Y".
    # Diagnosis: TCF_LEF↔CTNNB1 — "beta-catenin's interactions with TCF/Lef".
    RelationPattern(
        pattern_id="bind.possessive_interaction",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})['']s\s+interactions?\s+with\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X's interaction with Y",
    ),
    # R2: "X forms (a) complex/heterodimer/dimer with Y".
    # Q-phase regression: AGO2-RAD51 ("Ago2 forms a complex with Rad51").
    RelationPattern(
        pattern_id="bind.forms_complex_with",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+forms?\s+(?:an?\s+)?"
            rf"(?:complex|heterodimer|dimer|heterotrimer)\s+with\s+"
            rf"(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X forms a complex with Y",
    ),
    # R2: coordinated "X and Y complex/heterodimer/dimer (formation)".
    # Q-phase regression: TECPR1-ATG5 ("Atg5 and TECPR1 complex formation").
    # Permissive on trailing nominalizer (formation/interaction/binding) so
    # the pattern matches both "X and Y complex" and "X and Y complex
    # formation". Restrict the connector to "and"/"with"/"-"/"/" to avoid
    # accidental coordination with unrelated tokens.
    RelationPattern(
        pattern_id="bind.coord_complex",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+(?:and|with)\s+(?P<Y>{_NAME})\s+"
            rf"(?:complex|heterodimer|dimer|heterotrimer|interaction)"
            rf"(?:\s+(?:formation|interaction|binding|formation))?",
            re.IGNORECASE,
        ),
        surface_form_label="X and Y complex (formation)",
    ),
    # R2: "binding (affinity) of X for/to Y".
    # Q-phase regression: CDKN1A-PCNA ("binding affinity of p21 for PCNA").
    # The existing bind.binding_of_to handles "binding of X to Y"; this
    # adds the "for" preposition and the "binding affinity" nominal head.
    RelationPattern(
        pattern_id="bind.binding_affinity_for",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\bbinding\s+(?:affinity\s+)?of\s+(?P<X>{_NAME})\s+"
            rf"(?:for|to)\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="binding (affinity) of X for/to Y",
    ),
    # R2: "X bound to Y" (passive participle).
    # Q-phase regression: MED17-ERCC3 ("XPB and XPG bound to hMED17").
    # The existing bind.gerund_to handles "X binding to Y" (gerund form);
    # this adds the past participle "X bound to Y".
    RelationPattern(
        pattern_id="bind.bound_to",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+bound\s+(?:tightly\s+|directly\s+|"
            rf"strongly\s+)?to\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X bound to Y",
    ),
    # R2: passive "X is/are associated with Y".
    # Q-phase regression: APOA1-PLTP ("PLTP is associated with apoA-I").
    # The existing bind.interacts_with handles "X interacts/associates
    # with Y" (active form); this adds the passive form which dominates
    # in academic writing. Includes "complexed with" since coordinated
    # "X complexed with Y" is a passive-binding construction.
    RelationPattern(
        pattern_id="bind.passive_associated",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+(?:is|are|was|were)\s+"
            rf"(?:directly\s+|stably\s+|tightly\s+|physically\s+)?"
            rf"(?:associated|complexed|conjugated|coupled|linked)\s+with\s+"
            rf"(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X is associated with Y",
    ),
    # R2: coordinated "X1 and X2 bound to Y" — captures X2 (the second
    # entity in coord). The complement pattern bind.bound_to captures X
    # in single position; this adds coverage for the second entity in
    # an "and"-coordination.
    # Q-phase regression: MED17-ERCC3 ("XPG and XPB bound to hMED17");
    # ERCC3 alias = "XPB" lives in the second coord slot.
    RelationPattern(
        pattern_id="bind.coord_bound_to",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\b\w+\s+and\s+(?P<X>{_NAME})\s+bound\s+to\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X1 and X bound to Y",
    ),
    # R2: "X (multi-word modifier) binding to Y" — same gerund as
    # bind.gerund_to but with an optional modifier word between X and
    # "binding". Catches "14-3-3 protein binding to Par1b" where
    # _NAME captures "14-3-3" and "protein" sits between as filler.
    # Q-phase regression: p14_3_3-MARK2.
    RelationPattern(
        pattern_id="bind.gerund_modifier_to",
        axis="binding", sign="neutral",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+(?:protein\s+|family\s+|complex\s+)"
            rf"binding\s+to\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X (protein|family|complex) binding to Y",
    ),
]


# ============================================================================
# activity + positive (Activation)
# Diagnosis FNs: NOTCH1→NFkappaB (chain), DEFA1→NFkappaB, FGF→ERK,
# HGF→DGKA, TCR→IL13 (G-class), TGFB1→p38 (also F class via -induced).
# ============================================================================
ACTIVITY_POSITIVE: list[RelationPattern] = [
    RelationPattern(
        pattern_id="act_pos.active_verb",
        axis="activity", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+{_ACT_VERB_FORM}\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X activates/stimulates/induces/triggers Y",
    ),
    RelationPattern(
        pattern_id="act_pos.induced_adj",
        axis="activity", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})-induced\s+(?P<Y>{_NAME})\s+{_ACT_NOUN_FORM}",
            re.IGNORECASE,
        ),
        surface_form_label="X-induced Y activation",
    ),
    RelationPattern(
        pattern_id="act_pos.mediated_adj",
        axis="activity", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})-mediated\s+(?P<Y>{_NAME})\s+{_ACT_NOUN_FORM}",
            re.IGNORECASE,
        ),
        surface_form_label="X-mediated Y activation",
    ),
    RelationPattern(
        pattern_id="act_pos.nominalized_by",
        axis="activity", sign="positive",
        regex=re.compile(
            rf"\b{_ACT_NOUN_FORM}\s+of\s+(?P<Y>{_NAME})\s+by\s+(?P<X>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="activation of Y by X",
    ),
    RelationPattern(
        pattern_id="act_pos.compound_nominal",
        axis="activity", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+{_ACT_NOUN_FORM}\s+of\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X activation of Y",
    ),
    RelationPattern(
        pattern_id="act_pos.triggered_act",
        axis="activity", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+(?:triggered|induced|stimulated|drives|drove)\s+"
            rf"(?:the\s+)?{_ACT_NOUN_FORM}\s+of\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X triggered the activation of Y",
    ),
    # R2: "X-induced activation of Y (and Z, ...)" — coordinated targets.
    # The existing act_pos.induced_adj matches "X-induced Y activation"
    # (Y before noun). This adds the nominalized form "X-induced
    # activation of Y" which dominates in cytokine/ligand passages, and
    # is permissive on a trailing coord so Y can appear in either
    # position of "of A and B".
    # Q-phase regressions: IGF1-SHC ("IGF-I-induced activation of IRS-1,
    # Shc, PI3K, and MAPK"); TNFSF10-CASP9 ("TRAIL-induced activation
    # of caspase-3 and caspase-9").
    RelationPattern(
        pattern_id="act_pos.x_induced_act_of_y",
        axis="activity", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})-(?:induced|mediated|driven|dependent|"
            rf"stimulated|triggered)\s+{_ACT_NOUN_FORM}\s+of\s+"
            rf"(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X-induced activation of Y",
    ),
    RelationPattern(
        pattern_id="act_pos.x_induced_act_of_coord_last",
        axis="activity", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})-(?:induced|mediated|driven|dependent|"
            rf"stimulated|triggered)\s+{_ACT_NOUN_FORM}\s+of\s+"
            rf"[\w-]+(?:\s*,\s*[\w-]+){{0,3}}"
            rf"\s+(?:and|or)\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X-induced activation of A, B, and Y (Y last)",
    ),
    # R2: "X induced (the) release/secretion/production of Y" — Q-phase
    # regression TCR-IL4 ("TCR induced release of TNF-alpha, IL-4,
    # IL-13 and IL-10"). Captures Y as either first or any position in
    # the coord; the simpler form here covers Y as first slot.
    RelationPattern(
        pattern_id="act_pos.x_induced_release_of",
        axis="activity", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})(?:-induced|\s+induced|\s+stimulates?|"
            rf"\s+stimulated|\s+triggered)\s+(?:the\s+)?"
            rf"(?:release|secretion|production)\s+of\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X induced (the) release of Y",
    ),
    RelationPattern(
        pattern_id="act_pos.x_induced_release_coord_last",
        axis="activity", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})(?:-induced|\s+induced|\s+stimulates?|"
            rf"\s+stimulated|\s+triggered)\s+(?:the\s+)?"
            rf"(?:release|secretion|production)\s+of\s+"
            rf"[\w-]+(?:\s*,\s*[\w-]+){{0,3}}"
            rf"\s+(?:and|or)\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X induced release of A, B, and Y (Y last)",
    ),
]


# ============================================================================
# amount + positive (IncreaseAmount)
# Diagnosis FNs: NFkappaB→LCN2 (also D class), IL4→IL4R, TCR→IL13 (axis G).
# ============================================================================
AMOUNT_POSITIVE: list[RelationPattern] = [
    RelationPattern(
        pattern_id="amt_pos.upregulates",
        axis="amount", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+(?:upregulates?|up-regulates?|increases?|enhances?|"
            rf"elevates?|augments?|amplifies?|boosts?)\s+(?P<Y>{_NAME})"
            rf"(?:\s+(?:expression|levels|amount[s]?|production|abundance|protein))?",
            re.IGNORECASE,
        ),
        surface_form_label="X upregulates/increases Y (expression)",
    ),
    RelationPattern(
        pattern_id="amt_pos.induces_expression",
        axis="amount", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+induces?\s+(?P<Y>{_NAME})\s+"
            rf"(?:expression|production|levels|protein)",
            re.IGNORECASE,
        ),
        surface_form_label="X induces Y expression",
    ),
    RelationPattern(
        pattern_id="amt_pos.passive_upregulated",
        axis="amount", sign="positive",
        regex=re.compile(
            rf"\b(?P<Y>{_NAME})\s+(?:expression|levels?|amount[s]?|production)?"
            rf"\s*(?:is|was|are|were)\s+"
            rf"(?:upregulated|up-regulated|increased|elevated|enhanced|"
            rf"induced|augmented|elevated)\s+by\s+(?P<X>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="Y (expression) is upregulated by X",
    ),
    RelationPattern(
        pattern_id="amt_pos.induced_adj",
        axis="amount", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})-induced\s+(?P<Y>{_NAME})\s+(?:expression|production|levels?)",
            re.IGNORECASE,
        ),
        surface_form_label="X-induced Y expression",
    ),
    RelationPattern(
        pattern_id="amt_pos.required_to_elevate",
        axis="amount", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME}).{{0,40}}required\s+to\s+(?:elevate|increase|induce|"
            rf"upregulate|enhance)\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X required to elevate Y",
    ),
    # R2: "X induces (the) expression of Y" — different word order
    # from amt_pos.induces_expression ("X induces Y expression"). Allows
    # an optional adverb between subject and verb.
    # Q-phase regression: TP53-GLS2 ("p53 also induces the expression
    # of GLS2"); CENPJ-NFkappaB shape via mod variant above.
    RelationPattern(
        pattern_id="amt_pos.x_induces_expression_of_y",
        axis="amount", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})"
            rf"(?:\s+(?:also|then|further|subsequently|directly|"
            rf"here|moreover|in\s+turn))?"
            rf"\s+(?:induces?|increases?|enhances?|elevates?|"
            rf"upregulates?|up-regulates?|drives?|promotes?)\s+"
            rf"(?:the\s+)?(?:expression|production|levels?|abundance|"
            rf"transcription)\s+of\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X induces the expression of Y",
    ),
    # R2: coordinated targets — "X induces Y and Z (...) expression"
    # captures Y at the start of the coord; the symmetric sibling
    # pattern below captures Y at the end of the coord.
    # Q-phase regression: TNF-ADAMTS12 ("TNF induces ADAMTS-7 and
    # ADAMTS-12 expression").
    RelationPattern(
        pattern_id="amt_pos.coord_first",
        axis="amount", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+(?:induces?|increases?|enhances?|"
            rf"upregulates?|stimulates?|elevates?)\s+"
            rf"(?P<Y>{_NAME})"
            rf"(?:\s*,\s*[\w-]+){{0,3}}"
            rf"\s+(?:and|or)\s+[\w-]+"
            rf"\s+(?:expression|production|levels?|abundance|secretion|release)",
            re.IGNORECASE,
        ),
        surface_form_label="X induces Y and Z expression (Y first)",
    ),
    RelationPattern(
        pattern_id="amt_pos.coord_last",
        axis="amount", sign="positive",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+(?:induces?|increases?|enhances?|"
            rf"upregulates?|stimulates?|elevates?)\s+"
            rf"[\w-]+(?:\s*,\s*[\w-]+){{0,3}}"
            rf"\s+(?:and|or)\s+(?P<Y>{_NAME})"
            rf"\s+(?:expression|production|levels?|abundance|secretion|release)",
            re.IGNORECASE,
        ),
        surface_form_label="X induces A and Y expression (Y last)",
    ),
]


# ============================================================================
# U-phase U8: ACTIVITY_NEGATIVE — semantic-equivalent verb taxonomy
# (Intervention §3.6). Closes axis_mismatch FNs where the LLM probe
# rejected "X decreases activation of Y" / "X inhibits Y signaling" as
# wrong-axis despite the verb compound being semantically equivalent
# to Inhibition. Adding the substrate match short-circuits the LLM
# call: router emits direct_sign_match (negative axis) at substrate
# tier for these CATALOG entries.
#
# Stems chosen to NOT collide with AMOUNT_DOWN (which already has
# decreas/reduc/diminish/suppress/attenuat/inhibit). The discriminator
# is the OBJECT PHRASE: amount-down patterns target "Y expression /
# Y mRNA / Y protein levels"; activity-negative targets "Y activation
# / Y signaling / Y activity". The downstream noun is what makes the
# axis explicit; verbs alone are ambiguous.
# ============================================================================
_ACT_NEG_VERB_STEM = r"(?:decreas|reduc|inhibit|abolish|suppress|diminish|attenuat|block)"
# Suffix set covers present-tense -s ("inhibits", "blocks"), -es ("reduces",
# "abolishes", "suppresses"), past/participle -ed ("inhibited"), gerund -ing
# ("inhibiting"), and bare -e for "decreas-e", "reduc-e" base forms.
_ACT_NEG_VERB_FORM = rf"{_ACT_NEG_VERB_STEM}(?:s|es|ed|ing|e)?"
_ACT_NEG_NOUN_FORM = (
    r"(?:decrease|reduction|inhibition|suppression|attenuation|blockade)"
)

ACTIVITY_NEGATIVE: list[RelationPattern] = [
    RelationPattern(
        # "X decreases the activation of Y" / "X inhibits Y signaling"
        pattern_id="act_neg.verb_activation_of",
        axis="activity", sign="negative",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+{_ACT_NEG_VERB_FORM}\s+(?:the\s+)?"
            rf"(?:activation|signaling|activity)\s+of\s+(?P<Y>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="X decreases/inhibits the activation of Y",
    ),
    RelationPattern(
        # "X decreases Y activation" / "X reduces Y signaling"
        pattern_id="act_neg.verb_y_activation",
        axis="activity", sign="negative",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})\s+{_ACT_NEG_VERB_FORM}\s+(?P<Y>{_NAME})\s+"
            rf"(?:activation|signaling|activity)",
            re.IGNORECASE,
        ),
        surface_form_label="X decreases/inhibits Y activation",
    ),
    RelationPattern(
        # "decrease in Y activation by X" / "reduction in Y signaling by X"
        pattern_id="act_neg.nominalized_by",
        axis="activity", sign="negative",
        regex=re.compile(
            rf"\b{_ACT_NEG_NOUN_FORM}\s+(?:in|of)\s+(?P<Y>{_NAME})\s+"
            rf"(?:activation|signaling|activity)\s+by\s+(?P<X>{_NAME})",
            re.IGNORECASE,
        ),
        surface_form_label="decrease/reduction of Y activation by X",
    ),
    RelationPattern(
        # "X-mediated decrease in Y activation"
        pattern_id="act_neg.mediated_decrease",
        axis="activity", sign="negative",
        regex=re.compile(
            rf"\b(?P<X>{_NAME})-mediated\s+{_ACT_NEG_NOUN_FORM}\s+"
            rf"(?:in|of)\s+(?P<Y>{_NAME})\s+(?:activation|signaling|activity)",
            re.IGNORECASE,
        ),
        surface_form_label="X-mediated decrease/inhibition of Y activation",
    ),
]


# ============================================================================
# CATALOG — ordered iteration target for context_builder.
# Adding a new (axis, sign) catalog: append the constant here.
# ============================================================================
CATALOG: tuple[RelationPattern, ...] = tuple(
    p for group in (
        MODIFICATION_POSITIVE,
        BINDING_NEUTRAL,
        ACTIVITY_POSITIVE,
        ACTIVITY_NEGATIVE,
        AMOUNT_POSITIVE,
    ) for p in group
)


def iter_patterns(axis: Axis | None = None,
                  sign: Sign | None = None) -> list[RelationPattern]:
    """Filter the catalog by axis/sign. Returns the full catalog when
    both args are None — used by the builder to scan everything."""
    out = list(CATALOG)
    if axis is not None:
        out = [p for p in out if p.axis == axis]
    if sign is not None:
        out = [p for p in out if p.sign == sign]
    return out
