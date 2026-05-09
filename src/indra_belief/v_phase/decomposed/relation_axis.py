"""Decomposed relation_axis curator — three-step sub-question chain.

The single-shot V6g+curator-prompt-v2 path collapses three orthogonal
decisions (relation-presence, axis, sign) into one categorical pick. On
gemini-3.1-pro-preview it produced:

  direct_sign_match (n=273):    P=0.912 R=0.853 F1=0.891
  no_relation       (n=53):     P=0.527 R=0.736 F1=0.614  (over-predicted)
  direct_axis_mismatch (n=50):  P=0.558 R=0.580 F1=0.569
  direct_sign_mismatch (n=16):  P=0.600 R=0.188 F1=0.286  (stuck minority)

Hypothesis: when the entities and axis match, the model defaults to
`direct_sign_match` and treats sign-flipping as too noisy to commit to.
Decomposing into narrower sub-questions:

  Q1 relation_present  — does the evidence link the claim's subject and
                          object (under any alias) at all?
       → {yes, no, via_intermediate}
  Q2 evidence_axis     — what axis is the asserted relation on?
       → {modification, activity, amount, binding, localization,
          gtp_state, conversion}
  Q3 evidence_sign     — what sign is the asserted relation?
       → {positive, negative, neutral}

Claim axis and claim sign are computed deterministically from
`stmt_type` via `derive_axis` / `derive_sign` in `decomposed_curator`.

Assembly logic (deterministic):
  if relation_present == "no":           return "no_relation"
  if relation_present == "via_intermediate": return "direct_sign_match"
  if evidence_axis  != claim_axis:       return "direct_axis_mismatch"
  if claim_sign     == "neutral":        return "direct_sign_match"
  if evidence_sign  != claim_sign:       return "direct_sign_mismatch"
  return "direct_sign_match"

Notes on the via_intermediate→direct_sign_match mapping: the curator-v2
gold-alignment treats indirect-but-asserted relations between the
claim's entities as a match (because the U2 gold tag set has no
`via_mediator` anymore — those records were re-tagged to `correct` or
`wrong_relation` during the audit, see research/v6g_gold_audit.md).
This mirrors the single-shot prompt's permissive direct_sign_match
default for "X via Y mediates Z" patterns.

Sub-question prompts use synthetic placeholder names per the
contamination guard at memory/feedback_fewshot_contamination.md.

Per-record cost: usually 3 sub-calls (relation_present + axis + sign).
Records that skip later steps (no_relation, axis-mismatch, neutral
claim) cost 1-2 sub-calls. Average ~2.3 calls/record on U2 mix.
"""
from __future__ import annotations

from typing import Any

from indra_belief.v_phase.decomposed_curator import (
    CallSubQ,
    derive_axis,
    derive_sign,
    register,
)


# ── Q1: relation_present ─────────────────────────────────────────────

_RELATION_PRESENT_SYSTEM = """\
You decide whether an evidence sentence describes a relation between \
the CLAIM SUBJECT and the CLAIM OBJECT (under any alias). You are NOT \
deciding what kind of relation it is, what axis or sign — only \
whether such a relation is asserted.

★★★ DEFAULT to `yes` whenever a verb, nominalization, or a binding \
phrase connects the claim's subject and object (or their aliases) in \
the SAME clause or in a clause that the claim entities both anchor. \
The bar for `yes` is LOW: declarative ("X activates Y"), passive ("Y \
is phosphorylated by X"), nominalization ("X-mediated Y activation"), \
binding/complex ("X-Y interaction", "Y binds X"), and presupposition \
("the X-Y relationship is required for...") all count as `yes`. ★★★

★★★ ALIAS TOLERANCE: family names (ERK ↔ MAPK1/2, AKT ↔ PKB, NFkB ↔ \
NFKB1/RELA), aliases (p53 ↔ TP53, TRAIL ↔ TNFSF10), parenthetical \
expansions, and surface-form variations ALL count as mentions. Match \
on aliases — do NOT pick `no` because surface forms differ. ★★★

Answer ONE of:
  yes              — the evidence directly asserts a relation between \
    the claim's subject and the claim's object (or their aliases). \
    INCLUDES: passive voice, nominalization, symmetric binding in any \
    word order, embedded "X-Y" relation in pathway descriptions.
  via_intermediate — the evidence asserts a relation between subject \
    and object, but ONLY through a NAMED intermediate (X → Z → Y, \
    "X-mediated Z action induces Y", "via Z, X regulates Y"). The \
    relation IS present but indirect.
  no               — both entities are mentioned (or one/both are \
    absent), but no relation between them is asserted: \
      • parallel-list/co-mention positions ("X, Y, and Z are induced") \
        with no governing verb between X and Y; \
      • methods/setup descriptions ("To investigate the X-Y \
        interaction we transfected..."); \
      • loading-control mentions ("levels normalized to Y"); \
      • genetic-association statements ("variation in X is associated \
        with Y responsiveness"); \
      • one or both entities literally absent from the sentence and \
        no alias appears.

CRITICAL — relation must connect THE CLAIM'S subject and object. If \
the evidence asserts X-W and Y-Z but the claim is X-Y, that is `no`.

Output ONE JSON object: {"answer": <yes|no|via_intermediate>, \
"rationale": <short_phrase>}. The "rationale" is a 5-15 word phrase \
quoting the relevant words. NO prose outside the JSON."""


_RELATION_PRESENT_FEW_SHOTS: list[tuple[str, str]] = [
    # yes — declarative active voice.
    (
        "CLAIM: Activation(MAPK1, JUN)\n"
        "EVIDENCE: MAPK1 activates JUN in stimulated cells.",
        '{"answer": "yes", "rationale": "MAPK1 activates JUN — direct relation"}',
    ),
    # yes — passive voice with "by".
    (
        "CLAIM: Phosphorylation(KinaseA, TargetB)\n"
        "EVIDENCE: TargetB is phosphorylated by KinaseA in resting cells.",
        '{"answer": "yes", "rationale": "TargetB phosphorylated by KinaseA — passive, relation asserted"}',
    ),
    # yes — symmetric binding in reverse order.
    (
        "CLAIM: Complex(EntityP, EntityQ)\n"
        "EVIDENCE: The EntityQ-EntityP interaction is essential for complex assembly.",
        '{"answer": "yes", "rationale": "EntityP-EntityQ interaction asserted, symmetric binding"}',
    ),
    # yes — nominalization.
    (
        "CLAIM: Activation(KinaseA, TargetB)\n"
        "EVIDENCE: KinaseA-mediated activation of TargetB drives the response.",
        '{"answer": "yes", "rationale": "KinaseA-mediated activation of TargetB — nominalized relation"}',
    ),
    # yes — alias tolerance: claim uses canonical, evidence uses synonym.
    (
        "CLAIM: Phosphorylation(MAPK1, JUN)\n"
        "EVIDENCE: ERK2 phosphorylates c-Jun at Ser63 in stimulated cells.",
        '{"answer": "yes", "rationale": "ERK2 (MAPK1) phosphorylates c-Jun (JUN) — alias resolution"}',
    ),
    # via_intermediate — named mediator chain.
    (
        "CLAIM: Activation(ReceptorR, GeneZ)\n"
        "EVIDENCE: ReceptorR activates KinaseS, which then activates GeneZ.",
        '{"answer": "via_intermediate", "rationale": "ReceptorR -> KinaseS -> GeneZ — named intermediate KinaseS"}',
    ),
    # via_intermediate — "X-mediated Z action on Y".
    (
        "CLAIM: Inhibition(FactorR, TargetA)\n"
        "EVIDENCE: AdaptorP is recruited by FactorR to suppress TargetA activity.",
        '{"answer": "via_intermediate", "rationale": "FactorR recruits AdaptorP to suppress TargetA — chain via AdaptorP"}',
    ),
    # no — parallel-list co-mention.
    (
        "CLAIM: Activation(SignalK, GeneZ)\n"
        "EVIDENCE: SignalK, GeneZ, and KinaseN are all elevated in disease tissue.",
        '{"answer": "no", "rationale": "parallel list of elevated factors, no relation between SignalK and GeneZ asserted"}',
    ),
    # no — methods/setup only.
    (
        "CLAIM: Complex(GeneZ, KinaseN)\n"
        "EVIDENCE: To investigate the interaction of KinaseN with GeneZ in vivo, "
        "FLAG-tagged KinaseN was transfected with GeneZ.",
        '{"answer": "no", "rationale": "methods setup describing transfection, no relation result asserted"}',
    ),
    # no — claim entities not present (no alias either).
    (
        "CLAIM: Activation(EntityA, EntityB)\n"
        "EVIDENCE: ReceptorR activates KinaseS which then induces GeneZ expression.",
        '{"answer": "no", "rationale": "EntityA and EntityB not in evidence, no aliases visible"}',
    ),
    # no — loading control.
    (
        "CLAIM: Activation(KinaseN, ProteinM)\n"
        "EVIDENCE: KinaseN levels were normalized to ProteinM by Western blot.",
        '{"answer": "no", "rationale": "ProteinM is loading control, no relation asserted"}',
    ),
]


# ── Q2: evidence_axis ────────────────────────────────────────────────

_EVIDENCE_AXIS_SYSTEM = """\
You identify the AXIS of the relation that the evidence sentence \
asserts between the CLAIM SUBJECT and the CLAIM OBJECT (or their \
aliases). You have already established that a relation IS asserted; \
your job is to pick the axis the verb/nominalization is on.

★★★ Look at the actual verb or noun connecting the claim's subject \
and object in the sentence. Ignore decorative descriptors (cell type, \
mechanism-of-action notes) — focus on the linking action. ★★★

Axes (use the lowercase keyword exactly):
  modification — phosphorylation, dephosphorylation, methylation, \
    acetylation, ubiquitination, sumoylation, hydroxylation, \
    glycosylation, palmitoylation, etc. Verbs/nouns: \
    "phosphorylates", "is phosphorylated by", "phosphorylation of", \
    "ubiquitinates", "acetylated by".

  activity — activation/inhibition of a protein's catalytic or \
    signaling activity. Verbs/nouns: "activates", "inhibits", \
    "stimulates activity of", "suppresses activity", \
    "activator/inhibitor of", "is required for ... signaling".

  amount — change in expression / abundance / level / induction / \
    mRNA or protein level. Verbs/nouns: "induces", "induced", \
    "upregulates", "downregulates", "increased ... mRNA", \
    "decreased ... protein levels", "overexpression of X increased \
    Y", "X-mediated degradation of Y", "depletion of X decreased Y", \
    "expression of X by Y", "secretion of Y by X", "X promotes Y \
    degradation".

  binding — protein-protein binding / complex formation. Verbs/nouns: \
    "binds", "associates with", "interacts with", "X-Y complex", \
    "interaction between X and Y", "co-immunoprecipitates with". \
    Use this for protein-protein binding only — TF binding to DNA or \
    binding to lipids belongs under modification/amount/conversion as \
    appropriate.

  localization — change in subcellular location. Verbs/nouns: \
    "translocates", "exports to", "imports into nucleus", \
    "translocation of X to Y compartment".

  gtp_state — GEF/GAP regulation of GTP-binding state. Verbs/nouns: \
    "GTP-loads X", "GAP for X", "promotes GDP exchange on X".

  conversion — chemical conversion of one substance to another. \
    Verbs/nouns: "converts X to Y", "metabolizes X to Y".

★★★ DISAMBIGUATION RULES (memorize these): ★★★

Rule A — AMOUNT BEATS ACTIVITY when the verb is amount-axis. If the \
evidence reads "X induces Y", "X overexpression increased Y mRNA", \
"X-mediated degradation of Y", "depletion of X decreased Y", \
"X promotes Y secretion / Y release / Y expression", the axis is \
`amount`, NOT `activity`. Pick `amount` even if you can imagine an \
activity-axis interpretation.

Rule B — MODIFICATION BEATS ACTIVITY when the verb is a modification \
verb. "X phosphorylates Y" is `modification`, not `activity`, even \
though phosphorylation often regulates activity. Pick `modification` \
when the verb is the modification itself.

Rule C — TF BINDING TO PROMOTER / DNA / RNA is NOT `binding`. If the \
evidence describes "X binds the Y promoter" or "X binds the Y gene", \
that is functionally an amount/regulatory event — pick the axis the \
larger sentence implies (often `amount` if the consequence is \
expression). When the only verb is "binds the promoter" with no \
expression consequence, prefer `amount` (this matches U2 gold's \
treatment of TF-DNA evidence under Complex claims).

Rule D — LOSS-OF-FUNCTION OPERATIONS are amount-axis when the verb \
that follows them is amount-axis. "X-knockdown reduced Y mRNA" → \
amount. "X-RNAi inhibited Y expression" → amount. The knockdown \
itself doesn't create a new axis; the OUTCOME-VERB does.

Rule E — DEGRADATION verbs are amount-axis. "X degrades Y", \
"X-mediated degradation of Y", "X promotes Y proteolysis" all → \
`amount` (they describe an abundance change).

Output ONE JSON object: {"answer": <one_of_above>, "rationale": \
<short_phrase>}. The "rationale" is a 5-15 word phrase quoting the \
linking verb. NO prose outside the JSON."""


_EVIDENCE_AXIS_FEW_SHOTS: list[tuple[str, str]] = [
    # modification.
    (
        "CLAIM: Activation(KinaseA, TargetB)\n"
        "EVIDENCE: KinaseA phosphorylates TargetB at Ser63 in stimulated cells.",
        '{"answer": "modification", "rationale": "KinaseA phosphorylates TargetB — modification verb"}',
    ),
    # activity (clean).
    (
        "CLAIM: Activation(MAPK1, JUN)\n"
        "EVIDENCE: MAPK1 activates JUN in stimulated cells.",
        '{"answer": "activity", "rationale": "activates — activity verb"}',
    ),
    # activity — inhibits.
    (
        "CLAIM: Activation(FactorR, TargetA)\n"
        "EVIDENCE: FactorR inhibits TargetA via its kinase domain.",
        '{"answer": "activity", "rationale": "inhibits — activity verb (mechanism descriptor ignored)"}',
    ),
    # amount — Rule A: induction.
    (
        "CLAIM: Activation(SignalK, CytokineY)\n"
        "EVIDENCE: SignalK induced the production of CytokineY in primary monocytes.",
        '{"answer": "amount", "rationale": "induced production of CytokineY — amount-axis induction"}',
    ),
    # amount — Rule A: mRNA/protein level change.
    (
        "CLAIM: Activation(FactorR, GeneZ)\n"
        "EVIDENCE: FactorR overexpression increased GeneZ mRNA and protein levels.",
        '{"answer": "amount", "rationale": "increased GeneZ mRNA and protein levels — amount-axis"}',
    ),
    # amount — Rule D: knockdown + amount outcome verb.
    (
        "CLAIM: DecreaseAmount(EnzymeM, CytokineY)\n"
        "EVIDENCE: Lentiviral EnzymeM RNAi inhibited CytokineY expression in macrophages.",
        '{"answer": "amount", "rationale": "RNAi inhibited CytokineY expression — outcome verb is amount"}',
    ),
    # amount — Rule E: degradation.
    (
        "CLAIM: Inhibition(KinaseN, ProteinM)\n"
        "EVIDENCE: KinaseN-mediated degradation of ProteinM is required for the apoptotic response.",
        '{"answer": "amount", "rationale": "KinaseN-mediated degradation of ProteinM — amount-axis (Rule E)"}',
    ),
    # binding — protein-protein.
    (
        "CLAIM: Complex(EntityP, EntityQ)\n"
        "EVIDENCE: The EntityQ-EntityP interaction is essential for complex assembly at the membrane.",
        '{"answer": "binding", "rationale": "EntityP-EntityQ interaction — protein-protein binding"}',
    ),
    # binding — symmetric word order.
    (
        "CLAIM: Complex(KinaseA, AdaptorP)\n"
        "EVIDENCE: AdaptorP binds KinaseA in vitro and in vivo by co-immunoprecipitation.",
        '{"answer": "binding", "rationale": "AdaptorP binds KinaseA — protein-protein binding"}',
    ),
    # NOT binding — Rule C: TF-DNA.
    (
        "CLAIM: Complex(TF_X, GeneZ)\n"
        "EVIDENCE: TF_X binds to the GeneZ proximal promoter region in EMSA assays.",
        '{"answer": "amount", "rationale": "TF_X binds GeneZ promoter — TF-DNA, not protein-protein (Rule C)"}',
    ),
    # localization.
    (
        "CLAIM: Translocation(TF_X, nucleus)\n"
        "EVIDENCE: TF_X translocates to the nucleus following stress stimulation.",
        '{"answer": "localization", "rationale": "translocates to the nucleus — localization verb"}',
    ),
    # modification — Rule B: phosphorylation under activity claim.
    (
        "CLAIM: Activation(MAPK1, JUN)\n"
        "EVIDENCE: MAPK1 phosphorylates JUN at Ser63; activity not measured.",
        '{"answer": "modification", "rationale": "phosphorylates JUN — modification verb (Rule B)"}',
    ),
]


# ── Q3: evidence_sign ────────────────────────────────────────────────

_EVIDENCE_SIGN_SYSTEM = """\
You identify the SIGN (positive, negative, or neutral) of the relation \
the evidence asserts between the CLAIM SUBJECT and the CLAIM OBJECT \
(or their aliases). You already know the relation IS asserted and \
that the axis matches the claim — your only job is to read the \
direction.

★★★ COMPUTE THE EVIDENCE SIGN by reading the polarity of the linking \
verb plus any sign-flipping operations. Loss-of-function operations \
(knockdown, RNAi, silencing, depletion, knockout, deletion, \
"lentiviral X RNAi") FLIP X's effective effect. ★★★

Sign labels:
  positive — the evidence asserts that the subject's effect on the \
    object is INCREASING / ACTIVATING / PHOSPHORYLATING (in the \
    "addition" direction). Verbs: activates, phosphorylates, \
    increases, induces, upregulates, stabilizes, promotes (in \
    abundance/activity), enhances.

  negative — the evidence asserts that the subject's effect on the \
    object is DECREASING / INHIBITING / DEPHOSPHORYLATING (in the \
    "removal" direction). Verbs: inhibits, suppresses, blocks, \
    reduces, downregulates, abrogates, dephosphorylates, deacetylates, \
    degrades, represses, antagonizes.

  neutral — the relation has no inherent direction. Use only for \
    binding/Complex evidence ("X binds Y", "X-Y interaction"), \
    localization (translocation), and conversion. Do NOT use for \
    activity/modification/amount evidence.

★★★ SIGN-FLIPPING RULES (apply BEFORE picking a label): ★★★

Rule F — LOSS-OF-FUNCTION + outcome-verb. If the evidence describes \
loss-of-function of the claim SUBJECT (knockdown, RNAi, silencing, \
depletion, knockout, deletion) followed by an outcome on the object, \
the EFFECTIVE sign of the subject is the OPPOSITE of the outcome verb:
  - "X-knockdown decreased Y" → X normally INCREASES Y → positive
  - "X-RNAi inhibited Y expression" → X normally INCREASES Y → positive
  - "X-knockout increased Y mRNA" → X normally DECREASES Y → negative
  - "loss of X results in upregulation of Y" → X DECREASES Y → negative
  - "depletion of X reduced Y" → X INCREASES Y → positive

Rule G — DOUBLE NEGATION collapses. "X abrogated downregulation of Y" \
= X INCREASES Y → positive. "X inhibited Y degradation" = X stabilizes \
Y → positive (effect is increase). "X blocks suppression of Y" = X \
INCREASES Y → positive.

Rule H — NEGATIVELY/POSITIVELY-REGULATES. "X negatively regulates Y" \
→ negative. "X positively regulates Y" → positive. "X-mediated \
downregulation of Y" → negative. "X-mediated activation of Y" → \
positive.

Rule I — SUFFICIENT vs NECESSARY. "X is required for Y activation" — \
sign of X→Y is positive (X is needed for Y to go up). "X is required \
for Y suppression" — sign of X→Y is negative (X is needed for Y to go \
down).

Rule J — neutralize on binding/translocation/conversion verbs only. \
For modification/activity/amount, you MUST pick positive or negative.

Output ONE JSON object: {"answer": <positive|negative|neutral>, \
"rationale": <short_phrase>}. The "rationale" is a 5-15 word phrase \
quoting the verb plus any flippers applied. NO prose outside JSON."""


_EVIDENCE_SIGN_FEW_SHOTS: list[tuple[str, str]] = [
    # positive — clean activity.
    (
        "CLAIM: Activation(MAPK1, JUN)\n"
        "EVIDENCE: MAPK1 activates JUN in stimulated cells.",
        '{"answer": "positive", "rationale": "activates — positive activity verb"}',
    ),
    # positive — amount, induction.
    (
        "CLAIM: IncreaseAmount(SignalK, CytokineY)\n"
        "EVIDENCE: SignalK induced the production of CytokineY in primary monocytes.",
        '{"answer": "positive", "rationale": "induced production — positive amount"}',
    ),
    # negative — clean inhibition.
    (
        "CLAIM: Inhibition(FactorR, TargetA)\n"
        "EVIDENCE: FactorR inhibits TargetA via its kinase domain.",
        '{"answer": "negative", "rationale": "inhibits — negative activity verb"}',
    ),
    # negative — amount downregulation.
    (
        "CLAIM: DecreaseAmount(FactorR, GeneZ)\n"
        "EVIDENCE: FactorR overexpression downregulates GeneZ at both mRNA and protein levels.",
        '{"answer": "negative", "rationale": "downregulates GeneZ — negative amount"}',
    ),
    # Rule F — RNAi flips sign (target is sign of normal X effect).
    (
        "CLAIM: DecreaseAmount(EnzymeM, CytokineY)\n"
        "EVIDENCE: Lentiviral EnzymeM RNAi inhibited CytokineY expression in macrophages.",
        '{"answer": "positive", "rationale": "RNAi inhibited CytokineY -> EnzymeM normally INCREASES CytokineY (Rule F)"}',
    ),
    # Rule F — knockdown decreases.
    (
        "CLAIM: IncreaseAmount(KinaseA, GeneZ)\n"
        "EVIDENCE: KinaseA-knockdown decreased GeneZ mRNA expression in HEK293 cells.",
        '{"answer": "positive", "rationale": "knockdown decreased GeneZ -> KinaseA normally INCREASES GeneZ (Rule F)"}',
    ),
    # Rule F — loss-of-X gives upregulation of Y.
    (
        "CLAIM: IncreaseAmount(TF_X, GeneZ)\n"
        "EVIDENCE: Loss of TF_X results in upregulation of GeneZ in primary cells.",
        '{"answer": "negative", "rationale": "loss of TF_X gives upregulation -> TF_X DECREASES GeneZ (Rule F)"}',
    ),
    # Rule G — double negation.
    (
        "CLAIM: Inhibition(TF_X, GeneZ)\n"
        "EVIDENCE: TF_X ectopic expression abrogated drug-induced downregulation of GeneZ.",
        '{"answer": "positive", "rationale": "abrogates downregulation -> TF_X INCREASES GeneZ (Rule G double-negation)"}',
    ),
    # Rule H — negatively regulates.
    (
        "CLAIM: Activation(KinaseA, TargetB)\n"
        "EVIDENCE: KinaseA negatively regulates TargetB activity in resting cells.",
        '{"answer": "negative", "rationale": "negatively regulates — explicit negative (Rule H)"}',
    ),
    # neutral — binding.
    (
        "CLAIM: Complex(EntityP, EntityQ)\n"
        "EVIDENCE: EntityP and EntityQ form a stable complex in vitro.",
        '{"answer": "neutral", "rationale": "form a complex — binding is neutral"}',
    ),
    # negative — modification: dephosphorylation under phosphorylation claim.
    (
        "CLAIM: Phosphorylation(EnzymeM, TargetB)\n"
        "EVIDENCE: EnzymeM dephosphorylates TargetB at Ser45 in resting cells.",
        '{"answer": "negative", "rationale": "dephosphorylates — negative modification"}',
    ),
    # positive — Rule G: stabilization via blocked degradation.
    (
        "CLAIM: IncreaseAmount(KinaseA, TargetB)\n"
        "EVIDENCE: KinaseA blocks TargetB degradation, leading to TargetB accumulation.",
        '{"answer": "positive", "rationale": "blocks degradation -> KinaseA INCREASES TargetB (Rule G)"}',
    ),
]


# ── Composition ──────────────────────────────────────────────────────


def _claim_string(stmt_type: str, subject: str | None, obj: str | None) -> str:
    return f"{stmt_type}({subject or '?'}, {obj or '?'})"


def _user_msg(rec: dict, question: str) -> str:
    """Render a self-contained user message for a sub-question."""
    claim = _claim_string(
        rec["stmt_type"], rec.get("subject"), rec.get("object")
    )
    evidence = (rec.get("evidence_text") or "").strip()
    return (
        f"CLAIM: {claim}\n"
        f"EVIDENCE: {evidence}\n"
        f"QUESTION: {question}"
    )


def _err_response(stage: str, sub_responses: dict) -> dict:
    """Format a partial-failure return per the runner contract."""
    return {
        "error": f"sub_call_failed:{stage}",
        "sub_responses": sub_responses,
    }


async def score(rec: dict, call_subq: CallSubQ) -> dict[str, Any]:
    """Decomposed relation_axis scorer.

    Chain:
      1. relation_present  ∈ {yes, no, via_intermediate}
      2. evidence_axis     ∈ {modification, activity, amount, binding,
                              localization, gtp_state, conversion}
                             (only if relation_present != "no")
      3. evidence_sign     ∈ {positive, negative, neutral}
                             (only if evidence_axis matches claim_axis
                              AND claim_sign != "neutral")

    Deterministic assembly per the module docstring.
    """
    sub_answers: dict[str, str | None] = {
        "relation_present": None,
        "evidence_axis": None,
        "evidence_sign": None,
    }
    sub_responses: dict[str, dict] = {}

    claim_axis = derive_axis(rec.get("stmt_type", ""))
    claim_sign = derive_sign(rec.get("stmt_type", ""))

    # ── Step 1: relation_present ──
    rp_resp = await call_subq(
        _RELATION_PRESENT_SYSTEM,
        _RELATION_PRESENT_FEW_SHOTS,
        _user_msg(
            rec,
            "Does the evidence describe a relation between the claim's "
            "SUBJECT and the claim's OBJECT (under any alias)?",
        ),
        ["yes", "no", "via_intermediate"],
    )
    sub_responses["relation_present"] = rp_resp
    if "error" in rp_resp:
        return _err_response("relation_present", sub_responses)
    rp = rp_resp.get("answer")
    sub_answers["relation_present"] = rp

    if rp == "no":
        return {
            "answer": "no_relation",
            "sub_answers": sub_answers,
            "sub_responses": sub_responses,
            "claim_axis": claim_axis,
            "claim_sign": claim_sign,
        }

    if rp == "via_intermediate":
        # Curator-v2 alignment: indirect-but-asserted resolves to
        # direct_sign_match (no via_mediator class in U2 gold).
        return {
            "answer": "direct_sign_match",
            "sub_answers": sub_answers,
            "sub_responses": sub_responses,
            "claim_axis": claim_axis,
            "claim_sign": claim_sign,
        }

    # rp == "yes" → continue to axis check.

    # ── Step 2: evidence_axis ──
    ax_resp = await call_subq(
        _EVIDENCE_AXIS_SYSTEM,
        _EVIDENCE_AXIS_FEW_SHOTS,
        _user_msg(
            rec,
            "What axis does the evidence relation between the claim's "
            "SUBJECT and OBJECT describe? Pick exactly one axis keyword.",
        ),
        [
            "modification",
            "activity",
            "amount",
            "binding",
            "localization",
            "gtp_state",
            "conversion",
        ],
    )
    sub_responses["evidence_axis"] = ax_resp
    if "error" in ax_resp:
        return _err_response("evidence_axis", sub_responses)
    ev_axis = ax_resp.get("answer")
    sub_answers["evidence_axis"] = ev_axis

    if ev_axis != claim_axis:
        return {
            "answer": "direct_axis_mismatch",
            "sub_answers": sub_answers,
            "sub_responses": sub_responses,
            "claim_axis": claim_axis,
            "claim_sign": claim_sign,
        }

    # Axes match. If the claim sign is neutral (binding, localization,
    # conversion, RegulateActivity/Amount), sign is not material — call
    # it a sign_match.
    if claim_sign == "neutral":
        return {
            "answer": "direct_sign_match",
            "sub_answers": sub_answers,
            "sub_responses": sub_responses,
            "claim_axis": claim_axis,
            "claim_sign": claim_sign,
        }

    # ── Step 3: evidence_sign ──
    sg_resp = await call_subq(
        _EVIDENCE_SIGN_SYSTEM,
        _EVIDENCE_SIGN_FEW_SHOTS,
        _user_msg(
            rec,
            "What sign does the evidence relation describe between the "
            "claim's SUBJECT and OBJECT? Apply loss-of-function flippers "
            "before answering.",
        ),
        ["positive", "negative", "neutral"],
    )
    sub_responses["evidence_sign"] = sg_resp
    if "error" in sg_resp:
        return _err_response("evidence_sign", sub_responses)
    ev_sign = sg_resp.get("answer")
    sub_answers["evidence_sign"] = ev_sign

    if ev_sign == "neutral":
        # The evidence-sign probe should pick neutral only for
        # binding/localization/conversion. If we're here, axes already
        # match a non-neutral claim, so a neutral evidence-sign is
        # treated as a degenerate match (no sign info to mismatch on).
        final = "direct_sign_match"
    elif ev_sign != claim_sign:
        final = "direct_sign_mismatch"
    else:
        final = "direct_sign_match"

    return {
        "answer": final,
        "sub_answers": sub_answers,
        "sub_responses": sub_responses,
        "claim_axis": claim_axis,
        "claim_sign": claim_sign,
    }


# Self-register on import.
register("relation_axis", score)
