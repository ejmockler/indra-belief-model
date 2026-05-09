"""relation_axis probe — does the evidence describe a relation between
the claim's subject and object on the claim's axis with claim's sign?

Closed answer set (8 values; see doctrine §2.3):
  direct_sign_match, direct_sign_mismatch, direct_axis_mismatch,
  direct_partner_mismatch, via_mediator, via_mediator_partial,
  no_relation, abstain.

Substrate fast-path (router) handles CATALOG-aligned cases. The LLM
escalation handles nominalization, cross-sentence aggregation, chain
disambiguation, and partner-type checks for binding axis.

Few-shot curriculum covers the 8 answer values with one exemplar each
(some shots may double-up to reduce prompt length while preserving
discriminative coverage).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from indra_belief.scorers.probes._llm import llm_classify
from indra_belief.scorers.probes.types import ProbeRequest, ProbeResponse

if TYPE_CHECKING:
    from indra_belief.model_client import ModelClient


_ANSWER_SET = frozenset({
    "direct_sign_match",
    "direct_amount_match",     # U7: amount-axis explicit match
    "direct_sign_mismatch",
    "direct_axis_mismatch",
    "direct_partner_mismatch",
    "via_mediator",
    "via_mediator_partial",
    "no_relation",
    # T-phase Fix A: "abstain" removed. The probe must commit to one of
    # the eight substantive labels. Underdetermined evidence projects to
    # "no_relation" via failure_default in llm_classify().
})


_SYSTEM_PROMPT = """\
You classify whether an evidence sentence describes a relation between \
two named entities on a specified AXIS with a specified SIGN.

The CLAIM is given as: subject, object, axis, sign. Axes are: \
modification (phosphorylation, methylation, etc.), activity \
(activation, inhibition), amount (increase, decrease in expression), \
binding (Complex/protein-protein), localization (translocation). \
Signs: positive, negative, neutral (binding/translocation).

CRITICAL — clause-localized evaluation: Evaluate the relation between \
the claim's SUBJECT and OBJECT using ONLY the clause(s) where they \
co-occur or are connected by the relation verb. If the same sentence \
contains a different clause whose negation or wrong-axis verb does \
NOT govern the claim's subject-object pair, IGNORE that clause. \
Negation of a sibling proposition does not propagate to the claim. \
Sign and axis labels apply to the claim-relevant span, not the \
sentence as a whole.

DO NOT use direct_axis_mismatch when the claim's axis verb appears \
within the claim-relevant clause. The axis_mismatch label is for the \
narrow case where the evidence ONLY describes a different axis between \
these entities (e.g., the evidence describes phosphorylation but the \
claim is activity-axis, AND no activity verb is in the relevant span). \
If both axes appear (the claim's axis AND another axis), use \
direct_sign_match (or direct_amount_match) on the claim's matching axis.

Answer ONE of:
  direct_sign_match     — the evidence directly asserts the SAME \
    axis+sign between subject and object. (Sign may have been inverted \
    upstream by perturbation propagation; trust the claim's stated sign.) \
    Use this for activity, modification, binding, localization, conversion, \
    and gtp_state axes.
  direct_amount_match   — the evidence describes EXPRESSION/ABUNDANCE \
    change of the object by the subject (upregulation, downregulation, \
    increased/decreased protein/mRNA levels, transcriptional regulation). \
    Use this when the evidence verbs are amount-axis, REGARDLESS of the \
    claim axis. The adjudicator decides whether this is a match for the \
    claim's stated axis. Examples: "X overexpression increased Y mRNA", \
    "Y protein levels were reduced by X", "X induced Y expression".
  direct_sign_mismatch  — the evidence asserts the same axis but \
    OPPOSITE sign (claim says inhibits, evidence says activates).
  direct_axis_mismatch  — the evidence asserts a relation on a \
    DIFFERENT axis from BOTH the claim AND amount (claim says Activation, \
    evidence says Phosphorylation only — neither activity nor amount).
  direct_partner_mismatch — binding axis only: the evidence asserts \
    binding but to a non-protein partner (DNA/RNA/lipid) when claim \
    is protein-protein Complex.
  via_mediator          — the relation between subject and object is \
    INDIRECT, mediated by a named intermediate (X -> Z -> Y).
  via_mediator_partial  — the sentence carries chain markers (thereby, \
    leads to, mediated by) but no named intermediate is extractable.
  no_relation           — both entities are mentioned but no relation \
    between them is asserted. Use this ONLY when the entities co-occur \
    without any asserted relationship — e.g., one is mentioned in a \
    list, methods description, or unrelated clause. If the relation IS \
    asserted (even briefly, hedged, in a longer description, or as part \
    of a chain), use the appropriate direct_*_match or via_mediator label.

You MUST pick one of the eight labels above. There is no "abstain" \
option. Decision priority:
  1. If the relation between subject and object is explicitly asserted \
     (with appropriate axis verb), use direct_sign_match (or \
     direct_amount_match for expression-axis evidence).
  2. If the relation is asserted but with a chain mediator, use \
     via_mediator (if mediator is named) or via_mediator_partial.
  3. If the relation appears with the OPPOSITE sign, use \
     direct_sign_mismatch.
  4. If the evidence describes a different axis verb between the \
     entities (e.g., phosphorylation when claim is activity), use \
     direct_axis_mismatch.
  5. If both entities are mentioned but NO relation is asserted between \
     them, use no_relation.

When uncertain between via_mediator and via_mediator_partial, prefer \
via_mediator_partial.

Output ONE JSON object: {"answer": <one_of_above>, "rationale": <short_phrase>}.
The "rationale" is a 5-15 word phrase quoting the relevant words. NO \
prose outside the JSON."""


_FEW_SHOTS: list[tuple[str, str]] = [
    (
        "CLAIM: subject=MAPK1, object=JUN, axis=activity, sign=positive\n"
        "EVIDENCE: MAPK1 activates JUN in stimulated cells.",
        '{"answer": "direct_sign_match", '
        '"rationale": "MAPK1 activates JUN — direct activation"}',
    ),
    (
        "CLAIM: subject=KinaseA, object=ProteinB, axis=activity, sign=positive\n"
        "EVIDENCE: KinaseA controls cellular response via the activation "
        "of ProteinB, which then drives downstream effects.",
        '{"answer": "direct_sign_match", '
        '"rationale": "via the activation of ProteinB — direct activation (nominalization)"}',
    ),
    (
        "CLAIM: subject=MAPK1, object=JUN, axis=activity, sign=positive\n"
        "EVIDENCE: MAPK1 inhibits JUN at high concentrations.",
        '{"answer": "direct_sign_mismatch", '
        '"rationale": "MAPK1 INHIBITS JUN — opposite sign"}',
    ),
    (
        "CLAIM: subject=MAPK1, object=JUN, axis=activity, sign=positive\n"
        "EVIDENCE: MAPK1 phosphorylates JUN at Ser63; activity not measured.",
        '{"answer": "direct_axis_mismatch", '
        '"rationale": "modification not activity"}',
    ),
    (
        "CLAIM: subject=p53, object=DNA-binding-element, axis=binding, "
        "sign=neutral\n"
        "EVIDENCE: p53 binds the consensus DNA element in the promoter.",
        '{"answer": "direct_partner_mismatch", '
        '"rationale": "DNA binding, not protein-protein Complex"}',
    ),
    (
        "CLAIM: subject=FactorR, object=TargetA, axis=activity, sign=negative\n"
        "EVIDENCE: AdaptorP is recruited by FactorR to drive expression by "
        "suppressing TargetA activity in cell models.",
        '{"answer": "via_mediator", '
        '"rationale": "FactorR -> AdaptorP -> suppresses TargetA — indirect chain"}',
    ),
    (
        "CLAIM: subject=PKA, object=CREB, axis=activity, sign=positive\n"
        "EVIDENCE: PKA leads to CREB-mediated transcription via several "
        "downstream steps.",
        '{"answer": "via_mediator_partial", '
        '"rationale": "chain via several steps but no named intermediate"}',
    ),
    (
        "CLAIM: subject=KinaseN, object=ProteinM, axis=activity, sign=positive\n"
        "EVIDENCE: KinaseN levels were normalized to ProteinM by Western blot.",
        '{"answer": "no_relation", '
        '"rationale": "ProteinM is a loading control, no relation asserted"}',
    ),
    (
        "CLAIM: subject=KinaseN, object=ProteinM, axis=activity, sign=positive\n"
        "EVIDENCE: We characterized KinaseN substrates in cycling cells.",
        '{"answer": "no_relation", '
        '"rationale": "ProteinM not mentioned; KinaseN-ProteinM relation not asserted"}',
    ),
    # T-phase Fix C span-narrow few-shots: pairs that distinguish in-claim-clause
    # negation (sign_mismatch fires) from sibling-clause negation (does NOT fire).
    # Synthetic placeholder names per contamination guard.
    (
        "CLAIM: subject=KinaseA, object=TargetB, axis=modification, sign=positive\n"
        "EVIDENCE: Phosphorylation of TargetB by KinaseA is inhibited by an "
        "unrelated repressor protein RepC.",
        '{"answer": "direct_sign_match", '
        '"rationale": "phosphorylation of TargetB by KinaseA — direct '
        '(inhibition by RepC governs a different proposition)"}',
    ),
    (
        "CLAIM: subject=KinaseA, object=TargetB, axis=modification, sign=positive\n"
        "EVIDENCE: KinaseA does not phosphorylate TargetB under any tested "
        "condition; KinaseA does phosphorylate other substrates.",
        '{"answer": "direct_sign_mismatch", '
        '"rationale": "KinaseA does NOT phosphorylate TargetB — '
        'in-claim-clause negation"}',
    ),
    (
        "CLAIM: subject=ReceptorR, object=KinaseS, axis=activity, sign=positive\n"
        "EVIDENCE: ReceptorR activates KinaseS in stimulated cells, but this "
        "pathway does not affect downstream proliferation outcomes.",
        '{"answer": "direct_sign_match", '
        '"rationale": "ReceptorR activates KinaseS — sibling negation '
        'governs proliferation, not the claim relation"}',
    ),
    (
        "CLAIM: subject=KinaseA, object=TargetB, axis=activity, sign=positive\n"
        "EVIDENCE: KinaseA-mediated TargetB activation has been observed; "
        "subsequent steps are inhibited by drug X.",
        '{"answer": "direct_sign_match", '
        '"rationale": "KinaseA activates TargetB — inhibition is on '
        'subsequent steps, not the claim"}',
    ),
    # U-phase U7: activity vs amount disambiguation few-shots.
    # Synthetic placeholder names per contamination guard.
    (
        "CLAIM: subject=FactorR, object=GeneZ, axis=amount, sign=positive\n"
        "EVIDENCE: FactorR overexpression increased GeneZ mRNA and protein "
        "levels in HEK293 cells.",
        '{"answer": "direct_amount_match", '
        '"rationale": "FactorR overexpression → increased GeneZ levels — '
        'amount-axis change explicitly stated"}',
    ),
    (
        "CLAIM: subject=FactorR, object=GeneZ, axis=activity, sign=positive\n"
        "EVIDENCE: FactorR overexpression increased GeneZ mRNA and protein "
        "levels in HEK293 cells.",
        '{"answer": "direct_amount_match", '
        '"rationale": "evidence describes EXPRESSION change, not activity '
        'change — adjudicator will treat as axis_mismatch for activity claim"}',
    ),
    (
        "CLAIM: subject=KinaseA, object=TargetB, axis=activity, sign=positive\n"
        "EVIDENCE: KinaseA enhances TargetB activity in transfected cells.",
        '{"answer": "direct_sign_match", '
        '"rationale": "TargetB activity enhanced — activity-axis match"}',
    ),
    # U-phase U9: assertion-detection in dense / multi-clause evidence.
    # Counters the T-phase Fix A overshoot where "prefer no_relation"
    # made the LLM emit no_relation on records where the relation IS
    # asserted alongside other downstream descriptions. Synthetic
    # placeholder names per contamination guard.
    (
        "CLAIM: subject=ReceptorR, object=KinaseS, axis=binding, sign=neutral\n"
        "EVIDENCE: After ligand stimulation, ReceptorR forms a stable complex "
        "with KinaseS at the membrane, while a parallel pathway involving "
        "an unrelated transcription factor governs nuclear translocation.",
        '{"answer": "direct_sign_match", '
        '"rationale": "ReceptorR-KinaseS complex IS asserted; the parallel '
        'pathway describes a separate proposition"}',
    ),
    (
        "CLAIM: subject=AdaptorP, object=GeneZ, axis=activity, sign=positive\n"
        "EVIDENCE: When stimulated by growth factor, AdaptorP activates "
        "GeneZ, KinaseN, and FactorR; this activation is required for "
        "the downstream apoptosis program.",
        '{"answer": "direct_sign_match", '
        '"rationale": "AdaptorP activates GeneZ — listed alongside other '
        'targets, which does not weaken the assertion"}',
    ),
]


def answer(
    request: ProbeRequest, client: "ModelClient",
    *, reasoning_effort: str = "none",
) -> ProbeResponse:
    """Resolve a relation_axis probe via LLM closed-set classification.

    U3 selective reasoning: pass reasoning_effort="medium" to escalate
    on hard cases. Default "none" preserves first-pass behavior.
    """
    if request.kind != "relation_axis":
        raise ValueError(
            f"relation_axis.answer received kind={request.kind!r}"
        )
    user_msg_parts = [
        f"CLAIM: {request.claim_component}",
        f"EVIDENCE: {request.evidence_text.strip()}",
    ]
    if request.substrate_hint:
        user_msg_parts.append(f"SUBSTRATE HINT: {request.substrate_hint}")
    user_msg = "\n".join(user_msg_parts)

    answer_value, rationale, succeeded = llm_classify(
        system_prompt=_SYSTEM_PROMPT,
        few_shots=_FEW_SHOTS,
        user_message=user_msg,
        answer_set=_ANSWER_SET,
        kind="relation_axis",
        client=client,
        # T-phase Fix A: when LLM fails or returns out-of-set answer,
        # project to "no_relation" rather than picking an alphabetical
        # default. Doctrine §3.1.
        failure_default="no_relation",
        reasoning_effort=reasoning_effort,
    )
    return ProbeResponse(
        kind="relation_axis",
        answer=answer_value,
        # T-phase Fix A: source="abstain" still meaningful — it signals
        # the LLM call failed and the answer was projected. The
        # adjudicator's defensive scope-tiebreaker (former abstain
        # branch) handles this case.
        source="llm" if succeeded else "abstain",
        confidence="medium" if succeeded else "low",
        rationale=rationale,
    )
