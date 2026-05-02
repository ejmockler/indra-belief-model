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
    "direct_sign_mismatch",
    "direct_axis_mismatch",
    "direct_partner_mismatch",
    "via_mediator",
    "via_mediator_partial",
    "no_relation",
    "abstain",
})


_SYSTEM_PROMPT = """\
You classify whether an evidence sentence describes a relation between \
two named entities on a specified AXIS with a specified SIGN.

The CLAIM is given as: subject, object, axis, sign. Axes are: \
modification (phosphorylation, methylation, etc.), activity \
(activation, inhibition), amount (increase, decrease in expression), \
binding (Complex/protein-protein), localization (translocation). \
Signs: positive, negative, neutral (binding/translocation).

Answer ONE of:
  direct_sign_match     — the evidence directly asserts the SAME \
    axis+sign between subject and object. (Sign may have been inverted \
    upstream by perturbation propagation; trust the claim's stated sign.)
  direct_sign_mismatch  — the evidence asserts the same axis but \
    OPPOSITE sign (claim says inhibits, evidence says activates).
  direct_axis_mismatch  — the evidence asserts a relation on a \
    DIFFERENT axis (claim says Activation, evidence says \
    Phosphorylation only).
  direct_partner_mismatch — binding axis only: the evidence asserts \
    binding but to a non-protein partner (DNA/RNA/lipid) when claim \
    is protein-protein Complex.
  via_mediator          — the relation between subject and object is \
    INDIRECT, mediated by a named intermediate (X -> Z -> Y).
  via_mediator_partial  — the sentence carries chain markers (thereby, \
    leads to, mediated by) but no named intermediate is extractable.
  no_relation           — both entities are mentioned but no relation \
    between them is asserted (co-occurrence only).
  abstain               — the sentence underdetermines the relation \
    (e.g., one entity is absent, or the description is ambiguous).

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
        "CLAIM: subject=MAPK1, object=GAPDH, axis=activity, sign=positive\n"
        "EVIDENCE: MAPK1 levels were normalized to GAPDH by Western blot.",
        '{"answer": "no_relation", '
        '"rationale": "GAPDH is a loading control, no relation asserted"}',
    ),
    (
        "CLAIM: subject=MAPK1, object=JUN, axis=activity, sign=positive\n"
        "EVIDENCE: We characterized MAPK1 substrates in cycling cells.",
        '{"answer": "abstain", '
        '"rationale": "JUN not mentioned; sentence too general"}',
    ),
]


def answer(
    request: ProbeRequest, client: "ModelClient",
) -> ProbeResponse:
    """Resolve a relation_axis probe via LLM closed-set classification."""
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
    )
    return ProbeResponse(
        kind="relation_axis",
        answer=answer_value,
        source="llm" if succeeded else "abstain",
        confidence="medium" if succeeded else "low",
        rationale=rationale,
    )
