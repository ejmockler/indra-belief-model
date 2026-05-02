"""subject_role probe — what role does the claim's subject play in evidence?

Closed answer set: present_as_subject, present_as_object, present_as_mediator,
present_as_decoy, absent.

Substrate fast-path (router) handles the high-confidence cases:
  - CATALOG match with subject as agent → present_as_subject
  - subject's aliases not in evidence → absent
  - chain signal + subject in chain candidates → present_as_mediator

This module handles the LLM escalation: when substrate has a hint but
not an answer (entity present, role ambiguous, or substrate flagged
role-swap candidate), the LLM classifies into the closed set.

Few-shot curriculum:
  1. role-clear subject ("MAPK1 phosphorylates JUN" → present_as_subject)
  2. role-swap (claim subject is named as the TARGET of the relation)
  3. mediator (claim subject named in a chain but not at either end)
  4. decoy (claim subject mentioned but in a non-claim relation)
  5. absent (claim subject not mentioned despite hint claiming it is —
     forces LLM to override over-confident substrate hints)

Each shot exemplifies ONE answer; together they cover the closed set.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from indra_belief.scorers.probes._llm import llm_classify
from indra_belief.scorers.probes.types import ProbeRequest, ProbeResponse

if TYPE_CHECKING:
    from indra_belief.model_client import ModelClient


_ANSWER_SET = frozenset({
    "present_as_subject",
    "present_as_object",
    "present_as_mediator",
    "present_as_decoy",
    "absent",
})


_SYSTEM_PROMPT = """\
You classify the role of a single biomedical entity (the CLAIM SUBJECT) \
in an evidence sentence.

Answer ONE of:
  present_as_subject — the entity acts as the SUBJECT/AGENT of the \
    sentence's main relation (it does the action).
  present_as_object  — the entity is named as the TARGET/OBJECT of the \
    main relation (the action is done TO it). This catches role-swap.
  present_as_mediator — the entity sits in the middle of a chain \
    (X → ENTITY → Y), neither the upstream initiator nor the final \
    target; sentence frames it as an intermediate.
  present_as_decoy   — the entity is mentioned but in a relation other \
    than the claim's (e.g., as a control, co-treatment, bystander, or \
    in an adjacent claim that doesn't bear on the claim relation).
  absent             — the entity is not mentioned in the evidence.

Output ONE JSON object: {"answer": <one_of_above>, "rationale": <short_phrase>}.
Be terse. The "rationale" is a 5-15 word phrase quoting the relevant \
words from the evidence. NO prose outside the JSON."""


_FEW_SHOTS: list[tuple[str, str]] = [
    (
        "CLAIM SUBJECT: MAPK1\n"
        "EVIDENCE: MAPK1 phosphorylates JUN at Ser63 in stimulated cells.",
        '{"answer": "present_as_subject", '
        '"rationale": "MAPK1 phosphorylates"}',
    ),
    (
        "CLAIM SUBJECT: KinaseN\n"
        "EVIDENCE: ProteinM is known to be phosphorylated by cofactor X-"
        "kinase N catalytic subunit (KinaseN), which promotes downstream "
        "trafficking.",
        '{"answer": "present_as_subject", '
        '"rationale": "phosphorylated by ... (KinaseN) — apposition resolves to KinaseN as subject"}',
    ),
    (
        "CLAIM SUBJECT: TNF\n"
        "EVIDENCE: TNF receptor binding of TNFalpha activates downstream "
        "apoptosis pathways.",
        '{"answer": "present_as_object", '
        '"rationale": "TNF receptor BINDING OF TNFalpha — TNF is target"}',
    ),
    (
        "CLAIM SUBJECT: RegulatorM\n"
        "EVIDENCE: AdaptorP is recruited by RegulatorM to drive expression "
        "by suppressing TargetA activity through an intermediate step.",
        '{"answer": "present_as_mediator", '
        '"rationale": "RegulatorM acts via AdaptorP to affect TargetA"}',
    ),
    (
        "CLAIM SUBJECT: ELK1\n"
        "EVIDENCE: MAPK1 activates JUN; ELK1 was used as a control.",
        '{"answer": "present_as_decoy", '
        '"rationale": "ELK1 is the control, not the actor"}',
    ),
    (
        "CLAIM SUBJECT: GeneZ\n"
        "EVIDENCE: SignalK regulates the immune response in primary cells.",
        '{"answer": "absent", '
        '"rationale": "GeneZ not mentioned in evidence"}',
    ),
]


def answer(
    request: ProbeRequest, client: "ModelClient",
) -> ProbeResponse:
    """Resolve a subject_role probe via LLM closed-set classification.

    Returns a ProbeResponse with source="llm" when the LLM successfully
    answered, source="abstain" on any failure (transport, JSON, schema).
    On failure the fallback answer is "absent" — the most conservative
    value that does not cause the adjudicator to commit a false-positive
    relationship.
    """
    if request.kind != "subject_role":
        raise ValueError(
            f"subject_role.answer received kind={request.kind!r}"
        )
    user_msg_parts = [
        f"CLAIM SUBJECT: {request.claim_component}",
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
        kind="subject_role",
        client=client,
    )
    return ProbeResponse(
        kind="subject_role",
        answer=answer_value if succeeded else "absent",
        source="llm" if succeeded else "abstain",
        confidence="medium" if succeeded else "low",
        rationale=rationale,
    )
