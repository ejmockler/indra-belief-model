"""V6g curator prompts — overrides for the Gemini 3.1 Pro labeler path.

Production probe prompts (in `indra_belief.scorers.probes.*`) are tuned
for the live scorer. The V6g run revealed that Gemini 3.1 Pro fails the
≥90% per-probe gate on three of four probes due to systematic patterns
this module addresses by overriding the system prompt and few-shots
for the curator path only:

  relation_axis: After the first iteration (system prompt designed
    around "axes match → sign_match"), Pro lifted micro accuracy to
    71.2% but recall on the two minority gold classes collapsed:
    direct_axis_mismatch recall = 2.0% (1/50), direct_sign_mismatch
    recall = 18.8% (3/16). The dominant failure was Pro using the
    out-of-gold label `direct_amount_match` for cross-axis amount
    evidence (gold convention: those are `direct_axis_mismatch`).
    The second iteration (this file) restricts the answer set to the
    four labels that actually appear in U2 gold
    (direct_sign_match, direct_sign_mismatch, direct_axis_mismatch,
    no_relation), routes the cross-axis-amount pattern to
    `direct_axis_mismatch`, and adds explicit decision-table ordering
    (axis-mismatch fires BEFORE sign-match) plus loss-of-function
    sign-flipping rules. See `research/v6g_relation_axis_iteration.md`
    for diagnostics and expected impact.

  subject_role / object_role (54.8% / 59.6%): Symmetric Complex
    evidence ("X-Y interaction", "Y binds X") triggers role-swap. Pro
    treats syntactic position in the evidence as the role, ignoring
    that the CLAIM SUBJECT/OBJECT are stipulated.

  scope (87.2%, close to gate): Pro abstains when entities don't
    match verbatim, and over-picks `hedged` on words like "model is",
    "plausible", "suggesting" even when the relation is asserted.

The dicts below are looked up in `scripts/v6g_gemini_validation.py` and
substituted in preference to the production system prompt + few-shots.
Probes not present here fall back to production. Synthetic placeholder
entity names per the contamination guard
(`memory/feedback_fewshot_contamination.md`).
"""
from __future__ import annotations


# ── relation_axis: amount-axis claim + amount evidence = direct_sign_match ──

_RELATION_AXIS_SYSTEM = """\
You classify whether an evidence sentence describes a relation between \
two named entities on a specified AXIS with a specified SIGN.

The CLAIM is given as `StmtType(subject, object)`. Map the StmtType \
to its axis as follows:
  - Phosphorylation, Dephosphorylation, Methylation, Acetylation, \
    Ubiquitination, Sumoylation, Hydroxylation, Farnesylation, \
    Glycosylation, Ribosylation: axis=modification
  - Activation, Inhibition: axis=activity
  - IncreaseAmount, DecreaseAmount: axis=amount
  - Complex: axis=binding
  - Translocation: axis=localization
  - GtpActivation, GtpInactivation: axis=gtp_state
  - Conversion: axis=conversion
  - Autophosphorylation: axis=modification (subject == object)

Sign comes from the StmtType: `Increase*`, `Activation`, \
`Phosphorylation` etc. → positive; `Decrease*`, `Inhibition`, \
`Dephosphorylation` → negative; `Complex`, `Translocation` → neutral.

★★★ DECISION TABLE — apply IN ORDER. The first matching rule wins. ★★★

Step 1 — IDENTIFY AXES.
  CLAIM AXIS = the axis of the StmtType (table above).
  EVIDENCE AXIS = the axis of the verb that connects the claim's \
    SUBJECT to the claim's OBJECT (or their aliases) in the sentence. \
    Look at the actual verb/noun that links them:
      - "phosphorylates", "is phosphorylated by", "phosphorylation \
        of", "ubiquitinates" → modification
      - "activates", "inhibits", "stimulates GnRH secretion", \
        "activator of", "suppresses activity" → activity
      - "induces", "induced", "upregulates", "downregulates", \
        "increased ... mRNA/protein expression", "decreased ... \
        levels", "expression of X by Y", "X-mediated degradation \
        of Y", "depletion of X decreased Y", "loss of X results \
        in upregulation of Y", "X promotes Y degradation" → amount
      - "binds", "associates with", "interaction between", \
        "X-Y complex" → binding
      - "translocates", "exports to", "imports into nucleus" → \
        localization

Step 2 — AXIS-MISMATCH CHECK (FIRES BEFORE SIGN-MATCH CHECK).
  IF CLAIM AXIS != EVIDENCE AXIS:
    → answer = `direct_axis_mismatch`. STOP.
  This includes the COMMON cross-axis pattern: claim is Activation, \
  Inhibition, Phosphorylation, Dephosphorylation, or Complex, but \
  evidence describes only an amount/expression/level/induction \
  effect. Even if the SIGN agrees in spirit (e.g., claim=Activation \
  and evidence shows X induces Y mRNA), the AXIS DOES NOT MATCH, so \
  the answer is `direct_axis_mismatch`. Do NOT pick \
  `direct_sign_match` and do NOT route to any other label.

  Common cross-axis patterns that ALL resolve to `direct_axis_mismatch`:
    - claim=Activation/Inhibition, evidence=amount/expression \
      (induction, upregulation, secretion, mRNA/protein level change, \
      X-mediated degradation, depletion-decreases-Y).
    - claim=IncreaseAmount/DecreaseAmount, evidence=activity (X \
      activates Y, X is required for Y signaling).
    - claim=Phosphorylation/Dephosphorylation, evidence=activity or \
      amount (without an explicit phosphorylation verb).
    - claim=Complex (protein-protein binding), evidence=binding to \
      a DNA/RNA/lipid promoter (chromatin binding, "binds to the X \
      promoter") or evidence=co-localization or coexpression list.
    - claim=Activation, evidence=phosphorylation (modification axis) \
      with no activity verb.

Step 3 — SIGN-MISMATCH CHECK.
  ELSE IF CLAIM AXIS == EVIDENCE AXIS AND signs disagree:
    → answer = `direct_sign_mismatch`. STOP.
  Compute the EVIDENCE SIGN by reading the polarity of the verb plus \
  any sign-flipping operations (knockdown/silencing/depletion of X \
  flips X's effective effect; "X-RNAi inhibited Y expression" means \
  X normally INCREASES Y; "X-knockdown decreased Y" means X normally \
  INCREASES Y; "loss of X results in upregulation of Y" means X \
  normally DECREASES Y; "X-mediated downregulation of Y" → X \
  decreases Y; "X-mediated degradation/repression of Y" → X \
  decreases Y).
  Then compare to the claim sign:
    - claim=DecreaseAmount(X, Y), evidence="X-RNAi inhibited Y" → \
      X normally INCREASES Y → SIGN MISMATCH.
    - claim=Inhibition(X, Y), evidence="X reduced Y activity" → \
      claim says inhibits, evidence agrees → sign MATCH (not \
      mismatch). But evidence "X activates Y" under claim=Inhibition \
      → MISMATCH.
    - claim=Activation(X, Y), evidence="X promotes degradation of \
      Y antagonist DEPTOR" → if Y is on the DEPTOR side this is \
      indirect; check whether subject and object are linked.

Step 4 — DEFAULT (axes and signs both match).
  ELSE:
    → answer = `direct_sign_match`.

Step 5 — NO-RELATION OVERRIDE (only after Steps 2–4 fail).
  If the claim's subject AND object both appear (after alias \
  resolution) but no verb/relation connects them — e.g., they appear \
  in parallel-list positions, in methods/setup descriptions ("To \
  investigate the X-Y interaction we transfected..."), as loading \
  controls, in genetic-association statements ("variation in X is \
  associated with Y responsiveness"), or in pure co-occurrence \
  enumerations:
    → answer = `no_relation`.
  But if the relation IS asserted (even hedged, even within a longer \
  pathway, even via aliases), DO NOT pick `no_relation` — pick from \
  Steps 2–4 instead.

CRITICAL — clause-localized evaluation. Evaluate the relation between \
the claim's subject and object using ONLY the clause(s) where they \
co-occur or are connected by the relation verb. Sibling clauses with \
different verbs that do NOT govern the claim's subject-object pair \
are IGNORED.

CRITICAL — alias tolerance. The claim names entities by canonical \
symbol (e.g., MAPK1). Evidence may use synonyms or family forms \
(e.g., ERK, ERK2, p42, MAP kinase). Treat alias-vs-alias relations \
as direct. Do NOT pick `no_relation` because surface forms differ.

CRITICAL — clause sign flippers. When computing EVIDENCE SIGN for \
Step 3, account for: (a) loss-of-function operations (knockdown, \
RNAi, silencing, depletion, knockout, deletion, "lentiviral X RNAi") \
flip X's effective sign; (b) loss-of-X-rescues-Y / loss-of-X-causes-Y \
phrasing — interpret as "X normally does the OPPOSITE"; (c) "X \
abrogated downregulation of Y" → X INCREASES Y (double negation).

Answer ONE of (the only valid labels):
  direct_sign_match     — same axis as claim, same sign.
  direct_sign_mismatch  — same axis as claim, opposite sign \
    (account for loss-of-function flippers).
  direct_axis_mismatch  — different axis from claim. INCLUDES \
    cross-axis amount evidence under non-amount claims, modification \
    evidence under activity claims, and binding-to-DNA-promoter \
    evidence under protein-protein Complex claims.
  no_relation           — both entities mentioned but no claim \
    relation asserted (parallel lists, methods, controls).

DO NOT use any other label. The labels `direct_amount_match`, \
`direct_partner_mismatch`, `via_mediator`, `via_mediator_partial` \
are NOT in the allowed answer set for this probe.

Output ONE JSON object: {"answer": <one_of_above>, "rationale": <short_phrase>}.
The "rationale" is a 5-15 word phrase quoting the relevant words. NO \
prose outside the JSON."""


_RELATION_AXIS_FEW_SHOTS: list[tuple[str, str]] = [
    # ── direct_sign_match anchors (positive controls) ──
    # Anchor — clean activity-axis sign match.
    (
        "CLAIM: Activation(MAPK1, JUN)\n"
        "EVIDENCE: MAPK1 activates JUN in stimulated cells.",
        '{"answer": "direct_sign_match", "rationale": "MAPK1 activates JUN -- activity axis matches, sign matches"}',
    ),
    # Amount-axis claim + amount evidence → direct_sign_match.
    (
        "CLAIM: IncreaseAmount(FactorR, GeneZ)\n"
        "EVIDENCE: FactorR overexpression increased GeneZ mRNA and protein "
        "levels in HEK293 cells.",
        '{"answer": "direct_sign_match", "rationale": "amount claim, amount evidence (mRNA/protein increased), signs match"}',
    ),
    # Modification-axis claim + modification evidence → direct_sign_match.
    (
        "CLAIM: Phosphorylation(KinaseA, KinaseB)\n"
        "EVIDENCE: The Raf/MEK/KinaseA pathway, in which KinaseA "
        "phosphorylates KinaseB1 and KinaseB2 on Thr-Glu-Tyr residues.",
        '{"answer": "direct_sign_match", "rationale": "KinaseA phosphorylates KinaseB -- modification axis matches via aliases"}',
    ),
    # Symmetric binding — direct_sign_match regardless of word order.
    (
        "CLAIM: Complex(EntityP, EntityQ)\n"
        "EVIDENCE: The EntityQ-EntityP interaction is essential for complex "
        "assembly at the membrane.",
        '{"answer": "direct_sign_match", "rationale": "EntityP-EntityQ binding axis matches, Complex is symmetric"}',
    ),
    # Activity claim, activity evidence (with mechanism descriptor).
    (
        "CLAIM: Inhibition(FactorR, TargetA)\n"
        "EVIDENCE: FactorR inhibits TargetA via its kinase domain in a "
        "phosphorylation-dependent manner.",
        '{"answer": "direct_sign_match", "rationale": "FactorR inhibits TargetA -- activity axis, sign matches; via-domain is a mechanism descriptor"}',
    ),

    # ── direct_axis_mismatch (cross-axis: amount evidence under non-amount claim) ──
    # Activation claim, but evidence is amount/expression (mRNA, protein
    # levels, induction, secretion) — gold convention says axis_mismatch.
    (
        "CLAIM: Activation(FactorR, GeneZ)\n"
        "EVIDENCE: FactorR overexpression increased GeneZ mRNA and protein "
        "levels in HEK293 cells.",
        '{"answer": "direct_axis_mismatch", "rationale": "claim is activity but evidence is amount (mRNA/protein levels) -- different axes"}',
    ),
    # Activation claim, evidence is induction-of-secretion (amount axis).
    (
        "CLAIM: Activation(SignalK, CytokineY)\n"
        "EVIDENCE: SignalK induced the production of CytokineY in primary "
        "monocytes after LPS stimulation.",
        '{"answer": "direct_axis_mismatch", "rationale": "claim activity but evidence is induction/production (amount axis)"}',
    ),
    # Inhibition claim, evidence is amount/expression-decrease — axis mismatch.
    (
        "CLAIM: Inhibition(FactorR, TargetA)\n"
        "EVIDENCE: FactorR overexpression reduced TargetA mRNA expression in "
        "primary cells.",
        '{"answer": "direct_axis_mismatch", "rationale": "claim is activity (Inhibition) but evidence is amount (reduced mRNA expression)"}',
    ),
    # Inhibition claim, evidence is mediated-degradation (amount axis).
    (
        "CLAIM: Inhibition(KinaseN, ProteinM)\n"
        "EVIDENCE: KinaseN-mediated degradation of ProteinM is required for "
        "the apoptotic response.",
        '{"answer": "direct_axis_mismatch", "rationale": "claim activity but evidence is degradation (amount), different axes"}',
    ),
    # Activation claim, evidence is phosphorylation-only (modification axis).
    (
        "CLAIM: Activation(MAPK1, JUN)\n"
        "EVIDENCE: MAPK1 phosphorylates JUN at Ser63; activity was not "
        "measured.",
        '{"answer": "direct_axis_mismatch", "rationale": "claim activity but evidence is modification only -- different axes"}',
    ),
    # Complex claim, but evidence is binding to a DNA promoter (not protein-
    # protein complex axis).
    (
        "CLAIM: Complex(TF_X, GeneZ)\n"
        "EVIDENCE: TF_X binds to the GeneZ proximal promoter region in EMSA "
        "assays.",
        '{"answer": "direct_axis_mismatch", "rationale": "claim is protein-protein binding but evidence is TF binding to DNA promoter -- different axes"}',
    ),
    # Complex claim, but evidence is increased-expression (amount under
    # binding claim).
    (
        "CLAIM: Complex(KinaseN, AdaptorP)\n"
        "EVIDENCE: AdaptorP protein levels are upregulated in cells "
        "overexpressing KinaseN.",
        '{"answer": "direct_axis_mismatch", "rationale": "claim is binding but evidence is amount (protein levels) -- different axes"}',
    ),
    # IncreaseAmount claim, evidence is activity-only (no level change).
    (
        "CLAIM: IncreaseAmount(SignalK, EnzymeM)\n"
        "EVIDENCE: SignalK activates EnzymeM in stimulated lymphocytes.",
        '{"answer": "direct_axis_mismatch", "rationale": "claim is amount but evidence is activity -- different axes"}',
    ),
    # Modification claim, evidence is amount (reduced expression cannot
    # stand in for phosphorylation).
    (
        "CLAIM: Phosphorylation(KinaseA, TargetB)\n"
        "EVIDENCE: TargetB protein levels are reduced in KinaseA-knockout "
        "cells.",
        '{"answer": "direct_axis_mismatch", "rationale": "claim is modification but evidence is amount (protein level) -- different axes"}',
    ),

    # ── direct_sign_mismatch (same axis, opposite sign) ──
    # Activity-axis sign flip.
    (
        "CLAIM: Activation(MAPK1, JUN)\n"
        "EVIDENCE: MAPK1 inhibits JUN at high concentrations in HEK293 cells.",
        '{"answer": "direct_sign_mismatch", "rationale": "claim Activation but evidence says inhibits -- same axis, opposite sign"}',
    ),
    # Sign-mismatch via knockdown-flips-direction on amount axis.
    # (claim=DecreaseAmount, evidence shows X-RNAi inhibited Y → X normally
    # increases Y → mismatch.)
    (
        "CLAIM: DecreaseAmount(EnzymeM, CytokineY)\n"
        "EVIDENCE: Lentiviral EnzymeM RNAi inhibited CytokineY expression in "
        "macrophages.",
        '{"answer": "direct_sign_mismatch", "rationale": "RNAi of EnzymeM reduced CytokineY -- EnzymeM normally INCREASES CytokineY; claim says DecreaseAmount, opposite sign"}',
    ),
    # Sign-mismatch via "negatively regulates" / downregulation phrasing
    # under an IncreaseAmount claim.
    (
        "CLAIM: IncreaseAmount(FactorR, GeneZ)\n"
        "EVIDENCE: We previously reported that FactorR overexpression "
        "downregulates GeneZ at both the mRNA and protein levels in HEK293 "
        "cells.",
        '{"answer": "direct_sign_mismatch", "rationale": "claim Increase but evidence says FactorR overexpression downregulates GeneZ -- same axis, opposite sign"}',
    ),
    # Sign-mismatch via abrogated-downregulation (double negative collapses
    # to increase).
    (
        "CLAIM: Inhibition(TF_X, GeneZ)\n"
        "EVIDENCE: TF_X ectopic expression abrogated drug-induced "
        "downregulation of GeneZ in cancer cells.",
        '{"answer": "direct_sign_mismatch", "rationale": "TF_X abrogates downregulation of GeneZ -> TF_X INCREASES GeneZ; claim says Inhibition, opposite sign"}',
    ),
    # Modification-axis sign flip (Phosphorylation claim, but evidence shows
    # dephosphorylation by the claim subject).
    (
        "CLAIM: Phosphorylation(EnzymeM, TargetB)\n"
        "EVIDENCE: EnzymeM dephosphorylates TargetB at residue Ser45 in "
        "resting cells.",
        '{"answer": "direct_sign_mismatch", "rationale": "claim Phosphorylation but evidence says EnzymeM dephosphorylates TargetB -- same axis, opposite sign"}',
    ),

    # ── no_relation (only when no relation between subj+obj is asserted) ──
    # Co-IP / methods sentence — no relation asserted.
    (
        "CLAIM: Complex(GeneZ, KinaseN)\n"
        "EVIDENCE: To investigate the interaction of KinaseN with GeneZ in "
        "vivo, FLAG-tagged KinaseN was transfected with GeneZ in HEK293 "
        "cells.",
        '{"answer": "no_relation", "rationale": "describes experimental setup, no result asserted"}',
    ),
    # Loading-control mention — no_relation.
    (
        "CLAIM: Activation(KinaseN, ProteinM)\n"
        "EVIDENCE: KinaseN levels were normalized to ProteinM by Western "
        "blot.",
        '{"answer": "no_relation", "rationale": "ProteinM is loading control, no relation asserted"}',
    ),
    # Hallucination guard: claim entities NOT in evidence → no_relation.
    (
        "CLAIM: Activation(EntityA, EntityB)\n"
        "EVIDENCE: ReceptorR activates KinaseS which then induces "
        "transcription factor T to drive expression of GeneZ.",
        '{"answer": "no_relation", "rationale": "EntityA and EntityB not in evidence (no aliases)"}',
    ),

    # ── direct_sign_match positive controls — guards against over-firing of
    # axis_mismatch. ──
    # Dense pathway sentence with claim relation IS asserted on the right axis.
    (
        "CLAIM: Activation(ReceptorR, KinaseS)\n"
        "EVIDENCE: After ligand stimulation, ReceptorR activates KinaseS "
        "and downstream kinases, while a parallel pathway involving an "
        "unrelated transcription factor governs nuclear translocation.",
        '{"answer": "direct_sign_match", "rationale": "ReceptorR activates KinaseS -- activity axis matches"}',
    ),
    # Coordinated activity-axis target list.
    (
        "CLAIM: Activation(AdaptorP, GeneZ)\n"
        "EVIDENCE: When stimulated by growth factor, AdaptorP activates "
        "GeneZ, KinaseN, and FactorR.",
        '{"answer": "direct_sign_match", "rationale": "AdaptorP activates GeneZ -- activity axis, sign matches"}',
    ),
    # Conditional negation in sibling clause does not propagate.
    (
        "CLAIM: Phosphorylation(KinaseA, TargetB)\n"
        "EVIDENCE: KinaseA-mediated TargetB phosphorylation has been "
        "observed; subsequent steps are inhibited by drug X.",
        '{"answer": "direct_sign_match", "rationale": "KinaseA phosphorylates TargetB -- modification axis matches; sibling drug-X inhibition does not govern claim"}',
    ),
]


# ── subject_role / object_role: claim subject/object is stipulated ──

_SUBJECT_ROLE_SYSTEM = """\
You classify the role of the CLAIM SUBJECT in an evidence sentence.

★★★ The CLAIM SUBJECT is given to you. The CLAIM stipulates that this \
entity is the AGENT of the relation (the one doing the action). Your \
job is to check whether the evidence sentence treats this entity in a \
way consistent with that agent role, OR whether the evidence \
syntactically inverts roles (calling the claim subject the target/object \
of the relation), OR places it elsewhere. ★★★

The evidence's syntactic position does NOT override the claim. A claim \
of `Phosphorylation(KinaseA, TargetB)` stipulates KinaseA-as-agent; if \
the evidence reads "TargetB is phosphorylated by KinaseA", KinaseA is \
still the agent (passive voice does not flip the role). If the \
evidence reads "TargetB phosphorylates KinaseA", that IS a role-swap \
(present_as_object).

SYMMETRIC RELATIONS — for binding/Complex statements, BOTH partners \
are syntactically swappable. "X-Y interaction", "Y binds X", "X binds \
Y", "the X-Y complex", "Y interacts with X" all describe the same \
symmetric binding. A Complex claim's subject is `present_as_subject` \
in any of these phrasings, not `present_as_object`. The CLAIM \
designates which partner is the subject; the evidence asserts the \
binding regardless of word order.

Answer ONE of:
  present_as_subject — the entity acts as the AGENT of the claim's \
    relation. INCLUDES: passive-voice mentions ("X is phosphorylated \
    by KinaseA" → KinaseA agent), symmetric-relation evidence \
    regardless of word order, and apposition/parenthetical name \
    resolutions.
  present_as_object  — the entity is named as the TARGET of the \
    claim's relation in a way that contradicts its claim-stipulated \
    agent role. ONLY use this when there is a CLEAR role swap (the \
    sentence makes the OTHER entity the agent and this entity the \
    direct target/recipient of the action). Do NOT use for symmetric \
    binding evidence.
  present_as_mediator — the entity sits in the middle of a NAMED \
    chain (X → ENTITY → Y), not the upstream initiator nor the final \
    target.
  present_as_decoy   — the entity is mentioned but in a separate \
    proposition (control, co-treatment, bystander, parallel claim) \
    that does NOT bear on the claim relation.
  absent             — the entity is not mentioned and no alias of it \
    appears.

ALIAS TOLERANCE: family names (ERK ↔ MAPK1/2, AKT ↔ PKB), aliases \
(p53 ↔ TP53), parenthetical expansions, and surface-form variations \
all count as mentions of the entity.

Output ONE JSON object: {"answer": <one_of_above>, "rationale": <short_phrase>}.
The "rationale" is a 5-15 word phrase quoting the relevant words. NO \
prose outside the JSON."""


_SUBJECT_ROLE_FEW_SHOTS: list[tuple[str, str]] = [
    # Direct active voice.
    (
        "CLAIM SUBJECT: MAPK1\n"
        "EVIDENCE: MAPK1 phosphorylates JUN at Ser63 in stimulated cells.",
        '{"answer": "present_as_subject", '
        '"rationale": "MAPK1 phosphorylates — direct agent"}',
    ),
    # Passive voice, agent named with "by" — STILL present_as_subject.
    (
        "CLAIM SUBJECT: KinaseA\n"
        "EVIDENCE: TargetB is phosphorylated by KinaseA in resting cells.",
        '{"answer": "present_as_subject", '
        '"rationale": "phosphorylated by KinaseA — passive voice, KinaseA '
        'still the agent"}',
    ),
    # Symmetric binding, evidence reverses word order — STILL present_as_subject.
    (
        "CLAIM SUBJECT: EntityP\n"
        "EVIDENCE: The EntityQ-EntityP interaction is essential for complex "
        "assembly.",
        '{"answer": "present_as_subject", '
        '"rationale": "EntityP-EntityQ binding is symmetric; subject role '
        'preserved"}',
    ),
    # Symmetric binding via "binds to" with swap.
    (
        "CLAIM SUBJECT: EntityP\n"
        "EVIDENCE: EntityQ binds to EntityP in vitro and in vivo.",
        '{"answer": "present_as_subject", '
        '"rationale": "Complex/binding is symmetric — EntityP is subject by claim"}',
    ),
    # Complex with multi-partner list.
    (
        "CLAIM SUBJECT: TF_X\n"
        "EVIDENCE: GeneA-mediated TF_X interaction with NuclearY occurs in "
        "the nuclear fraction during stress.",
        '{"answer": "present_as_subject", '
        '"rationale": "TF_X named in interaction; claim subject preserved"}',
    ),
    # True role-swap: claim says KinaseA is agent, evidence says it's TARGET.
    (
        "CLAIM SUBJECT: KinaseA\n"
        "EVIDENCE: TargetB phosphorylates KinaseA at multiple residues "
        "during the cell cycle.",
        '{"answer": "present_as_object", '
        '"rationale": "TargetB phosphorylates KinaseA — KinaseA is target '
        '(true role swap)"}',
    ),
    # Apposition resolves to subject role.
    (
        "CLAIM SUBJECT: KinaseN\n"
        "EVIDENCE: ProteinM is phosphorylated by the cofactor X-kinase "
        "(KinaseN) which then promotes downstream effects.",
        '{"answer": "present_as_subject", '
        '"rationale": "phosphorylated by ... (KinaseN) — apposition resolves '
        'to KinaseN as agent"}',
    ),
    # Mediator — named middle of chain.
    (
        "CLAIM SUBJECT: NodeB\n"
        "EVIDENCE: NodeA activates NodeB, which then activates NodeC.",
        '{"answer": "present_as_mediator", '
        '"rationale": "NodeB sits between NodeA and NodeC in named chain"}',
    ),
    # Decoy — control protein.
    (
        "CLAIM SUBJECT: ELK1\n"
        "EVIDENCE: MAPK1 activates JUN; ELK1 was used as a control.",
        '{"answer": "present_as_decoy", '
        '"rationale": "ELK1 used as control, not in claim relation"}',
    ),
    # Absent.
    (
        "CLAIM SUBJECT: GeneZ\n"
        "EVIDENCE: SignalK regulates the immune response in primary cells.",
        '{"answer": "absent", '
        '"rationale": "GeneZ not mentioned, no alias visible"}',
    ),
    # Passive voice with different syntactic ordering.
    (
        "CLAIM SUBJECT: TF_X\n"
        "EVIDENCE: The induction of GeneZ, GeneY, and CytokineW expression "
        "by TF_X depended on the NF-kB pathway.",
        '{"answer": "present_as_subject", '
        '"rationale": "induction ... by TF_X — TF_X is the agent of induction"}',
    ),
    # Apposition in coordinated list.
    (
        "CLAIM SUBJECT: KinaseFamily\n"
        "EVIDENCE: Activation of NodeC by KinaseFamily members KinaseA and "
        "KinaseB drives downstream signaling.",
        '{"answer": "present_as_subject", '
        '"rationale": "Activation ... by KinaseFamily members — agent role"}',
    ),
]


_OBJECT_ROLE_SYSTEM = """\
You classify the role of the CLAIM OBJECT in an evidence sentence.

★★★ The CLAIM OBJECT is given to you. The CLAIM stipulates that this \
entity is the TARGET of the relation (the one being acted upon). Your \
job is to check whether the evidence treats it consistently with that \
target role, or whether the evidence inverts it (calling the claim \
object the agent), or places it elsewhere. ★★★

The evidence's syntactic position does NOT override the claim. A \
claim of `Phosphorylation(KinaseA, TargetB)` stipulates TargetB-as-target; \
if the evidence reads "TargetB is phosphorylated by KinaseA", TargetB \
is still the target. If the evidence reads "TargetB phosphorylates X" \
where X is unrelated to KinaseA, that IS a role-swap.

SYMMETRIC RELATIONS — for binding/Complex statements, BOTH partners \
are syntactically swappable. "X-Y interaction", "Y binds X", "X binds \
Y", "the X-Y complex" all describe the same symmetric binding. The \
CLAIM OBJECT in a Complex claim is `present_as_object` regardless of \
the evidence's word order. The CLAIM designates which partner is the \
object; the evidence asserts the binding regardless of who appears \
first.

Answer ONE of:
  present_as_object  — the entity is the TARGET of the claim's relation. \
    INCLUDES: passive-voice with "by", symmetric-relation evidence \
    regardless of word order, coordinated lists of targets.
  present_as_subject — the entity is the AGENT in a way that \
    contradicts its claim-stipulated target role. ONLY use this when \
    there is a CLEAR role swap. Do NOT use for symmetric binding.
  present_as_mediator — the entity sits in the middle of a NAMED \
    chain (X → ENTITY → Y).
  present_as_decoy   — the entity is mentioned in a separate \
    proposition (control, bystander, parallel claim) not in the \
    claim relation.
  absent             — the entity is not mentioned and no alias appears.

ALIAS TOLERANCE: family names, aliases, parenthetical expansions count \
as mentions.

Output ONE JSON object: {"answer": <one_of_above>, "rationale": <short_phrase>}.
The "rationale" is a 5-15 word phrase quoting the relevant words. NO \
prose outside the JSON."""


_OBJECT_ROLE_FEW_SHOTS: list[tuple[str, str]] = [
    # Direct active voice — target.
    (
        "CLAIM OBJECT: JUN\n"
        "EVIDENCE: MAPK1 phosphorylates JUN at Ser63 in stimulated cells.",
        '{"answer": "present_as_object", '
        '"rationale": "phosphorylates JUN — JUN is target"}',
    ),
    # Passive voice — STILL present_as_object.
    (
        "CLAIM OBJECT: TargetB\n"
        "EVIDENCE: TargetB is phosphorylated by KinaseA in resting cells.",
        '{"answer": "present_as_object", '
        '"rationale": "TargetB is phosphorylated by KinaseA — passive, '
        'TargetB is still the target"}',
    ),
    # Symmetric binding with reversed surface order — STILL present_as_object.
    (
        "CLAIM OBJECT: EntityQ\n"
        "EVIDENCE: The EntityQ-EntityP interaction is essential for complex "
        "assembly.",
        '{"answer": "present_as_object", '
        '"rationale": "binding is symmetric — EntityQ is object by claim"}',
    ),
    # Symmetric binding "binds to".
    (
        "CLAIM OBJECT: EntityP\n"
        "EVIDENCE: EntityP binds to EntityQ in vitro and in vivo.",
        '{"answer": "present_as_object", '
        '"rationale": "Complex is symmetric — EntityP-as-object preserved"}',
    ),
    # Coordinated list of targets.
    (
        "CLAIM OBJECT: CytokineY\n"
        "EVIDENCE: ReceptorX-induced release of CytokineW, CytokineY, "
        "CytokineZ and CytokineV into the supernatants was measured.",
        '{"answer": "present_as_object", '
        '"rationale": "release of ... CytokineY — target in coordinated list"}',
    ),
    # True role-swap: claim says X is target, evidence says X acts.
    (
        "CLAIM OBJECT: MAPK1\n"
        "EVIDENCE: MAPK1 in turn phosphorylates RSK and several other "
        "downstream kinases.",
        '{"answer": "present_as_subject", '
        '"rationale": "MAPK1 phosphorylates RSK — MAPK1 is agent (true swap)"}',
    ),
    # Mediator — named middle of chain.
    (
        "CLAIM OBJECT: AKT\n"
        "EVIDENCE: PI3K activates PDK1 which phosphorylates AKT to drive "
        "downstream survival signaling.",
        '{"answer": "present_as_mediator", '
        '"rationale": "AKT mid-chain: PI3K -> PDK1 -> AKT -> survival"}',
    ),
    # Decoy — loading control.
    (
        "CLAIM OBJECT: GAPDH\n"
        "EVIDENCE: MAPK1 phosphorylates JUN; GAPDH was used as a loading "
        "control on the Western blot.",
        '{"answer": "present_as_decoy", '
        '"rationale": "GAPDH is loading control, not target"}',
    ),
    # Absent.
    (
        "CLAIM OBJECT: TP53\n"
        "EVIDENCE: MAPK1 phosphorylates JUN at Ser63 in stimulated cells.",
        '{"answer": "absent", '
        '"rationale": "TP53 not mentioned"}',
    ),
    # Symmetric binding: claim object first, evidence reverses.
    (
        "CLAIM OBJECT: EntityP\n"
        "EVIDENCE: We investigated whether EntityP and EntityQ interact in "
        "vivo using co-immunoprecipitation.",
        '{"answer": "present_as_object", '
        '"rationale": "EntityP-EntityQ interaction studied — symmetric, '
        'object role preserved"}',
    ),
    # Receptor-ligand binding (symmetric Complex).
    (
        "CLAIM OBJECT: ReceptorR\n"
        "EVIDENCE: LigandL regulates several functions by binding to its "
        "receptor ReceptorR.",
        '{"answer": "present_as_object", '
        '"rationale": "binding to ReceptorR — target of binding"}',
    ),
]


# ── scope: nudge away from over-cautiousness; alias-tolerance for "abstain" ──

_SCOPE_SYSTEM = """\
You classify the EPISTEMIC SCOPE of the relation between two named \
entities in an evidence sentence.

The CLAIM names a relation between subject and object. Your job is to \
classify how the sentence FRAMES that relation — independent of \
whether you can verify it.

★★★ DEFAULT TO `asserted` when the relation is stated as fact in any \
clause of the sentence. The bar for `asserted` is LOW: declarative \
("X activates Y"), passive ("Y is activated by X"), nominalization \
("X-mediated activation of Y"), and presupposed/embedded framings \
("the X-Y interaction is required for...") ALL count as asserted. \
Pathway descriptions stating "X→Y is the pathway" assert the X-Y \
relation. ★★★

★★★ HEDGE WORDS that DO NOT trigger `hedged` when the relation itself \
is stated as fact: "suggesting" (used to introduce a conclusion that \
IS the relation), "model is/proposes that", "it is plausible/likely \
that" used to FRAME a stated relation, presence of "may" or "might" \
within a different clause that does NOT govern the claim relation, \
"thought to" + a separate factual statement of the relation. \
Hedge-adjacent vocabulary is NOT enough — only when the relation \
itself is FRAMED hypothetically does `hedged` apply. ★★★

★★★ ALIAS TOLERANCE: do not pick `abstain` because the entities \
appear under aliases (ERK1/2 vs MAPK1, p53 vs TP53, family names, \
parenthetical expansions). If the relation is stated between aliases \
of the claim subject and object, that is `asserted`. Pick `abstain` \
ONLY when the relation between the entities (or their aliases) is \
genuinely not described by the sentence — not just because surface \
forms differ. ★★★

Answer ONE of:
  asserted — the sentence directly affirms the relation in any form: \
    declarative, passive, nominalization, presupposition, "the X-Y \
    interaction", "X→Y pathway is activated when X does Y", "X by \
    Y-mediated mechanism" (asserts X-Y relation), conclusions \
    introduced by "suggesting that X does Y" or "our data show X \
    does Y", embedded models like "favored model is that X does Y" \
    that present the relation as held by the authors.
  hedged   — the relation itself is exploratory/hypothetical/putative: \
    "X may activate Y", "X might bind Y", "we hypothesized that X \
    activates Y", "putative X-Y interaction", "we tested whether X \
    activates Y" (test-frame, no result), "it remains unclear if X \
    activates Y", "X is THOUGHT to activate Y" (folk belief framing \
    of the relation itself).
  asserted_with_condition — the relation IS asserted on a QUALIFIED \
    form (e.g., wild-type, full-length, specific variant) with the \
    COMPLEMENTARY form negated: "X binds wild-type Y, but not the 3G \
    mutant", "Z phosphorylates the active form of A but not the \
    inactive form".
  negated  — the sentence explicitly DENIES the relation \
    UNCONDITIONALLY: "X did not activate Y", "Y was not phosphorylated \
    by X", "no effect of X on Y".
  abstain  — the sentence does not commit to the claim relation at \
    all: it describes only methods/setup ("To investigate whether X \
    binds Y, we transfected..."), describes only one entity, or \
    describes a different relation entirely. Use `abstain` ONLY when \
    no clause of the sentence frames the X-Y relation in any way.

CRITICAL — the CLAIM RELATION is the focus. Hedging or negation \
governing a DIFFERENT proposition does NOT propagate. "X activates Y, \
but Z was not affected" → `asserted` for X→Y.

CONDITIONAL NEGATION on a variant does NOT propagate. "X binds Y, but \
the mutant form does not" → if claim is about X-Y wild-type, scope \
is `asserted` (use `asserted_with_condition` only when both halves \
are explicit).

Output ONE JSON object: {"answer": <one_of_above>, "rationale": <short_phrase>}.
The "rationale" is a 5-15 word phrase quoting the relevant words. NO \
prose outside the JSON."""


_SCOPE_FEW_SHOTS: list[tuple[str, str]] = [
    # Direct asserted.
    (
        "CLAIM: relation between MAPK1 and JUN\n"
        "EVIDENCE: MAPK1 activates JUN in stimulated cells.",
        '{"answer": "asserted", '
        '"rationale": "direct affirmation: MAPK1 activates JUN"}',
    ),
    # "Suggesting that..." used as conclusion-introducer — ASSERTED, not hedged.
    (
        "CLAIM: relation between FactorR and TargetA\n"
        "EVIDENCE: Together, the binding assays and pulldown experiments "
        "support direct interaction, suggesting that FactorR inhibits "
        "TargetA in resting cells.",
        '{"answer": "asserted", '
        '"rationale": "suggesting that FactorR inhibits TargetA — '
        'conclusion-introducer, relation is stated"}',
    ),
    # "Our model is that..." — author's stated position — ASSERTED.
    (
        "CLAIM: relation between KinaseA and TargetB\n"
        "EVIDENCE: However, our favored model is that KinaseA phosphorylates "
        "TargetB at Ser1045 after detecting a block in replication, that "
        "phosphorylated TargetB is then translocated to the repair site.",
        '{"answer": "asserted", '
        '"rationale": "favored model is that KinaseA phosphorylates TargetB '
        '— authors hold this position"}',
    ),
    # Pathway-pattern with embedded relation — ASSERTED.
    (
        "CLAIM: relation between KinaseA and KinaseB\n"
        "EVIDENCE: The KinaseA-KinaseB-KinaseC pathway is activated when "
        "KinaseA phosphorylates KinaseB on Thr-Glu-Tyr residues.",
        '{"answer": "asserted", '
        '"rationale": "KinaseA phosphorylates KinaseB — embedded in pathway '
        'description"}',
    ),
    # Truly hedged — speculation/exploration.
    (
        "CLAIM: relation between CCR7 and AKT\n"
        "EVIDENCE: CCR7 may activate Akt in T-cells, but this remains to "
        "be confirmed.",
        '{"answer": "hedged", '
        '"rationale": "may activate ... remains to be confirmed"}',
    ),
    # Hedged via "we tested whether".
    (
        "CLAIM: relation between FactorR and TargetA\n"
        "EVIDENCE: We tested whether FactorR activates TargetA in primary "
        "cells, but did not analyze downstream consequences.",
        '{"answer": "hedged", '
        '"rationale": "we tested whether — exploratory frame, no result"}',
    ),
    # Negated unconditionally.
    (
        "CLAIM: relation between MAPK1 and JUN\n"
        "EVIDENCE: MAPK1 did not activate JUN under any tested condition.",
        '{"answer": "negated", '
        '"rationale": "did not activate ... under any condition"}',
    ),
    # Sibling clause negation does NOT propagate.
    (
        "CLAIM: relation between MAPK1 and JUN\n"
        "EVIDENCE: MAPK1 activates JUN robustly, but ELK1 was not affected.",
        '{"answer": "asserted", '
        '"rationale": "negation governs ELK1, not MAPK1-JUN"}',
    ),
    # Methods-setup only — abstain.
    (
        "CLAIM: relation between KinaseN and TargetA\n"
        "EVIDENCE: To investigate the interaction of TargetA with KinaseN "
        "in vivo, FLAG-tagged KinaseN was transfected with TargetA in "
        "HEK293 cells.",
        '{"answer": "abstain", '
        '"rationale": "methods setup only, no result asserted"}',
    ),
    # Asserted_with_condition — wild-type vs mutant.
    (
        "CLAIM: relation between FactorR and TargetX\n"
        "EVIDENCE: Endogenous TargetX binds wild-type FactorR, but does not "
        "bind the catalytically-dead FactorR mutant.",
        '{"answer": "asserted_with_condition", '
        '"rationale": "wild-type binding asserted; mutant negation qualifies"}',
    ),
    # Alias evidence — claim symbol differs but alias appears — ASSERTED.
    (
        "CLAIM: relation between KinaseA and KinaseB\n"
        "EVIDENCE: The Raf/MEK/KinaseA-family pathway phosphorylates "
        "KinaseB1 and KinaseB2 on canonical residues.",
        '{"answer": "asserted", '
        '"rationale": "KinaseA family phosphorylates KinaseB family — '
        'aliases assert relation"}',
    ),
    # "Suggesting that ... is regulated by" — passive conclusion — ASSERTED.
    (
        "CLAIM: relation between KinaseE and FactorN\n"
        "EVIDENCE: Reporter assays showed that insulin activated FactorN, "
        "which was enhanced by inhibitor U, suggesting that the PI3K-"
        "mediated FactorN activation is negatively regulated by KinaseE.",
        '{"answer": "asserted", '
        '"rationale": "suggesting that FactorN activation is regulated by '
        'KinaseE — conclusion stated"}',
    ),
    # "Plausible that X does Y" — embedded position, but stated — ASSERTED.
    (
        "CLAIM: relation between FactorI and SIRT_X\n"
        "EVIDENCE: Given context-dependent regulation, it is plausible that "
        "FactorI inhibits SIRT_X via mediator M.",
        '{"answer": "asserted", '
        '"rationale": "plausible that FactorI inhibits SIRT_X — author '
        'frames position; conditional but committed"}',
    ),
    # Truly hypothetical — "we hypothesized".
    (
        "CLAIM: relation between TF_X and GeneZ\n"
        "EVIDENCE: We hypothesized that TF_X induces GeneZ expression in "
        "response to stress signaling.",
        '{"answer": "hedged", '
        '"rationale": "we hypothesized — pure exploratory frame"}',
    ),
]


# ── Public dicts, consumed by scripts/v6g_gemini_validation.py ──

CURATOR_SYSTEM_PROMPTS: dict[str, str] = {
    "relation_axis": _RELATION_AXIS_SYSTEM,
    "subject_role": _SUBJECT_ROLE_SYSTEM,
    "object_role": _OBJECT_ROLE_SYSTEM,
    "scope": _SCOPE_SYSTEM,
}


CURATOR_FEW_SHOTS: dict[str, list[tuple[str, str]]] = {
    "relation_axis": _RELATION_AXIS_FEW_SHOTS,
    "subject_role": _SUBJECT_ROLE_FEW_SHOTS,
    "object_role": _OBJECT_ROLE_FEW_SHOTS,
    "scope": _SCOPE_FEW_SHOTS,
}
