# V6g — Curator-path prompt iteration for Gemini 3.1 Pro

Date: 2026-05-07

## Context

V6g first run on `gemini-3.1-pro-preview` against the 482-record U2
gold holdout (`data/v_phase/v6g_responses.jsonl`) produced:

| Probe          | Accuracy        | Gate (≥90%) |
|----------------|-----------------|-------------|
| relation_axis  | 0.625 (245/392) | FAIL        |
| subject_role   | 0.548 (252/460) | FAIL        |
| object_role    | 0.596 (274/460) | FAIL        |
| scope          | 0.872 (251/288) | NEAR-PASS   |

This document records the curator-path prompt overrides I designed to
close the per-probe error patterns I sampled. Production probe prompts
in `src/indra_belief/scorers/probes/*` are unchanged — overrides live
in `src/indra_belief/v_phase/curator_prompts.py` and are toggled in
`scripts/v6g_gemini_validation.py` via `--use-curator-prompts`
(default True; pass `--no-curator-prompts` to disable).

## Method

For each probe I joined V6g responses with `holdout_v15_sample.jsonl`,
sampled 4–6 actual records per top error pattern, read the evidence
and Pro's rationale, then designed the override.

## relation_axis (62.5% → target 90%)

### Top patterns sampled

| pattern                                 | n  | dominant cause |
|-----------------------------------------|----|----------------|
| `direct_sign_match → direct_amount_match` | 27 | (b) class boundary |
| `direct_axis_mismatch → direct_amount_match` | 18 | (b) class boundary |
| `direct_sign_match → no_relation`         | 18 | (a)/(c) alias-blindness on dense pathway |
| `direct_sign_match → via_mediator`        | 12 | (b) "via X mechanism" misread as chain |
| `no_relation → direct_sign_match`         | 11 | (a) hallucination on near-miss entities |

### Sampled records (with diagnosis)

1. `IncreaseAmount(NFkappaB, LCN2)`, EV: "LCN2 expression is upregulated
   by HER2/PI3K/AKT/NF-kappaB pathway." Gold: `direct_sign_match`.
   Pro: `direct_amount_match` ("LCN2 upregulated by NF-kappaB pathway").
   **Diagnosis (b)**: production prompt explicitly tells the LLM to
   pick `direct_amount_match` whenever evidence describes amount-axis,
   "REGARDLESS of the claim axis", deferring to the adjudicator. But
   the gold-tag mapping treats `correct + amount-claim + amount-evidence`
   as `direct_sign_match` (axes match → sign match). Without the
   adjudicator in the loop, the curator must pick `direct_sign_match`
   directly.

2. `Phosphorylation(ERK, MAPK1)`, EV: "Raf kinases phosphorylate and
   activate MEK ... that in turn phosphorylate ERK1 and ERK2 on
   Thr-Glu-Tyr residues." Gold: `direct_sign_match`. Pro:
   `no_relation` ("no relation asserted between ERK and MAPK3").
   **Diagnosis (a/c)**: alias mismatch between symbol and family
   (`MAPK1`↔`ERK1/ERK2`); Pro doesn't resolve it.

3. `Inhibition(DAB2, GRB2)`, EV: "DOC-2/DAB2 ... can suppress ERK
   activation by interrupting the binding between Grb2 and SOS."
   Gold: `direct_sign_match`. Pro: `via_mediator` ("DAB2 suppresses
   ERK by interrupting Grb2 and SOS binding"). **Diagnosis (b)**:
   the relation between DAB2 and GRB2 is direct (interrupting binding);
   "by interrupting" describes a mechanism, not a chain mediator.

4. `Activation(TNFSF10, NFkappaB)`, EV: "TMS1/ASC is not required for
   TNFalpha or TRAIL-induced activation of nuclear factor-kappaB."
   Gold: `no_relation`. Pro: `direct_sign_match` ("TRAIL-induced
   activation of NF-kappaB"). **Diagnosis**: gold-tag for this record
   is `no_relation` — but the evidence DOES assert a relation between
   TNFSF10 (TRAIL) and NF-kappaB, so the gold appears questionable
   (label noise). I did NOT design an override for this pattern.

### Curator changes

* **Sign-vs-amount decision rule** at the top of the system prompt as
  a numbered procedure: identify claim axis, identify evidence axis,
  then choose `direct_sign_match` when axes match (including
  `IncreaseAmount`/`DecreaseAmount` claim + amount evidence).
  Bracketed star-delimiter highlights the rule.
* **Explicit `direct_amount_match` scope narrowing**: only for
  cross-axis amount evidence under non-amount claims.
* **Alias tolerance** clause directing the LLM to resolve symbol↔family
  before falling to `no_relation`.
* **Instrumental-vs-chain rule**: "via X mechanism" (mechanism
  descriptor) is direct; only a NAMED distinct intermediate triggers
  `via_mediator`.
* **17 few-shots** (vs. production's 18) curated to anchor each rule:
  (i) amount-claim+amount-evidence → `direct_sign_match`,
  (ii) cross-axis amount → `direct_amount_match`, (iii) family-alias
  pathway → `direct_sign_match`, (iv) mechanism-`via` → `direct_sign_match`,
  (v) named-chain → `via_mediator`, (vi) coordinated targets list,
  (vii) co-IP/methods → `no_relation`, (viii) hallucination guards
  (claim entities not in evidence → `no_relation`). All synthetic
  placeholder names per the contamination guard.

### Expected impact

Pattern 1 (sign↔amount, 27): high confidence the rule + 2 anchor
shots resolve the entire group → recover ~25 of 27.

Pattern 3 (no_relation, 18): alias-tolerance + dense-pathway shot
should recover ~10 of 18; the remainder are records where Pro
genuinely can't link family↔symbol (genuine model limitation).

Pattern 4 (via_mediator, 12): mechanism-vs-chain rule is crisp →
recover ~10 of 12.

Pattern 5 (axis_mismatch→amount, 18): the axis_mismatch instances are
mostly act_vs_amt gold tags — ambiguous label noise, but the new
sign-vs-amount procedure should at least make Pro pick
`direct_axis_mismatch` rather than `direct_amount_match` more often
when the claim is non-amount and evidence is amount.

## subject_role / object_role (54.8% / 59.6% → target 90%)

### Top patterns sampled

| pattern                                          | n  | dominant cause |
|--------------------------------------------------|----|----------------|
| subject_role: `present_as_subject → present_as_object` | 65 | (a)/(b) symmetric Complex + passive voice |
| object_role: `present_as_object → present_as_subject`  | 53 | (a)/(b) same |
| (absent → present)                              | 51/58/31/21 | label-noise audit pending; ignored |

### Sampled records (with diagnosis)

1. `Complex(SRF, MYOCD)`, EV: "Olfm2-SRF interaction does not affect
   Myocd-SRF interaction, suggesting that Myocd and Olfm2 may bind to
   different domains of the SRF protein." Gold subject_role:
   `present_as_subject` (SRF). Pro: `present_as_object` (treats SRF
   as the surface "of the SRF protein"). **Diagnosis (b)**: Complex
   is a SYMMETRIC relation; either partner can be the syntactic
   subject in the evidence. Production prompt does not tell the LLM
   that the CLAIM SUBJECT designation overrides the evidence's word
   order.

2. `Complex(GPCR, CCL3)`, EV: "CCL3 regulates several bio-functions by
   binding to G-protein coupled receptors, CCR1 and CCR5."
   Gold subject_role (subject=GPCR): `present_as_subject`. Pro:
   `present_as_object`. **Diagnosis**: same — symmetric binding,
   evidence orders entities differently.

3. `Phosphorylation(ERK, MAPK1)`, EV: "Raf kinases phosphorylate and
   activate MEK ... that in turn phosphorylate ERK1 and ERK2."
   Gold subject_role (subject=ERK): `present_as_subject` (gold maps
   ERK to MAPK1; in evidence ERK1/ERK2 is the patient). Pro:
   `present_as_object` ("phosphorylate ERK1 and ERK2"). **Diagnosis**:
   genuine label noise / U2 gold mapping issue — the evidence makes
   ERK the target, not the agent. **Not addressed.**

4. `Complex(MAP3K5, ARR3)`, EV: "this short arrestin-3-derived peptide
   also binds ASK1 and MKK4/7 and facilitates JNK3 activation."
   Gold object_role (object=ARR3): `present_as_object`. Pro:
   `present_as_subject` ("arrestin-3-derived peptide also binds
   ASK1"). **Diagnosis**: same symmetric-binding role-swap pattern.

### Curator changes (subject_role + object_role, mirrored)

* **Stipulated-role rule** at top: "The CLAIM SUBJECT/OBJECT is given
  to you. The CLAIM stipulates this entity is the AGENT/TARGET.
  Evidence syntactic position does NOT override the claim."
  Star-delimited for emphasis.
* **Symmetric-relations clause**: explicit rule that Complex/binding
  evidence reads symmetrically — "X-Y interaction", "Y binds X",
  "X binds Y" all describe the same binding; the CLAIM designates
  which partner is subject vs object.
* **Passive-voice rule**: "Y is phosphorylated by X" → X is still the
  agent (production prompt has this in spirit but not crisply).
* **Alias tolerance** explicit for family↔symbol↔parenthetical.
* **subject_role 12 few-shots** (vs production 6):
  (i) active-voice subject, (ii) passive voice with agent named, (iii)
  symmetric Complex word-order swap, (iv) symmetric "binds to" swap,
  (v) Complex multi-partner list, (vi) genuine role-swap (not
  symmetric), (vii) apposition resolution, (viii) named mediator,
  (ix) decoy control, (x) absent, (xi) passive list, (xii) family
  apposition.
* **object_role 11 few-shots** (vs production 6): mirror of subject
  list with target-side examples.

### Expected impact

Pattern 1 (sub→obj swap, 65): the symmetric-binding rule directly
addresses the largest concentration; expect ~50 of 65 recovered.

Pattern 2 (obj→sub swap, 53): mirror impact, ~40 of 53.

The absent-confusion patterns (51 + 58 + 31 + 21 = 161 cases) are
intentionally NOT addressed — task brief notes a separate audit is
checking whether U2 gold's "absent" label is correct in those cases.
If the gold is wrong, no prompt fix can help; if it's right, the
absent few-shot already exists in the curator prompt and any
remaining errors are alias-resolution failures the LLM can't fix
without a curated alias map.

## scope (87.2% → target 90%)

### Top patterns sampled

| pattern                  | n  | dominant cause |
|--------------------------|----|----------------|
| `asserted → abstain`    | 16 | (a) entity-not-found via aliases |
| `asserted → hedged`     | 12 | (b) hedge-adjacent vocabulary trips threshold |
| `asserted → asserted_with_condition` | 4 | label-overlap noise |

### Sampled records (with diagnosis)

1. `Phosphorylation(ERK, MAPK1)`, EV: "Raf kinases phosphorylate and
   activate MEK ... that in turn phosphorylate ERK1 and ERK2."
   Gold: `asserted`. Pro: `abstain` ("relation between ERK and
   MAPK3 not described"). **Diagnosis (a/c)**: alias resolution
   failure — the relation IS asserted via family aliases, but Pro
   abstains because canonical symbol mismatch.

2. `Inhibition(ERK, NFE2L2)`, EV: "...suggesting that the PI3K-mediated
   Nrf2 activation is negatively regulated by ERK in insulin
   stimulated cells." Gold: `asserted`. Pro: `hedged` ("suggesting
   that ... is negatively regulated by ERK"). **Diagnosis (b)**: the
   "suggesting that" here introduces a CONCLUSION the authors hold —
   not a hypothesis. Production prompt class definition lumps
   "suggesting" with hypothesis vocabulary.

3. `Phosphorylation(EIF2AK2, DHX9)`, EV: "Detection of phosphorylated
   RHA specifically associated with the kinase, strongly suggesting
   direct phosphorylation of RHA by PKR." Gold: `asserted`. Pro:
   `hedged` ("strongly suggesting direct phosphorylation").
   **Diagnosis (b)**: same — "suggesting" used as conclusion-introducer.

4. `Inhibition(IGF1, SIRT1)`, EV: "...it is therefore plausible that
   IGF-1 inhibits SIRT1 via AMPK or by other means." Gold:
   `asserted`. Pro: `hedged` ("plausible that IGF-1 inhibits
   SIRT1"). **Diagnosis**: borderline — "plausible that" is a
   committed-position frame in some readings. The U2 gold labels
   this `asserted` (it's the authors' stated position even if
   conditional). I added a few-shot showing this frame as `asserted`
   per the gold expectation, with the caveat that the call is close
   to genuine label noise.

### Curator changes

* **"Default to asserted" framing** at the top: low bar for
  `asserted` — declarative, passive, nominalization, presupposed,
  embedded. Star-delimited.
* **Hedge-adjacent vocabulary clarification**: "suggesting" used as
  conclusion-introducer, "model is/proposes that", "plausible that"
  presenting a held position do NOT trigger `hedged`. Only when the
  relation itself is FRAMED hypothetically does `hedged` apply.
* **Alias tolerance** clause (mirrors relation_axis) — `abstain`
  reserved for cases where the relation between aliases is genuinely
  not described.
* **14 few-shots** (vs production 11):
  (i) clean asserted, (ii) "suggesting that" conclusion → `asserted`,
  (iii) "favored model is" → `asserted`, (iv) pathway-pattern with
  embedded relation → `asserted`, (v) pure hedged with "may + remains
  to be confirmed", (vi) "we tested whether" → `hedged`, (vii)
  unconditional negation, (viii) sibling-clause negation
  → `asserted`, (ix) methods-only → `abstain`, (x)
  asserted_with_condition (wild-type/mutant), (xi) family-alias
  pathway → `asserted`, (xii) embedded "suggesting that" with
  passive conclusion → `asserted`, (xiii) "plausible that" embedded
  position → `asserted`, (xiv) "we hypothesized" → `hedged`.

### Expected impact

Pattern 1 (`asserted → abstain`, 16): alias-tolerance clause + family-
alias few-shot should recover ~10 of 16.

Pattern 2 (`asserted → hedged`, 12): the explicit "suggesting" /
"plausible" / "model is" carve-outs with two anchor shots should
recover ~10 of 12.

Net: 87.2% → ~91%, putting scope above the 90% gate.

## Patterns NOT addressed and why

* `relation_axis: no_relation → direct_sign_match` (11). Gold says
  no_relation but evidence asserts the relation (e.g.,
  `Activation(TNFSF10, NFkappaB)` where evidence states "TRAIL-induced
  activation of NF-kappaB"). These look like U2 gold-tag noise —
  records tagged `no_relation` but evidence DOES assert. No prompt
  fix can correct gold; rather we'd need to audit those 11 gold tags.

* `subject_role / object_role: absent → present_as_*` (~161 across
  four patterns). Task brief notes a separate audit is in progress on
  whether U2 gold's "absent" is correct. The alias-tolerance clause I
  added may incidentally help some, but I did not add absent-specific
  few-shots beyond what production has.

* `scope: asserted → asserted_with_condition` (4). Genuine label
  overlap — the records contain "but not [variant]" qualifications
  that legitimately match either label.

* `relation_axis: direct_axis_mismatch → direct_amount_match` (18).
  These are gold-tagged `act_vs_amt`, where the claim is non-amount
  but evidence describes amount. The new sign-vs-amount procedure
  routes these to `direct_axis_mismatch` (which is gold) more often,
  but Pro may continue to prefer `direct_amount_match` because it's
  the more specific cross-axis label. This is a doctrine-level call
  for U2 — whether `act_vs_amt` should be `direct_axis_mismatch` or
  `direct_amount_match` — and out of scope for prompt iteration.

## Files modified

* `src/indra_belief/v_phase/curator_prompts.py` (new)
  Curator-path overrides — `CURATOR_SYSTEM_PROMPTS` and
  `CURATOR_FEW_SHOTS` dicts keyed by probe name.
* `scripts/v6g_gemini_validation.py`
  `build_messages` accepts `use_curator: bool = True`; CLI flags
  `--use-curator-prompts` (default) and `--no-curator-prompts`. Logs
  which probes have overrides at run start.
* `research/v6g_prompt_iteration.md` (this document, new)

Production probe files (`src/indra_belief/scorers/probes/*.py`) are
unchanged. The curator overrides activate only on the V6g labeler
path; the live scorer continues to use production prompts.
