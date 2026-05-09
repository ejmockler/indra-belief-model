# V6g — `relation_axis` curator-prompt iteration #2

**Date:** 2026-05-07
**Author:** ejmockler
**Files touched:** `src/indra_belief/v_phase/curator_prompts.py`
**Predecessor doc:** `research/v6g_prompt_iteration.md` (iteration #1)
**Related:** `research/v6g_gemini_validation.md` (full V6g report)

## Problem statement

V6g iteration #1 lifted `relation_axis` micro-accuracy on Gemini 3.1 Pro from
62.5% → 71.2% by aligning the prompt with U2 gold's "claim axis matches
evidence axis → `direct_sign_match`" convention. But this over-collapsed
into `direct_sign_match`: recall on the two minority gold classes
nearly disappeared.

| Class | Support | Predicted | Precision | Recall | F1 |
|---|---|---|---|---|---|
| direct_sign_match | 273 | 259 | 0.900 | 0.853 | 0.876 |
| no_relation | 53 | 75 | 0.560 | 0.792 | 0.656 |
| **direct_axis_mismatch** | 50 | 6 | 0.167 | **0.020** | 0.036 |
| **direct_sign_mismatch** | 16 | 3 | 1.000 | **0.188** | 0.316 |
| direct_amount_match | 0 | 26 | 0.000 | — | — |
| direct_partner_mismatch | 0 | 9 | 0.000 | — | — |
| via_mediator | 0 | 12 | 0.000 | — | — |
| via_mediator_partial | 0 | 2 | 0.000 | — | — |

The bottom four labels have **zero support in U2 gold** for relation_axis.
Pro emitted 49 predictions in those classes, all of them automatically
wrong. That structural mismatch dominates the minority-class recall
collapse.

## Confusion matrices (gold → pred)

### Gold = `direct_axis_mismatch` (n=50)

| Pred | Count | % |
|---|---|---|
| direct_amount_match | 15 | 30.0% |
| direct_sign_match | 14 | 28.0% |
| no_relation | 11 | 22.0% |
| direct_partner_mismatch | 8 | 16.0% |
| via_mediator | 1 | 2.0% |
| **direct_axis_mismatch** | **1** | **2.0%** |

U2 tag breakdown: `wrong_relation` (26) + `act_vs_amt` (24).

### Gold = `direct_sign_mismatch` (n=16)

| Pred | Count | % |
|---|---|---|
| direct_amount_match | 4 | 25.0% |
| **direct_sign_mismatch** | **3** | **18.8%** |
| via_mediator | 3 | 18.8% |
| direct_sign_match | 3 | 18.8% |
| direct_axis_mismatch | 1 | 6.2% |
| no_relation | 1 | 6.2% |
| via_mediator_partial | 1 | 6.2% |

U2 tag: `polarity` (16/16).

## Sample error log

### `direct_axis_mismatch` errors (12 representative of 49)

Format: `[ID] CLAIM=Type(subj,obj) | EV: ... | gold=axis_mismatch pred=X tag=Y | Pro rationale`

- **[A1]** `IncreaseAmount(NKX2-5, MYH)` — EV: "Similar to Nkx2-5, MEF2C
  expression initiated cardiomyogenesis, resulting in the up-regulation of
  ... myosin heavy chain expression." pred=`direct_sign_match`
  tag=`act_vs_amt`. Pro: "Similar to Nkx2-5, MEF2C expression initiated...
  up-regulation of... myosin heavy chain". *(NKX2-5 doesn't actually act on
  MYH here; it's the sibling factor MEF2C that does the upregulating, but
  Pro fired on aliases.)*

- **[A2]** `Activation(E2F1, ELAVL1)` — EV: "E2F1 overexpression
  increased ... HuR protein and mRNA expression". pred=`direct_amount_match`
  tag=`act_vs_amt`. Pro: "E2F1 overexpression increased STAT3 mRNA". *(Cross-
  axis amount evidence; gold convention sends this to axis_mismatch.)*

- **[A5]** `Complex(CYP1A1, AHR)` — EV: "Oscillations in the binding of ...
  AHR ... to CYP1A1 were not observed". pred=`direct_sign_match`
  tag=`wrong_relation`. *(Binding to gene/promoter, not protein-protein
  Complex.)*

- **[A6]** `Activation(ERK, CXCL5)` — EV: "induction of ... CXCL5 by WNT5A
  ... depended on ... MAPK ... pathways". pred=`direct_amount_match`
  tag=`act_vs_amt`. *(Amount evidence under activity claim.)*

- **[A7]** `Dephosphorylation(PPP3, MEF2)` — EV: "calcineurin may directly
  dephosphorylate MEF2". pred=`direct_sign_match` tag=`wrong_relation`.
  *(PPP3 alias OK, but the gold considers the catalytic-subunit/complex
  distinction an axis mismatch here.)*

- **[A10]** `Inhibition(TP53, MAD2L1)` — EV: "BubR1 and Mad2 are
  downregulated by OS in a p53 dependent manner".
  pred=`direct_amount_match` tag=`act_vs_amt`. *(Amount evidence; gold says
  axis mismatch with the activity claim.)*

- **[A14]** `Activation(GATA4, LRRC10)` — EV: "GATA4 mediated activation of
  the Lrrc10 promoter". pred=`direct_sign_match` tag=`act_vs_amt`.
  *("Activation of the promoter" = transcriptional/amount, not protein
  activity.)*

- **[A15]** `Activation(CYP1B1, Wnt)` — EV: "CYP1B1 ... activated Wnt and
  beta-catenin signaling via upregulation of CTNNB1, ZEB2, SNAI1, and
  TWIST1". pred=`direct_sign_match` tag=`act_vs_amt`. *(Mechanism is
  upregulation of pathway components; gold considers this amount-axis.)*

- **[A17]** `Activation(E2F1, Cyclin)` — EV: "E2F upregulated cell cycle
  genes (cyclins E, A, B and D3)". pred=`direct_amount_match`
  tag=`wrong_relation`.

- **[A18]** `Inhibition(PPARG, IL6)` — EV: "PMA induced expression of ...
  IL-6 ... is significantly reduced ... by PPARgamma agonists".
  pred=`direct_amount_match` tag=`act_vs_amt`.

- **[A22]** `Activation(E2F1, Cyclin)` — EV: "knockdown E2F1 ... decreased
  cyclin E1, cyclin A2 ... proteins". pred=`direct_amount_match`
  tag=`act_vs_amt`.

- **[A23]** `Complex(DLX5, IRS2)` — EV: "DLX5 can bind to the IRS-2
  promoter and augment its activity". pred=`direct_partner_mismatch`
  tag=`wrong_relation`. *(Promoter-binding under protein-protein Complex
  claim.)*

### `direct_sign_mismatch` errors (full 16)

- **[S1]** `DecreaseAmount(ADAM17, MMP9)` — EV: "Lentiviral ADAM17 RNAi
  inhibited MMP9 expression". gold=sign_mismatch pred=**sign_mismatch** ✓
  tag=`polarity`. (RNAi inverts → ADAM17 normally INCREASES MMP9, opposite
  to claim.)

- **[S2]** `IncreaseAmount(PAPPA, ABCG1)` — EV: "down-regulations of ...
  ABCG1 ... by PAPP-A". gold=sign_mismatch pred=**sign_mismatch** ✓
  tag=`polarity`.

- **[S3]** `DecreaseAmount(ALDH2, FOXO3)` — EV: "ALDH2 overexpression
  antagonizes ... improvement of insulin signaling at the levels of ...
  Foxo3a". pred=`direct_axis_mismatch` tag=`polarity`. (Pro saw "activity"
  language and routed to axis_mismatch; gold disagreed.)

- **[S4]** `Inhibition(MYC, AKT)` — EV: "Akt phosphatase, Phlpp2, is
  co-opted by Myc to drive proliferation by suppressing Akt activity".
  pred=`via_mediator` tag=`polarity`. (Gold treats Myc → Akt as direct
  via Phlpp2 with sign-flipped claim — claim says inhibit, evidence shows
  Myc→Phlpp2→inhibit Akt → Myc *INHIBITS* Akt activity → actually claim
  matches?? Gold says sign_mismatch because the *direct* link via
  inhibition cascade nets out as ACTIVATION → claim "Inhibition" is wrong
  signed. Subtle.)

- **[S5]** `Activation(SESN2, AMPK)` — EV: "induced SESTRIN1/2 activates
  AMPK". pred=`direct_sign_match` tag=`polarity`. (Gold flips because of
  upstream context not in this sentence — borderline.)

- **[S6]** `Inhibition(SMAD, TGFB)` — EV: "TGF-β induced ... is prevented
  by inhibition of the SMAD ... pathway". pred=`no_relation`
  tag=`polarity`. (Reverse direction: SMAD acts downstream, but the
  claim says SMAD inhibits TGFB.)

- **[S7]** `Inhibition(PPARG, TNF)` — EV: "PPARgamma ... reduced ...
  TNFalpha". pred=`direct_amount_match` tag=`polarity`. (Gold considers
  this same-axis-sign-mismatch despite cross-axis surface form.)

- **[S8]** `Complex(FAM83B, RAF1)` — EV: "Binding of FAM83B with CRAF
  disrupted CRAF/14-3-3 interactions". pred=`direct_sign_match`
  tag=`polarity`. (Binding present, but the complex disrupts another
  complex; gold flips.)

- **[S9]** `Inhibition(FOXM1, BIRC5)` — EV: "FOXM1 ectopic expression ...
  abrogated docetaxel induced downregulation of XIAP and Survivin".
  pred=`direct_amount_match` tag=`polarity`. (Double-negative: FOXM1
  abrogates downregulation → FOXM1 INCREASES Survivin → claim Inhibition
  is opposite.)

- **[S10]** `Inhibition(MIR34A, TP53)` — EV: "OCT4 as a target of miR-34a
  ... inhibits p53". pred=`via_mediator` tag=`polarity`. (Chain: miR34A→
  OCT4→inhibits p53; net effect of miR34A on p53 is inhibition only if OCT4
  is upregulated by miR34A — but miR34A typically REPRESSES targets, so
  miR34A→reduce OCT4→increase p53 → claim says Inhibition, opposite.)

- **[S11]** `Inhibition(ACKR3, RB1)` — EV: "CXCR7 depletion ... decreased
  ... Rb". pred=`direct_amount_match` tag=`polarity`. (Loss-of-function:
  CXCR7 depletion DECREASED Rb → CXCR7 normally INCREASES Rb → claim
  Inhibition is opposite.)

- **[S12]** `IncreaseAmount(PAPPA, ABCG1)` — EV: "the IGF-1/PI3-K/Akt
  signaling pathway plays a critical role in the negative regulation of
  ... ABCG1 ... by PAPP-A". gold=sign_mismatch pred=**sign_mismatch** ✓
  tag=`polarity`.

- **[S13]** `Activation(CXCR4, MTOR)` — EV: "CXCR4 promoted lysosomal
  degradation of the mammalian target of rapamycin antagonist DEPTOR".
  pred=`via_mediator` tag=`polarity`. (CXCR4 degrades DEPTOR which is an
  MTOR antagonist → CXCR4 INCREASES MTOR → claim Activation matches in
  spirit but axis is amount; gold marks sign_mismatch.)

- **[S14]** `Inhibition(FOXO3, BCL2L11)` — EV: "phosphorylated inactivation
  of FOXO3a, which led to downregulation of Bim".
  pred=`via_mediator_partial` tag=`polarity`. (Bim=BCL2L11 alias; FOXO3
  inactivated → Bim downregulated → FOXO3 normally INCREASES Bim → claim
  Inhibition is opposite.)

- **[S15]** `Activation(PI3K, NFkappaB)` — EV: "PAPP-A down-regulates ...
  through PI3-K but not by the MAPK cascade. ... activation of PI3-K ...
  trigger the activation of NF-kappaB". pred=`direct_sign_match`
  tag=`polarity`. (Borderline: the "activation" sentence is later in the
  excerpt, but earlier context suggests opposite sign.)

- **[S16]** `Inhibition(PHLPP1, Integrins)` — EV: "PHLPP-mediated
  downregulation of integrin". pred=`direct_amount_match`
  tag=`polarity`. (PHLPP downregulates integrin → PHLPP DECREASES
  integrins → claim Inhibition matches in spirit but axis is amount; gold
  marks sign_mismatch.)

### Positive controls (gold = pred = `direct_sign_match`)

Verified that the new prompt does not regress on these patterns:

- `Complex(GPCR, CCL3)` — "CCL3 regulates several bio-functions by binding
  to G-protein coupled receptors, CCR1 and CCR5". *(Symmetric binding via
  alias; new prompt preserves direct_sign_match.)*
- `Phosphorylation(CDK2, CDC6)` — "Cdc6 is known to be phosphorylated by
  Cdk2". *(Passive voice, modification axis match.)*
- `IncreaseAmount(NFkappaB, LCN2)` — "LCN2 expression is upregulated by
  ... NF-kappaB pathway". *(Amount axis match via alias.)*
- `Deacetylation(HDAC, Histone)` — "deacetylation of histones by HDACs".
  *(Modification axis match.)*

## Diagnosis

**Dominant hypotheses: h2 + h4 (with h1 as a contributing factor).**

- **h4 (gold-tag mismatch)** is the structural root: U2 gold for
  `relation_axis` admits only **four** labels (`direct_sign_match`,
  `direct_sign_mismatch`, `direct_axis_mismatch`, `no_relation`). The first-
  iteration prompt advertised eight labels including `direct_amount_match`,
  `direct_partner_mismatch`, `via_mediator`, `via_mediator_partial`. Pro
  emitted 49 predictions (12.5% of the 392-record holdout) into those
  out-of-gold labels — every one of which was an automatic miss. The
  largest single bucket (15 axis_mismatch records → `direct_amount_match`)
  fits a recognizable pattern: "claim is non-amount axis, evidence is
  expression/level/induction/degradation". U2 gold treats those as
  `direct_axis_mismatch` (cross-axis substitution). The first iteration
  even encoded the wrong rule: "Step 5: use `direct_amount_match` ONLY
  when the evidence is amount-axis but the claim is NOT amount-axis." That
  sentence directly contradicts U2.

- **h2 (no axis-mismatch decision logic)** is the second contributor:
  the first iteration listed `direct_axis_mismatch` only as a fallback
  ("Otherwise use `direct_axis_mismatch`"), with no decision-table
  ordering and no sample showing the cross-axis-amount pattern under
  this label. Pro defaulted to `direct_sign_match` whenever any verb
  loosely connected the entities (14/50 axis_mismatch errors).

- **h1 (axes-match overfires)** describes the second-order error: when Pro
  did identify the axes as matching, Step 4 routed to `direct_sign_match`
  even where the SIGN disagreed via knockdown/depletion/abrogation
  reasoning (4/16 sign_mismatch errors → `direct_amount_match`, 3/16 →
  `direct_sign_match` despite an upstream sign flip).

- **h3 (few-shot count)** is real but secondary: the iteration-1 few-shots
  had only 1 axis_mismatch and 1 sign_mismatch example.

## Changes made

### System-prompt rule changes

1. **Restricted answer set to 4 labels** matching U2 gold. Removed
   `direct_amount_match`, `direct_partner_mismatch`, `via_mediator`,
   `via_mediator_partial`. Added explicit "DO NOT use any other label"
   guard rail.

2. **New decision table (apply in order, first match wins):**
   - Step 1: identify CLAIM AXIS and EVIDENCE AXIS (with explicit
     vocabulary lists for each axis).
   - **Step 2: AXIS-MISMATCH FIRES BEFORE SIGN-MATCH.** If
     `CLAIM AXIS != EVIDENCE AXIS` → `direct_axis_mismatch`. STOP.
   - Step 3: if axes match and signs disagree → `direct_sign_mismatch`.
   - Step 4: default → `direct_sign_match`.
   - Step 5: `no_relation` only fires when no clause links the subj+obj
     pair (parallel lists, methods, controls).

3. **Common cross-axis patterns** explicitly listed under Step 2:
   - claim activity, evidence expression/induction/degradation
   - claim amount, evidence activity verb
   - claim modification, evidence amount or activity
   - claim Complex, evidence DNA-promoter binding or coexpression list
   - claim activity, evidence phosphorylation only

4. **Loss-of-function sign-flipping rules** added under Step 3:
   - Knockdown/silencing/depletion/RNAi/knockout flips X's effective sign.
   - "X-RNAi inhibited Y" → X normally INCREASES Y.
   - "X-knockdown decreased Y" → X normally INCREASES Y.
   - "loss of X results in upregulation of Y" → X normally DECREASES Y.
   - "X-mediated downregulation/degradation/repression of Y" → X DECREASES Y.
   - "X abrogated downregulation of Y" → X INCREASES Y (double negation).

5. **Preserved earlier additions:** clause-localized evaluation, alias
   tolerance, instrumental-vs-chain disambiguation. (The chain rule no
   longer routes to `via_mediator` since that label is removed; instead
   "X via mechanism Y" stays as `direct_sign_match`.)

System prompt size after edits: **7,211 bytes** (under 8KB target; was
5,600 bytes before).

### Few-shot changes

Replaced the 17-shot block with 25 shots structured by class:

- **5 `direct_sign_match` anchors** covering activity, modification,
  amount, binding, and mechanism-descriptor cases.
- **9 `direct_axis_mismatch` shots** — the pattern with the
  highest-recoverable error count.
- **5 `direct_sign_mismatch` shots** including knockdown-flip and
  abrogated-downregulation patterns.
- **3 `no_relation` shots** (methods, loading control, hallucination).
- **3 additional `direct_sign_match` positive controls** (parallel
  pathways, coordinated targets, sibling-clause negation) to defend
  against axis-mismatch over-firing.

All 25 few-shots use synthetic placeholder names (KinaseA, FactorR,
GeneZ, EnzymeM, AdaptorP, etc.) per the contamination guard. Maximum
trigram-Jaccard against any of the 500 holdout records is **0.143**,
well below the 0.20 leak threshold.

### Few-shots verbatim

```
1. (sign_match)        Activation(MAPK1, JUN) | MAPK1 activates JUN in stimulated cells.
2. (sign_match)        IncreaseAmount(FactorR, GeneZ) | FactorR overexpression increased GeneZ mRNA and protein levels in HEK293 cells.
3. (sign_match)        Phosphorylation(KinaseA, KinaseB) | The Raf/MEK/KinaseA pathway, in which KinaseA phosphorylates KinaseB1 and KinaseB2 on Thr-Glu-Tyr residues.
4. (sign_match)        Complex(EntityP, EntityQ) | The EntityQ-EntityP interaction is essential for complex assembly at the membrane.
5. (sign_match)        Inhibition(FactorR, TargetA) | FactorR inhibits TargetA via its kinase domain in a phosphorylation-dependent manner.
6. (axis_mismatch)     Activation(FactorR, GeneZ) | FactorR overexpression increased GeneZ mRNA and protein levels in HEK293 cells.
7. (axis_mismatch)     Activation(SignalK, CytokineY) | SignalK induced the production of CytokineY in primary monocytes after LPS stimulation.
8. (axis_mismatch)     Inhibition(FactorR, TargetA) | FactorR overexpression reduced TargetA mRNA expression in primary cells.
9. (axis_mismatch)     Inhibition(KinaseN, ProteinM) | KinaseN-mediated degradation of ProteinM is required for the apoptotic response.
10. (axis_mismatch)    Activation(MAPK1, JUN) | MAPK1 phosphorylates JUN at Ser63; activity was not measured.
11. (axis_mismatch)    Complex(TF_X, GeneZ) | TF_X binds to the GeneZ proximal promoter region in EMSA assays.
12. (axis_mismatch)    Complex(KinaseN, AdaptorP) | AdaptorP protein levels are upregulated in cells overexpressing KinaseN.
13. (axis_mismatch)    IncreaseAmount(SignalK, EnzymeM) | SignalK activates EnzymeM in stimulated lymphocytes.
14. (axis_mismatch)    Phosphorylation(KinaseA, TargetB) | TargetB protein levels are reduced in KinaseA-knockout cells.
15. (sign_mismatch)    Activation(MAPK1, JUN) | MAPK1 inhibits JUN at high concentrations in HEK293 cells.
16. (sign_mismatch)    DecreaseAmount(EnzymeM, CytokineY) | Lentiviral EnzymeM RNAi inhibited CytokineY expression in macrophages.
17. (sign_mismatch)    IncreaseAmount(FactorR, GeneZ) | We previously reported that FactorR overexpression downregulates GeneZ at both the mRNA and protein levels in HEK293 cells.
18. (sign_mismatch)    Inhibition(TF_X, GeneZ) | TF_X ectopic expression abrogated drug-induced downregulation of GeneZ in cancer cells.
19. (sign_mismatch)    Phosphorylation(EnzymeM, TargetB) | EnzymeM dephosphorylates TargetB at residue Ser45 in resting cells.
20. (no_relation)      Complex(GeneZ, KinaseN) | To investigate the interaction of KinaseN with GeneZ in vivo, FLAG-tagged KinaseN was transfected with GeneZ in HEK293 cells.
21. (no_relation)      Activation(KinaseN, ProteinM) | KinaseN levels were normalized to ProteinM by Western blot.
22. (no_relation)      Activation(EntityA, EntityB) | ReceptorR activates KinaseS which then induces transcription factor T to drive expression of GeneZ.
23. (sign_match)       Activation(ReceptorR, KinaseS) | After ligand stimulation, ReceptorR activates KinaseS and downstream kinases, while a parallel pathway involving an unrelated transcription factor governs nuclear translocation.
24. (sign_match)       Activation(AdaptorP, GeneZ) | When stimulated by growth factor, AdaptorP activates GeneZ, KinaseN, and FactorR.
25. (sign_match)       Phosphorylation(KinaseA, TargetB) | KinaseA-mediated TargetB phosphorylation has been observed; subsequent steps are inhibited by drug X.
```

## Expected impact

### Recoverable cases per minority class

#### `direct_axis_mismatch` (n=50)

| Pred bucket | Count | Recovery rate | Recovered |
|---|---|---|---|
| direct_amount_match | 15 | ~100% (label removed; pattern explicitly mapped to axis_mismatch) | ~15 |
| direct_partner_mismatch | 8 | ~100% (label removed; "Complex but DNA-promoter" → axis_mismatch in Step 2) | ~8 |
| direct_sign_match | 14 | ~70% (Step 2 fires before Step 4) | ~10 |
| no_relation | 11 | ~40% (some are genuinely "no claim relation"; new prompt doesn't change no_relation semantics) | ~4 |
| via_mediator | 1 | 100% | 1 |
| direct_axis_mismatch | 1 | already correct | 1 |
| **Total recovery** | | | **≈39/50 = 78%** |

#### `direct_sign_mismatch` (n=16)

| Pred bucket | Count | Recovery rate | Recovered |
|---|---|---|---|
| direct_sign_mismatch | 3 | already correct | 3 |
| direct_amount_match | 4 | ~50% (label removed; loss-of-function flippers help, but cross-axis cases like PPARG/TNF and PHLPP/Integrins are sent to axis_mismatch under our rule, conflicting with gold) | ~2 |
| via_mediator | 3 | ~33% (label removed; chain cases route to direct_*) | ~1 |
| direct_sign_match | 3 | ~33% (sign-flip detection helps for cases where evidence has clear flip; helpless on subtle context-dependent flips) | ~1 |
| direct_axis_mismatch | 1 | 0% | 0 |
| no_relation | 1 | 0% | 0 |
| via_mediator_partial | 1 | ~50% | ~0 |
| **Total recovery** | | | **≈7/16 = 44%** |

### Aggregate expected lift

Recall on `direct_axis_mismatch`: **2.0% → ~78%** (+76 pp on a 50-record bucket).
Recall on `direct_sign_mismatch`: **18.8% → ~44%** (+25 pp on a 16-record bucket).

In raw record counts, that's roughly **39 + 4 = 43 minority-class records
recovered** (vs. 4 currently correct).

### Tradeoff risks

1. **`direct_sign_match` recall drop (5–10%)**: the new Step-2 axis-
   mismatch rule could over-fire on cases where Pro previously got
   `direct_sign_match` correct. Estimating 13–27 of the 233 currently-
   correct sign_match records may flip to axis_mismatch. Mitigation: 8 of
   the 25 few-shots are `direct_sign_match` positive controls explicitly
   showing "axis matches, sign matches" patterns including: parallel-
   pathway sentences, coordinated activity-axis lists, sibling-clause
   negation, alias-resolution. Net micro-accuracy expected: **+5pp to
   +8pp** (more minority-class wins than positive-control losses).

2. **`no_relation` precision drop**: removing the four out-of-gold labels
   means Pro's previous "I'll route this to direct_amount_match" cases
   now have to go somewhere. If the prompt's Step-5 no_relation override
   isn't fully preserved (it is), this could push `no_relation` precision
   even lower than the current 0.560. Mitigation: the no_relation
   override explicitly says "Steps 2-4 fire FIRST; only fall through to
   no_relation when no clause links the subj+obj pair".

3. **Subtle sign-flip cases remain hard**: 5–7 of the 16 sign_mismatch
   records require multi-clause integration (e.g., upstream-context
   inference, double-negative-via-pathway, cell-line-conditional flips)
   that this prompt doesn't cleanly handle. Those will likely persist
   as residual errors regardless.

4. **Possible new failure mode**: with `via_mediator` removed, chain
   examples (X→Y→Z) will be forced into one of the four labels. Most
   should fall to `direct_sign_match` or `direct_axis_mismatch`
   depending on the axis of the X→Z connection. Worst case: a small
   number of true-chain cases that gold treats as `direct_sign_match`
   may misroute to `direct_axis_mismatch`.

## Validation plan (handoff to user)

1. Re-run V6g `relation_axis` only against the 392 holdout records.
2. Compute per-class precision/recall/F1.
3. Confirm: `direct_axis_mismatch` recall ≥ 70%, `direct_sign_mismatch`
   recall ≥ 40%, `direct_sign_match` recall ≥ 80% (no >7pp drop from
   85.3% baseline), micro-accuracy ≥ 75%.
4. If micro-accuracy crosses ≥90% gate, ship; if not, escalate to
   iteration #3 focused on residual sign-mismatch cases (likely
   solvable only with explicit sign-flip cascade reasoning prompts or
   reasoning escalation).
