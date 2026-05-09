# V5 brutalist gate

Date: 2026-05-06
Verdict: **FAIL** (revise before V6)

## Summary

V5 has the right shape — five-probe label spaces, multi-source LFs, Snorkel
LabelModel aggregation, holdout exclusion, V7 hand-validation — but four
load-bearing issues will break V6 if implemented as written. Two are
specification errors (Snorkel API misuse, missing `direct_amount_match` LF
class space mismatch with §3.1 versus §1); two are doctrinal violations the
audit clause (§10) was supposed to catch but failed because V5 trusted that
substrate code is "belief-clean" without tracing the call paths.

The biggest problem is silent **substrate→holdout-tuning leakage**: V5's
"LF_substrate_*" entries route through `context_builder.py`, `relation_patterns.py`,
M9/M10/M11 detectors. The substrate code itself is belief-free — but it has
been **explicitly tuned ON the 482-record holdout's false-fires and false-negatives**
across S/T/U-phases. Every `_CYTOKINE_LIGAND_HGNC` entry, every CATALOG
RelationPattern, every M10 hedge-marker addition was chosen because the holdout
showed the previous version produce a wrong answer. V5 §10's "audit each LF
must be traceable to V4.5 inventory signals" doesn't catch this — the signals
are inventory-clean but the **decision boundaries that consume them are
holdout-derived**. This is a concrete contamination path that will inflate
V8 hand-validation accuracy and V11 feasibility numbers, then collapse on V21.

The verify_grounding training-data shape is also under-specified, the rare
classes have no LF coverage path, and V8 GATE thresholds at 75% are noise
floors at the planned sample size.

Recommend CONDITIONAL pass only after the four BLOCKING items below are
resolved. None of the issues are catastrophic; all are addressable with
~1-2 days of doctrine revision and ~3-5 days of V6 design work before
implementation begins.

## Findings

### Finding A — class-space alignment

**Severity**: MAJOR (specification mismatch between V5 sections)

**Evidence**:

V5 §1's table lists for relation_axis: `direct_sign_match`, `direct_amount_match`,
`direct_sign_mismatch`, `direct_axis_mismatch`, `direct_partner_mismatch`,
`via_mediator`, `via_mediator_partial`, `no_relation` — 8 classes. This MATCHES
`/Users/noot/Documents/indra-belief-model/src/indra_belief/scorers/probes/types.py:43-65`
exactly (8 values; `abstain` was T-phase Fix A removed).

V5 §1's table for scope lists 5: `asserted, hedged, asserted_with_condition,
negated, abstain`. This matches `types.py:67-79`. ✓

V5 §1's table for verify_grounding lists 4: `mentioned, equivalent, not_present,
uncertain`. This matches `grounding.py:42-47`. ✓

**The mismatches are inside V5 itself**:

1. V5 §3.1 LF table has 9 LFs but only one of them (`LF_amount_lexical`) is
   the routing path for `direct_amount_match` — and it's listed only in the
   "Notes" prose, NOT in the table. The 8-class space promises a label V5's
   own table doesn't show how to vote for. (`LF_amount_lexical` should be a
   first-class table row.)
2. V5 §3.1 lists `LF_chain_no_terminal → via_mediator_partial`. Good. But
   nothing in V5 §3.1 votes for `direct_axis_mismatch` except possibly
   `LF_reach_found_by_axis` in the "Translocation→`direct_axis_mismatch`"
   parse mapping. That's an extremely narrow path — REACH found_by parses
   per V4.5 §c carry axis explicitly, so this would only fire when REACH
   parsed the evidence as `Translocation` but the claim is e.g. Activation.
   The other axis-mismatch-producing combinations (claim=Phosphorylation,
   evidence=Activation; claim=Activation, evidence=Phosphorylation;
   claim=Activation, evidence=DecreaseAmount; etc.) are not addressed.
3. V5 §3.1 mentions `direct_partner_mismatch` "requires DNA-binding vs
   Complex/protein detection — substrate-driven LF using M11 signal." But
   there's no LF table row for it. M11 in `relation_patterns.py` matches
   binding-VERB patterns; it does not produce a "this Complex claim has a
   DNA partner" signal as a positive vote. The substrate uses
   `_binding_admissible_for(stmt_type)` to GATE binding patterns, not to
   emit a partner-mismatch label. There is no existing code path that
   converts "evidence text says X binds DNA element when claim is X-Y
   Complex" into a vote — V6 would have to write it from scratch, and
   V5 doesn't acknowledge this.

**Recommended action**: V5 §3.1 must add explicit LF-table rows for every
class. For each rare class (`direct_amount_match`, `direct_axis_mismatch`,
`direct_partner_mismatch`, `via_mediator_partial`), name at least TWO LFs
that can vote for it (one substrate, one extractor-side). If the doctrine
cannot name them, the class is undeliverable in §6's 3K-per-class target
and either the class space shrinks or the doctrine needs synthetic-oracle
generation called out in §10.

---

### Finding B — LF class coverage and class-frequency floors

**Severity**: BLOCKING (multiple classes have ≤1 LF path or no path)

**Evidence**:

Per-class LF inventory (counting only the LFs in §3 tables):

| Probe | Class | Voting LFs | Issue |
|---|---|---|---|
| relation_axis | direct_sign_match | 5 | OK |
| relation_axis | direct_amount_match | 1 (in prose) | Single LF — fragile |
| relation_axis | direct_sign_mismatch | 1 (LF_substrate_negation_regex) | Single LF |
| relation_axis | direct_axis_mismatch | partial (only Translocation parse) | Effectively zero broad coverage |
| relation_axis | direct_partner_mismatch | 0 (only "M11 signal" prose) | NO concrete LF |
| relation_axis | via_mediator | 1 (LF_epistemics_direct_false) | Single LF — narrow |
| relation_axis | via_mediator_partial | 1 (LF_chain_no_terminal) | Single LF |
| relation_axis | no_relation | 1 (LF_no_entity_overlap) | Single LF — and biased toward extreme cases |
| subject_role | absent | 1 (LF_no_grounded_match) | Single LF |
| subject_role | present_as_decoy | 1 (LF_substrate_decoy) | Single LF — narrow |
| subject_role | present_as_mediator | 1 (LF_substrate_chain_position) | Single LF |
| scope | asserted_with_condition | 1 (LF_conditional_clause) | Single LF |
| scope | abstain | 1 (LF_indecisive_text) | Single LF |
| scope | negated | 1 (LF_substrate_negation_explicit) | Single LF |
| scope | hedged | 1 (LF_substrate_hedge_marker) | Single LF — substrate-only |
| verify_grounding | uncertain | 1 (LF_evidence_too_short) | Single LF |
| verify_grounding | not_present | 1 (LF_gilda_no_match) | Single LF |
| verify_grounding | equivalent | 2 (LF_gilda_alias, LF_gilda_family_member) | OK |
| verify_grounding | mentioned | 1 (LF_gilda_exact_symbol) | Single LF |

**14 of 22 non-trivial classes have exactly 1 LF voting for them.** This
makes Snorkel's LabelModel mathematically incapable of estimating that LF's
accuracy from agreement structure — accuracy is identifiable only from
disagreement with OTHER LFs voting on the same record.

**Class-frequency floors**: §6 requires 3K examples per class to train.
Several classes will be naturally rare:
- `direct_partner_mismatch`: requires DNA-binding-vs-Complex confusion. In
  the corpus, Complex statements with DNA partners are a small fraction
  of binding-verb evidence — perhaps 1-3% of binding evidence. If binding
  evidence is ~15-20% of corpus (per V4.5 §a stmt_type distributions), that
  gives at most 0.3-0.6% of (statement, evidence) pairs even **before**
  Snorkel's confidence threshold (§4 step 4: drop records with proba<0.5).
  Realistic surviving count: **<500 records**, far below 3K target.
- `via_mediator_partial`: requires "thereby/leads to/mediated by" markers
  WITHOUT a named intermediate. This is a tiny fraction of via_mediator
  candidates. Expected <1K.
- `direct_axis_mismatch`: V5's only concrete path is REACH-Translocation-
  parses-claim-Activation. Per V4.5 §c, Translocation patterns are ~few-K
  out of 1.29M found_by patterns — and the additional constraint of "claim
  is on different axis" cuts this further. Expected <2K.
- `abstain` (scope): undefined how often "evidence too short or generic"
  fires; depends on §10 contamination filter's 30%-overlap definition (see
  Finding G). Could be very rare or very common.
- `present_as_decoy`: depends on substrate decoy-detector definition,
  which V5 doesn't pin down. Could be undefined — there's no
  `_substrate_decoy` function in the current codebase.
- `present_as_mediator`: same — no current substrate code returns this
  signal as a discrete vote, V5 promises "M11 chain-position" but
  `relation_patterns.py` doesn't expose a per-entity middle-node detector.

**Recommended action**:
1. Per probe, add at LEAST 2 LFs voting for every class. If a class can
   only be reached by one substrate function, write a complementary
   lexical or extractor-side LF (e.g., found_by-pattern parse, source_api
   convention, alias presence/absence).
2. For each class, estimate the natural frequency BEFORE writing V6. If
   <1K natural records, V5 must call out synthetic-oracle generation in
   §10 as the planned mitigation, with a target count (200-500 placeholder
   examples per rare class), AND specify that those examples are excluded
   from the contamination check (since they're synthetic).
3. Verify with V6 dry-run that LabelModel converges on each probe's Λ
   matrix before committing to the V8 gate.

---

### Finding C — Snorkel LabelModel API misuse and cardinality realism

**Severity**: BLOCKING (specification refers to API that doesn't exist)

**Evidence**:

V5 §10 row 3: "*declare known-correlated pairs explicitly via `dependencies=`
argument*". Snorkel's actual `LabelModel.fit()` signature (snorkel 0.9.x,
the only released line of the data-programming aggregator):

```python
def fit(self, L_train, Y_dev=None, class_balance=None, **kwargs)
```

There is **no `dependencies=` parameter**. Snorkel's old `DependencySelector`
class (Snorkel 0.6/0.7 era) exposed a separate API, was deprecated, and is
not part of the `LabelModel` constructor either. Correlation diagnosis is via
`snorkel.labeling.LFAnalysis.lf_conflicts()` and `lf_summary()`; remediation
is by writing an LF that aggregates the correlated set, not by passing
correlation info into `fit()`. The doctrine prescribes a control knob that
doesn't exist — V6 implementation will discover this and have to invent
an unspecified workaround.

**Cardinality 8 risk** (relation_axis):
- The Ratner et al. 2017 LabelModel paper benchmarks K=2; the original
  Snorkel paper extends to K=3,5. The model's identifiability proofs depend
  on the per-class LF accuracy being estimable from **pairwise agreement
  matrices** — which gets harder as K grows because per-pair-per-class
  estimates require more data, and the LFs themselves get sparser per class
  (cf. Finding B: half the relation_axis classes have 1 LF).
- At K=8 with imbalanced LF coverage and ~1-2 LFs voting on rare classes,
  the LabelModel is more likely to collapse those classes to the majority
  class than to recover accurate per-LF coefficients. The output will
  "look fine" (proba mass concentrated on majority) but mask the failure.

**Stratified subsampling ordering** (V5 §6):
> "stratified subsampling AFTER LabelModel fit"

This is the wrong order if the goal is calibrated probabilistic labels.
LabelModel's per-LF accuracy estimates are derived from the natural class
distribution of the Λ matrix (it implicitly assumes the empirical class
balance OR you pass `class_balance=`). If you fit on the natural distribution
then subsample to balance afterwards, you get correctly-calibrated proba
scores from a model fit to the imbalanced corpus, applied to a balanced
training set — the resulting LoRA adapter inherits the calibration of the
imbalanced source distribution, not the balanced target. For rare classes
this means the model under-confidences correct rare-class predictions.

The right order: pass `class_balance=` to `LabelModel.fit()` reflecting
either the natural distribution or the EQUAL distribution you want at
training time — depending on whether you want the soft labels to reflect
prior beliefs or to be balanced. V5 §6 doesn't acknowledge this choice.

**Recommended action**:
1. Strike the `dependencies=` reference from §10. Replace with: "diagnose
   high-correlation LF pairs via `LFAnalysis.lf_summary()` BEFORE fit; if
   correlation > 0.5, merge the pair into a single LF or drop the redundant
   one."
2. For relation_axis at K=8: V5 must specify either (a) a fallback to
   per-axis-pair binary classifiers chained via decision rules, or (b) a
   reduced cardinality (e.g., merge `direct_amount_match` + `direct_sign_match`
   under a single "direct" head and let the adjudicator do the activity-vs-
   amount split via U7 logic at inference time). Option (b) preserves the
   §1 production label space at output but reduces V6 training cardinality.
3. §6 must specify `class_balance` parameter to `LabelModel.fit()` and
   reorder: fit→soft-labels→stratified subsample, with explicit
   acknowledgment that the resulting LoRA adapter is calibrated to whichever
   class_balance was passed.

---

### Finding D — substrate→holdout-tuning leakage (the audit clause failed)

**Severity**: BLOCKING — doctrine violation of V0 §4.4

**Evidence**:

V5 §10 row 4 audits "each LF must be traceable to V4.5 inventory signals;
reject any LF that touches `belief` field." V5 §3 names many `LF_substrate_*`
entries. Tracing one:

`LF_substrate_negation_regex` consumes substrate output from
`/Users/noot/Documents/indra-belief-model/src/indra_belief/scorers/relation_patterns.py`
which contains the CATALOG.

`LF_substrate_chain_position` and `LF_substrate_decoy` would consume
detection logic in
`/Users/noot/Documents/indra-belief-model/src/indra_belief/scorers/context_builder.py`.

**The substrate code does NOT directly read `statement.belief`** — verified
via grep. So the §10 audit, taken literally, passes.

**But the substrate code IS holdout-tuned**. Direct quotes from the source:

`context_builder.py:452-453`:
> "Conservative starting set. Each entry is a canonical HGNC symbol whose
> product is a cytokine, secreted ligand, or growth factor that INDRA
> curators treat as a valid upstream-attribution agent. **Expanding this
> set is L9-gated: only after the holdout shows no FPs from the current
> set do we add more.**"

`context_builder.py:632-634`:
> "L7-fix for C3b: known protein-family / non-site tokens that match the
> letter-form regex but are NOT modification sites. **Conservative starting
> set — expand as holdout shows new false-fires.**"

`context_builder.py:836-849` (M9):
> "The diagnosis traced ~20 FNs and several FPs to perturbation language
> the parser missed: FN MEK→MAPK1: 'inhibiting MEK ... blocked ERK
> phosphorylation' ... FP HDAC→AR: 'HDAC inhibitors induced AR acetylation'"

`context_builder.py:740-746` (M10):
> "Diagnosis FP: CCR7→AKT — 'CCR7 may activate Akt' is a hypothesis-level
> claim, not asserted."

`context_builder.py:765-769` (N2 fix):
> "Diagnosis: M13 surfaced 9 hedging_hypothesis regressions where the
> 60-char window crossed semicolons..."

`relation_patterns.py:19`:
> "Each pattern is justified by at least one diagnosis FN it must close;"
(followed by lines 105, 235, 436, 560, 672 each citing diagnosis FN claim
identifiers — these come from running the scorer on the 482-record
holdout through S/T/U-phases).

**The pattern is unambiguous**: the substrate code's decision boundaries
were chosen because earlier holdout runs showed where they were wrong.
That's holdout-derived training. A V5 LF that consumes
`ctx.detected_relations` or `ctx.subject_perturbation_marker` or
`ctx.explicit_hedge_markers` is downstream of decisions tuned ON the
benchmark V-phase will be evaluated on. The signal is "belief-free" but
not "evaluation-blind."

**Why V5 §10 misses this**: the audit checks for `belief` field reads, not
for tuning-against-holdout. The contamination is in the regex thresholds,
denylists, marker windows, and family-member sets — not in the data-flow
graph V5 inspects.

**Concrete consequence**: an LF that uses `LF_substrate_hedge_marker` to
vote `hedged` will be RIGHT on holdout-overlapping CCR7→AKT records (because
the substrate was tuned against them) and on similar holdout-paraphrase
records — but generalize less than V8 will measure. The LabelModel will
learn an inflated accuracy for these LFs. The LoRA adapter will inherit the
inflated calibration. V21 holdout accuracy will be lower than V11 estimates
predict.

**KG signal note (separate but related)**: `kg_signal.py:142` reads
`belief = float(stmt_json.get("belief", 0.0))` directly. The orchestrator
calls `_kg_signal.get_signal()` which exposes `max_belief` to the
adjudicator. V5 doesn't propose using kg_signal as an LF — but it's worth
flagging defensively in V6 review that NO LF can transitively call
`get_signal()` even via the orchestrator code path.

**Recommended action**:
1. Add a new audit clause to V5 §10: "**no LF may consume a substrate
   feature whose definition was modified after Date X based on holdout
   evaluation.**" Pick X = first commit after the 482 holdout was extracted.
2. Concretely: list the substrate features that are holdout-tuned
   (`_CYTOKINE_LIGAND_HGNC`, `_SITE_DENYLIST`, `_HEDGE_MARKERS`,
   `_LOF_PATTERNS`, every CATALOG entry past the M5 baseline) and either:
   - Forbid LFs from reading them, OR
   - Quantify the leakage: run `LF_substrate_*` LFs with the substrate
     reset to its M5 baseline (pre-holdout-tuning state) and compare the
     vote distributions. The delta is the contamination floor.
3. If option (a): V5 must replace `LF_substrate_*` with `LF_lexical_*`
   that re-implements the surface-form detection from scratch using ONLY
   V4.5 inventory + canonical biomedical lexicons (not holdout-fitted
   denylists/allowlists).
4. Document the contamination guard in `feedback_substrate_holdout_tuning.md`
   (NEW file, sibling to `feedback_substrate_vs_llm_lever.md`).

---

### Finding E — Snorkel realism (covered partially in C, additional class-balance issue)

**Severity**: MAJOR — see also Finding C bullet 3

**Evidence**: see C above. Additional point: V5 §4 claims the LabelModel
"learns per-LF accuracy and per-pair correlation purely from agreement
structure (no gold labels needed)." This is partially misleading — Snorkel's
LabelModel needs the agreement structure to be **non-singular**. With
singletons (single-LF classes per Finding B), the corresponding rows of the
agreement matrix have only auto-agreement (LF agrees with itself trivially)
and no cross-LF signal. The model will collapse those classes to whatever
prior the optimization initializes from. V5 §10 says "min 4 LFs per probe
with non-singular agreement structure" — but the constraint should be
**per-class**, not per-probe. A probe with 4 LFs but where 3 vote on the
majority class is not non-singular for the rare classes.

**Recommended action**: see Finding B (add LFs per class) and Finding C
(specify class_balance).

---

### Finding F — verify_grounding training-data shape is undefined

**Severity**: MAJOR (silent data-shape ambiguity)

**Evidence**:

`grounding.py:223` shows `verify_grounding(claim_entity, evidence_text, client)`
takes a single `GroundedEntity`. The orchestrator
(`probes/orchestrator.py:295-297`) loops over entities:
```python
for e in entities:
    grounding_list.append(verify_grounding(e, evidence_text, client))
```
For a typical claim, `entities` includes the subject AND the object — sometimes
more if the claim is multi-agent (e.g., Complex with 3+ members). So per
(statement, evidence) pair, verify_grounding produces 2-N records, not 1.

V5 §2 says "Each (statement, evidence) pair generates up to 5 training records
— one per probe." This implicitly says 1 verify_grounding record per pair.
But production calls verify_grounding 2-N times. Which is the V6 training
shape?

Two unclean options if V5 doesn't pick:
- (a) Generate one verify_grounding training record per (statement, evidence,
  entity) triple. That's correct production fidelity but blows up the count
  estimate in §6 (12K target × ~2.3 entities = 28K actual records).
- (b) Generate one training record per (statement, evidence) pair using the
  SUBJECT entity only. That's wrong — production also queries the object,
  and at inference time the LoRA adapter will be invoked twice with
  identical-format prompts; both calls need to behave correctly.

**Prompt-format question**: V5 §8's training schema says "matches production
prompt verbatim." Production's prompt (`grounding.py:_build_user_message`)
takes a `GroundedEntity` and renders structured fields:
```
Claim entity: <name>
Grounding: <db>:<db_id>  (or "<none — possibly a generic class noun>")
Family members: <list>  (if is_family)
is_pseudogene: true  (if is_pseudogene)
Aliases: <list>
Gilda score (low confidence): <float>  (if is_low_confidence)

Evidence: "<text>"
Does the evidence reference this entity?
```

V5 doesn't say how V6 will reconstruct GroundedEntity (with Gilda annotations,
family members, pseudogene flags) for each training record. The corpus's
INDRA Statements have agent objects but not GroundedEntity wrappers — Gilda
must be re-run per agent during V6 derivation, OR a Gilda-cache must be
built. V5 also doesn't address how GroundedEntity's `is_low_confidence` and
`gilda_score` thresholds are set (these affect prompt content and therefore
training-time-vs-inference-time format match).

**Recommended action**:
1. V5 §2 must specify: verify_grounding generates ONE training record per
   (statement, evidence, entity) triple, where entity ∈ claim agents.
2. §6 sample-size targets must be updated: 12K verify_grounding target →
   28K actual at production multiplicity (or pick a lower target acknowledging
   the multiplier).
3. §8 must specify: V6 derivation runs Gilda per agent and renders the
   `_build_user_message` format verbatim, with the **same** thresholds
   used at production-inference time. V6 dry-run validates that for a
   sample of holdout-excluded records, the rendered prompt is byte-equal
   to what production would emit.
4. LF assignment for verify_grounding (§3.4) must be per-entity, not
   per-pair: an LF firing for the subject doesn't carry over to the object.

---

### Finding G — contamination guard specification is too soft

**Severity**: MAJOR (false-positive AND false-negative risk)

**Evidence**:

V5 §10 row 5: "≥30% n-gram overlap" — V5 doesn't specify n.
- If unigram (n=1): biomedical evidence sentences from the same domain share
  high unigram overlap by default (gene names, "phosphorylation",
  "cells", "expression"). 30% unigram overlap would drop large fractions of
  legitimate training data.
- If trigram or higher (n=3): paraphrase contamination slips through. A
  holdout sentence "MAPK1 phosphorylates JUN at Ser63" vs a corpus sentence
  "MAPK1 phosphorylates the substrate JUN at S63 residue" shares few exact
  trigrams ("MAPK1 phosphorylates the", "phosphorylates the substrate",...)
  and might pass <30% trigram overlap despite being a clear paraphrase.

V4.5 reports no measurement of natural overlap distribution in the corpus.
Without that baseline, "30%" is unanchored — V6 won't know whether it's
filtering 0.1% of records (too lax) or 50% (too strict).

**Recommended action**:
1. V5 must specify: n (suggest n=3 trigram), normalization (lowercase,
   strip punctuation, drop stopwords + biomedical-stopword list), and the
   overlap measure (Jaccard? cosine on count vectors?).
2. V6 must include a calibration step: compute the natural n-gram-overlap
   distribution between corpus pairs, set the threshold at the 95th
   percentile of natural overlap, then drop training records exceeding
   that threshold against any holdout evidence.
3. Threshold should be class-conditional if possible: same-stmt-type
   pairs naturally overlap more than cross-type pairs.
4. Cross-check with `scripts/check_contamination.py` (the tool from
   feedback_contamination_guard.md) to ensure the V6 contamination filter
   uses the same algorithm the existing pipeline uses, not a parallel
   reimplementation that drifts.

---

### Finding H — V8 GATE thresholds are statistical noise

**Severity**: MAJOR (can pass V8 by random sampling at planned N)

**Evidence**:

V5 §7 prescribes 50-100 hand-labeled records, stratified 30/50/20 by confidence
across 5 probes. Worst case: 50 total records / 5 probes = 10 records per
probe. Best case: 100 / 5 = 20. The 30% confident bucket gives 3-6 confident
records per probe. The 20% rare-class bucket gives 2-4 records per probe.

V8 GATE: "≥75% LF accuracy on confident predictions per probe."

At N=5 confident samples per probe, a binomial 95% CI for an observed
75% (4/5 correct) is approximately [29%, 96%] — half the [0,1] range.
At N=10 (best case), 75% (7-8 correct) gives 95% CI ≈ [40%, 94%]. The
threshold doesn't distinguish the LF set from random noise.

For "no class with <30% accuracy", with 2-4 rare-class samples, the
threshold is even worse: a single bad luck draw fails the gate; a single
good draw passes. The gate emits a yes/no signal that is almost entirely
sampling-driven.

**Power calculation** (for ≥75% target with 95% CI half-width ≤10pp):
- Per probe, per stratum: need ≈75 confident samples for 75% with ±10pp
  half-width. Total: 75 × 5 probes × 3 strata ≈ 1125 hand-labels.
- At ±15pp tolerance (looser): need ≈30 per stratum per probe = 450 total.
- At ±20pp: ≈18 per stratum per probe = 270 total.

**Recommended action**:
1. V5 §7 must specify either:
   - (a) Larger sample (≥250 total, ≥50 per probe), accepting the labeling
     cost. This is feasible per V0 §10's "~80K unique pairs" headroom and
     the user's stated availability.
   - (b) Use U2's per-probe gold (`data/benchmark/probe_gold_holdout.jsonl`,
     482 records with curator-derived per-probe labels — see Finding I)
     for V8 LF-calibration on a held-back portion of the holdout instead
     of fresh hand-labeling. This is a fundamentally better signal than
     synthetic hand-labels.
2. V5 §7 GATE thresholds must include CI half-widths, not just point
   estimates: "75% LF accuracy with 95% CI lower bound ≥ 65%" is testable;
   "≥75%" alone is not.
3. The 30/50/20 stratification needs justification — if the goal is to
   detect failure modes, oversampling rare classes (the 20% bucket) makes
   sense, but it should be expanded (50% rare, 30% borderline, 20%
   confident is the inverse and arguably better for catching bugs that
   matter — confident-bucket failures are rare; rare-class-bucket failures
   are common and load-bearing).

---

### Finding I — V5 ignores U2's per-probe gold (missed opportunity, calibration risk)

**Severity**: MAJOR (calibration evidence available, V5 doesn't use it)

**Evidence**:

`research/u2_kg_analysis.md` documents that the 482-record holdout has
curator-assigned `gold_tag` values (12 categories: correct, no_relation,
grounding, wrong_relation, act_vs_amt, polarity, hypothesis, negative_result,
etc.). U2 saved `data/benchmark/probe_gold_holdout.jsonl` carrying per-record
fields: `gold_target`, `gold_tag`, `claim_axis`, `claim_sign`, plus KG
context. These ARE per-probe labels: `polarity` ↔ relation_axis sign;
`act_vs_amt` ↔ relation_axis amount-vs-activity; `hypothesis` ↔ scope hedged;
`negative_result` ↔ scope negated; `grounding` + `entity_boundaries` ↔
verify_grounding.

V5 doesn't reference this artifact. Three concrete missed uses:

1. **LF accuracy calibration (zero training-set cost)**: For each LF, compute
   its accuracy on the U2-gold subset. This gives Snorkel a Y_dev to pass
   to `LabelModel.fit(L_train=Λ, Y_dev=Y_holdout)` — the LabelModel API
   ACCEPTS this. (`Y_dev` is the optional gold-label vector for a held-out
   subset of training records.) That converts the data-programming approach
   from blind to semi-supervised, with no risk of train-test leakage if the
   holdout is excluded from L_train (which V5 §5 already does).

   **However**, this conflicts with V0 §4.4's "no use of curator gold for
   training." Reconcilation: U2 gold is FOR EVALUATION, but Snorkel's `Y_dev`
   is used for VALIDATION of LF accuracy estimates, not for training labels.
   This is a defensible interpretation — Y_dev tunes LabelModel's internal
   parameters, not the LoRA adapter. V5 should make this explicit and
   either permit it or forbid it intentionally.

2. **Cross-validation of LabelModel accuracy estimates**: After fitting,
   compare Snorkel's learned per-LF accuracies (`get_weights()`) against
   what U2 gold computes. Disagreement >10pp on any LF is a red flag that
   Snorkel's identifiability collapsed.

3. **V8 evaluation**: Use U2 gold on the holdout as a SECOND signal alongside
   V7 hand-labels. This addresses Finding H's power problem — 482 curator-
   gold labels >> 50-100 fresh hand-labels.

**Recommended action**:
1. V5 must add a §3.5 or new section explicitly addressing U2 gold:
   either use it (defining how) or forbid it (defining why). Silence is
   the worst option because V6 implementers won't know which is intended.
2. Recommended use: pass a holdout-subset `Y_dev` to `LabelModel.fit()` for
   accuracy calibration WITHOUT including those records in `L_train`. This
   satisfies both V0 §4.4 (no curator gold in training labels) and improves
   LabelModel's per-LF estimates.
3. V8 GATE should consume U2 gold for LF-accuracy validation alongside
   V7 hand-labels.

---

### Finding J — substrate→holdout-tuning quantification (formalization of D)

**Severity**: BLOCKING (need a measurement before V6 ships)

**Evidence**: see Finding D above. The contamination is real but its
magnitude is unknown. Until quantified, V5's risk register understates the
likely V21 holdout regression.

**Recommended action**:
1. V6 derivation script must include a "substrate-baseline" mode: run all
   `LF_substrate_*` LFs with substrate reverted to its first-commit state
   (no holdout-tuning entries in `_CYTOKINE_LIGAND_HGNC`, no L7-fix
   denylist, only M5-era CATALOG patterns).
2. Compare the LF vote distributions on the holdout subset between
   "tuned-substrate" and "baseline-substrate" modes. The delta in vote
   counts per class is the contamination magnitude.
3. If delta > 10pp on any class for any holdout-overlap subset: V8 fails
   and V5 must be revised to use baseline-substrate LFs only.
4. This measurement becomes the new V8 gate criterion alongside the
   existing thresholds.

---

## What V5 must change before V6 implementation

These are the **load-bearing fixes**. The doctrine is binding once these
are addressed.

1. **Re-write §3.1 and §3.2 LF tables to enumerate ≥2 LFs per class**
   (Finding A + B). For each class with no concrete LF (`direct_partner_mismatch`,
   `present_as_decoy`, `present_as_mediator`), either name a new LF and
   describe its detection logic, or remove the class from §1. Specifically:
   - Add `LF_amount_lexical` (currently in §3.1 prose) as a first-class
     LF table row.
   - Add ≥1 LF reaching `direct_axis_mismatch` beyond the narrow
     Translocation-claim-Activation case (e.g., `LF_reach_pattern_mismatch`
     using found_by → claim's stmt_type axis comparison).
   - Concretely define the M11 partner-mismatch detector or remove the
     class from §1.
   - Concretely define `LF_substrate_chain_position` and `LF_substrate_decoy`
     or remove `present_as_mediator` and `present_as_decoy` from §1.

2. **Strike `dependencies=` argument from §10** (Finding C). Replace with
   `LFAnalysis.lf_summary()` for diagnosis + LF-merge for remediation.
   Specify Snorkel version (`snorkel==0.9.9` or pinned).

3. **Specify `class_balance` for `LabelModel.fit()`** (Finding C+E) and
   reorder §6: pass class_balance reflecting the EQUAL distribution if
   targeting balanced training, else the natural distribution. Document
   the tradeoff.

4. **Audit clause for substrate→holdout-tuning** (Finding D + J). Add to
   §10: "no LF may consume a substrate feature whose entries were added
   to handle a holdout false-fire/false-negative." Either forbid such LFs
   and replace them with from-scratch lexical LFs, or add §10's quantified
   substrate-baseline measurement as a V8 gate.

5. **Specify verify_grounding training shape** (Finding F). Confirm
   per-(statement, evidence, entity) record. Update §6 sample-size targets
   to match. Specify Gilda re-run + GroundedEntity reconstruction in V6.
   Per-entity LFs in §3.4.

6. **Define n-gram contamination guard precisely** (Finding G). Specify n,
   normalization, threshold derivation from natural-distribution baseline.

7. **Increase V8 sample size or use U2 gold** (Finding H). Either ≥250
   hand-labels with confidence-interval-based gates, or use U2's 482-record
   per-probe gold for LF calibration.

8. **Reference U2 per-probe gold** (Finding I). New section deciding either
   to use Y_dev calibration in LabelModel.fit() or to forbid it. Recommended:
   permit it for accuracy calibration only, not for label generation.

## What V5 already gets right

These are the load-bearing decisions to PRESERVE:

- **The five-probe label space is correct**. §1 matches `types.py` and
  `grounding.py` exactly. The probe count and answer sets are right.
- **The multi-source weak-supervision frame is the right approach**. V4.5's
  empirical inventory supports the decision; the alternative of fully
  manual labeling at this scale is infeasible.
- **Holdout-exclusion on (source_hash, matches_hash, pmid+entity_pair)** is
  defensible. V4.5's count of ~2,500 unique-after-union confirms the union
  isn't dropping more than necessary, and the conscious choice to exclude
  pmid-only is correct (would lose 1% of corpus for marginal contamination
  reduction).
- **§4 LabelModel proba threshold of 0.5 with drop-on-fail** is right. Better
  to train on confident records than to noisy-fit borderline cases.
- **§8 emitting both `completion` and `soft_labels`** preserves the option
  for KL-distillation in V9. Good design.
- **§10's contamination guard with synthetic placeholder names** matches
  the May 2026 contamination-guard memory and prevents repeat of
  feedback_fewshot_contamination incident.
- **§11 explicit non-commitments** (LoRA hparams in V9 not V5, prompt
  wording = production verbatim) keep V5 focused on data-prep contract.
- **V0 §4.4 "no belief score" enforcement** at the call-graph level.
  Confirmed `kg_signal.py` is the only `belief`-reading site in scorers/;
  no V5 LF references it. The strict commitment holds at this level (the
  separate substrate-tuning issue is a refinement of the audit).
- **V5 §12 ship criteria are testable**. The conditions for V6 unblock are
  unambiguous.

The doctrine is structurally sound. The load-bearing flaws are in the
LF inventory specifics, the Snorkel API misstatement, and the audit
clause's coverage gap — all addressable in 1-2 days of revision.
