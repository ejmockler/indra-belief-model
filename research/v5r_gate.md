# V5r brutalist re-gate
Date: 2026-05-06
Verdict: **CONDITIONAL** (two BLOCKING factual errors; remainder of fixes are sound or close-enough; V6 may proceed once the two blockers are corrected by an in-place edit to V5r)

## Summary

V5r resolves 6 of 8 V5 gate findings cleanly and resolves Fix 4's structural concern about substrate-tuning quantification. But two of the eight fixes contain factual errors that must be corrected before V6c can compile against the spec: (1) Fix 2's Snorkel API description still mis-describes what `LFAnalysis.lf_summary()` produces (it does NOT report pairwise correlations; Snorkel 0.9.9 has no public correlation utility), and (2) Fix 4's "baseline-substrate = M5-era CATALOG, empty `_CYTOKINE_LIGAND_HGNC`, empty `_SITE_DENYLIST`" is not a recoverable git state — `relation_patterns.py` and `context_builder.py` were both introduced in commit 70d821f as a single S-phase ship, so there is no "first commit" pre-tuning state to revert to. V5r §7c requires V6a to construct a baseline that the doctrine names but does not define. Two new MAJOR issues from revision-introduced material: (3) the Y_dev semantics in V5r §4 step 4 mis-state what Y_dev does in Snorkel 0.9.9 (it tunes class_balance, NOT per-LF accuracy), and (4) the §4 step 4 Y_dev path implicitly does the curator-gold filtering V5r §11 forbids.

## Per-fix verification

### Fix 1 (Findings A + B): >=2 LFs per class

**Status**: RESOLVED with one MINOR residual + one NEW concern about LF realism

**Evidence**: V5r §3.1-§3.4 add 7 named LFs versus V5 (`LF_chain_with_named_intermediate`, `LF_partner_dna_lexical`, `LF_decoy_lexical`, `LF_amount_lexical`, `LF_amount_keyword_negative`, `LF_role_swap_lexical`, `LF_chain_position_lexical`, `LF_evidence_contains_official_symbol`, `LF_fragment_processed_form`, `LF_text_too_short`, `LF_negation_lexical`, `LF_hedge_lexical`, `LF_conditional_lexical`). The class-coverage tables at the end of each subsection (§3.1, §3.2, §3.3, §3.4) audit each class. After review:

- relation_axis: 5/8 classes have >=2 LFs; `direct_amount_match`, `via_mediator_partial`, `no_relation` have 1 LF each. V5r §6 explicitly tags all three as "synthetic-oracle augmentation if natural <3K". Compliant.
- subject_role / object_role: 4/5 classes have >=2 LFs; `absent` has 1 LF. V5r §3.2 says "augment via per-statement deterministic check: claim entity name + all aliases not found in evidence text → absent". This is a SECOND LF in disguise but isn't tagged; should be promoted to a named row `LF_absent_alias_check`.
- scope: 4/5 classes have >=2 LFs; `asserted` has 1 LF. V5r §3.3 says "Plus implicit: when no other scope LF fires AND claim is in evidence, default to `asserted` via LabelModel calibration". This is NOT a second LF — it is a claim about how Snorkel's LabelModel will collapse-vote on records where no LF fires. But Snorkel's LabelModel doesn't auto-cast a "default" class on all-ABSTAIN records; it will simply emit a uniform-prior probability and the §4 step 6 threshold (>=0.5) will drop those records. So `asserted` will be UNDER-represented in training, not "implicit-defaulted". This is a residual MAJOR if uncorrected, but easily fixed by adding a real `LF_asserted_default` LF that fires when none of {hedge, negation, condition} fire AND claim verb is present — which is what `LF_clean_assertion` already is. Functionally OK; just rename or remove the misleading "implicit" comment.
- verify_grounding: 4/4 classes have >=2 LFs (`not_present` has 1 named LF + an "implicit when no other LF fires" comment that is false for the same reason as `asserted` — but `LF_gilda_no_match` is a positive-firing LF, so the implicit-collapse comment is just unnecessary noise, not a hole).

**LF realism check** (the V5g task asks specifically about V5r-introduced LFs):

- `LF_chain_with_named_intermediate` — pattern `<claim_subj> {verb} <X>{ thereby|that|which then } {verb} <claim_obj>` where X != claim entities and X is "a recognizable biomedical entity". Realistic if "recognizable biomedical entity" delegates to Gilda. V6 implementation must spell out the Gilda call here, but this is implementable in pure Python regex + Gilda lookup. Acceptable.
- `LF_partner_dna_lexical` — pattern matching `promoter`, `enhancer`, `binding site` etc. anchored on the claim subject. Implementable as regex. Acceptable.
- `LF_decoy_lexical` — "Claim entity present in evidence but in a relation pattern that does NOT match claim's stmt_type axis". This is the MOST suspect. To detect "in a relation pattern that does NOT match claim's stmt_type axis", the LF needs to know what "the relation pattern in evidence" is — which is exactly what M11/M12 detect. Either this LF must call into substrate (then it's not `[clean]` — it's substrate-tuned) or V6 must reimplement an axis-tagging pattern matcher from scratch. V5r tags it `[clean]` but the description implies substrate dependency. **MINOR/MAJOR risk**: V5r should either re-tag as `[substrate-tuned]` or specify an independent regex catalog (which would itself need to be tuned somehow).
- `LF_amount_lexical` — straightforward regex; realistic.
- `LF_amount_keyword_negative` — fires when "Claim is IncreaseAmount/DecreaseAmount AND evidence describes activity-axis (`phosphorylat`, `activates`, `binds`) NOT amount". Implementable. Acceptable.
- `LF_role_swap_lexical` — "Lexical pattern `<claim_obj> {active-verb} <claim_subj>`". Realistic but the scope of "active-verb" needs a closed list. V6 must spell out which verbs.
- `LF_chain_position_lexical` — `via X`, `through X`, `mediated by X` patterns. Realistic.
- `LF_fragment_processed_form` — pre-baked list of `cleaved <X>`, `phosphorylated <X>`, `Aβ` for APP, etc. Realistic but the "Aβ for APP" specifically comes from substrate's M-phase tuning experience (the holdout has Aβ→APP cases). This LF is `[clean]` in V5r tagging but its lexicon was selected because of holdout audit — same contamination pathway as Finding D. **MINOR**: the "synthetic-placeholder" or "M5-era only" discipline on lexicons is silently violated here. V5r §13 should call out lexicon provenance.

### Fix 2 (Finding C): Snorkel API correctness

**Status**: PARTIALLY RESOLVED — the `dependencies=` reference is gone, but V5r introduces a NEW factual error about `lf_summary()` capabilities

**Evidence**: V5r §4 step 2 says: "Run `LFAnalysis(L=Λ, lfs=lf_list).lf_summary()` to diagnose **pairwise correlations** and per-LF coverage". 

I verified against Snorkel 0.9.9 source (https://raw.githubusercontent.com/snorkel-team/snorkel/v0.9.9/snorkel/labeling/analysis.py): `lf_summary()` returns a DataFrame with columns `Polarity, Coverage, Overlaps, Conflicts` (and optionally `Correct, Incorrect, Empirical Accuracy, Learned Weight` if Y/est_weights provided). It does **NOT** compute pairwise correlations. None of LFAnalysis's 10 public methods compute pairwise LF correlation; they compute per-LF or global aggregates only. The pairwise correlation tool that was in pre-0.9 Snorkel (`DependencySelector`) is gone.

Concretely: **V5r's "if any pair has correlation > 0.5: merge ..." rule cannot be evaluated using `lf_summary()`**. V6c will discover this and either need to (a) compute correlations manually from the Λ matrix (numpy `np.corrcoef`), or (b) drop the rule. The doctrine prescribes a diagnostic that doesn't exist in the API it cites. This is the SAME class of error as V5's `dependencies=` — a fictional API control knob — just at a different layer.

`Y_dev` argument: verified — `LabelModel.fit(L_train, Y_dev=None, class_balance=None, **kwargs)` is the correct signature in 0.9.9. V5r's call shape is correct.

**Severity**: This is a BLOCKING factual error in the doctrine. V5g recommends in-place fix to V5r §4 step 2: "Compute pairwise LF agreement from Λ via `np.corrcoef` on the (n_records × n_LFs) integer-vote matrix (treating ABSTAIN as a sentinel value, or filtering to records where both LFs fire). If correlation > 0.5: merge..." OR drop the rule entirely.

### Fix 3 (Finding C/E): class_balance specified

**Status**: RESOLVED with one weakly-justified choice

**Evidence**: V5r §4 step 3 commits per-probe class_balance:
- relation_axis: equal
- subject_role / object_role: equal
- scope: equal
- verify_grounding: natural

The relation_axis / subject_role / object_role / scope choices are coherent: balanced training makes the LoRA adapter equally sensitive across rare and common classes, which is what V8/V9 measure (per-class accuracy floors).

The verify_grounding "natural" choice is justified weakly: "entity grounding is naturally skewed `mentioned`-heavy and that's correct prior". This argument is REASONABLE but conflicts with the §6 logic for the other 4 probes — V5r doesn't acknowledge that the asymmetry exists. If the V8 gate requires per-class accuracy floors on verify_grounding (as it does on the other 4), training under "natural" will under-train `not_present` and `uncertain` cases. **MINOR**: V5r should either (a) commit to a uniform "equal" choice across all 5 probes for consistency with V8 per-class gate, or (b) explicitly acknowledge that verify_grounding's V8 gate is asymmetric and explain why.

### Fix 4 (Finding D + J): substrate→holdout-tuning quantification

**Status**: BLOCKING — "baseline mode" is not concretely recoverable

**Evidence**: V5r §3 tags `LF_substrate_*` rows with `[substrate-tuned]` (Compliant). V5r §7c specifies V7c as a measurement comparing `tuned mode` vs `baseline mode`, where baseline is "M5-era CATALOG, empty `_CYTOKINE_LIGAND_HGNC`, empty `_SITE_DENYLIST`, no diagnosis-fix entries" with a 10pp per-class delta gate.

I verified the git history:
```
git log --all --oneline -- src/indra_belief/scorers/relation_patterns.py
70d821f arch: ship S-phase four-probe scorer (74.58%, +1.72pp vs R-phase)

git log --all --oneline -- src/indra_belief/scorers/context_builder.py
70d821f arch: ship S-phase four-probe scorer (74.58%, +1.72pp vs R-phase)
```

Both files were introduced in a SINGLE COMMIT (70d821f). All M-phase markers (M5 FPLX backfill, M9 perturbation detection, M10 hedge marker, M11 binding patterns, M13 hedging-hypothesis, etc.) are inline annotations describing intermediate sub-phases that all landed in this commit. There is no `git checkout` of the "M5-era state" — that state lives only in commit messages of the deleted history (`r_phase` archive scripts), not in current scorers/.

Worse: `relation_patterns.py` and `grounding.py` are **untracked in git** (status `??`). They exist on disk, but pre-S-phase versions cannot be retrieved at all. V5r §7c's reproducibility hinges on V6a being able to construct "M5-era CATALOG", but the doctrine doesn't define that artifact concretely. Possible interpretations:

- (a) Manually scope by code comment: include only RelationPattern entries whose docstring says "M1" or "M2" (the original catalog), exclude entries whose docstrings cite M9/M10/M11/M12 diagnosis FNs. **Implementable** but tedious and ambiguous (some entries combine multiple sub-phase rationales).
- (b) Use the parent skill `git checkout`: not possible because pre-S-phase versions don't exist as commits.
- (c) Delete entries from `_CYTOKINE_LIGAND_HGNC`, `_SITE_DENYLIST`, `_HEDGE_MARKERS`: easy if those sets are immutable frozensets (they are: line 454, 641 in `context_builder.py`).

V5r commits to (c) in the "empty `_CYTOKINE_LIGAND_HGNC`, empty `_SITE_DENYLIST`" phrasing but says nothing about `_HEDGE_MARKERS`, `_LOF_PATTERNS`, the M11 binding-admissibility logic, the M9 perturbation detector, the M13 60-char-window fix, etc. — all of which are tuned on the holdout per the same logic.

**The 10pp threshold**: V5g cannot evaluate "is 10pp the right number" without the V7c measurement actually being defined. With the baseline ambiguity above, V7c will report different deltas depending on which interpretation V6a picks. If interpretation (c) is taken literally and only the two named frozensets are emptied, the delta will be SMALLER than reality (other holdout-tuned signals still active) → false-negative on the contamination gate.

**Severity**: BLOCKING. V5r §7c must spell out which substrate features are reset and how. Recommended in-place fix: enumerate every holdout-tuned constant (`_CYTOKINE_LIGAND_HGNC`, `_SITE_DENYLIST`, `_HEDGE_MARKERS`, `_LOF_PATTERNS`, the bilateral-ambiguity guard, the M11 partner-admissibility map, the M13 hedging-window) and specify each as "empty" or "M1/M2-era only" or "removed" in the V7c baseline.

The 10pp threshold itself is reasonable as a magnitude — the broader question is whether 10pp is a per-class shift OR a marginal accuracy shift. V5r says "per-class vote-distribution delta" which is the right metric (vote share across classes, not accuracy), but the doctrine should clarify: 10pp on the LF's vote share for that class (e.g., LF voted `direct_sign_match` 30% of the time tuned, 22% baseline → 8pp delta, passes), not 10pp on the LabelModel's output for that class.

### Fix 5 (Finding F): verify_grounding shape

**Status**: RESOLVED with one MINOR concern

**Evidence**: V5r §2 commits per-(stmt, evidence, entity) records (1 record per claim agent). The §6 multiplier (28K target with 2.3x average) is consistent with the §2 spec.

**byte-equality** check on Gilda re-run: V5r §2 says "V6 dry-run validates byte-equality on a holdout-excluded sample". This is the correct fidelity criterion for the LoRA adapter — if the prompt at training time differs by even one byte from the inference-time prompt, the adapter has learned a different distribution. However, byte-equality is BRITTLE: Gilda's match scores are floats, and tiny score variations (e.g., 0.7521 vs 0.7522 across reruns or library updates) would render bytes unequal. V5r should specify whether `gilda_score` is rendered to a fixed decimal precision (e.g., 2 decimals) or whether the prompt format is robust to score variation. **MINOR**: V5r §2 should explicitly say "round Gilda score to 2 decimal places before rendering" or "exclude gilda_score from the byte-equality check" — otherwise the dry-run will fail spuriously on numerical noise. Looking at the production prompt format in `grounding.py:_build_user_message`, it does include `Gilda score (low confidence): <float>` for `is_low_confidence` cases — so the float-precision question is real, not academic.

GroundedEntity reconstruction: V5r commits V6 to "running Gilda lookup with the same thresholds used at inference". This is the right discipline. The hardcoded thresholds for `is_low_confidence` are in `grounding.py` — V5r §2 should specify that V6 must call the SAME GroundedEntity factory function used at inference time, not re-implement it. **MINOR**: a single function-call dependency between V6 and production is the cleanest way to enforce this.

### Fix 6 (Finding G): contamination guard parameters

**Status**: PARTIALLY RESOLVED — MAJOR concern about adapt-vs-impose

**Evidence**: V5r §12 specifies n=3, normalization (lowercase, strip punctuation, drop stopwords + biomedical-stopword list), Jaccard similarity, threshold = 95th-percentile of natural overlap.

I cross-checked against `scripts/check_contamination.py`. The existing script does NOT use trigram Jaccard. It uses:
- `_norm`: collapse whitespace, strip trailing punctuation, casefold (NO stopword removal, NO biomedical-stopword list)
- Exact match, substring containment, paraphrase_overlap (sliding 50-char window with 5-char stride), pair match — all string operations, NO n-gram, NO Jaccard.

V5r says "If the existing script computes overlap differently, V5r imposes the trigram-Jaccard contract and the existing script is updated to match." This is BACKWARD. The existing script's algorithm is well-tested (CI-guarded per the contamination guard memory; caught the 2026-05-02 paraphrase incident). V5r is proposing to REPLACE a working algorithm with a parallel reimplementation that hasn't been validated against any historical contamination event.

**Why backward**: the existing script's substring + 50-char-window + pair-match catches a class of contamination (verbatim quotes, prefix/suffix containment, exact-pair overlap) that trigram-Jaccard handles WORSE than substring matching. Trigram-Jaccard is designed for fuzzy paraphrase detection — but the script's `paraphrase_overlap` already does that with the sliding window. V5r's "impose the trigram-Jaccard contract" would lose the substring detection.

**Severity**: MAJOR. V5r should either:
- (a) Adopt the existing script's algorithm (substring + sliding-window + pair-match) as the V6d contract, OR
- (b) Specify that V6d runs BOTH algorithms (existing script's checks + V5r's trigram-Jaccard) and drops a record if EITHER flags it. This is conservative and preserves backward compatibility.

V5r currently picks neither — it imposes a brand-new algorithm and orphans the existing one. The threshold derivation step (95th percentile of natural overlap) is sound IN PRINCIPLE but applies only to V5r's algorithm, not to the existing script's checks.

### Fix 7 (Finding H): V8 sample size

**Status**: PARTIALLY RESOLVED — math is mostly right but the bucket-level CI is mis-stated

**Evidence**: V5r §7b prescribes 50 records per probe, stratified 50% rare-class / 30% borderline / 20% confident. So per probe:
- confident bucket: **n=10**
- borderline bucket: **n=15**
- rare-class bucket: **n=25**

V5r §10 says "V5r §7 V7b at N=50/probe gives 95% CI half-width <= 14pp". This is correct ONLY if accuracy is computed on the POOLED 50 records (no bucketing). Wilson 95% CI at p=0.75, n=50: half-width = 11.7pp. ✓ Math is right at the pool level.

But V5r §7b gates the BUCKETS separately:
- "lower 95% CI bound >= 65%" on confident bucket (n=10): At p=0.85, n=10, normal-approx CI half-width = 22pp. Wilson lower bound at p=0.85, n=10 is roughly 0.55. **Even at p=0.85 this barely fails the 65% bar**. At p=0.90 (9/10 correct), Wilson lower bound is roughly 0.60 — still fails. The threshold is unachievable at this sample size with realistic accuracies.
- "lower 95% CI bound >= 35%" on borderline bucket (n=15): At p=0.50, n=15, Wilson lower bound ≈ 0.27. Threshold is unachievable.

V5r §7b's gate is therefore mostly testing whether p_hat is HIGH ENOUGH to overcome small-N noise — the CI floors will fail unless accuracy is essentially perfect. **MAJOR concern**: the gate is over-strict at this sample size because it stratifies before computing CI.

Fix is one of:
- (a) Increase to N=100 per probe (i.e., 20 confident / 30 borderline / 50 rare-class). Then p=0.85, n=20 has Wilson lower bound ≈ 0.65 — gates at 65% pass cleanly with realistic accuracies.
- (b) Compute the CI on the POOLED 50 records and apply per-class minimums separately (e.g., "no bucket has p_hat below 50%" without CI requirement on tiny buckets).
- (c) Use the V7a U2-gold validation as the LF-accuracy gate (n=482, much higher power) and treat V7b as a sanity check at N=50 pooled with looser thresholds.

V5r's current text mixes these and the bucket-level gates aren't reachable.

**Severity**: MAJOR but not BLOCKING — the gate is conservative-failing (will trip even when LFs are OK), so it's biased toward false-fail rather than false-pass. Worst case V6 ships a useful LoRA but V8 fails, requiring a re-spec. The right move is to fix V5r §7b in-place to either route via V7a (high power) or pool the bucket CIs.

### Fix 8 (Finding I): U2 gold use

**Status**: RESOLVED (wording precise) — but see Issue 4 below for a hidden filtering path

**Evidence**: V5r §11 has 4 explicit clauses:
- PERMITTED: U2 gold as Y_dev for `LabelModel.fit()` calibration
- PERMITTED: U2 gold for V7a LF accuracy measurement
- FORBIDDEN: U2 gold as label source for V6d JSONL records
- FORBIDDEN: U2 gold to select/filter training records

The phrasing is unambiguous. V0r §4.4 ("STRICT: do NOT use INDRA's belief score") is preserved — U2 gold is curator labels, not INDRA belief. V5r's distinction between "tune LabelModel's class_balance prior" and "select training records" is principled.

But **Issue 4 below** finds a hidden filtering path that V5r does not acknowledge.

## New issues introduced by V5r

### Issue 1 (Snorkel `lf_summary()` does not compute pairwise correlations)
**Severity**: BLOCKING (factual error in spec; same class of error V5 had with `dependencies=`)
**Evidence**: V5r §4 step 2 says "Run `LFAnalysis(L=Λ, lfs=lf_list).lf_summary()` to diagnose pairwise correlations and per-LF coverage". The Snorkel 0.9.9 source confirms `lf_summary()` returns Polarity, Coverage, Overlaps, Conflicts (plus optional Correct/Incorrect/Empirical Accuracy with Y, plus Learned Weight with est_weights). Conflicts (LFs vote opposite) and Overlaps (LFs both fire) are NOT pairwise correlations. Snorkel 0.9.9 has no public correlation utility; the pre-0.9 `DependencySelector` is removed.
**Recommended action**: V5r §4 step 2 must change to one of:
- "Compute pairwise LF correlation manually from Λ via `numpy.corrcoef` (treating ABSTAIN as a sentinel; filter to records where both LFs fire). If correlation > 0.5..."
- Drop the correlation rule and rely on `lf_summary()`'s Conflicts column to detect over-agreement (high overlap + low conflict ≈ correlated, but only as a proxy).

### Issue 2 (Y_dev semantics mis-stated in V5r §4 step 4)
**Severity**: MAJOR (V5r conflates two different uses of Y_dev)
**Evidence**: V5r §4 step 4 says "Y_dev passes the U2 per-probe gold for the holdout-overlap subset (see §11) to anchor LF-accuracy estimates". Per Snorkel 0.9.9 source: when Y_dev is supplied, it is used to compute `class_balance` via `_set_class_balance()` — i.e., Y_dev's role is to set the class-prior parameter, NOT to anchor per-LF accuracies. Per-LF accuracies in Snorkel 0.9.x LabelModel are still derived from agreement structure (Λ); Y_dev does not feed into the mu-parameter optimization directly.

This matters because:
- V5r §11 PERMITS Y_dev for "calibrating per-LF accuracy estimates" — but Y_dev does no such thing. It calibrates class_balance.
- If V5r §4 step 3 has already explicitly committed `class_balance=` to a fixed array, then `Y_dev=` is REDUNDANT — Snorkel will ignore the dev-gold for class-balance computation when a class_balance is supplied. **Both arguments together are non-orthogonal**.
- V5r should pick: either pass `class_balance=` (the explicit choices in step 3) OR pass `Y_dev=` (let Snorkel compute class_balance from gold) — not both.

**Recommended action**: V5r §4 step 4 + §11 must be reconciled. Pick one:
- (a) Drop Y_dev entirely; rely only on the explicit class_balance in step 3. V7a's U2-gold validation runs separately, not via Y_dev.
- (b) Drop the explicit class_balance for natural-distribution probes (just verify_grounding currently); pass Y_dev there. Equal-distribution probes still use explicit class_balance and don't pass Y_dev.

### Issue 3 (Y_dev is empty for L_train per V5r §5 holdout exclusion)
**Severity**: MAJOR (structural confusion in V5r §4 step 4)
**Evidence**: V5r §5 commits to dropping all holdout-overlapping records from training. V5r §11 says Y_dev = "U2-gold subset of holdout that is NOT used for L_train". But in Snorkel's `LabelModel.fit(L_train=Λ, Y_dev=Y, ...)`, Y_dev's expected shape is **(n_dev,)** with corresponding rows from Λ_dev (a separate label matrix for the dev set). Snorkel's API expects you to pass BOTH `Λ_dev` and `Y_dev`, or for `Y_dev` to align with a subset of `L_train`. V5r says Y_dev records are "excluded from L_train" — this means there's no Λ_train row for them. Snorkel won't error, but the `_set_class_balance` from Y_dev will compute from records that have NO LF votes in Λ.

Concretely: the path V5r prescribes either (a) requires building a SEPARATE Λ_dev for the holdout records (using the SAME LFs, on records EXCLUDED from training) which is a step V5r doesn't mention, OR (b) is a misapplication where Y_dev is passed but doesn't connect to Λ_train rows.

**Recommended action**: V5r §4 step 4 must specify: "compute Λ_dev separately from L_train, applying all LFs to the holdout records (excluded from training), and pass `Y_dev=Y_holdout_gold` alongside the implicit Λ_dev that the LabelModel can read". OR drop the Y_dev path entirely (Issue 2 fix).

### Issue 4 (Hidden filtering path: Y_dev → LF weights → 0.5 threshold → record drop)
**Severity**: MAJOR (silently violates V5r §11's "FORBIDDEN: filter training records via U2 gold")
**Evidence**: V5r §11 forbids "Use U2 gold to select/filter training records". V5r §4 step 4 PERMITS Y_dev anchoring. The pipeline path:
1. U2 gold → `Y_dev` argument
2. `Y_dev` → influences `class_balance` (per Snorkel 0.9.9 source)
3. `class_balance` → influences LabelModel's mu-parameter optimization
4. Mu-parameters → influences `predict_proba(Λ_record)` for every record
5. `predict_proba >= 0.5` threshold (V5r §4 step 6) → drops records below threshold

So U2 gold INDIRECTLY filters which training records survive into V6d's JSONL. A record's survival depends on `predict_proba`, which depends on class_balance, which depends on U2 gold. This is a hidden filtering path that V5r §11's plain text doesn't anticipate.

The contamination is small in magnitude — class_balance is a 1-D vector, not per-record gold leakage — but it does mean the V11 holdout accuracy is upward-biased relative to a hypothetical class_balance computed only from Λ structure.

**Recommended action**: V5r should acknowledge this in §11 (add: "Indirect: Y_dev influences class_balance which influences predict_proba which influences which records survive the 0.5 threshold. The magnitude of this leakage is bounded by class_balance's degrees of freedom (K-1 floats per probe) and is far smaller than per-record gold leakage. We accept this as the price of using Y_dev for calibration."). This makes the trade-off explicit instead of hidden.

### Issue 5 (Synthetic-oracle generation scope is unrealistic)
**Severity**: MINOR (operational, not architectural)
**Evidence**: V5r §6 requires synthetic-oracle generation for at least 4 classes (`direct_amount_match`, `via_mediator_partial`, `no_relation`, `direct_partner_mismatch`) at 200-500 placeholder examples each. This is 800-2000 hand-crafted synthetic records. V5r §3.2 also says `present_as_decoy` and `present_as_mediator` rely on synthetic augmentation if natural counts fall short, potentially adding 400-1000 more.

Total synthetic budget: 1200-3000 placeholder records. At 5 minutes per record (write + verify + tag), this is **100-250 hours of curator effort** — well beyond V6's task-graph budget (V6d is one P-phase node).

The doctrine should either:
- (a) Specify a specific lower bound (e.g., 100 per class instead of 200-500) to cap the budget at ~20-40 hours.
- (b) Specify a generation methodology (e.g., LLM-assisted templating from a set of N seed structures × M placeholder names = NM records) that reduces per-record cost to seconds.

**Recommended action**: V5r §6 should commit to a generation METHOD (not just a count), and that method should be auditable (e.g., the templating approach can be regenerated and diffed deterministically).

### Issue 6 (Fragment-form LF lexicon is silently substrate-tuned)
**Severity**: MINOR
**Evidence**: V5r §3.4 introduces `LF_fragment_processed_form` with examples `cleaved <X>`, `phosphorylated <X>`, `<X> peptide`, `Aβ` for APP, `<X>-CTD`. These specific patterns (especially `Aβ` for APP) are derived from the holdout's audit trail — the same path Finding D flagged. V5r tags this LF `[clean]` but its lexicon was selected because the holdout showed APP/Aβ confusion.

**Recommended action**: V5r §3.4 should re-tag this LF as `[substrate-tuned]` (because its lexicon provenance is holdout-derived) OR explicitly limit it to a domain-general fragment-form taxonomy (e.g., "any uppercase letter + Greek + digit pattern paired with a parent symbol"). The current spec falls between these two options.

### Issue 7 (verify_grounding "absent class" auxiliary detection mis-stated)
**Severity**: MINOR
**Evidence**: V5r §3.2 row for `subject_role` `absent`: 1 LF (no_grounded_match). The augmentation note says "augment via per-statement deterministic check: claim entity name + all aliases not found in evidence text → absent". This second check is a separate LF in disguise. It should be promoted to a named row (`LF_absent_alias_check`) so the class-coverage table actually shows 2/2.

**Recommended action**: V5r §3.2 add a named row `LF_absent_alias_check` to make the count visible and the check explicit.

## Verdict justification

V5r resolves 6/8 fixes cleanly (Fix 1 with one residual MINOR, Fix 3 with one MINOR, Fix 5 with one MINOR, Fix 7 partially, Fix 8 cleanly with Issue 4 caveat) and resolves Fix 4 STRUCTURALLY but FAILS on the implementation detail (BLOCKING: M5-baseline not recoverable from git). Fix 2 PARTIALLY resolves by removing `dependencies=` but introduces a NEW factual error about `lf_summary()`'s correlation-reporting capability (BLOCKING).

The 2 BLOCKING issues are both small in-place edits to V5r (not structural rewrites) — Issue 1 needs a single sentence change in §4 step 2 (use np.corrcoef or drop the rule), and the M5-baseline definition needs a 1-paragraph spec in §7c enumerating which substrate features are reset. Issues 2, 3, 4, 6 are MAJOR but addressable in V5r §4 step 4 + §11 in <1 hour of editing.

**What unblocks V6**: (1) fix Issue 1's Snorkel correlation reference; (2) define V7c baseline concretely (which constants are emptied); (3) reconcile Y_dev-vs-class_balance redundancy (Issues 2-4) by picking ONE path; (4) decide contamination filter direction (Fix 6) — adopt existing script OR specify both algorithms in parallel.

**What does NOT unblock V6**: the remaining MINOR issues (synthetic-oracle scope, fragment-form lexicon provenance, absent-class LF naming) can be resolved during V6 implementation as inline doctrine edits, not pre-V6 blockers.

V5g recommends **CONDITIONAL pass**: V5r is structurally sound and most fixes hold; the 2 BLOCKING items are easy in-place corrections. V6a + V6c implementers will hit the corrections immediately if not pre-fixed, so fixing them in V5r is cheaper than discovering them in implementation.
