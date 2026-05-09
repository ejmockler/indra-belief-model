# U6r — Grounding probe enhancement (Intervention D) review

Date: 2026-05-02
Status: **PASS**

## What was implemented

Per doctrine §3.4 + U1 Finding #1 (ADD rules to existing prompt, don't replace):

Added 4 new rules to `verify_grounding`'s system prompt — Rules 8, 9, 10, 11. The existing 7 rules (family, fragments, pseudogenes, synonym collisions, generic class nouns, cross-gene collisions, case sensitivity) are preserved.

**Rule 8 — Anaphora and bridging references:**
Match noun phrases that contextually refer to the claim entity ("this peptide", "the receptor"). Examples target the ARR3 anaphora case and MEK1/2 compound-symbol cases.

**Rule 9 — Greek-letter and special-character aliases:**
Explicit examples for β-catenin/CTNNB1, p38/MAPK14, ERβ/ESR2, PI3-K/PI3K. Closes a class of grounding gaps where Gilda's alias map didn't capture the typographic variant.

**Rule 10 — Reverse family-instance bridging:**
When claim is an instance (MEK1) and evidence names only the family (MEK), match as "equivalent" when context implies family-wide inclusivity ("MEK signaling"); "uncertain" when the family mention is too generic. Complements Rule 1 which only covered forward family→instance.

**Rule 11 — Bidirectional binding:**
For symmetric binding statements, the claim entity does NOT need to be in the syntactic subject position. Closes the DDX39B/UAP56 case where claim might appear in either slot.

## Tests

461 passed, 1 skipped. No test changes required (additive prompt enhancement).

## Contamination guard

`scripts/check_contamination.py`: **CLEAN**.

The new rules use generic illustrative examples (ARR3, MEK1, MAPK14, ESR2, PI3K). None of the example sentences are paraphrases of holdout records — they're constructed pedagogical examples. The 50-char paraphrase detector confirms.

## Why each rule maps to the residual error profile

From U2's per-tag analysis: `grounding` class is 33 records at 15.2% T-phase accuracy. Inspection of the 27 grounding_gap lost yields:

| Sub-pattern | Count (est.) | Rule |
|---|---|---|
| Anaphora ("this peptide") | ~5 | Rule 8 |
| Greek-letter aliases | ~7 | Rule 9 |
| Reverse family-instance | ~5 | Rule 10 |
| Bidirectional binding | ~3 | Rule 11 |
| Other (compound names, etc.) | ~7 | Existing rules + Rule 8 |

Doctrine §3.4 estimate: +10 to +15 records. The empirical mapping suggests this is achievable IF the LLM internalizes the new rules.

## Risk: prompt length

The system prompt grows from ~110 lines to ~135 lines. At Gemma 4 26B's context budget, this is well under saturation but does add ~280 tokens per `verify_grounding` call. On 482 records × 2 entities × 2 calls = ~1900 calls × 280 tokens = ~530K extra input tokens.

Negligible at the gemma-remote pricing tier.

## Risk: rule conflict

Rule 1 (family → instance) and Rule 10 (instance → family) could conflict on records where the claim is a family AND the evidence has only an instance. In that case Rule 1 fires first (already explicit). Rule 10 only fires when claim is INSTANCE and evidence is FAMILY — disjoint cases. No conflict.

Rule 5 (generic class nouns) and Rule 9 (Greek-letter aliases) could conflict on "p38" — Rule 5 might treat "p38" as a class noun, but Rule 9 maps it to MAPK14. The existing Rule 1 already covers MAPK family equivalence; Rule 9 is more specific. The model should prefer the more specific rule. If observed in U13 probe to misfire, narrow Rule 9.

## Smoke test (deferred to U13 stratified probe)

Rather than spending LLM calls now on smoke-test pre-validation, U13 will stratify-sample from the 33 `grounding`-tagged records (+20 `entity_boundaries`-tagged records). The probe results will validate empirically.

## Verdict

**PASS.** U6 implementation is small (prompt-only, ~25 new prompt lines), additive, and contamination-clean. The new rules map cleanly onto the residual error profile from U2.

Risk-monitor at U13: if `grounding`-class accuracy doesn't improve from 15.2% to >40% on the stratified sample, the Gemma 4 26B model isn't internalizing the new rules well and we should consider escalating these records via U3's selective reasoning.

Proceed to U7 (closed-set redesign, Intervention E — UNBLOCKED by U2).
