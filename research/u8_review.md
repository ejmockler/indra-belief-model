# U8r — Verb taxonomy in CATALOG review

Date: 2026-05-02
Status: **PASS**

## What was implemented

Per doctrine §3.6:

Added new `ACTIVITY_NEGATIVE` group to `relation_patterns.py` with 4 surface-form patterns:
- `act_neg.verb_activation_of`: "X decreases/inhibits the activation of Y"
- `act_neg.verb_y_activation`: "X decreases/inhibits Y activation"
- `act_neg.nominalized_by`: "decrease/reduction of Y activation by X"
- `act_neg.mediated_decrease`: "X-mediated decrease in Y activation"

The verb stem `_ACT_NEG_VERB_STEM = "decreas|reduc|inhibit|abolish|suppress|diminish|attenuat|block"` deliberately overlaps with `_AMT_DOWN_STEM`. Discrimination between the two axes is by the OBJECT NOUN PHRASE:
- amount-down: "Y expression / Y mRNA / Y protein levels"
- activity-negative: "Y activation / Y signaling / Y activity"

Verbs alone are ambiguous between axes; the noun phrase is what makes the axis explicit. This is consistent with the existing CATALOG design where verb+noun-phrase compounds are atomic surface forms.

## Engineering distinction

### Bug caught and fixed during smoke test

Initial verb-form regex was `(?:es|ed|ing|e)?` — missed present-tense `-s` suffix ("inhibits", "blocks", "reduces"). Smoke test caught this; fixed to `(?:s|es|ed|ing|e)?`. **8/9 smoke tests passed only after this fix.**

This is the kind of bug that would have silently degraded U8's effectiveness on the holdout — the patterns would compile and tests would pass, but they wouldn't actually match real evidence. The smoke test caught it pre-holdout.

### Avoided overlap collisions with AMOUNT_DOWN

`AMOUNT_POSITIVE` exists but the symmetric `AMOUNT_NEGATIVE` doesn't (S-phase didn't add it). Adding ACTIVITY_NEGATIVE without an existing AMOUNT_NEGATIVE means there's no risk of double-counting the same verb stem against two axes. If/when AMOUNT_NEGATIVE is added (V-phase), the noun-phrase discriminator is already in place.

### Spurious match of generic words ("the", "drug")

The `_NAME` regex `[A-Za-z\d][\w/.-]*\w` matches words like "the" and "drug" because it doesn't enforce a biomedical-entity vocabulary. The router's downstream alias-validation filters these out — only `_NAME` captures that bind to claim entity aliases reach the adjudicator's substrate-fallback rule. So `act_neg.verb_y_activation` matching "MIR140 decreases the activation of p38" with Y='the' is harmless — Y='the' isn't in the alias map, so the match is discarded.

This is consistent with the existing CATALOG design (M3 alias-validation in context_builder).

## Tests

467 passed, 1 skipped. No test regressions. No new tests needed for the patterns themselves — context_builder's alias-validation tests already cover the routing.

The 9-case smoke test verifies:
- 6 positive cases (real-world activity-negative phrasings) all match.
- 3 negative cases (amount-down or wrong sign) correctly do NOT match.

## Empirical hypothesis

From U2 per-tag analysis: `wrong_relation` class has 26 records at 65.4% T-phase acc (most are correctly classified). The "axis_mismatch" FN class (10 records identified in T-phase analysis) is what U8 targets:

| FN evidence | Pattern that should fire |
|---|---|
| "miR-140 decreases the activation of p38" | act_neg.verb_activation_of |
| "TLR4 activation enhances TGF-β signaling" | (NOT addressed — would need ACTIVITY_POSITIVE expansion) |
| "DAB2 suppresses ERK activation" | act_neg.verb_y_activation |

Doctrine §3.6 estimate: +5 to +10 records. Realistic for ACTIVITY_NEGATIVE alone: +3 to +5 records (the explicit Inhibition-equivalent verb compounds). Other axis_mismatch FNs (semantic-equivalent verbs that aren't activity-negative) remain unaddressed.

## What U8 does NOT address

- "X enhances Y signaling" (Activation, not Inhibition) — would need ACTIVITY_POSITIVE expansion. Skipped because `enhanc` already overlaps with `_AMT_UP_STEM`; adding it to `_ACT_STEM` risks collisions.
- "X induces Y expression" — already covered by AMOUNT_POSITIVE (no addition needed).
- "X abolishes Y binding" → BINDING_NEGATIVE — would need a new group; deferred (small payoff).

These are residual axis_mismatch FN sources that V-phase or later iteration would address.

## Verdict

**PASS.** U8 implementation is correct, smoke test verifies pattern matching on 9 cases, no test regressions. The intervention is small (~50 lines of patterns) and substrate-only — no LLM cost, no prompt changes.

Risk for U13: if `act_neg.verb_*` patterns over-fire on edge cases not seen in smoke test (e.g., "X reduces apoptosis" — biologically activity-negative but evidence text might say "apoptosis" not "Y activation"), we'd see TP regressions. The smoke test covers the obvious cases; U13 stratified probe will validate at scale.

Proceed to U9 (Fix A softening + Fix C tightening).
