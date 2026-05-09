# U1 brutalist gate — U-phase doctrine review

Date: 2026-05-02
Status: **PASS WITH FINDINGS** (3 actionable items, 1 critical)

## Verification of doctrine claims against current code

| Claim | Status | Evidence |
|---|---|---|
| `_llm.py` has `reasoning_effort="none"` hardcoded | ✓ CONFIRMED | one occurrence at line 80 |
| CorpusIndex importable with 894K statements | ✓ CONFIRMED | import works; methods: build_records, get, load |
| `router._route_relation_axis` exists | ✓ CONFIRMED | function present |
| `effective_sign` computation in router | ✓ CONFIRMED | already used elsewhere |
| Perturbation handling in router | ✓ CONFIRMED | already detected |
| `verify_grounding` function exists | ✓ CONFIRMED | with detailed prompt + 4-status answer set |
| `direct_sign_match` in RelationAxisAnswer | ✓ CONFIRMED | targeted by U7 |
| ScopeAnswer has 4 values | ✓ CONFIRMED | {asserted, hedged, negated, abstain} |
| relation_patterns.py exists | ✓ CONFIRMED | 693 lines |
| Fix A's `prefer no_relation` instruction present | ✓ CONFIRMED | targeted by U9 |
| Fix C CRITICAL block present | ✓ CONFIRMED | targeted by U9 |
| adjudicate function | ✓ CONFIRMED | 344 lines, decision table extensible |

## Critical findings (must address in U-phase)

### Finding #1 — verify_grounding's prompt is already detailed; U6 changes the EDIT, not REPLACE strategy
**Severity: ADVISORY for U6.**

The doctrine §3.4 implies starting from a thinner grounding prompt. In reality, `verify_grounding` already has 5 detailed rules covering family claims, fragments/processed forms, pseudogenes, synonym collisions, and generic class nouns.

**Implication for U6:** the intervention is to ADD anaphora/Greek-letter/family-bridge handling on top of these existing rules, not to replace them. Specifically, add Rule 6 covering:
- Anaphora ("this peptide", "the kinase", "the target")
- Greek-letter aliases (β-catenin, p38, ERβ)
- Reverse family-instance (claim is instance, evidence has family — currently Rule 1 only covers forward direction)
- Bidirectional binding (UAP56 = DDX39B etc.)

Plus 4-6 paired few-shots. The existing structure is preserved.

### Finding #2 — U2 may have low INDRA KG coverage on holdout pairs
**Severity: BLOCKING for U7 if coverage too low.**

The doctrine §3.9 acknowledges "Coverage will be partial". U7's closed-set redesign assumes per-probe gold for ≥30% of holdout records. **This needs to be measured FIRST in U2 before committing to U7.**

**Action:** U2 must report coverage stratified by stmt_type. If <20% coverage on the affected error classes (Activation+IncreaseAmount FPs, conditional-negation FNs), defer U7 to V-phase.

Adding to U2's success criteria: emit a coverage report (records-with-curated-triples / total) per stmt_type before declaring U2 complete.

### Finding #3 — Selective reasoning's max_tokens budget needs explicit override
**Severity: BLOCKING for U3 implementation.**

Per `model_client.py` doctrine: gemma-remote with `reasoning_effort="medium"` burns 12000+ reasoning tokens. The model's `max_tokens` config is 12000, meaning content (the JSON answer) can get truncated. This is exactly the silent-truncation failure mode noted in the model_client.py comments.

**Action for U3:**
- When escalating to `reasoning_effort="medium"`, also override `max_tokens` to a higher value (16000 or 20000) to leave room for both reasoning AND content.
- Alternatively, request `reasoning_effort="low"` first (a less aggressive escalation tier) and only go to "medium" if "low" still abstains.
- Add a finish_reason="length" detection to the escalation logic — if the escalated call truncates, fall back to the first-pass answer rather than emitting garbage.

### Finding #4 — U10's "probe_inconsistency" reason isn't a registered ReasonCode
**Severity: TECHNICAL for U10.**

`commitments.py:ReasonCode` literal enumerates the valid reason codes. U10 emits `probe_inconsistency` which doesn't exist. Adding it would make U10 work but adds another reason code to the surface area.

**Alternative (simpler):** U10 doesn't add a new reason code; instead it appends to the rationale and downgrades confidence. The composed_scorer reads confidence; downstream consumers don't need a new reason.

**Recommended:** make U10 confidence-only (no new ReasonCode). Cleaner.

## Non-critical observations

### Observation #5 — Stacking interactions need empirical validation, not just additive estimates
The doctrine §5 estimates +66 to +104 with independence assumption, then notes "+30 to +50 realistic with overlap". This is honest but vague. **U13 stratified probe must explicitly test:**
- Records where U3 (selective reasoning) alone might solve a case U6 (grounding) targets — does U6 add anything beyond U3?
- Records where U4 (KG verifier) confidence boost may collide with U10 (consistency check) confidence downgrade — what's the priority order?

Recommend U11 add a "pairwise interaction matrix" review item before declaring implementation gate PASS.

### Observation #6 — U5 perturbation propagation may double-apply in adjudicator
Adjudicator §5.1 already calls `_effective_claim_sign(claim.sign, perturbation)` (informationally only — line 251 in current code). After U5, the relation_axis probe receives the effective sign. **Risk:** if the adjudicator ALSO inverts in its decision table, we double-flip and break correct cases.

**Action for U5:** before propagating effective_sign, audit adjudicator §5.2 to confirm sign comparison is between (probe answer's effective_sign reading) and (claim.sign). If adjudicator independently inverts, remove that inversion when U5 ships.

### Observation #7 — U9 risks regressing T-phase's Inhibition gain
T-phase's Inhibition lift (+14.3pp) was driven by Fix A's class-prior bias on the (no reason) class. U9 softens that bias. **Risk:** Inhibition could drop if the LLM, given the new "prefer direct_sign_match" instruction, overshoots in the opposite direction.

**Action for U9r:** smoke test Inhibition records specifically. If Inhibition decision-only drops below 50% (from current 57.1%), iterate before proceeding to U10.

### Observation #8 — Doctrine doesn't specify what happens to T11 push
The deleted T11 task was "push T-phase to origin/main". U-phase doctrine §1 implies U-phase ships on TOP of T-phase (combined). If U-phase fails, do we still want to ship T-phase alone? The doctrine should clarify the fallback.

**Recommendation:** if U13 probe gate fails badly (e.g., raw acc <60% — below T-phase), ITERATE specific failing interventions; never revert all of U-phase. If U15 ship verdict is ITERATE, the T-phase changes stay shippable on their own as a fallback.

## Verdict

**PASS with 4 findings.**

- Finding #1: U6 implementation refinement (advisory).
- Finding #2: BLOCKING gate on U2 coverage report before U7.
- Finding #3: BLOCKING for U3 implementation — must handle truncation.
- Finding #4: TECHNICAL for U10 — recommend confidence-only, no new ReasonCode.

Action items:
- [Finding #1] Update U6 to "Add Rule 6" rather than redesign.
- [Finding #2] U2 must emit per-stmt-type coverage; U7 gate decision based on coverage ≥20%.
- [Finding #3] U3 must use max_tokens=16000+ on escalation and check finish_reason.
- [Finding #4] U10 implements as confidence-only.
- [Observation #6] Audit adjudicator before U5 ships to prevent double-flip.
- [Observation #7] U9r explicit Inhibition regression test.
- [Observation #8] If U13 fails, iterate specific intervention; if U15 ITERATE, T-phase remains shippable.

Doctrine is internally consistent and empirically grounded. Proceed to U2 (the diagnostic + critical-path prerequisite for U7).
