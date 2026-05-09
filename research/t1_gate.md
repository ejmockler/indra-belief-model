# T1 brutalist gate — doctrine review

Date: 2026-05-02
Status: **PASS WITH FINDINGS** (4 actionable items for T2-T4)

## Verification of doctrine claims against current code

| Claim | Status | Evidence |
|---|---|---|
| `adjudicator.py:198-199` has `if ra == "abstain"` branch | ✓ FOUND | `if ra == "abstain": return "abstain", None, "relation underdetermined"` |
| `adjudicator.py:187-188` has `via_mediator` direct-claim abstain branch | ✓ FOUND | `return "abstain", "indirect_chain", ...` |
| `relation_axis.py` lacks CRITICAL span-narrowing block | ✓ CONFIRMED | Fix C will ADD this block |
| `scope.py` already has a CRITICAL span-narrowing block | ✓ CONFIRMED | Fix C will STRENGTHEN this block |
| `commitments.py` has no `upstream_attribution` ReasonCode | ✓ CONFIRMED | T3 must add it |
| `_llm.py` has no transport-level fallback to `abstain` | ✓ CONFIRMED | But `_failure_fallback` does — see Finding #1 |

## Critical findings (must address in T2-T4)

### Finding #1 — `_failure_fallback` projects wrong default after Fix A
**Severity: BLOCKER for Fix A.**

`src/indra_belief/scorers/probes/_llm.py:107-115`:
```python
def _failure_fallback(answer_set, rationale):
    if "abstain" in answer_set:
        return "abstain", rationale, False
    return sorted(answer_set)[0], rationale, False
```

After Fix A drops `"abstain"` from `RelationAxisAnswer`, the fallback picks `sorted(answer_set)[0]` — alphabetically first. That's `direct_axis_mismatch`, NOT `no_relation` as the doctrine specifies.

**Required change in T2:**
- Add an optional `failure_default: str | None = None` parameter to `llm_classify`.
- Each probe module passes its preferred fallback (relation_axis → `"no_relation"`).
- `_failure_fallback` uses `failure_default` if provided; falls back to current logic otherwise.

This is the cleanest fix — explicit per-probe rather than alphabetical default.

### Finding #2 — `relation_axis.py` few-shot with `answer="abstain"` must be replaced
**Severity: BLOCKER for Fix A.**

`relation_axis.py:130-134`:
```python
(
    "CLAIM: ...\nEVIDENCE: We characterized MAPK1 substrates in cycling cells.",
    '{"answer": "abstain", "rationale": "JUN not mentioned; sentence too general"}',
),
```

This few-shot demonstrates abstain as a valid answer — directly contradicts Fix A.

**Required change in T2:**
- Replace this few-shot with one demonstrating `no_relation` projection from an underdetermined sentence.
- Suggested replacement (synthetic placeholder names per contamination guard):
```python
(
    "CLAIM: subject=KinaseN, object=ProteinM, axis=activity, sign=positive\n"
    "EVIDENCE: We characterized KinaseN substrates in cycling cells.",
    '{"answer": "no_relation", '
    '"rationale": "ProteinM not mentioned; KinaseN-ProteinM relation not asserted"}',
),
```

### Finding #3 — Test churn larger than implied
**Severity: WARNING for T2 effort estimate.**

`grep -rn "relation_axis.*abstain"` shows 8+ references across:
- `tests/test_probe_adjudicator.py:321` (`test_relation_axis_abstain_source_forces_overall_abstain`)
- `tests/test_probe_adjudicator.py:373`
- `tests/test_probes.py:219` (`test_relation_axis_legitimate_abstain`)
- `tests/test_orchestrator.py:115`
- `tests/test_probe_types.py:210`

T2 must not just delete these tests — it must REPLACE the assertions to test the new behavior:
- The "legitimate abstain" test becomes "legitimate no_relation projection" or similar.
- The "abstain forces overall abstain" test becomes "Fix A's scope-tiebreaker handles unreachable abstain branch defensively".
- Type tests must remove `"abstain"` from the validated value set.

Estimated test work: 5-8 test functions touched, ~30-50 lines changed.

### Finding #4 — Fix C "strengthen" instruction is too vague
**Severity: ADVISORY for T4 prompt-design discipline.**

Doctrine §3.3 says:
> To `scope.py` system prompt, the existing CRITICAL block already instructs span-narrowing. Strengthen with: ...

But scope.py's existing CRITICAL block is already well-formed:
```
CRITICAL: focus on the CLAIM RELATION specifically. Hedging or
negation that governs a DIFFERENT proposition in the same sentence
does NOT propagate. "X activates Y, but Z was not affected" → asserted
for X→Y; the negation governs Z.
```

T4 should STRENGTHEN by:
- Adding the conditional-mutant clause: `"If the sentence contains 'X does Y, but mutant Z does not do Y' and the claim is about X (not the mutant), the scope is asserted for the X-Y relation."`
- Adding 2-4 new few-shots specifically for conditional negation cases (synthetic names: variant/mutant placeholders).
- Existing few-shot at scope.py:78-84 already covers the basic case ("MAPK1 activates JUN robustly, but ELK1 was not affected" → asserted). Don't delete it; add to it.

For relation_axis.py (which lacks any CRITICAL block), T4 ADDS:
```
CRITICAL: Evaluate the relation between the claim's subject and
object ONLY within the clause where they co-occur. If a different
clause contains a negation or wrong-axis verb that does not connect
these two entities, ignore it.
```
Plus 4 paired few-shots.

## Non-critical observations

### Observation #5 — Honest framing on decision-only accuracy
Doctrine §4 acknowledges decision-only accuracy will likely DROP because we're committing on harder records (Fix A converts abstains into decisions, ~63% of which are correct vs 74.58% on current decision set). This is the correct framing; the gain is in raw accuracy and abstain-rate reduction, NOT in conditional precision.

Ship gate at T10 should NOT require decision-only ≥ 74.58% — instead it should require:
- Raw accuracy ≥ S-phase + 3pp (≥57.77%).
- No more abstains than 10% of records.
- Decision-only ≥ ~70% (lower bar than S-phase, on a much bigger denominator).

The doctrine's §6 ship gates as written are slightly too aggressive on decision-only. **Recommend amending §6 ship gate to "decision-only ≥ 70% on the bigger denominator" rather than "≥ S-phase".**

### Observation #6 — Inhibition lift expected to overshoot success bar
Success criterion: "Inhibition lifts by ≥3pp via Fix A's class-prior bias."

Empirical breakdown of Fix A on Inhibition: 1 gold-correct + 6 gold-incorrect among the 7 (no reason) Inhibition records. Force-incorrect: 6 TN gain, 1 FN loss, net +5. On 38 total Inhibition records, that's +13.2pp. Expected to easily clear the 3pp bar.

If actual gain is <3pp, that signals Fix A under-performed broadly; treat as ITERATE trigger.

### Observation #7 — Scope tiebreaker semantics
Doctrine §3.1 specifies the scope tiebreaker:
- `scope=asserted` → `correct/match`
- `scope=negated` → `incorrect/contradicted`
- otherwise → `incorrect/relation_underdetermined`

This handles 3 of 4 scope values. The 4th (`scope=hedged`) gets `incorrect/relation_underdetermined`, which may be too aggressive — if axis is genuinely underdetermined but scope is hedged, the record is closer to "we don't know" than "definitely wrong". 

**Suggestion (non-blocking):** consider `scope=hedged` → `correct/low` (lean positive at low confidence), consistent with the existing §5.7 hedged-with-direct-match rule. This preserves the spirit of "hedged means the relation IS asserted at low confidence."

T2 should pick one and document; both are defensible. Recommend `incorrect/relation_underdetermined` as proposed (more conservative; commits to incorrect when nothing else is informative).

## Verdict

**PASS with 4 actionable findings.** Findings #1 and #2 are blockers for T2 (Fix A). Findings #3 and #4 are scope-clarification warnings for T2-T4. Observation #5 amends §6 ship gates. Observations #6, #7 are advisory.

Doctrine is internally consistent, empirically grounded, and parsimonious as claimed. Proceed to T2 with the four findings folded in.

### Action items for T2
- [Finding #1] Add `failure_default` parameter to `llm_classify`; relation_axis passes `"no_relation"`.
- [Finding #2] Replace the relation_axis abstain few-shot with a no_relation projection example.
- [Finding #3] Update 5+ test functions across 4 test files; do NOT just delete.
- [Observation #7] Document the chosen `scope=hedged` policy in the implementation comment.

### Action items for T4
- [Finding #4] ADD CRITICAL block to relation_axis.py (it has none). STRENGTHEN scope.py's existing block (add conditional-mutant clause + new few-shots).

### Action item for T10 (ship gate)
- [Observation #5] Amend ship gates to use raw accuracy + abstain rate as primary criteria; decision-only as secondary on the larger denominator.
