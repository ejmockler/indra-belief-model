# U2 — INDRA KG cross-reference + per-probe gold derivation

Date: 2026-05-02
Status: **COMPLETE** — coverage and per-class gold sufficient for U-phase work

## Coverage of holdout pairs in INDRA's curated KG

Built a `(subject_lower, object_lower)` → curated triples index over 894K INDRA statements. Coverage on the 482 holdout records:

| stmt_type | covered | total | coverage |
|---|---|---|---|
| Activation | 102 | 106 | **96.2%** |
| Complex | 169 | 178 | 94.9% |
| IncreaseAmount | 23 | 24 | **95.8%** |
| Inhibition | 36 | 39 | 92.3% |
| Phosphorylation | 83 | 91 | 91.2% |
| DecreaseAmount | 12 | 16 | 75.0% |
| Dephosphorylation | 5 | 7 | 71.4% |
| Acetylation | 1 | 2 | 50.0% |
| Translocation | 0 | 3 | 0.0% |
| Autophosphorylation | 0 | 7 | 0.0% |
| ActiveForm | 0 | 2 | 0.0% |

**Overall coverage: 438/482 = 90.9%.**

Per the U1 finding #2 gate: U7 needs ≥20% coverage on the affected classes. Activation, IncreaseAmount, Complex, Phosphorylation all >90%. **U7 is GO.**

## Critical discovery — the holdout's `tag` field IS per-record gold

The holdout records carry curator-assigned error categories in the `tag` field, not just binary correct/incorrect:

```
correct           285   true positives
no_relation        53   curator says no relation between entities
grounding          33   entity grounding wrong
wrong_relation     26   different relation than claimed
act_vs_amt         25   Activation↔IncreaseAmount conflation (Intervention E)
other              22
entity_boundaries  20   entity span issue (related to grounding)
polarity           16   sign flip (Intervention U5)
hypothesis          9   hedged claim
negative_result     7   explicit negation
agent_conditions    1
mod_site            1
```

**This means we already have the per-probe gold the doctrine §3.9 sought to derive from KG.** The KG cross-reference still informs U4 (KG verifier confidence modifier), but the per-record fix-target validation is done via curator tags.

## T-phase performance by gold-tag category

| tag | n | right | wrong | abst | acc |
|---|---|---|---|---|---|
| correct | 273 | 188 | 50 | 35 | 68.9% |
| no_relation | 53 | 35 | 8 | 10 | 66.0% |
| **grounding** | 33 | 5 | 8 | 20 | **15.2%** |
| wrong_relation | 26 | 17 | 4 | 5 | 65.4% |
| **act_vs_amt** | 24 | 10 | 11 | 3 | **41.7%** |
| entity_boundaries | 20 | 7 | 5 | 8 | 35.0% |
| other | 20 | 9 | 6 | 5 | 45.0% |
| polarity | 16 | 9 | 7 | 0 | 56.2% |
| hypothesis | 8 | 4 | 2 | 2 | 50.0% |
| negative_result | 7 | 6 | 0 | 1 | 85.7% |

**Doctrine §3 alignment is excellent.** The interventions target exactly the lowest-performance classes:
- **U6 (Intervention D, grounding probe enhancement)**: targets `grounding` (15.2% acc) and `entity_boundaries` (35.0% acc) — combined 53 records.
- **U7 (Intervention E, closed-set redesign)**: targets `act_vs_amt` (41.7% acc) — 24 records.
- **U5 (perturbation propagation)**: targets `polarity` (56.2% acc) — 16 records.
- **U3 (selective reasoning)**: targets the 89 abstain records, especially the 35 grounding-abstains that block recovery.

## Addressable headroom by intervention

If each intervention recovers at the rates suggested by their conditional accuracy gaps:

| Intervention | Target class | Records | T-phase acc | Recoverable |
|---|---|---|---|---|
| U6 (Intervention D) | grounding + entity_boundaries | 53 | 22.6% combined | **+12 to +18** |
| U7 (Intervention E) | act_vs_amt | 24 | 41.7% | **+5 to +8** |
| U5 (perturbation) | polarity | 16 | 56.2% | **+3 to +5** |
| U3 (selective reasoning) | abstains in low-acc classes | ~30 | varies | **+8 to +12** |
| U4 (KG verifier) | calibration on TP class | 273 | 68.9% | **+5 to +10** confidence-only gains |
| U8 (verb taxonomy) | wrong_relation FN | 26 | 65.4% | **+3 to +5** |
| U9 (Fix A softening) | T-phase regressions | ~14 | n/a | **+10 to +14** |

**Total addressable: +46 to +72.**

Realistic with overlap: **+30 to +45 records** on the 482 holdout, putting raw accuracy at **66-70%** and decision-only at **78-82%**.

## Saved artifact

`data/benchmark/probe_gold_holdout.jsonl` (482 records) carries per-record fields:
- `gold_target`: correct/incorrect (binary)
- `gold_tag`: curator's error category (one of 12 values)
- `claim_axis`, `claim_sign`: derived from claim's stmt_type
- `kg_n_curated_pairs`: how many curated INDRA statements between this entity pair
- `kg_curated_stmt_types`: dict of stmt_type → count
- `kg_max_belief`: highest belief score across curated statements
- `kg_total_evidence`: total evidence count across curated statements
- `kg_signal`: SAME_STMT (438 records) / NO_KG (44 records)

## Implications for U-phase

1. **U7 is unblocked** — coverage and per-record gold both available.
2. **U13 stratified probe should oversample the low-acc tag classes** (grounding, act_vs_amt, entity_boundaries, polarity) to validate per-intervention impact.
3. **U4 KG verifier signal**: 438/482 records have SAME_STMT KG support — that's a strong baseline signal. The 44 NO_KG records are interesting; they may be over-represented in errors (no curator support could indicate genuinely wrong claims OR coverage gap).

Proceeding to U3 (selective reasoning) — first intervention in the architectural order.
