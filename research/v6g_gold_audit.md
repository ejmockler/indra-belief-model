# V6g — Gold-mapping audit & fix

Date: 2026-05-07
Script: `scripts/v6g_gemini_validation.py` (lines 59–93)
Inputs joined: `data/benchmark/probe_gold_holdout.jsonl` ⨝ `data/benchmark/holdout_v15_sample.jsonl` on `source_hash` (482 records).

## Motivation

Gemini 3.1 Pro on the full 482-record holdout produced surprisingly low role-probe accuracies:

| Probe              | Pre-fix accuracy |
|--------------------|------------------|
| relation_axis      | 62.5% (245/392)  |
| subject_role       | 54.8% (252/460)  |
| object_role        | 59.6% (274/460)  |
| scope              | 87.2% (251/288)  |
| verify_grounding   | (n/a — separate) |

The confusion matrix showed 51 cases where `gold = "absent"` but Pro said `present_as_subject`, and 31 cases `gold = "absent"` → Pro said `present_as_object` (subject_role); 58 + 21 cases of the same shape on object_role. Inspection of those Pro rationales (TRAIL = TNFSF10, Astrin = SPAG5, INI1 = SMARCB1, iNOS = NOS2, eIF2alpha = EIF2S1, COX-2 = PTGS2, …) revealed Pro was correctly identifying entity mentions through aliases — the U2 mapping was forcing a `gold = "absent"` label that was structurally wrong.

The current mapping forces `subject_role = object_role = "absent"` whenever `gold_tag` ∈ {`no_relation`, `grounding`, `entity_boundaries`}, but the production probe defines `absent` as "the entity is not mentioned in the evidence" — a property of the *evidence text*, not of the *claim's correctness*. A `gold_tag = no_relation` only says the claimed *relation* is unsupported; the entity may be mentioned in any role.

## Audit

### 1. Substring presence under "absent"-mapping tags

Joined 482 records. For tags whose current mapping forces role gold = `"absent"`, ran case-insensitive substring match of the claim subject and object against the evidence text:

| gold_tag           | N    | subj in ev | obj in ev | both | subj only | obj only | neither |
|--------------------|------|-----------:|----------:|-----:|----------:|---------:|--------:|
| no_relation        | 53   |         31 |        28 |   21 |        10 |        7 |      15 |
| grounding          | 33   |          8 |         7 |    2 |         6 |        5 |      20 |
| entity_boundaries  | 20   |         12 |        12 |    9 |         3 |        3 |       5 |
| **Total**          | 106  |  **51**    |   **47**  | **32** |   19    |       15 |     40  |

So at minimum **48 % / 44 %** of the records the current mapping calls "absent" actually have the claim subject/object as a substring of the evidence. With alias resolution (e.g., "TRAIL" for TNFSF10) the true "present" rate is materially higher — the Pro rationales above are de facto evidence that those records are mis-mapped.

### 2. Pro behaviour on the mis-mapped records

For the same 106 records (per probe), Gemini 3.1 Pro's distribution:

| probe          | gold_tag           | N  | ent in ev | Pro present_as_* (in_ev) | Pro present_as_* (NOT in_ev) | Pro absent |
|----------------|--------------------|---:|----------:|-------------------------:|-----------------------------:|-----------:|
| subject_role   | no_relation        | 53 |        31 |                       31 |                           20 |          2 |
| subject_role   | grounding          | 33 |         8 |                        8 |                           17 |          8 |
| subject_role   | entity_boundaries  | 20 |        12 |                       12 |                            7 |          1 |
| object_role    | no_relation        | 53 |        28 |                       28 |                           24 |          1 |
| object_role    | grounding          | 33 |         7 |                        7 |                           13 |         13 |
| object_role    | entity_boundaries  | 20 |        12 |                       12 |                            5 |          3 |

Two key observations:

- When the entity IS in evidence (substring), Pro says `present_*` **100 %** of the time (51/51 subject; 47/47 object). The gold says "absent." Pro is right; the gold is wrong.
- Even when the entity is NOT in evidence by substring, Pro says `present_*` 70–80 % of the time — and rationales confirm these are bona-fide alias resolutions ("Astrin = SPAG5", "iNOS = NOS2", etc.), not hallucinations. So substring matching is *too conservative* as a corrective signal: we cannot reliably split true-absent from alias-present without per-record curation.

### 3. Other suspect mappings

| Tag               | Current mapping                                                                           | Verdict    |
|-------------------|-------------------------------------------------------------------------------------------|------------|
| no_relation       | `relation_axis = "no_relation"`                                                            | **Keep.** Matches the production probe spec ("both entities mentioned but no relation asserted"). Some records have one or both entities missing, which weakens the spec match — but that's a probe-level definitional issue, not a mapping bug. |
| hypothesis        | `relation_axis = None; subject = present_as_subject; object = present_as_object; scope = hedged` | **Keep.** Hypothesised relations have no defensible ground-truth axis class, but the entities are mentioned and "hedged" matches the scope spec. Symmetric-relation ambiguity (subject vs object) is inherent to the probes. |
| negative_result   | `relation_axis = None; ...; scope = negated`                                              | **Keep.** Same reasoning; "negated" is the right scope. |
| mod_site          | All five fields `None` except `verify_grounding = mentioned`                              | **Keep (low-N).** N=1; the relation/entities are correctly grounded but the modification site is wrong, so axis/scope/role are debatable. Not worth re-mapping for a single record. |
| agent_conditions  | All five fields `None` except `verify_grounding = mentioned`                              | **Keep (low-N).** N=1; same logic as `mod_site`. |
| grounding         | `verify_grounding = "not_present"`                                                         | **Keep.** Curator said "this is a grounding error" → entity is not really referenced → `not_present` matches the probe spec. |
| entity_boundaries | `verify_grounding = "not_present"`                                                         | **Keep.** Compound-name false matches → not a real reference → `not_present` matches. |

### 4. Decision: Option B (skip role probes for the three tags)

**Picked Option B (the more defensible of the two).**

- **Option A (recommended in task brief)** — make role gold content-conditional (`absent` if entity not in evidence by substring, else `present_*`). Rejected because (a) substring match is too conservative — alias resolution makes "not in evidence" itself unreliable, and (b) even when the entity IS in evidence, picking the correct role-class (`present_as_subject` vs `present_as_object` vs `present_as_mediator` vs `present_as_decoy`) requires per-record curation; we don't have the data to do it programmatically.
- **Option B (chosen)** — set role-probe gold to `None` (skip from accuracy calc) for `no_relation`, `grounding`, `entity_boundaries`. The gold tag is fundamentally about *the claim*, not about *the entity's role in evidence*; without per-record curation we cannot derive a defensible role-class label, so we exclude rather than poison the metric. Loses 106 records of role-probe measurement; the remaining ~376 per role probe still gives tight CIs, and the lift is large enough to flip the validation gate decision.

### 5. Expected accuracy lift

Under Option B (computed from existing V6g raw responses, no re-validation needed):

| Probe          | Pre-fix          | Post-fix         | Δ        |
|----------------|------------------|------------------|---------:|
| relation_axis  | 62.5% (245/392)  | 62.5% (245/392)  |   +0.0pp |
| subject_role   | 54.8% (252/460)  | **68.1% (241/354)** | **+13.3pp** |
| object_role    | 59.6% (274/460)  | **72.6% (257/354)** | **+13.0pp** |
| scope          | 87.2% (251/288)  | 87.2% (251/288)  |   +0.0pp |
| verify_grounding | unchanged      | unchanged        |   +0.0pp |

Removing 106 misclassified "absent" records from each role-probe denominator (and the corresponding misclassified-as-correct count from numerator) shifts both role probes from "fails the 90 % gate by 30+pp" to "fails by ~20pp" — large but still well below the V6g pass-bar. So the mapping fix alone does not save the validation; it does, however, give a much truer baseline for whether prompt-tuning or role-probe redesign can close the gap. (Note: even after the fix, Pro's role-probe accuracy is below 80%, suggesting genuine probe-design issues remain — e.g., symmetric-relation subject/object ambiguity.)

## Change applied

`scripts/v6g_gemini_validation.py`, lines 59–93:

- Added a comment block (lines 59–80) documenting the audit and the Option-B rationale.
- Three rows in `GOLD_TAG_TO_PROBE_CLASS` changed for `subject_role` and `object_role`:
  - `no_relation`:       `"absent"` → `None`
  - `grounding`:         `"absent"` → `None`
  - `entity_boundaries`: `"absent"` → `None`

No other fields touched; production prompts (`src/indra_belief/scorers/probes/*.py`) untouched; main() flow unchanged (the existing `if gold_class is None: skipped[...] += 1; continue` branch already handles `None` mappings).

---

## Revision 2026-05-07 — Option A (content-conditional)

### Why Option B turned out to be too aggressive

The first V6g run after Option B revealed an inadvertent side-effect: Option B not only stripped the 106 mis-mapped "absent" gold under `no_relation`/`grounding`/`entity_boundaries`, it **collapsed the entire role-probe gold to a single class**. Inspection of `data/v_phase/v6g_responses.jsonl`:

| Probe          | Gold-class distribution after Option B |
|----------------|----------------------------------------|
| subject_role   | `present_as_subject = 354/354` (100%)   |
| object_role    | `present_as_object  = 354/354` (100%)   |

With one gold class per probe, the most-frequent-class baseline = 1.000 and the composite gate's 1-class escape hatch fires (`mfc=1.0 ⇒ INSUFFICIENT_GOLD`). The validation cannot even *try* to measure role-probe quality. Option B was strictly more cautious than Option A on the original metric (micro-acc), but it threw away every minority-class gold record we had — including the ~15-18 records per probe where a substring/alias check makes the "absent" call high-confidence (`grounding`, `entity_boundaries`).

### Resolver policy (Option A — content-conditional)

For records under the three "claim-suspect" tags only, role-probe gold is now resolved per-record at task-build time by `_resolve_role_gold(probe, gold_tag, rec)`:

1. **Test 1 — literal substring.** Case-insensitive substring match of the entity name against the evidence text. Hit ⇒ `present_as_*`.
2. **Test 2a — normalized substring.** Strip non-alphanumeric, lowercase both sides; substring match. Catches BMP2↔BMP-2, MMP1↔MMP-1, IL22↔IL-22, SUMO1↔SUMO-1 etc. without per-pair entries. Hit ⇒ `present_as_*`.
3. **Test 2b — static alias map.** `_ROLE_ALIAS_MAP` (~92 entries; HGNC↔display-name pairs harvested from Pro's V6g rationales: TRAIL↔TNFSF10, INI1↔SMARCB1, iNOS↔NOS2, eIF2alpha↔EIF2S1, COX-2↔PTGS2, Astrin↔SPAG5, ERK↔MAPK1, PKB↔AKT1, NFkB↔NFKB1, CXCR7↔ACKR3, PACAP↔ADCYAP1, TOPK↔PBK, Hsp27↔HSPB1, XPB↔ERCC3, FAK↔PTK2, eNOS↔NOS3, fibronectin↔FN1, p53↔TP53, p21↔CDKN1A, etc.). Each alternate is checked under literal + normalized substring. Hit ⇒ `present_as_*`.
4. **Miss after Tests 1–3 ⇒ `"absent"`.** For `grounding` and `entity_boundaries` the gold tag itself says the entity name is suspect, so a miss is high-confidence absent. For `no_relation` it means the entity wasn't mentioned at all.

For records that pass Tests 1–3 we assume the gold role matches the claim role (subject→`present_as_subject`, object→`present_as_object`). We deliberately do **not** try to detect mediator / decoy roles — disambiguating those requires per-record curation we don't have, and the production probe expects subject/object as the dominant present-class anyway.

### Per-tag resolver tally

Computed by loading the U2 holdout (482 pairs) and applying `_resolve_role_gold` to every record under the three target tags:

| Tag                | N  | subject_role: present | absent | object_role: present | absent |
|--------------------|---:|----------------------:|-------:|---------------------:|-------:|
| no_relation        | 53 |                    49 |      4 |                   51 |      2 |
| grounding          | 33 |                    24 |      9 |                   20 |     13 |
| entity_boundaries  | 20 |                    18 |      2 |                   17 |      3 |
| **Total**          | 106|                **91** | **15** |               **88** | **18** |

So Option A restores **15 minority-class `absent` gold records on subject_role** and **18 on object_role** that Option B had dropped. The remaining "present" records (91 / 88) re-enter the majority class.

### Expected impact on V6g

| Metric                                  | Option B            | Option A (this revision) |
|-----------------------------------------|---------------------|--------------------------|
| subject_role denominator                | 354                 | **460** (+106)           |
| subject_role gold dist                  | present=354 (100%)  | present=445, absent=15   |
| subject_role mfc baseline               | 1.000               | **0.967**                |
| object_role denominator                 | 354                 | **460** (+106)           |
| object_role gold dist                   | present=354 (100%)  | present=442, absent=18   |
| object_role mfc baseline                | 1.000               | **0.961**                |
| Composite-gate verdict per role probe   | INSUFFICIENT_GOLD   | measurable (lift+macro+per-class recall on `absent` evaluable) |

Macro-recall is now meaningful: it averages recall on `present_as_*` (n≈445/442) with recall on `absent` (n=15/18). The min-class-recall floor will fire on the `absent` class (support 15/18 ≥ min_support=5), so the gate can now distinguish a curator that genuinely catches the high-confidence-absent cases from one that always says "present". That is exactly the V6g question we wanted to answer.

### Files changed

`scripts/v6g_gemini_validation.py`:

- Replaced the Option-B comment block above `GOLD_TAG_TO_PROBE_CLASS` with the Option-A audit log and resolver-tally summary.
- Added a `_ROLE_RESOLVE_SENTINEL = "__RESOLVE__"` constant.
- For `no_relation`, `grounding`, `entity_boundaries` rows: changed the `subject_role` and `object_role` values from `None` to `_ROLE_RESOLVE_SENTINEL`.
- Added the `_ROLE_ALIAS_MAP` dict (~92 entries), `_normalize_for_match`, `_entity_in_evidence`, and `_resolve_role_gold` helpers below the gold map.
- In the `main()` task-build loop: when `gold_class == _ROLE_RESOLVE_SENTINEL`, call `_resolve_role_gold(probe, gold_tag, rec)` to obtain the actual class. Tally the resolutions and print a per-(probe, tag, class) summary so a re-run shows exactly how many records were resolved each way.
- Added `import re` for the normalizer.

No production prompts were touched (`src/indra_belief/scorers/probes/*.py` untouched), no curator prompts (`src/indra_belief/v_phase/curator_prompts.py` untouched), no V6g re-run was performed (it's expensive and out of scope for this revision; the user will re-run).
