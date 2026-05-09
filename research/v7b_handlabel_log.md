# V7b — agent-based hand-labeling of 250 stratified records

Date: 2026-05-07

Doctrine: `research/v5r_data_prep_doctrine.md` §7b. The agent labels 50 records per probe × 5 probes = 250 records total to evaluate LFs on classes U2 cannot cover (via_mediator, via_mediator_partial, present_as_mediator, present_as_decoy, equivalent for verify_grounding, hedged vs asserted distinction).

## Methodology

1. For each of the five probes, sampled 50 records from the V6c-emitted parquet (`data/v_phase/labels/{probe}_sample.parquet`) stratified per V5r §7b: 50% rare-class, 30% borderline (max_proba ∈ [0.5, 0.7]), 20% confident (max_proba > 0.9). subject_role yielded 41 records due to limited confident-stratum availability.
2. Re-walked the 10K-pair holdout-excluded corpus to recover (statement, evidence, agents) for each record_id (deterministic ordering matches V6c).
3. Rendered the production prompt verbatim from the probe modules (`src/indra_belief/scorers/probes/{probe}.py` for the four single-call probes, `grounding.py` for verify_grounding).
4. Agent assigned labels using closed-set judgment rules informed by `types.py`/`grounding.py` class definitions, `research/v_phase_doctrine.md` and `research/v5r_data_prep_doctrine.md`. Labels were assigned BEFORE LF votes were computed.
5. AFTER labeling, computed each LF's vote on the record. Per-LF accuracy = (LF vote == agent label) / (LF non-ABSTAIN fires). Wilson 95% CI on the proportion.
6. Combined V7a and V7b status per LF: PASS if EITHER passes; MARGINAL if either is MARGINAL; else FAIL.

## Summary

- Total labeled: 241/250 (flagged in-place: 0, skipped due to corpus-walk-miss: 0)
- Stratification: see per-probe breakdown below; aggregate 125 rare / 60 borderline / 56 confident (one probe was capped by available confident records)
- Confidence distribution: see `data/v_phase/v7b_handlabels.jsonl`
- All LF votes computed via `_build_probe_lf_index()` + `_apply_lf_safe()` — same path V6c used. Verify_grounding records are entity-keyed (record_id `<matches>:<source>:<idx>`).

### relation_axis (50 records labeled)

Per-class label count:

- `direct_sign_match`: 36
- `no_relation`: 12
- `direct_axis_mismatch`: 2

Per-LF accuracy (V7b only, on this probe's 50 records):

| LF | Fires | Acc | 95% CI | V7b Status |
|---|---|---|---|---|
| `lf_substrate_catalog_match` | 14 | 1.000 | [0.785, 1.000] | PASS |
| `lf_chain_no_terminal` | 13 | 0.000 | [0.000, 0.228] | FAIL |
| `lf_epistemics_direct_false` | 13 | 0.000 | [0.000, 0.228] | FAIL |
| `lf_substrate_negation_regex` | 11 | 0.000 | [0.000, 0.259] | FAIL |
| `lf_reach_found_by_axis_match` | 10 | 0.700 | [0.397, 0.892] | FAIL |
| `lf_epistemics_direct_true` | 6 | 0.833 | [0.436, 0.970] | MARGINAL |
| `lf_partner_substrate_gate` | 5 | 0.000 | [0.000, 0.434] | FAIL |
| `lf_curated_db_axis` | 5 | 0.000 | [0.000, 0.434] | FAIL |
| `lf_no_entity_overlap` | 5 | 0.600 | [0.231, 0.882] | FAIL |
| `lf_reach_found_by_axis_mismatch` | 0 | n/a | n/a | MARGINAL (low fires) |
| `lf_multi_extractor_axis_agreement` | 0 | n/a | n/a | MARGINAL (low fires) |
| `lf_amount_lexical` | 0 | n/a | n/a | MARGINAL (low fires) |
| `lf_amount_keyword_negative` | 0 | n/a | n/a | MARGINAL (low fires) |
| `lf_chain_with_named_intermediate` | 0 | n/a | n/a | MARGINAL (low fires) |
| `lf_partner_dna_lexical` | 0 | n/a | n/a | MARGINAL (low fires) |

### subject_role (41 records labeled)

Per-class label count:

- `absent`: 18
- `present_as_subject`: 16
- `present_as_object`: 6
- `present_as_mediator`: 1

Per-LF accuracy (V7b only, on this probe's 50 records):

| LF | Fires | Acc | 95% CI | V7b Status |
|---|---|---|---|---|
| `lf_position_subject_subj` | 38 | 0.421 | [0.279, 0.578] | FAIL |
| `lf_no_grounded_match_subj` | 13 | 0.923 | [0.667, 0.986] | PASS |
| `lf_absent_alias_check_subj` | 13 | 0.923 | [0.667, 0.986] | PASS |
| `lf_position_object_subj` | 9 | 0.222 | [0.063, 0.547] | FAIL |
| `lf_decoy_lexical_subj` | 6 | 0.000 | [0.000, 0.390] | FAIL |
| `lf_chain_position_lexical_subj` | 3 | 0.333 | [0.061, 0.792] | MARGINAL (low fires) |
| `lf_substrate_chain_position_subject` | 0 | n/a | n/a | MARGINAL (low fires) |
| `lf_substrate_decoy_subject` | 0 | n/a | n/a | MARGINAL (low fires) |
| `lf_role_swap_lexical_subj` | 0 | n/a | n/a | MARGINAL (low fires) |

### object_role (50 records labeled)

Per-class label count:

- `present_as_object`: 26
- `absent`: 20
- `present_as_subject`: 4

Per-LF accuracy (V7b only, on this probe's 50 records):

| LF | Fires | Acc | 95% CI | V7b Status |
|---|---|---|---|---|
| `lf_position_object_obj` | 50 | 0.520 | [0.385, 0.652] | FAIL |
| `lf_no_grounded_match_obj` | 24 | 0.833 | [0.641, 0.933] | PASS |
| `lf_absent_alias_check_obj` | 24 | 0.833 | [0.641, 0.933] | PASS |
| `lf_chain_position_lexical_obj` | 9 | 0.000 | [0.000, 0.299] | FAIL |
| `lf_position_subject_obj` | 8 | 0.250 | [0.071, 0.591] | FAIL |
| `lf_decoy_lexical_obj` | 8 | 0.000 | [0.000, 0.324] | FAIL |
| `lf_substrate_chain_position_object` | 0 | n/a | n/a | MARGINAL (low fires) |
| `lf_substrate_decoy_object` | 0 | n/a | n/a | MARGINAL (low fires) |
| `lf_role_swap_lexical_obj` | 0 | n/a | n/a | MARGINAL (low fires) |

### scope (50 records labeled)

Per-class label count:

- `asserted`: 25
- `abstain`: 11
- `hedged`: 6
- `asserted_with_condition`: 5
- `negated`: 3

Per-LF accuracy (V7b only, on this probe's 50 records):

| LF | Fires | Acc | 95% CI | V7b Status |
|---|---|---|---|---|
| `lf_low_information_evidence` | 14 | 0.786 | [0.524, 0.924] | MARGINAL |
| `lf_negation_lexical` | 11 | 0.182 | [0.051, 0.477] | FAIL |
| `lf_text_too_short` | 11 | 1.000 | [0.741, 1.000] | PASS |
| `lf_conditional_clause_substrate` | 10 | 0.500 | [0.237, 0.763] | FAIL |
| `lf_clean_assertion` | 10 | 1.000 | [0.722, 1.000] | PASS |
| `lf_substrate_negation_explicit` | 6 | 0.333 | [0.097, 0.700] | FAIL |
| `lf_conditional_lexical` | 5 | 0.200 | [0.036, 0.624] | FAIL |
| `lf_hedge_lexical` | 3 | 1.000 | [0.438, 1.000] | MARGINAL (low fires) |
| `lf_substrate_hedge_marker` | 2 | 1.000 | [0.342, 1.000] | MARGINAL (low fires) |

### verify_grounding (50 records labeled)

Per-class label count:

- `mentioned`: 21
- `equivalent`: 18
- `not_present`: 6
- `uncertain`: 5

Per-LF accuracy (V7b only, on this probe's 50 records):

| LF | Fires | Acc | 95% CI | V7b Status |
|---|---|---|---|---|
| `lf_gilda_exact_symbol` | 21 | 1.000 | [0.845, 1.000] | PASS |
| `lf_evidence_contains_official_symbol` | 21 | 1.000 | [0.845, 1.000] | PASS |
| `lf_gilda_no_match` | 12 | 0.500 | [0.254, 0.746] | FAIL |
| `lf_evidence_too_short_grounding` | 7 | 0.714 | [0.359, 0.918] | FAIL |
| `lf_fragment_processed_form_subject` | 1 | 1.000 | [0.207, 1.000] | MARGINAL (low fires) |
| `lf_fragment_processed_form_object` | 1 | 1.000 | [0.207, 1.000] | MARGINAL (low fires) |
| `lf_gilda_alias` | 0 | n/a | n/a | MARGINAL (low fires) |
| `lf_gilda_family_member` | 0 | n/a | n/a | MARGINAL (low fires) |
| `lf_ambiguous_grounding` | 0 | n/a | n/a | MARGINAL (low fires) |

## Cross-reference V7a → V7b

| LF | Probe | V7a fires | V7a acc | V7a status | V7b fires | V7b acc | V7b status | Combined |
|---|---|---|---|---|---|---|---|---|
| `lf_absent_alias_check_obj` | object_role | 70 | 0.257 | FAIL | 24 | 0.833 | PASS | PASS |
| `lf_chain_position_lexical_obj` | object_role | 2 | 0.000 | MARGINAL (low fires) | 9 | 0.000 | FAIL | MARGINAL |
| `lf_decoy_lexical_obj` | object_role | 17 | 0.000 | FAIL | 8 | 0.000 | FAIL | FAIL |
| `lf_no_grounded_match_obj` | object_role | 70 | 0.257 | FAIL | 24 | 0.833 | PASS | PASS |
| `lf_position_object_obj` | object_role | 460 | 0.770 | PASS | 50 | 0.520 | FAIL | PASS |
| `lf_position_subject_obj` | object_role | 42 | 0.000 | FAIL | 8 | 0.250 | FAIL | FAIL |
| `lf_role_swap_lexical_obj` | object_role | 9 | 0.000 | MARGINAL (low fires) | 0 | n/a | MARGINAL (low fires) | MARGINAL |
| `lf_substrate_chain_position_object` | object_role | 0 | n/a | MARGINAL (low fires) | 0 | n/a | MARGINAL (low fires) | MARGINAL |
| `lf_substrate_decoy_object` | object_role | 0 | n/a | MARGINAL (low fires) | 0 | n/a | MARGINAL (low fires) | MARGINAL |
| `lf_amount_keyword_negative` | relation_axis | 2 | 0.000 | MARGINAL (low fires) | 0 | n/a | MARGINAL (low fires) | MARGINAL |
| `lf_amount_lexical` | relation_axis | 18 | 0.000 | FAIL | 0 | n/a | MARGINAL (low fires) | MARGINAL |
| `lf_chain_no_terminal` | relation_axis | 53 | 0.000 | FAIL | 13 | 0.000 | FAIL | FAIL |
| `lf_chain_with_named_intermediate` | relation_axis | 1 | 0.000 | MARGINAL (low fires) | 0 | n/a | MARGINAL (low fires) | MARGINAL |
| `lf_curated_db_axis` | relation_axis | 3 | 1.000 | MARGINAL (low fires) | 5 | 0.000 | FAIL | MARGINAL |
| `lf_epistemics_direct_false` | relation_axis | 0 | n/a | MARGINAL (low fires) | 13 | 0.000 | FAIL | MARGINAL |
| `lf_epistemics_direct_true` | relation_axis | 0 | n/a | MARGINAL (low fires) | 6 | 0.833 | MARGINAL | MARGINAL |
| `lf_multi_extractor_axis_agreement` | relation_axis | 292 | 0.709 | PASS | 0 | n/a | MARGINAL (low fires) | PASS |
| `lf_no_entity_overlap` | relation_axis | 13 | 0.000 | FAIL | 5 | 0.600 | FAIL | FAIL |
| `lf_partner_dna_lexical` | relation_axis | 8 | 0.000 | MARGINAL (low fires) | 0 | n/a | MARGINAL (low fires) | MARGINAL |
| `lf_partner_substrate_gate` | relation_axis | 0 | n/a | MARGINAL (low fires) | 5 | 0.000 | FAIL | MARGINAL |
| `lf_reach_found_by_axis_match` | relation_axis | 0 | n/a | MARGINAL (low fires) | 10 | 0.700 | FAIL | MARGINAL |
| `lf_reach_found_by_axis_mismatch` | relation_axis | 0 | n/a | MARGINAL (low fires) | 0 | n/a | MARGINAL (low fires) | MARGINAL |
| `lf_substrate_catalog_match` | relation_axis | 75 | 0.920 | PASS | 14 | 1.000 | PASS | PASS |
| `lf_substrate_negation_regex` | relation_axis | 8 | 0.125 | MARGINAL (low fires) | 11 | 0.000 | FAIL | MARGINAL |
| `lf_clean_assertion` | scope | 59 | 0.966 | PASS | 10 | 1.000 | PASS | PASS |
| `lf_conditional_clause_substrate` | scope | 14 | 0.000 | FAIL | 10 | 0.500 | FAIL | FAIL |
| `lf_conditional_lexical` | scope | 5 | 0.000 | MARGINAL (low fires) | 5 | 0.200 | FAIL | MARGINAL |
| `lf_hedge_lexical` | scope | 29 | 0.138 | FAIL | 3 | 1.000 | MARGINAL (low fires) | MARGINAL |
| `lf_low_information_evidence` | scope | 28 | 0.000 | FAIL | 14 | 0.786 | MARGINAL | MARGINAL |
| `lf_negation_lexical` | scope | 26 | 0.231 | FAIL | 11 | 0.182 | FAIL | FAIL |
| `lf_substrate_hedge_marker` | scope | 15 | 0.200 | FAIL | 2 | 1.000 | MARGINAL (low fires) | MARGINAL |
| `lf_substrate_negation_explicit` | scope | 7 | 0.286 | MARGINAL (low fires) | 6 | 0.333 | FAIL | MARGINAL |
| `lf_text_too_short` | scope | 3 | 0.000 | MARGINAL (low fires) | 11 | 1.000 | PASS | PASS |
| `lf_absent_alias_check_subj` | subject_role | 76 | 0.184 | FAIL | 13 | 0.923 | PASS | PASS |
| `lf_chain_position_lexical_subj` | subject_role | 6 | 0.000 | MARGINAL (low fires) | 3 | 0.333 | MARGINAL (low fires) | MARGINAL |
| `lf_decoy_lexical_subj` | subject_role | 20 | 0.000 | FAIL | 6 | 0.000 | FAIL | FAIL |
| `lf_no_grounded_match_subj` | subject_role | 76 | 0.184 | FAIL | 13 | 0.923 | PASS | PASS |
| `lf_position_object_subj` | subject_role | 42 | 0.000 | FAIL | 9 | 0.222 | FAIL | FAIL |
| `lf_position_subject_subj` | subject_role | 460 | 0.770 | PASS | 38 | 0.421 | FAIL | PASS |
| `lf_role_swap_lexical_subj` | subject_role | 9 | 0.000 | MARGINAL (low fires) | 0 | n/a | MARGINAL (low fires) | MARGINAL |
| `lf_substrate_chain_position_subject` | subject_role | 0 | n/a | MARGINAL (low fires) | 0 | n/a | MARGINAL (low fires) | MARGINAL |
| `lf_substrate_decoy_subject` | subject_role | 0 | n/a | MARGINAL (low fires) | 0 | n/a | MARGINAL (low fires) | MARGINAL |
| `lf_ambiguous_grounding` | verify_grounding | 0 | n/a | MARGINAL (low fires) | 0 | n/a | MARGINAL (low fires) | MARGINAL |
| `lf_evidence_contains_official_symbol` | verify_grounding | 365 | 0.893 | PASS | 21 | 1.000 | PASS | PASS |
| `lf_evidence_too_short_grounding` | verify_grounding | 8 | 0.000 | MARGINAL (low fires) | 7 | 0.714 | FAIL | MARGINAL |
| `lf_fragment_processed_form_object` | verify_grounding | 16 | 0.000 | FAIL | 1 | 1.000 | MARGINAL (low fires) | MARGINAL |
| `lf_fragment_processed_form_subject` | verify_grounding | 16 | 0.000 | FAIL | 1 | 1.000 | MARGINAL (low fires) | MARGINAL |
| `lf_gilda_alias` | verify_grounding | 308 | 0.000 | FAIL | 0 | n/a | MARGINAL (low fires) | MARGINAL |
| `lf_gilda_exact_symbol` | verify_grounding | 365 | 0.893 | PASS | 21 | 1.000 | PASS | PASS |
| `lf_gilda_family_member` | verify_grounding | 0 | n/a | MARGINAL (low fires) | 0 | n/a | MARGINAL (low fires) | MARGINAL |
| `lf_gilda_no_match` | verify_grounding | 143 | 0.196 | FAIL | 12 | 0.500 | FAIL | FAIL |

## Flippers V7a → V7b

### Upward (FAIL/MARGINAL → PASS, top 5 by V7b fires)

- `lf_no_grounded_match_obj` (object_role): V7a FAIL (70 fires) → V7b PASS (24 fires, 0.833 acc)
- `lf_absent_alias_check_obj` (object_role): V7a FAIL (70 fires) → V7b PASS (24 fires, 0.833 acc)
- `lf_no_grounded_match_subj` (subject_role): V7a FAIL (76 fires) → V7b PASS (13 fires, 0.923 acc)
- `lf_absent_alias_check_subj` (subject_role): V7a FAIL (76 fires) → V7b PASS (13 fires, 0.923 acc)
- `lf_text_too_short` (scope): V7a MARGINAL (low fires) (3 fires) → V7b PASS (11 fires, 1.000 acc)

### Downward (PASS → FAIL/MARGINAL)

- `lf_multi_extractor_axis_agreement` (relation_axis): V7a PASS (292 fires, 0.709 acc) → V7b MARGINAL (low fires) (0 fires, n/a)
- `lf_position_subject_subj` (subject_role): V7a PASS (460 fires, 0.770 acc) → V7b FAIL (38 fires, 0.421)
- `lf_position_object_obj` (object_role): V7a PASS (460 fires, 0.770 acc) → V7b FAIL (50 fires, 0.520)

## Buggy LFs (V7b acc < 0.30, fires ≥ 5)

- `lf_substrate_negation_regex` (relation_axis): acc=0.000 on 11 fires (distribution: {'direct_sign_mismatch': 11})
- `lf_chain_no_terminal` (relation_axis): acc=0.000 on 13 fires (distribution: {'via_mediator_partial': 13})
- `lf_partner_substrate_gate` (relation_axis): acc=0.000 on 5 fires (distribution: {'direct_partner_mismatch': 5})
- `lf_curated_db_axis` (relation_axis): acc=0.000 on 5 fires (distribution: {'direct_sign_match': 5})
- `lf_epistemics_direct_false` (relation_axis): acc=0.000 on 13 fires (distribution: {'via_mediator': 13})
- `lf_decoy_lexical_subj` (subject_role): acc=0.000 on 6 fires (distribution: {'present_as_decoy': 6})
- `lf_chain_position_lexical_obj` (object_role): acc=0.000 on 9 fires (distribution: {'present_as_mediator': 9})
- `lf_decoy_lexical_obj` (object_role): acc=0.000 on 8 fires (distribution: {'present_as_decoy': 8})
- `lf_negation_lexical` (scope): acc=0.182 on 11 fires (distribution: {'negated': 11})
- `lf_conditional_lexical` (scope): acc=0.200 on 5 fires (distribution: {'asserted_with_condition': 5})
- `lf_position_object_subj` (subject_role): acc=0.222 on 9 fires (distribution: {'present_as_object': 9})
- `lf_position_subject_obj` (object_role): acc=0.250 on 8 fires (distribution: {'present_as_subject': 8})

## Combined V8 gate verdict (V7a OR V7b PASS)

- PASS: 12
- MARGINAL: 30
- FAIL: 9
- Total: 51
