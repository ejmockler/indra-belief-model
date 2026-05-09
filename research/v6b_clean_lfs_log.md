# V6b — clean (non-substrate) LFs (implementation log)

Date: 2026-05-06
Doctrine: research/v5r_data_prep_doctrine.md §3 (LF tables tagged [clean]), §10 (no holdout-tuned features), §11 (no curator gold leakage)
Predecessor: V6a (research/v6a_substrate_lfs_log.md)

## File structure

- `src/indra_belief/v_phase/clean_lfs.py` — module
- `tests/test_v6_clean_lfs.py` — test file (65 tests)
- `scripts/v6b_sanity_check.py` — sanity-check runner over 100 holdout records

The `v_phase/` package now contains both `substrate_lfs.py` (V6a) and
`clean_lfs.py` (V6b). V6c (label matrix + Snorkel fit) and V6d (JSONL
emission) will land alongside.

## LF count per probe

38 clean LFs total (vs V6a's 13 substrate-tuned LFs = 51 LFs in
combined V6c label matrix).

| Probe | Clean LF count | Notes |
|---|---|---|
| relation_axis | 11 | curated_db, reach_found_by (match + mismatch), epistemics (true + false), multi_extractor, amount (lexical + keyword_negative), chain_named_intermediate, no_entity_overlap, partner_dna_lexical |
| subject_role | 7 | position_subject, position_object, role_swap, chain_position, decoy, no_grounded_match, absent_alias_check (each takes `role="subject"`) |
| object_role | 7 | same set with `role="object"` |
| scope | 6 | hedge_lexical, negation_lexical, conditional_lexical, clean_assertion, low_information_evidence, text_too_short |
| verify_grounding | 7 | gilda_exact_symbol, evidence_contains_official_symbol, gilda_alias, gilda_family_member, gilda_no_match, evidence_too_short, ambiguous_grounding |

All LFs return `-1` (ABSTAIN) on no-vote path. No `mode` parameter
(clean by construction). Class indices imported from substrate_lfs.py
(RELATION_AXIS_CLASSES, ROLE_CLASSES, SCOPE_CLASSES, GROUNDING_CLASSES).

## Lexicon sources

Per V5r §3 and §10 ("no holdout-tuned features"), each lexicon is
defined inline from open-source biomedical NLP resources, NOT from
the substrate `_HEDGE_MARKERS` / `_LOF_PATTERNS` etc.

| Lexicon | Provenance | Cue count |
|---|---|---|
| `_HEDGE_CUES_CLEAN` | LingScope-derived (compact subset that survives across-corpus benchmarks) | ~30 cues |
| `_NEGATION_RE_CLEAN` | BioScope-derived (V5r §3.3 phrasing verbatim) | 12 patterns |
| `_CONDITIONAL_RE_CLEAN` | V5r §3.3 phrasing verbatim | 1 regex |
| `_AMOUNT_LEXICON` | V5r §3.1 wording verbatim | 9 entries |
| `_ACTIVITY_LEXICON` | V5r §3.1 wording verbatim | 4 entries |
| `_CHAIN_MARKERS_CLEAN` | V5r §3.1 wording verbatim | 6 markers |
| `_DNA_BINDING_LEXICON` | V5r §3.1 wording + standard biomed terms | 8 entries |
| `_BOILERPLATE_PATTERNS` | V5r §3.3 wording verbatim | 4 patterns |
| `_REACH_AXIS_TOKENS` | REACH `found_by` pattern-ID dictionary (open) | 19 axis tokens |
| `_CURATED_DB_APIS` | V5r §3.1 fixed list (HPRD/BioPAX/SIGNOR/BEL/TRRUST) | 5 |

No lexicon is sourced from the substrate-tuned constants in
`context_builder.py`/`relation_patterns.py`. Imports from substrate are
limited to type maps and helpers (`RELATION_AXIS_CLASSES`,
`_claim_axis_sign`, `_claim_subject_object`, `_claim_stmt_type`,
`_evidence_text`) — no holdout-derived data.

Gilda is used for entity grounding lookups (`_gilda_aliases`,
`lf_gilda_*`) per V5r §3.4 — open-source resource, not holdout-tuned.

## Test coverage

`tests/test_v6_clean_lfs.py`: **65 tests, all passing**.

| Test class | Coverage |
|---|---|
| TestCuratedDBAxis | 2 tests: SIGNOR votes direct, REACH abstains |
| TestReachFoundBy | 5 tests: phosphorylation match, positive_activation match, sign mismatch, axis mismatch (Translocation), no annotations abstain |
| TestEpistemics | 3 tests: direct=True votes direct, direct=False votes via_mediator, missing abstains |
| TestMultiExtractor | 2 tests: multi-source votes, single-source abstains |
| TestAmountLFs | 4 tests: amount_lexical fires/abstains, amount_keyword_negative fires/abstains |
| TestChainNamedIntermediate | 2 tests: fires with named intermediate, no chain marker abstains |
| TestNoEntityOverlap | 2 tests: fires when neither entity present, abstains when entities present |
| TestPartnerDNALexical | 3 tests: fires on Complex+promoter, no DNA element abstains, non-Complex abstains |
| TestRolePosition | 3 tests: position_subject, position_object, role-conditional abstain |
| TestRoleSwap | 2 tests: swap detected, no swap abstains |
| TestChainPositionLexical | 2 tests: via pattern fires, no chain pattern abstains |
| TestDecoyLexical | 2 tests: axis mismatch fires decoy, axis match abstains |
| TestAbsentLFs | 3 tests: no_grounded_match fires/abstains, absent_alias_check fires |
| TestScopeLexical | 13 tests: hedge fires/abstains, negation fires/abstains, conditional fires/abstains, clean_assertion fires/abstains, low_information (3 cases), text_too_short (2 cases) |
| TestGroundingLFs | 11 tests: exact_symbol fires/abstains, evidence_contains, alias fires, family_member fires/abstains, no_match fires/abstains, too_short fires/abstains, ambiguous returns int-or-abstain |
| TestLFIndex | 4 tests: index runs, grounding runner, count == 38, kinds valid |
| TestDictInput | 2 tests: dict record relation_axis, dict role with kwarg |

Tests run on CPU in ~6s; no llama-server, no GPU.

Full repo test suite: **555 passed, 1 skipped** (no regressions).

## Sanity check on holdout-overlap fixtures

Ran all 38 clean LFs on the first 100 records of
`data/benchmark/holdout_v15_sample.jsonl` (same fixture used for V6a
sanity).

### Per-LF firing rate highlights

| LF | fires/100 | Notes |
|---|---|---|
| `lf_position_subject_subj` | 100% | All records have a subject in subject position by construction |
| `lf_position_object_obj` | 100% | (mirror) |
| `lf_multi_extractor_axis_agreement` | 75% | High coverage from `source_counts` field |
| `lf_gilda_exact_symbol::subject` | 50% | About half of holdout has exact-symbol mention |
| `lf_gilda_exact_symbol::object` | 46% | (mirror) |
| `lf_gilda_alias` (subj/obj) | 47% / 44% | Alias fallback paths are dense |
| `lf_ambiguous_grounding` (subj/obj) | 42% / 39% | Realistic — biomedical text has many uppercase tokens |
| `lf_clean_assertion` | 17% | Records with no hedge/negation/condition |
| `lf_hedge_lexical` | 15% | Hedge cues present in ~1/7 evidence sentences |
| `lf_low_information_evidence` | 7% | Short or boilerplate evidence |
| `lf_no_grounded_match` (subj/obj) | 13% / 14% | Subject/object name not in evidence |
| `lf_absent_alias_check` (subj/obj) | 13% / 14% | Stricter alias check; same rate |
| `lf_no_entity_overlap` | 2% | Tight criterion (no entity within ±20 tokens of any verb) |
| `lf_negation_lexical` | 2% | Genuine negation rare in this slice |
| `lf_curated_db_axis` | 0% | source_api in this slice = sparser/medscan/reach/trips/rlimsp/isi (no SIGNOR/HPRD/BEL/TRRUST/BioPAX in first 100) |
| `lf_reach_found_by_*` | 0% | Holdout records lack `annotations.found_by` field — present only in raw INDRA dumps, not the V0r benchmark schema |
| `lf_epistemics_direct_*` | 0% | Holdout records lack `epistemics` field |

Note: the curated-DB / found_by / epistemics LFs have 0 fires on the
holdout-overlap slice because the V0r benchmark JSONL strips those
fields. They will fire densely in V6c when LFs run against the full
INDRA corpus (where source_api distribution is broader and `epistemics`
+ `annotations.found_by` are populated by the original extractor). V6c
will validate this on a 5K-record corpus sample before LabelModel fit.

### Per-class vote distribution per probe

**relation_axis** (80 votes): heavily skewed to `direct_sign_match`
(94%) due to multi_extractor coverage; small `direct_amount_match`
contribution (4%); `no_relation` 2%. The skew is expected on the
holdout-overlap slice (curators select for clear positives). V6c's
balanced class_balance + V6d's stratified subsample will rebalance.

**subject_role** (137 votes): `present_as_subject` 73%, `absent` 19%,
`present_as_object` 4%, `present_as_decoy` 3%, `present_as_mediator` 1%.
Mediator/decoy density looks low but expected — V6c will pull from full
corpus where rare-class fixtures are denser; V5r §6 covers synthetic-
oracle augmentation if naturals stay <3K/class.

**object_role** (138 votes): `present_as_object` 72%, `absent` 20%,
plus mirror minor classes. Same shape as subject_role (expected).

**scope** (41 votes): `asserted` 41%, `hedged` 37%, `abstain` 17%,
`negated` 5%, `asserted_with_condition` 0% (none triggered on first
100 records). Conditional class will need either fuller corpus runs
or synthetic augmentation per V5r §6.

**verify_grounding** (391 votes — 2 LFs × 100 entities × 2 roles):
`mentioned` 49%, `equivalent` 23%, `uncertain` 21%, `not_present` 7%.
Distribution across all four classes; LabelModel has signal for each.

## Notable design choices

1. **Clean LFs do not import substrate-tuned data.** `clean_lfs.py`
   only imports type maps (`RELATION_AXIS_CLASSES`, etc.) and structural
   helpers (`_claim_subject_object`, `_evidence_text`) from
   `substrate_lfs.py` — never the `_CYTOKINE_LIGAND_HGNC` / `CATALOG` /
   `_HEDGE_MARKERS` constants flagged in V5r §7c.

2. **Verb position approximation.** The scope LFs need a "claim verb
   position" to apply 8-token windows. `_claim_verb_position()` finds
   the first active verb in the span between subject and object
   mentions; falls back to the first active verb in the text. This is
   a heuristic — substantial precision is gained by anchoring against
   the claim entities in evidence first.

3. **Token-window approximation.** `_within_token_window` converts
   8-token windows to character windows via 8 chars/token average. The
   threshold is robust because the search regex requires a word match;
   over-large windows over-fire (false positives), not silently miss.

4. **Family LF lazy resolution.** `lf_gilda_family_member` accepts
   either a pre-resolved entity dict (with `is_family` and
   `family_members` populated) OR a name-only entity dict, in which
   case it resolves via `GroundedEntity.resolve()`. This makes it
   reusable from V6c when records carry only the entity name.

5. **`lf_no_entity_overlap` is conservative.** It requires NEITHER
   claim entity (post-Gilda alias) within ±20 tokens of any verb
   before voting `no_relation`. This is intentionally narrow to
   avoid false negatives. The 2% firing rate is a feature, not a bug —
   the LF is targeted at the rare-class `no_relation` slot.

6. **REACH `found_by` parser is permissive.** Some pattern IDs lack the
   `_syntax_` suffix; the parser falls back to a head-token lookup so
   patterns like `Phosphorylation_1a` (without `syntax`) still parse.

7. **`lf_ambiguous_grounding` runs Gilda live.** This LF makes online
   Gilda calls per uppercase token in the evidence. The cost is bounded
   by Gilda's lru_cache; the sanity check ran 100 records in ~2s.

## LFs that couldn't be implemented cleanly (notes for V5r revision)

None blocked. Caveats for future V5r revisions:

1. **REACH `found_by` and `epistemics` LFs** depend on raw INDRA dump
   fields that are stripped in the V0r benchmark JSONL schema (the
   holdout records have no `annotations`/`epistemics`). On the
   benchmark slice these LFs always abstain; on the V6c full-corpus
   run they will fire (the corpus retains those fields). V5r §3.1 may
   want to mark this dependency explicitly so V6c reviewers don't
   misread "0% fires" on the V6b sanity check.

2. **`lf_chain_position_lexical` requires non-empty A and B context.**
   "Non-empty" was implemented as ≥5 chars before and after the
   matched pattern. V5r §3.2 says "non-empty" without a numerical
   threshold; we picked 5 chars because the pattern itself is at least
   "via X" (5 chars), so requiring 5 chars on each side guards against
   the pattern being at sentence boundary. V5r could pin this number.

3. **`lf_role_swap_lexical` is an English-syntax heuristic.** The LF
   detects `<obj> <verb> <subj>` ordering, which catches common
   passive/inverted constructions but misses compound sentences,
   coordinated clauses, and parenthetical inserts. Recall is bounded;
   precision is high. Low firing on holdout (0%) reflects that
   curated holdout records use the natural ordering. Real corpus runs
   will see more swaps; V6c will sample-verify.

4. **`lf_decoy_lexical` lexicon overlaps with the substrate CATALOG**
   in surface form (e.g., "phosphorylat" is both a clean axis cue and
   a CATALOG key). The clean version restricts to a self-contained
   regex list, NOT importing CATALOG. Coincidental overlap is fine
   per V5r §10 (the lexicon is open-source-derivable from any biomedical
   NLP resource).

## Reproducing the sanity check

```bash
.venv/bin/python scripts/v6b_sanity_check.py
```

Outputs the per-LF firing rate table + per-probe class distribution to
stdout. The first 100 records of `data/benchmark/holdout_v15_sample.jsonl`
are the canonical fixture (matches V6a's sanity sample).

## Next-step gating

V6b unblocks:
- V6c (Λ matrix + Snorkel fit) — V6a + V6b complete; V6c iterates the
  combined LF inventory (51 LFs across 5 probes) per V5r §4
- V7a (LF accuracy on U2 gold) — V6c provides per-LF firing matrix; V7a
  computes per-LF accuracy on `probe_gold_holdout.jsonl`

Ship gate items pending: V6c → V6d → V7 → V8 environment + data gate.
