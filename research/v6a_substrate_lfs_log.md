# V6a — substrate-tuned LFs in dual mode (implementation log)

Date: 2026-05-06
Doctrine: research/v5r_data_prep_doctrine.md §3 (LF tables), §7c (baseline-mode), §10 (substrate-tuning audit)
Predecessor commit: a43e1a1 (model setup README)

## File structure

- `src/indra_belief/v_phase/__init__.py` — V-phase package marker
- `src/indra_belief/v_phase/substrate_lfs.py` — module under test
- `tests/test_v6_substrate_lfs.py` — test file

The `v_phase/` subpackage is the home for V6a–V6d artifacts. V6b (clean
LFs), V6c (label-matrix builder + Snorkel fit), V6d (JSONL emission) will
land alongside as siblings.

## Swap mechanism

`substrate_baseline_mode()` is a `contextlib.contextmanager` that
monkey-patches seven attributes via module-level `setattr`:

| Module | Attribute | Tuned (live) | Baseline |
|---|---|---|---|
| `context_builder` | `_CYTOKINE_LIGAND_HGNC` | 73 HGNC symbols | `frozenset()` |
| `context_builder` | `_SITE_DENYLIST` | 7 denylist tokens | `frozenset()` |
| `context_builder` | `_HEDGE_MARKERS` | 20 cues | 5 universal cues (`may`, `might`, `suggest`, `propose`, `hypothesize`) |
| `context_builder` | `_HEDGE_PROXIMITY_CHARS` | 60 (M13 window) | 30 (default) |
| `context_builder` | `_LOF_PATTERNS` | 6 perturbation patterns | `()` |
| `relation_patterns` | `CATALOG` | 48 entries | 32 entries (16 holdout-tuned dropped) |
| `context_builder` | `_binding_admissible_for` | full type table | core gate (Complex→{protein}; everything else empty) |

The 16 dropped CATALOG entries are pattern-IDs with docstrings citing
"Q-phase regression", "R2:", "R5:", "M11 gap-fix", "U-phase" or
"Diagnosis FN/FP". Enumerated in `_TUNED_PATTERN_IDS` constant; the
audit script that derived this list runs against
`inspect.getsource(relation_patterns)` (non-load-bearing — the constant
is the source of truth).

Snapshot/restore is exception-safe (verified by
`test_swap_restores_on_exception`).

## LF count

13 LFs across 5 probes, all `[substrate-tuned]` per V5r §3:

| Probe | LF | Class index voted |
|---|---|---|
| relation_axis | `lf_substrate_catalog_match` | 0 (`direct_sign_match`) |
| relation_axis | `lf_substrate_negation_regex` | 2 (`direct_sign_mismatch`) |
| relation_axis | `lf_chain_no_terminal` | 6 (`via_mediator_partial`) |
| relation_axis | `lf_partner_substrate_gate` | 4 (`direct_partner_mismatch`) |
| subject_role | `lf_substrate_chain_position(entity="subject")` | 2 (`present_as_mediator`) |
| subject_role | `lf_substrate_decoy(entity="subject")` | 3 (`present_as_decoy`) |
| object_role | `lf_substrate_chain_position(entity="object")` | 2 (`present_as_mediator`) |
| object_role | `lf_substrate_decoy(entity="object")` | 3 (`present_as_decoy`) |
| scope | `lf_substrate_hedge_marker` | 1 (`hedged`) |
| scope | `lf_substrate_negation_explicit` | 3 (`negated`) |
| scope | `lf_conditional_clause_substrate` | 2 (`asserted_with_condition`) |
| verify_grounding | `lf_fragment_processed_form(entity="subject")` | 1 (`equivalent`) |
| verify_grounding | `lf_fragment_processed_form(entity="object")` | 1 (`equivalent`) |

Count matches V5r §3 substrate-tuned tally: 4 (relation_axis) + 4 (role,
2 per side) + 3 (scope) + 2 (grounding, 1 per agent) = 13.

All LFs return `-1` (ABSTAIN) when they cannot judge — this is the
Snorkel-LabelModel convention. Class indices match `RELATION_AXIS_CLASSES`
/ `ROLE_CLASSES` / `SCOPE_CLASSES` / `GROUNDING_CLASSES` in `substrate_lfs.py`,
which match `probes/types.py` and `grounding.py` exactly.

## Test coverage

`tests/test_v6_substrate_lfs.py`: **21 tests, all passing**.

| Test class | Coverage |
|---|---|
| `TestBaselineSwap` | 3 tests: swap changes constants, restores on exception, mode='baseline' arg routes through swap |
| `TestRelationAxisLFs` | 6 tests: catalog match aligned, no relation, unknown stmt_type, chain (with + without signal), partner gate |
| `TestRoleLFs` | 2 tests: chain position abstains, decoy abstains under aligned relation |
| `TestScopeLFs` | 5 tests: hedge fires, hedge absent, negation fires, conditional fires, conditional absent |
| `TestGroundingLFs` | 2 tests: Aβ tuned-only fires, no indicator abstains |
| `TestTunedVsBaselineDelta` | 2 tests: at least one LF differs on holdout-overlap fixtures, full LF index runs without error |
| `TestDictInput` | 1 test: dict-shaped statement+evidence round-trips through the LFs |

Tests run on CPU in 7s; no llama-server, no GPU.

Full repo test suite (`tests/`): **484 passed, 1 skipped** (parity
preserved, no regressions).

## Sanity check on holdout-overlap fixtures

Ran all 13 LFs in tuned + baseline modes on the first 100 records of
`data/benchmark/holdout_v15_sample.jsonl`. Results:

- 40 / 100 records had at least one LF firing (non-ABSTAIN) in either mode
- **7 / 100 records had at least one LF voting differently between modes**
- Per-LF diff counts (over 100 records):
  - `lf_substrate_negation_regex`: 3 (M9 LOF perturbation marker is the
    primary differential — `_LOF_PATTERNS` empties in baseline)
  - `lf_substrate_hedge_marker`: 4 (`_HEDGE_MARKERS` shrinks from 20 to 5
    in baseline; e.g., "could not" only fires under tuned)

Example differential records (from holdout_v15_sample):

| stmt_type | subj→obj | Evidence (truncated) | LF | tuned | baseline |
|---|---|---|---|---|---|
| IncreaseAmount | CTRL→SELE | "depletion of chymotrypsin-like protease ... eliminated ..." | `lf_substrate_negation_regex` | 2 (mismatch) | -1 (abstain) |
| Deacetylation | HDAC→AR | "histone deacetylase (HDAC) inhibitors ..." | `lf_substrate_negation_regex` | 2 | -1 |
| Phosphorylation | EGF→EGFR | "EGFR could not be phosphorylated by EGF ..." | `lf_substrate_hedge_marker` | 1 (hedged) | -1 |
| Inhibition | ACKR3→RB1 | "pharmacological inhibition of CXCR7 ..." | `lf_substrate_negation_regex` | 2 | -1 |

The differential is real and bounded: ~7% of holdout-overlap records
shift their substrate vote between tuned and baseline modes. V7c will
quantify the per-class vote-share delta against the full 482-record
holdout to determine which LFs are ACCEPTED with documented
contamination floor vs which need replacement in V6b.

## LFs that couldn't be implemented cleanly (notes for V5r revision)

None. All 13 substrate-tuned LFs from V5r §3 are implemented as thin
wrappers (avg ~20 lines per LF) around existing detectors in
`context_builder.py` and `relation_patterns.py`. No detector code was
duplicated; no detector code was modified.

Caveats / minor design choices that future V5r revisions may want to
flag:

1. **`lf_substrate_negation_regex`** votes only on the M9 entity-first
   LOF marker (subject perturbation); explicit verb-negation (e.g.,
   "X did NOT phosphorylate Y") is voted by `lf_substrate_negation_explicit`
   on the scope probe. Both LFs can fire on the same record. The
   relation_axis LF and scope LF target different probe questions —
   no double-counting.

2. **`lf_partner_substrate_gate`** fires on Activation/Inhibition/
   amount-axis claims when a binding CATALOG match is observed on the
   exact entity pair. The clean LF `LF_partner_dna_lexical` (V6b) covers
   the DNA-binding-element case; the substrate version covers the more
   general "binding observed but claim doesn't admit binding semantics"
   case.

3. **`lf_conditional_clause_substrate`** uses a regex extended slightly
   beyond V5r §3 phrasing (added bare adjective forms `\bwild-?type\b`
   and `\bmutant\b`) to capture the "X binds wild-type Y" surface form
   common in S-phase Intervention E test cases. The regex is still
   substrate-tuned per the V5r §3 tag.

4. **`lf_fragment_processed_form`** has two branches: (a) processed-form
   indicator within 30 chars of a claim entity alias, (b) standalone
   indicator (Aβ, -CTD, -NTD) that votes equivalent only in tuned mode.
   The second branch is the V7c contamination signal (the holdout
   addition of Aβ for APP fragment grounding); baseline drops it.

## Reproducing the audit

```python
# Identify which CATALOG entries are tuned (drop in baseline) vs core
import inspect, re
from indra_belief.scorers import relation_patterns as rp
TUNED_TOKENS = ("Diagnosis FN", "Diagnosis FP", "Q-phase regression",
                "R2:", "R5:", "N1:", "N2:", "M11 gap-fix",
                "U-phase", "U8")
src = inspect.getsource(rp).split("\n")
# walk source line-by-line to associate pattern_ids with comment blocks;
# emit `tuned` / `core` per pattern. (Source recipe in v6a_substrate_lfs_log.md
# implementation history.)
```

## Next-step gating

V6a unblocks:
- V6b (clean LFs) — no dependency, can run in parallel
- V6c (Λ matrix + Snorkel fit) — requires V6a + V6b complete
- V7c (substrate contamination delta) — requires V6a's
  `substrate_baseline_mode()` + V6c's per-LF firing matrix

Ship gate items pending: V8 environment + data gate (after V6d JSONL
emission lands).
