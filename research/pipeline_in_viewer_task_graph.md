# Pipeline-In-Viewer Task Hypergraph

Status: 2026-05-11. **U1 ✓ shipped** (heuristic coverage panel: per-probe substrate/llm/abstain/notrun rates + run-level summary). **U2.1–U2.3 ✓ shipped** (filesystem discovery + lazy shape preview + dashboard datasets section). **U2.4 pending** (ingest-status overlay). **U3.1 next** (architecture decision: subprocess vs sidecar).

Successor to `belief_instrument_task_graph.md` (closed — T-phase complete + brutalist P0 fixes shipped + dishonest-attribution refactor shipped). The T-phase made the **results** of scoring legible. This U-phase makes the **act** of scoring legible: ingesting, registering truth sets, kicking off a run, watching heuristics, comparing runs — all without leaving the SvelteKit surface.

## Frame

The current upstream workflow is:

```python
con = duckdb.connect(...); apply_schema(con); ingest_statements(con, stmts, ...)
run_id = score_corpus(con, stmts, client, ...); compute_validity(con, run_id)
export_beliefs(con, run_id, ...); model_card(con, run_id, ...)
```

Eight lines, Python REPL, then refresh the viewer. Every dataset-level action — *"score this corpus,"* *"register this benchmark as a truth_set,"* *"compare two runs,"* *"see what the substrate is doing"* — lives in a Python session the user maintains separately. The viewer is a passive consumer.

The inversion: **make the viewer the workflow.** A working biologist drops a JSON file in `data/corpora/`, opens the dashboard, sees it listed, previews its shape and cost, clicks `[ingest + score]`, watches the run advance in the run-feed, and lands on the focus card when it finishes. They never open Python. Heuristic coverage, P/R/F1 against benchmark labels, and run-vs-run deltas are sections of the same dashboard — answering *"what is the system doing"* with the same brutalist discipline already applied to *"what does this belief mean."*

Perceptual engineering is woven through every UI-bearing node. Brutalist review gates land after each phase.

---

## Inversions (what we tear down)

- **D1** — delete the implicit assumption that "running the pipeline" happens elsewhere. The viewer is no longer a read-only display of someone else's work.
- **D2** — delete the gap between *"a file exists on disk"* and *"the system knows about it"*. Filesystem state is surfaced.
- **D3** — delete the cost-projection panel's status as a calculator detached from any actual run. Cost becomes a preflight on a real action.
- **D4** — delete *"substrate fire rate is a number I have to compute in pandas"*. It becomes a panel.
- **D5** — delete the assumption that two runs can only be compared by squinting at the run-feed. Cross-run deltas become a section.

---

## Substrate edges (foundation capabilities — each unlocks ≥2 phase nodes)

### E0 — Pipeline-execution channel

The bridge between Node (SvelteKit server endpoints) and Python (the scorer + ingest + truth-set machinery). Two viable shapes:

- **Subprocess**: `node:child_process.spawn("python", ["-m", "indra_belief.worker", "<verb>", ...args])`. SSE for progress events.
- **Sidecar HTTP**: a tiny Python FastAPI on `localhost:5181` exposing `POST /ingest`, `POST /score`, `POST /register_truth_set`. SvelteKit proxies.

**Architectural decision deferred to U3.1.** Subprocess is simpler for a single-developer single-machine workflow; sidecar wins if anything else (notebooks, scripts, remote callers) needs the same API.

Either way: the *contract* is identical from the viewer's side — `POST /api/run/score`, `POST /api/datasets/<path>/ingest`, `POST /api/truth-sets`, with SSE for streaming.

### E1 — Dataset descriptor model

For each file in `data/corpora/*.json` and `data/benchmark/*.jsonl`, lazily compute `{path, kind, n_stmts, n_evidences, source_apis[], sample_biology[], file_mtime, ingest_status, last_run_id}`. `ingest_status` reflects how many of the file's `stmt_hash`es already exist in `corpus.duckdb`. Used by: dataset picker, cost preflight, ingest action.

### E2 — Run-lifecycle event stream

Streaming events from a worker to the viewer, narrated:
- `started(run_id, scorer_version, model_id, n_evidences_total, cost_estimate_usd)`
- `progress(n_done, n_total, cost_so_far_usd, latest_stmt_hash)`
- `done(run_id, mae, bias, n_stmts, n_evidences, cost_actual_usd)`
- `error(stage, message)`

Used by: live run-feed item, focus auto-rotate on completion, cost meter, ingest progress.

### E3 — Heuristic-coverage query layer

Pure DuckDB queries against persisted scorer_step state. Per probe (subject_role, object_role, relation_axis, scope):
- `substrate_rate = COUNT(is_substrate_answered=true) / total`
- `llm_rate = COUNT(source='llm') / total`
- `abstain_rate = COUNT(answer='abstain') / total`

Plus per-evidence aggregates: `all_substrate_rate = % of evidences where every probe was substrate-resolved`, `llm_calls_per_evidence_mean`, `cost_split_substrate_vs_llm`.

Used by: heuristic-coverage panel, run-narrative ("23% LLM-resolved vs prev run's 31%").

### E4 — Truth-set registrar

Python module exposing `register_truthset_from_jsonl(con, jsonl_path, truth_set_id, target_kind, field, value_extractor)`. Iterates records, extracts `(target_id, field, value)` triples, writes `truth_label` rows + a `truth_set` row. Idempotent on `(truth_set_id, target_kind, target_id, field)`. Called via the pipeline-execution channel (E0).

Used by: `[register as truth_set]` affordance, INDRA-benchmark wiring, custom-gold uploads.

---

## Phase U1 — heuristic coverage (read-only, no Python needed)

| node | deliverable | perceptual contract | depends |
|------|-------------|---------------------|---------|
| U1.1 | E3 DB queries in `viewer/src/lib/db.ts` → `getHeuristicCoverage(run_id)` | typed result: `{ per_probe: PerProbeCoverage[], all_substrate_rate, mean_llm_calls_per_evidence }` | — |
| U1.2 | `HeuristicCoverage.svelte` component | 4 per-probe pillbars (substrate / llm / abstain) + a one-sentence summary; same pillbar pattern as verdict shares | U1.1 |
| U1.3 | wire into validity section | section sibling: "what the system is doing" alongside "how is the system doing?" — *not* nested inside | U1.2 |

**Perceptual engineering check** (engage skill before writing U1.2): four pillbars stacked vertically with identical x-axis (0–100%). The reader's eye scans columns to compare probes. The one-sentence summary at the top ("LLM fired on 38% of evidences this run; scope was 92% substrate-resolved") carries the verdict.

**Brutalist gate G_U1** (after U1.3 ships): full-panel roast — does this section answer *"is the LLM doing real work or is the substrate doing everything?"* in one glance?

---

## Phase U2 — datasets surface (read-only, no Python needed)

| node | deliverable | perceptual contract | depends |
|------|-------------|---------------------|---------|
| U2.1 | filesystem discovery → `getDatasets()` | walks `data/corpora/*.json` and `data/benchmark/*.jsonl`; returns descriptors via E1 | E1 |
| U2.2 | lazy shape extraction | parses JSON/JSONL streaming; caches by mtime; returns `n_stmts`, `source_apis`, 3 sample sentences | E1 |
| U2.3 | `/datasets` page (or footer section on `/`) | one card per file; each card is a complete fact: name, shape preview, ingest status, last run, **no action buttons yet** — just legibility | U2.1, U2.2 |
| U2.4 | ingest-status overlay | `stmt_hash` membership check against `corpus.duckdb`; per-card badge: `[100% ingested]` / `[partial: 412/1847]` / `[not ingested]` | U2.3 |

**Perceptual engineering check**: each dataset card needs to answer four questions at a glance — *what is it · how big · have we touched it · what's the latest with it*. Avoid label-salad. Lead with the biology (3 sample sentences) so the reader sees what's inside, not just a count.

**Brutalist gate G_U2** (after U2.4): does a new visitor know what's actionable here, even without buttons?

---

## Phase U3 — pipeline-execution channel (architecture decision)

| node | deliverable | perceptual contract | depends |
|------|-------------|---------------------|---------|
| U3.1 | **decision: subprocess vs sidecar** | a 1-page brief in `research/` naming the choice + reversibility cost | — |
| U3.2 | minimum viable channel: `POST /api/datasets/<path>/ingest` | wraps `ingest_statements`; returns `{ ingested_stmt_count, duration_ms }` | E0, U3.1 |
| U3.3 | SSE event stream wired in: ingest emits `progress` events | each event renders a one-line update in a transient toast or in the dataset card | E2 |

**No perceptual engineering at U3.1** — this is an architecture call, not a UI choice. **Perceptual engineering for U3.3** (live progress display): single moving line, no spinners, no chartjunk. Bret-Victor-style: *"42 of 1,847 statements ingested · 12s elapsed · 8 min ETA"*.

**Brutalist gate G_U3** (after U3.3): challenge the architecture. *"You picked subprocess. Why? What's the reversibility cost if you need to add a second caller next month?"*

---

## Phase U4 — truth-set registrar

| node | deliverable | perceptual contract | depends |
|------|-------------|---------------------|---------|
| U4.1 | Python `register_truthset_from_jsonl(con, path, truth_set_id, ...)` | idempotent; writes `truth_label` rows; returns count | — |
| U4.2 | `POST /api/truth-sets` endpoint | calls U4.1 via E0; reports stats back | U3.2 |
| U4.3 | viewer affordance: `[register as truth_set]` button per benchmark dataset card | clicking opens a small modal with field-extractor configuration; submit triggers U4.2 | U2.3, U4.2 |
| U4.4 | validity panel auto-grows | after registration completes, `compute_validity` re-runs; the new P/R/F1 row appears in the next viewer poll | U4.3 |

**Perceptual engineering check**: the modal in U4.3 must keep the user inside the dashboard's voice — no jargon ("target_kind", "value_extractor"). Prose: *"these tags map to which scorer step? — aggregate / parse_evidence / grounding"*.

**Brutalist gate G_U4** (after U4.4): a holdout JSONL with 482 records. Register it. Confirm dashboard's validity section grows a row showing P/R/F1 in under 30 seconds total.

---

## Phase U5 — ingest + score from viewer

| node | deliverable | perceptual contract | depends |
|------|-------------|---------------------|---------|
| U5.1 | cost preflight modal | per-model projected cost based on actual evidence count from the descriptor; substrate-discount estimate; **opens before any spend** | U2.3 |
| U5.2 | `[ingest + score]` action on dataset cards | preflight → confirm → spawn worker | U3.2, U5.1 |
| U5.3 | live run-feed item with progress | run-feed gains a `[running ↻]` row that updates via SSE: `"scored 412 / 4,221 evidences · 13 min elapsed · ETA 41 min · cost so far $0.93"` | E2 |
| U5.4 | auto-focus rotation on completion | when SSE emits `done`, the focus card swaps to the highest-disagreement statement of the new run | U5.3 |
| U5.5 | cancel-in-flight | `[cancel]` on the running row; SIGTERM the worker; partial state preserved in DB | U5.3 |

**Perceptual engineering check** (before U5.1): the cost preflight has to communicate *"this is what you're about to spend"* — exact dollar figure, fan-out math visible, substrate-short-circuit caveat. Don't bury the number in fine print.

**Perceptual engineering check** (before U5.3): live progress must avoid the spinner trap. Sentence-style updates, no decorative motion. The `[running ↻]` row should feel like a typewriter, not a loading bar.

**Brutalist gate G_U5** (after U5.4): end-to-end test. Drop a fresh JSON in `data/corpora/`, refresh dashboard, click `[ingest + score]`, watch the run complete, land on a focus card — total user time < 1 minute (excluding actual scoring duration).

---

## Phase U6 — cross-run comparison *(deferred until U5 lands)*

| node | deliverable | perceptual contract | depends |
|------|-------------|---------------------|---------|
| U6.1 | `getRunComparison(run_a, run_b)` DB layer | reuses E2 narrative; adds per-stratum, per-truth_set deltas | — |
| U6.2 | `/runs/[run_id]/compare/[other]` page | two columns of validity sections side-by-side; deltas as the third column | U6.1 |
| U6.3 | dashboard run-feed `[compare to prev]` per row | one-click into U6.2 | U6.2 |

---

## Hyperedges (cross-cutting groupings)

- **H_dataset_action** = {U2, U4.3, U5.2} — every dataset-level affordance
- **H_python_call** = {U3, U4.2, U5.2, U5.5} — every place we cross the Node↔Python boundary
- **H_streaming** = {U3.3, U5.3, U5.4, U5.5} — every place we show live worker state
- **H_validity_growth** = {U1.3, U4.4} — every place the validity section grows new rows automatically
- **H_persistence_honesty** = {U1, U2.4} — every place that surfaces what's already on disk vs what isn't

A change to E0 (subprocess vs sidecar) propagates through H_python_call. A change to E2 event shape propagates through H_streaming. The hyperedges name the contracts that must stay consistent.

---

## Perceptual contracts (apply to every node, not relegated to T3)

Same eight from `belief_instrument_task_graph.md` carry over — P1 hierarchy of attention, P2 glanceability, P3 direct manipulation, P4 data:ink ratio, P5 delta primacy, P6 honest empties, P7 brutalist palette, P8 type discipline — plus three U-phase-specific:

- **P9 — Cost honesty.** Any UI element that triggers spend states the projected cost *before* the click, in the visual frame of the click. Buried fine print and post-hoc "actually it cost $X" are forbidden.
- **P10 — Live state ≥ static state.** A run in progress is more interesting than a run that finished an hour ago. The dashboard should naturally elevate in-flight runs without the user having to look for them.
- **P11 — Reversibility legibility.** When an action is destructive or hard to undo (registering a truth_set wrong, kicking off a multi-hour scoring run), the irreversibility is named in the same visual element as the action. Cancel affordances are real, not decorative.

---

## Review gates (with brutalist roast where indicated)

| gate | after node | gate kind | what it tests |
|------|------------|-----------|---------------|
| **G_U1** | U1.3 | brutalist roast (`design` domain, URL: dashboard) | does the heuristic-coverage section answer *"is the LLM doing real work?"* in one glance? |
| **G_U2** | U2.4 | perceptual-engineering eye-path audit + brutalist roast | does a new visitor know what's actionable on each dataset card without action buttons present? |
| **G_U3** | U3.1 | architecture review (no UI; `architecture` domain roast) | does the subprocess-vs-sidecar choice survive a 2nd-caller scenario? |
| **G_U3b** | U3.3 | brutalist on the progress display | does the live state avoid the spinner trap? |
| **G_U4** | U4.4 | end-to-end with a real JSONL: register holdout_d4 as a truth_set, confirm P/R/F1 row grows | the time from `[click register]` to `P/R/F1 row visible` |
| **G_U5** | U5.4 | brutalist + end-to-end | drop JSON → ingest + score → focus card lands. Under 1 minute user-time excluding scoring duration. |
| **G_U5b** | U5.5 | cancel-in-flight test | start a run, cancel mid-way, verify partial state and that re-running picks up cleanly |

When the brutalist roast names a blocker, it goes back on the hypergraph as a tracked node before the next phase starts.

---

## Acceptance criteria (run-level)

- **A1** — A new visitor opening `/` can answer *"how much LLM did this run use?"* in one glance. (G_U1)
- **A2** — A user can ingest+score a dataset without leaving the viewer; total user-time < 1 min excluding scoring duration. (G_U5)
- **A3** — A user can register a benchmark JSONL as a truth_set without leaving the viewer; P/R/F1 row visible within 30s. (G_U4)
- **A4** — Live runs are visible in the run feed with sentence-style progress updates. (G_U3b, G_U5)
- **A5** — P/R/F1 against any registered truth_set appears automatically in the validity panel; no second action required. (U4.4)
- **A6** — All cost-triggering actions show the projected cost before the click, inside the same visual frame. (P9, U5.1)

---

## Execution order

1. **U1** first (heuristic coverage) — no Python, no architecture choice; ships immediately and answers a long-standing question.
2. **U2.1 + U2.2 + U2.3** (datasets page, read-only) — still no Python; turns filesystem state into a perceptual surface.
3. **U3.1** (architecture decision) — small written brief; gate it with `architecture` brutalist roast.
4. **U3.2 + U3.3** (minimum-viable channel: ingest endpoint + SSE) — first Python-crossing wire-up.
5. **U4.1 → U4.4** (truth-set registrar end-to-end) — proves the channel + grows validity panel.
6. **U5.1 → U5.5** (full ingest+score from viewer) — the headline UX. Most engineering, most reversibility considerations.
7. **U6** when there are multiple real runs to compare.

Engage the **perceptual-engineering skill before** U1.2, U2.3, U3.3, U5.1, U5.3.
Engage the **brutalist roast after** U1.3, U2.4, U3.1, U3.3, U4.4, U5.4, U5.5.

Pause at each gate. When a roast names a P0 blocker, it becomes a tracked node; phase doesn't advance until it lands.
