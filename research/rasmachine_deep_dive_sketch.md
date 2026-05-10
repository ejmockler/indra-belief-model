# Single-Evidence Deep-Dive View — Sketch (Phase 1.4)

**Author:** ux-design-architect agent · **Date:** 2026-05-09
**Drives:** schema 2.1 (`output_json` shapes per step kind), 5c grammars, 5d motion choreography
**Register:** Bret Victor + Tufte in controlled chaos (D3 + D10)
**G1 review:** 2026-05-09 — multi-critic brutalist (claude/codex/gemini); resolutions below override the original sketch where they conflict.

## G1 brutalist resolutions (override conflicting text below)

1. **Disagreement is a mark, not a behavior.** Auto-expand-on-disagreement is killed. Disagreeing truth labels render as `!` in left margin + colored dot on the right rail. User does 1-click expand if they want detail. (Resolves contradiction between "calm summary" and "auto-expand on disagreement.")
2. **Reasons → matrix navigation is silent href-mutation, not clipboard-toast.** Clicking a reason chip in adjudicate silently sets the matrix back-link's href to carry the filter. The crumb shows `← matrix · filtered: <reason>` in muted weight when active. No toast, no clipboard, no surprise. One click navigates to the filtered matrix.
3. **Truth panel collapses absent rows.** Show present rows fully + a single roll-up line: `4 absent: gold_pool_v15, u2_per_probe_gold, source_db_curation, indra_supports_graph`. The 7-dot rail (peripheral on scroll) carries the spatial-expectation load. One source of truth for absence.
4. **Single-channel substrate-vs-LLM encoding.** Drop `◆`/`▲` taxonomy. Use one of: (a) same shape `●` filled vs hollow (substrate hollow, LLM filled — Tufte-pure), or (b) `S`/`L` text in 1ch left gutter. Position carries the load; the trace stays one shape. Drop the gradient edge-bar and the "atmosphere" rhetoric.
5. **Telemetry split.** Latency stays on collapsed LLM cards (perceptual; signals model strain). Tokens hides behind `[exp]` (operational; infrastructure leak). Render latency as a proportional bar `▎▎▎` rather than a number — Tufte-pure, glanceable.
6. **D10 operational constraints (made teeth-bearing):** no element changes position without explicit user action; no animation > 200ms; no interruption surfaces (toasts, modals) on canonical path; no data loads after first paint without a button. Audit spec against this every phase.
7. **Span-coordinate contract gap (Codex):** `ProbeResponse.span` is currently "optional substring," not stable char-offsets. Bidirectional span ↔ probe reactivity needs `(start, end)` offsets. Schema 2.1 must require probes report stable offset spans; backfill in scorer when emitting.
8. **G5a glance-test acceptance:** 3 reviewers, 5 traces of varying LLM-density, count # of LLM steps within ±1 in <1.5s peripheral glance. Pass = ≥80% accuracy. Fail = revert the substrate-vs-LLM encoding.
9. **Theatrical hedges removed:** Q3 (auto-expand) settled by #1; Q4 (`because:` inline) was already specced; Q5 (clipboard) settled by #2. Three open questions remain (verdict-first, truth-panel side, glyph color/encoding).

The bones (sticky evidence + spine trace + truth roll-up + rail-of-dots) survive. The poetry layer is cut.

---


---

## Felt situation

A scorer sits with one (statement, evidence) pair to **judge whether the machine got this right**. Their body is leaning in, not scanning. They are dwelling. The question is rarely "what does this say" — it is "where in this 9-step trace did belief get committed, on what evidence, and does any of seven truth_sets disagree?" The view is a **document**, not a dashboard. It rewards close reading; it does not reward saccades.

---

## Region layout

```
+------------------------------------------------------------------------------------------+
|  [<] matrix     stmt#3a4f1c · ev#5e2a9b              v15.3 · gpt-4 · 2026-05-09        |  <- crumb (sticky, 32px)
+------------------------------------------------------------------------------------------+
|                                                                                          |
|  Phosphorylation(MAP2K1, MAPK1)            verdict: correct/high  → 0.95  ░░░░░░░░░░    |  <- statement glyph + score
|  belief 0.82  ◇ ours 0.95  Δ +0.13         supports 4 ▸  supported_by 1 ▸               |     (sticky, 96px)
|                                                                                          |
+----------------------------------------+-------------------------------------------------+
|                                        |                                                 |
|  EVIDENCE                              |  TRUTH PANEL                          [absent]  |
|  source_api: reach  ·  pmid:18599499   |  ─────────────────────────────────             |
|                                        |  indra_published_belief    0.82      ◇         |
|  "MEK1 was found to phosphorylate      |  indra_epistemics                              |
|   ERK at threonine 202 in serum-       |    direct      true       agree                |
|   stimulated NIH3T3 cells."            |    negated     false      agree                |
|   ─────       ─────────────  ─── ───   |    curated     false      —                   |
|   subj-span   pred-span      site      |  indra_supports_graph                          |
|                                        |    supports[4]  hover ▸                        |
|  [hover any underline → highlights     |  indra_grounding (per agent)                   |
|   the probe(s) that consumed it]       |    MAP2K1 → HGNC:6840    agree                 |
|                                        |    MAPK1  → HGNC:6871    agree                 |
|  ─ epistemics ─────────────────────    |  source_db_curation         absent             |
|  direct: ✓   negated: ✗   curated: ✗   |  gold_pool_v15              absent             |
|                                        |  u2_per_probe_gold          absent             |
|  (sticky on scroll)                    |                                                 |
|                                        |  (collapses to rail when scrolled past)        |
+----------------------------------------+-------------------------------------------------+
|                                                                                          |
|  ┃ 1  parse_claim       Phosphorylation · subj=MAP2K1 · obj=MAPK1 · site=T202   det     |  <- TRACE RAIL
|  ┃                                                                                       |     (the spine; scrolls)
|  ┃ 2  build_context     1 chain · 3 hedges · 0 alias gaps                       det     |
|  ┃                                                                                       |
|  ┃ 3  substrate_route   subject_role:sub  object_role:sub  axis:LLM  scope:sub  3/4↓   |
|  ┃                                                                                       |
|  ┃ 4  subject_role_probe ◆ substrate · enzyme · 1.00 · "phosphorylate" span    [exp]   |
|  ┃ 5  object_role_probe  ◆ substrate · substrate · 1.00 · "ERK" span           [exp]   |
|  ┃ 6  relation_axis_probe ▲ LLM · phosphorylation · 0.92 · 412ms · 84tok       [exp]   |
|  ┃ 7  scope_probe        ◆ substrate · in-vitro · 0.88                         [exp]   |
|  ┃                                                                                       |
|  ┃ 8  grounding          MAP2K1✓ HGNC:6840 · MAPK1✓ HGNC:6871 · 0 flagged      det     |
|  ┃                                                                                       |
|  ┃ 9  adjudicate         ▣ correct / high · reasons[match, axis_match]          —       |
|  ┃                       lookup: correct·high → 0.95                                     |
|                                                                                          |
+------------------------------------------------------------------------------------------+
|  [ supports-graph snippet — lazy, opens to 200px panel below adjudicate when expanded ]  |
+------------------------------------------------------------------------------------------+
```

**Ratios:** crumb 32 / header 96 / left-panel 38ch / right-panel ≈ 32ch / trace fluid. Center column (the 9-step trace) is the strong center.

---

## Region rationales

**Crumb (sticky, top).** Provenance, not navigation chrome. `scorer_version · model_id · started_at` are the load-bearing tokens — without them the view is unreproducible. Tufte: chrome that earns its space because reproducibility is the spine.

**Header (sticky, just below crumb).** The verdict-and-score lives here, not at the bottom. **The conclusion is the first thing seen, the trace below justifies it.** This inverts dashboard convention but matches how a reviewer reads: "you said 0.95 — convince me." The score-bar is the strong center of the header. `belief 0.82 ◇ ours 0.95 Δ +0.13` is a single horizontal mark, not three numbers.

**Left column — Evidence (sticky on scroll within the trace region).** The text is the ground. It sticks because every probe below references spans inside it. Underlined spans are not decoration — they are the **shared substrate of probe consumption**. Hovering anywhere in the evidence text triggers reactivity downstream. Bret Victor: locus of reactive control lives next to what is being controlled.

**Right column — Truth panel (sticky → collapses to vertical rail on scroll).** Seven truth_sets, each as one row. **Present labels render as marks; absent labels render as the literal word `absent`** in muted weight. No row is hidden. The panel collapses to a 24px-wide rail of seven dots when the trace scrolls — peripheral awareness of "are we still in agreement" without reclaiming attention.

**Center — 9-step trace rail.** The agentic story, top-to-bottom, indented under a single vertical bar (`┃`). Each step is one row in summary mode. The bar visually unifies them as one continuous process; the step-numbers (1–9) anchor spatial memory.

**Below the rail — supports-graph snippet (lazy).** Not visible by default; opens inline when the user clicks `supports 4 ▸` or `supported_by 1 ▸` in the header. A 200px-tall panel showing the local 1-hop subgraph as text rows (not a node-link diagram at this scale — that lives in the matrix view). Each row: `→ stmt#hash · type · agents · belief`, hoverable for popover preview.

---

## The 9-step trace as a vertical story

**Default expansion:**
- Steps 1, 2, 8, 9 (deterministic foundation + grounding + adjudicate): collapsed to one summary line. They almost never carry the surprise.
- Step 3 (route): collapsed but visually heavy — the `3/4↓` glyph (3-of-4 substrate-resolved) tells the reader instantly "this evidence cost ~one LLM call."
- Steps 4–7 (probes): collapsed by default; one click expands. **Truth-disagreement on any step auto-expands that step.** (Open Q3.)
- Step 9 (adjudicate): always shows reasons inline — even collapsed — because reasons are the load-bearing payload.

**Substrate vs LLM visual distinction (the load-bearing affordance):**

| | substrate-answered | LLM-escalated |
|---|---|---|
| glyph | `◆` (filled diamond, no fill color) | `▲` (filled triangle, accent color) |
| label | `substrate` | `LLM` |
| telemetry strip | absent | `· 412ms · 84tok` always shown |
| left edge | thin solid bar | thicker bar with subtle gradient |
| rationale source | regex/Gilda pattern name | model_id + finish_reason |

Atmosphere: substrate-answered cards feel **quiet and inevitable**; LLM cards feel **alive and contingent**. A reader scrolling the rail can answer "where did the model spend?" peripherally — diamonds blur out, triangles do not.

**Path-rationale surface.** Each step has a one-line `because:` slot rendered in muted weight on its own line below the summary:
- `parse_claim` → `because: type=Phosphorylation, residue=T202 from regex`
- `substrate_route` → `because: subject_role had clear enzyme cue at "phosphorylate"`
- `relation_axis_probe` → `because: substrate axis-classifier ambiguous; escalated`
- `adjudicate` → `because: parse + axis + role all match; no flags`

When a step has telemetry beyond the summary (LLM call_log entries, prompt template hash), it hides behind `[exp]`.

---

## Concrete grammar walkthrough — `subject_role_probe`

**Substrate-answered (collapsed):**
```
┃ 4  subject_role_probe   ◆ substrate · enzyme · 1.00 · "phosphorylate" span   [exp]
```
- glyph `◆` = substrate; answer chip `enzyme`; confidence `1.00` as 3-char bar `███`
- the span `"phosphorylate"` is monospace; **hover → highlights same span in evidence text** (D10)
- no telemetry strip
- `[exp]` reveals: substrate pattern_id, candidate spans considered, `is_substrate_answered=true`

**Substrate-answered (expanded):**
```
┃ 4  subject_role_probe   ◆ substrate · enzyme · 1.00
┃    span:           "phosphorylate"   [hover → ev text]
┃    pattern_id:     ENZ_VERB_PRESENT
┃    candidates:     phosphorylate(1.00)  found(0.12 rejected)
┃    because:        present-tense enzyme verb in subject clause
┃    truth:          gold_pool_v15 → enzyme  agree         (only when gold present)
```

**LLM-answered (collapsed):**
```
┃ 4  subject_role_probe   ▲ LLM · enzyme · 0.78 · 391ms · 72tok   [exp]
```

**LLM-answered (expanded):**
```
┃ 4  subject_role_probe   ▲ LLM · enzyme · 0.78
┃    span:           "MEK1 was found to phosphorylate"   [hover → ev text]
┃    rationale:      "MEK1 acts as the kinase here; its action verb governs the predicate."
┃    model_id:       gpt-4-2024-08-06
┃    prompt_hash:    7a3c9e
┃    latency:        391ms        tokens: 28→72        finish: stop
┃    because:        substrate_route deferred — no clear enzyme-cue regex hit
┃    truth:          gold_pool_v15 → substrate  DISAGREE   (when present, marked)
```

The disagreement marker is a single character (`!`) in the truth row's left margin — Tufte minimum-ink. It also surfaces upward as a colored dot on the truth-panel rail.

---

## Truth-when-present rendering

**One mark per label, one row per truth_set.** Marks:
- `agree` → no mark, just the value (the absence of a mark IS the signal — Tufte)
- `disagree` → single character `!` in left margin, color-shifted
- `absent` → the word `absent` in muted weight

A truth label is never a separate annotation layer floating over a step — it is **inlined into the step card it judges**. The right-side panel is a **roll-up** of the same inlined marks, giving peripheral awareness.

**Reactivity:** hovering a row in the truth panel **highlights the scorer step(s) the label judges**. E.g., hovering `gold_pool_v15 → enzyme` flashes a thin border on `subject_role_probe`. Hovering `indra_supports_graph` flashes the supports-graph snippet button.

---

## Three reactive interactions (D10)

1. **Span ↔ probe (bidirectional).** Hover a span underline in evidence → probe(s) consuming it gain soft outline + their span chip pulses. Inverse: hover a span chip in any probe card → the span underlines in evidence. Evidence and probes share one span-coordinate space.

2. **Reasons code → matrix URL state.** Click a reason chip in `adjudicate` → does **not** open anything in this view; copies a URL to clipboard with 1.2s "matrix filter staged" toast. The crumb's `[<] matrix` back-link now carries the filter. Bret Victor: reactive control that crosses views.

3. **Supports-edge hover → preview popover.** Hovering a row in the supports-graph snippet opens a 280px popover with the supported statement's mini-render: type glyph, agents, our-score-vs-INDRA-belief delta, first 80 chars of one evidence. Click pins; second click replaces; ESC dismisses.

(Bonus, deferrable:) hovering an LLM probe's telemetry strip surfaces a sparkline of token-by-token latency from `call_log` — the only place raw LLM telemetry breaks the surface.

---

## Absence-as-absence

When `truth_label` rows are missing for this evidence on a given truth_set, the row reads literally:

```
gold_pool_v15              absent
u2_per_probe_gold          absent
```

No grayed-out checkbox, no `—`, no `n/a`. The word `absent` is the grammar. Muted weight, same row height as a present row — the absence occupies the same space presence would, so spatial expectation is honored. **Tufte: never an annotation when a mark of difference suffices** — the difference here is typographic weight, nothing more.

A view with **all** truth_sets absent (a pure no-truth corpus evidence) still renders the panel — just seven `absent` rows. The panel is never hidden. This matches G4's honest-failure clause.

---

## Position on the chaos ↔ order continuum

The matrix view (8.7k rows) is where chaos lives — controlled chaos at scale. **The deep-dive sits at the order pole.** It is one document, designed for dwelling, with strong-center spatial memory: the trace rail is in the same place every time. Reactivity is confined to **moves within this document** (span ↔ probe, truth-row ↔ step-card) plus one cross-view gesture (reasons → matrix URL).

The discipline: this view is allowed to be dense (7-truth-set panel, 9-step trace, telemetry on demand) but **not allowed to surprise**. Layout is stable; expansions are predictable; nothing animates without an action. The atmosphere is **a quiet study**, not an instrument panel. If the matrix view is a city, the deep-dive is a single page in a bound book.

---

## Deferred to later iterations

- Diff mode between two scorer_versions (compare v15 vs v16 trace side-by-side) — post-G5c.
- Inline evidence editing / re-score — out of scope; this is a viewer.
- Token-by-token LLM latency sparkline — useful for cost analysis, not judgment.
- Supports-graph as node-link diagram — text rows sufficient at 1-hop.
- Annotation / commenting — phase-7 concern.
- Keyboard navigation (j/k between steps, e to expand, t to focus truth panel) — should land with G5c.
- Mobile / narrow-viewport adaptation — defer; design is desktop-first.

---

## Open design questions (to surface to user before G5a)

1. **Header verdict-first vs. trace-first.** I put verdict + score in the sticky header. Alternative: verdict appears at the bottom as the consummation. Review (judge the conclusion) vs replay (follow the agent)?

2. **Truth panel right vs. left.** Right because evidence text reads left-to-right-then-jump-right (F-pattern). Argument for left: truth deserves dominant-eye placement.

3. **Auto-expand on disagreement.** Proposed: any probe with disagreeing truth label auto-expands. Failure modes: noisy when many disagree on same step; user loses calm summary. Should disagreement be a *mark*, not a *behavior*?

4. **`because:` line — load-bearing or decorative?** 1.2 says "rationale fields are informational." Inline default vs hide-until-expand?

5. **Reasons → matrix URL gesture.** Click-to-copy-with-toast vs contextual mini-link `[filter matrix]` opening new tab. Locus-respecting vs discoverable?

6. **Substrate-answered glyph color.** Monochrome diamond vs quiet semantic color. Quietness vs positive-spine-affordance.
