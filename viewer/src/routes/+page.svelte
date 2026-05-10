<script lang="ts">
	import type { PageData } from './$types';
	import { invalidateAll } from '$app/navigation';
	import { onMount, onDestroy } from 'svelte';
	import BeliefPrimitive from '$lib/components/BeliefPrimitive.svelte';
	import Validity from '$lib/components/Validity.svelte';

	let { data }: { data: PageData } = $props();
	const o = $derived(data.overview);
	const focus = $derived(data.focus);
	const findings = $derived(data.findings);
	const residuals = $derived(data.residuals);
	const narratives = $derived(data.narratives);

	function statusGlyph(status: string): string {
		if (status === 'succeeded') return '✓';
		if (status === 'running') return '↻';
		if (status === 'failed') return '✗';
		return '?';
	}

	function fmtCostSummary(cost: number): string {
		return cost < 0.01 ? '<$0.01' : cost < 1 ? '$' + cost.toFixed(2) : '$' + cost.toFixed(0);
	}

	const FINDING_LANES: Array<{ key: keyof NonNullable<typeof findings>; title: string; emptyMsg: string }> = [
		{ key: 'biggest_disagreement', title: 'biggest |Δ| vs INDRA', emptyMsg: 'no scored statements' },
		{ key: 'probe_split', title: 'probe split (axes disagree)', emptyMsg: 'no multi-probe statements' },
		{ key: 'low_confidence_high_stakes', title: 'low-confidence · n_ev ≥ 3', emptyMsg: 'no mid-range statements with multi-evidence' },
		{ key: 'verdict_regression', title: 'verdict regressions vs prev run', emptyMsg: 'no prev run or no regressions' },
		{ key: 'verdict_recovery', title: 'verdict recoveries vs prev run', emptyMsg: 'no prev run or no recoveries' }
	];

	function fmt(n: number): string {
		return n.toLocaleString('en-US');
	}

	// Cost projection for a "score the loaded corpus" run. Source of truth
	// for prices: src/indra_belief/corpus/cost.py::MODEL_PRICES_PER_M_TOKENS.
	// Mirrored client-side for the panel; keep in sync if rates change.
	const COST_MODELS: Array<[string, string, number, number]> = [
		// [display_name, model_id, in_per_m_usd, out_per_m_usd]
		['Flash', 'gemini-2.5-flash', 0.075, 0.30],
		['Haiku', 'claude-haiku-4-5', 0.80, 4.00],
		['Sonnet', 'claude-sonnet-4-6', 3.00, 15.00],
		['Opus', 'claude-opus-4-7', 15.00, 75.00]
	];
	const TOKENS_PER_LLM_CALL_IN = 330;
	const TOKENS_PER_LLM_CALL_OUT = 70;
	const LLM_CALLS_PER_EVIDENCE = 5;

	function projectCost(ev_count: number, in_per_m: number, out_per_m: number): number {
		const calls = ev_count * LLM_CALLS_PER_EVIDENCE;
		return calls * (TOKENS_PER_LLM_CALL_IN * in_per_m + TOKENS_PER_LLM_CALL_OUT * out_per_m) / 1_000_000;
	}

	// Phase 5d minimum-viable live-tail: poll the load function on a
	// fixed cadence (3s) so the dashboard refreshes counts / latest run /
	// validity without manual reload. Lighter than SSE; richer than a
	// static page. Bret Victor: the page is a live document, not a snapshot.
	//
	// Iter-3 brutalist BLOCKER #1: the live dot used to pulse unconditionally
	// (decoration, no signal). It now flashes only when data changed since
	// last poll — flash duration <200ms per D10 motion budget.
	let pollHandle: ReturnType<typeof setInterval> | null = null;
	let freshTimer: ReturnType<typeof setTimeout> | null = null;
	let lastSignature = $state<string>('');
	let dotFresh = $state(false);

	const currentSignature = $derived(
		`${o.statementCount}|${o.evidenceCount}|${o.scorerRuns.length}|${o.scorerRuns[0]?.run_id ?? ''}|${o.latestValidity?.run_id ?? ''}`
	);

	// Flash the dot for 800ms when data signature changes, then revert.
	// Previously this used `Date.now() - dataChangedAt < 800` inside $derived,
	// but Date.now() isn't reactive — once the inequality went true, nothing
	// triggered re-evaluation, so the dot stayed "fresh" forever until the
	// next change. Explicit setTimeout makes the 800ms revert real.
	$effect(() => {
		if (lastSignature && lastSignature !== currentSignature) {
			dotFresh = true;
			if (freshTimer) clearTimeout(freshTimer);
			freshTimer = setTimeout(() => { dotFresh = false; }, 800);
		}
		lastSignature = currentSignature;
	});

	// Empty-state pipeline snippet — held in a const so f-string `${...}`
	// interpolation doesn't conflict with Svelte's `{...}` template syntax.
	const PIPELINE_SNIPPET = `import duckdb
from indra.statements import stmts_from_json_file
from indra_belief.model_client import ModelClient
from indra_belief.corpus import (
    apply_schema, ingest_statements,
    score_corpus, export_beliefs, model_card,
)

# 1. Ingest INDRA Statements (lossless)
con = duckdb.connect("data/corpus.duckdb")
apply_schema(con)
stmts = stmts_from_json_file("data/corpora/latest_statements_rasmachine.json")
ingest_statements(con, stmts, source_dump_id="rasmachine_emmaa")

# 2. Score through the four-probe pipeline (auto-runs compute_validity)
client = ModelClient("claude-sonnet-4-6")  # or any ModelClient backend
run_id = score_corpus(con, stmts, client=client,
                      scorer_version="prod-v1", decompose=True)

# 3. Export INDRA-native JSON with our beliefs + model card
export_beliefs(con, run_id, f"data/exports/{run_id}_indra.json")
model_card(con, run_id, out_path=f"data/exports/{run_id}_card.json")
con.close()`;

	onMount(() => {
		lastSignature = currentSignature;
		pollHandle = setInterval(() => {
			invalidateAll();
		}, 3000);
	});

	onDestroy(() => {
		if (pollHandle) clearInterval(pollHandle);
		if (freshTimer) clearTimeout(freshTimer);
	});
</script>

<svelte:head>
	<title>INDRA Belief — Corpus</title>
</svelte:head>

<header>
	<div class="crumb">corpus<span class="sep"> / </span><strong>overview</strong></div>
	<div class="meta">
		<span class="live-indicator" title="dashboard polls every 3s; dot flashes when data changes">
			<span class="live-dot" class:live-dot-flash={dotFresh}></span>
			{dotFresh ? 'fresh' : 'live'}
		</span>
		<a href="/statements" class="nav-link">browse statements →</a>
		<span class="db-path" title={o.dbPath}>{o.dbPath.replace(/.*\//, '')}</span>
	</div>
</header>

<main id="main">
	<h1 class="visually-hidden">Corpus dashboard</h1>
	{#if !o.dbExists}
		<section class="empty">
			<h1>no corpus loaded</h1>
			<p class="lede">
				The viewer is wired to <code>{o.dbPath}</code>, but no DuckDB file exists there yet.
			</p>
			<p>
				Run the full pipeline from a Python REPL:
			</p>
			<pre>{PIPELINE_SNIPPET}</pre>
			<p class="hint">
				Or set <code>VIEWER_DUCKDB_PATH</code> to point at an existing <code>.duckdb</code> file.
			</p>
		</section>
	{:else}
		<section class="focus">
			{#if focus}
				<BeliefPrimitive
					stmt={focus.stmt}
					our_score={focus.our_score}
					indra_score={focus.indra_score}
					probes={focus.probes}
					evidences={focus.evidences}
					why_this_one={focus.why_this_one}
					mode="full"
				/>
				<p class="focus-deeplink">
					<a href={`/statements/${focus.stmt.stmt_hash}`}>open deep-dive →</a>
				</p>
			{:else}
				<div class="focus-empty">
					<p class="hint">no belief in focus yet · run <code>score_corpus</code> to produce one</p>
				</div>
			{/if}
		</section>

		{#if findings}
			{@const focusHash = focus?.stmt.stmt_hash ?? null}
			<section class="findings">
				<h2 class="findings-h">look here</h2>
				{#each FINDING_LANES as lane}
					{@const allRows = (findings[lane.key] as import('$lib/db').FindingRow[]) ?? []}
					{@const rows = allRows.filter((r) => r.stmt_hash !== focusHash)}
					{#if rows.length > 0}
						<div class="lane">
							<h3 class="lane-h">{lane.title} <span class="lane-n">({rows.length})</span></h3>
							<div class="lane-body">
								{#each rows as r}
									<BeliefPrimitive
										mode="compact"
										stmt={{ stmt_hash: r.stmt_hash, indra_type: r.indra_type, agents: r.agents }}
										our_score={r.our_score}
										indra_score={r.indra_score}
										evidences={Array(r.n_evidences).fill({ evidence_hash: '', source_api: null, text: null })}
										why_this_one={r.why_text}
										href={`/?focus=${r.stmt_hash}`}
									/>
								{/each}
							</div>
						</div>
					{/if}
				{/each}
			</section>
		{/if}

		{#if o.latestValidity}
			<Validity v={o.latestValidity} residuals={residuals} />
		{/if}

		<section class="grid">
			<article class="run-feed-article">
				<h2>runs</h2>
				{#if o.scorerRuns.length === 0}
					<p class="hint">no runs yet · invoke <code>score_corpus</code></p>
				{:else}
					<ul class="run-feed">
						{#each o.scorerRuns as r}
							{@const n = narratives[r.run_id]}
							<li class="run-row" class:run-row-failed={r.status === 'failed'} class:run-row-running={r.status === 'running'}>
								<span class="run-glyph" class:status-failed={r.status === 'failed'} class:status-running={r.status === 'running'} title={r.status}>{statusGlyph(r.status)}</span>
								<code class="run-hash" title={r.run_id}>{r.run_id.slice(0, 8)}</code>
								<span class="run-version" title={r.scorer_version}>{r.scorer_version.length > 22 ? r.scorer_version.slice(0, 21) + '…' : r.scorer_version}</span>
								<span class="run-when" title={r.started_at}>{r.started_at.replace(/\.\d+$/, '').replace(/^(\d{4}-\d{2}-\d{2}) /, '$1·')}</span>
								<span class="run-narrative">
									{#if r.status !== 'succeeded'}
										<span class="muted">{r.status}</span>
									{:else}
										<span class="run-n">{r.n_stmts ?? '—'} stmts</span>
										{#if r.cost_estimate_usd != null}
											<span class="muted">·</span>
											<span class="run-cost">{r.cost_estimate_usd < 0.01 ? '<$0.01' : '$' + r.cost_estimate_usd.toFixed(2)}</span>
										{/if}
										{#if n?.summary_sentence}
											<span class="muted">·</span>
											<span class="run-summary">{n.summary_sentence}</span>
										{:else if r.mae != null}
											<span class="muted">·</span>
											<span class="run-summary">MAE {r.mae.toFixed(3)}{#if r.bias != null} · bias {r.bias >= 0 ? '+' : '−'}{Math.abs(r.bias).toFixed(3)}{/if}</span>
										{/if}
									{/if}
								</span>
								<span class="run-exports">
									{#if r.hasIndraExport}<a href={`/export/${r.run_id}/indra`} class="dl-link" title="INDRA beliefs JSON">↓ beliefs</a>{:else}<span class="dl-link dl-missing" title="Run `export_beliefs(con, run_id, ...)` first">↓ beliefs</span>{/if}
									{#if r.hasCardExport}<a href={`/export/${r.run_id}/card`} class="dl-link" title="model card JSON">↓ card</a>{:else}<span class="dl-link dl-missing" title="Run `model_card(con, run_id, out_path=...)` first">↓ card</span>{/if}
								</span>
							</li>
						{/each}
					</ul>
				{/if}
			</article>

			<details class="cost-expander">
				<summary>next-run cost projection {#if o.evidenceCount > 0}<span class="muted">— Flash ≈ {fmtCostSummary(projectCost(o.evidenceCount, COST_MODELS[0][2], COST_MODELS[0][3]))} · Sonnet ≈ {fmtCostSummary(projectCost(o.evidenceCount, COST_MODELS[2][2], COST_MODELS[2][3]))}</span>{/if}</summary>
				{#if o.evidenceCount === 0}
					<p class="hint">no evidences loaded · run <code>ingest_statements</code> first</p>
				{:else}
					<table>
						<thead>
							<tr><th>model</th><th class="num">cost</th><th class="num">calls</th><th class="num">tokens</th></tr>
						</thead>
						<tbody>
							{#each COST_MODELS as [name, id, in_p, out_p]}
								{@const cost = projectCost(o.evidenceCount, in_p, out_p)}
								{@const calls = o.evidenceCount * LLM_CALLS_PER_EVIDENCE}
								{@const tokens = calls * (TOKENS_PER_LLM_CALL_IN + TOKENS_PER_LLM_CALL_OUT)}
								<tr>
									<td>{name} <span class="muted">{id}</span></td>
									<td class="num">${cost < 1 ? cost.toFixed(2) : cost.toFixed(0)}</td>
									<td class="num">{fmt(calls)}</td>
									<td class="num">{fmt(tokens)}</td>
								</tr>
							{/each}
						</tbody>
					</table>
					<p class="hint">
						Assumes ~{LLM_CALLS_PER_EVIDENCE} LLM calls/evidence (substrate short-circuits ≤2%) · {TOKENS_PER_LLM_CALL_IN}+{TOKENS_PER_LLM_CALL_OUT} tokens/call.
						Pass <code>cost_threshold_usd</code> to <code>score_corpus</code> to gate before spend.
					</p>
				{/if}
			</details>

			<!-- validity moved out of grid; rendered by Validity component above -->
		</section>

		<footer class="data-footer">
			<div class="df-line">
				<span class="df-count">{fmt(o.statementCount)} stmts</span>
				<span class="df-sep">·</span>
				<span class="df-count">{fmt(o.evidenceCount)} ev</span>
				<span class="df-sep">·</span>
				<span class="df-count">{fmt(o.agentCount)} agents</span>
				<span class="df-sep">·</span>
				<span class="df-count">{fmt(o.supportsEdgeCount)} supports</span>
				<span class="df-sep">·</span>
				<span class="df-count">{fmt(o.truthLabelCount)} truth labels</span>
			</div>
			{#if o.truthSets.length > 0}
				<div class="df-line df-line-sub">
					<span class="df-label">truth_sets</span>
					{#each o.truthSets as t, i}{#if i > 0}<span class="df-sep">·</span>{/if}<span class="df-item"><code>{t.id}</code> {fmt(t.rowCount)}</span>{/each}
				</div>
			{/if}
			{#if o.sourceDumps.length > 0}
				<div class="df-line df-line-sub">
					<span class="df-label">source_dump_id</span>
					{#each o.sourceDumps.slice(0, 6) as s, i}{#if i > 0}<span class="df-sep">·</span>{/if}<span class="df-item"><code>{s.source_dump_id ?? '<null>'}</code> {fmt(s.n)}</span>{/each}{#if o.sourceDumps.length > 6}<span class="df-sep">·</span><span class="df-item df-more">+{o.sourceDumps.length - 6} more</span>{/if}
				</div>
			{/if}
			{#if o.indraTypes.length > 0}
				<details class="df-details">
					<summary>indra_type breakdown ({o.indraTypes.length})</summary>
					<div class="df-line df-line-sub">
						{#each o.indraTypes as t, i}{#if i > 0}<span class="df-sep">·</span>{/if}<span class="df-item">{t.indra_type} {fmt(t.n)}</span>{/each}
					</div>
				</details>
			{/if}
		</footer>
	{/if}
</main>

<style>
	:global(:root) {
		--ink: #1a1a1a;
		--ink-muted: #6a6a6a;
		--ink-faint: #a8a8a8;
		--paper: #fdfcf8;
		--rule: #e6e2d6;
		--accent: #7d2a1a;
		--accent-wash: rgba(125, 42, 26, 0.04);
		--ok-green: #2a6f2a;
		--mono: ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace;
		--serif: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
		--sans: -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
	}

	:global(html, body) {
		background: var(--paper);
		color: var(--ink);
		font-family: var(--serif);
		font-size: 16px;
		line-height: 1.5;
		margin: 0;
		padding: 0;
	}

	header {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		padding: 0.6rem 1.5rem;
		border-bottom: 1px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
	}

	.crumb strong {
		color: var(--ink);
		font-weight: 500;
	}

	.crumb .sep {
		color: var(--ink-faint);
	}

	.db-path {
		color: var(--ink-faint);
	}

	.meta {
		display: flex;
		gap: 1.2rem;
		align-items: baseline;
	}

	.nav-link {
		color: var(--accent);
		text-decoration: none;
	}

	.nav-link:hover {
		text-decoration: underline;
	}

	main {
		max-width: 1200px;
		margin: 0 auto;
		padding: 2rem 1.5rem 4rem;
	}

	.empty {
		max-width: 60ch;
		margin: 4rem auto;
	}

	.empty h1 {
		font-family: var(--serif);
		font-weight: 400;
		font-size: 1.6rem;
		color: var(--ink);
		margin: 0 0 0.5rem;
	}

	.empty .lede {
		color: var(--ink-muted);
		margin-bottom: 1.2rem;
	}

	pre {
		background: transparent;
		border-left: 2px solid var(--accent);
		padding: 0.4rem 0 0.4rem 0.8rem;
		font-family: var(--mono);
		font-size: 0.82rem;
		color: var(--ink);
		overflow-x: auto;
	}

	code {
		font-family: var(--mono);
		font-size: 0.88em;
	}

	.hint {
		color: var(--ink-muted);
		font-style: italic;
		font-size: 0.92em;
	}


	.muted { color: var(--ink-faint); }

	.status-failed { color: var(--accent); font-weight: 500; }
	.status-running { color: var(--ink); font-style: italic; }

	.run-feed-article {
		grid-column: 1 / -1;
	}
	.run-feed {
		list-style: none;
		padding: 0;
		margin: 0;
		font-family: var(--mono);
		font-size: 0.78rem;
	}
	.run-row {
		display: grid;
		grid-template-columns: 1.4ch 9ch minmax(0, max-content) minmax(0, max-content) minmax(0, 1fr) auto;
		gap: 0.6rem;
		align-items: baseline;
		padding: 0.25rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.run-version, .run-when {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.run-row:last-child {
		border-bottom: none;
	}
	.run-row-failed { color: var(--accent); }
	.run-row-running { color: var(--ink); }
	.run-glyph {
		font-variant-numeric: tabular-nums;
		text-align: center;
	}
	.run-hash {
		color: var(--ink);
	}
	.run-version {
		color: var(--ink-muted);
	}
	.run-when {
		color: var(--ink-faint);
	}
	.run-narrative {
		color: var(--ink);
		font-variant-numeric: tabular-nums;
		min-width: 0;
	}
	.run-n, .run-cost {
		color: var(--ink);
	}
	.run-summary {
		color: var(--ink);
	}
	.run-exports {
		font-family: var(--mono);
		font-size: 0.72rem;
	}

	.cost-expander {
		grid-column: 1 / -1;
		padding: 0.4rem 0.8rem;
		margin-top: 0.4rem;
		border: 1px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
	}
	.cost-expander summary {
		cursor: pointer;
		font-size: 0.74rem;
		color: var(--ink);
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.cost-expander[open] summary {
		margin-bottom: 0.6rem;
	}
	.cost-expander table {
		width: 100%;
		font-family: var(--mono);
		font-size: 0.78rem;
	}
	.cost-expander td.num {
		font-variant-numeric: tabular-nums;
	}

	.live-indicator {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
		text-transform: lowercase;
		letter-spacing: 0.04em;
	}

	.live-dot {
		display: inline-block;
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--ink-faint);
		transition: background 200ms ease;
	}

	.live-dot.live-dot-flash {
		background: var(--accent);
		transform: scale(1.4);
	}

	.dl-link {
		color: var(--accent);
		text-decoration: none;
		margin-right: 0.6rem;
	}

	.dl-link:hover {
		text-decoration: underline;
	}

	.dl-missing {
		color: var(--ink-faint);
		cursor: help;
	}

	.focus {
		margin-top: 0.5rem;
		margin-bottom: 2.5rem;
	}

	.findings {
		margin: 0 0 2.5rem;
	}

	.findings-h {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.04em;
		margin: 0 0 0.8rem;
		font-weight: 500;
	}

	.lane {
		margin-bottom: 1.2rem;
	}

	.lane-h {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		font-weight: 400;
		margin: 0 0 0.2rem;
		border-bottom: 1px dotted var(--rule);
		padding-bottom: 0.2rem;
	}

	.lane-n {
		color: var(--ink-faint);
		font-weight: 400;
	}

	.lane-body {
		display: flex;
		flex-direction: column;
	}

	.focus-deeplink {
		font-family: var(--mono);
		font-size: 0.78rem;
		text-align: right;
		margin: 0.4rem 0 0;
	}
	.focus-deeplink a {
		color: var(--accent);
		text-decoration: none;
	}
	.focus-deeplink a:hover {
		text-decoration: underline;
	}
	.focus-empty {
		padding: 1.6rem;
		border-left: 3px solid var(--rule);
	}

	.data-footer {
		margin-top: 4rem;
		padding-top: 1rem;
		border-top: 1px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
	}
	.df-line {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem;
		align-items: baseline;
		line-height: 1.7;
	}
	.df-line-sub {
		color: var(--ink-muted);
		margin-top: 0.2rem;
	}
	.df-count {
		color: var(--ink);
	}
	.df-sep {
		color: var(--ink-faint);
	}
	.df-label {
		text-transform: lowercase;
		letter-spacing: 0.04em;
		color: var(--ink-faint);
		margin-right: 0.4rem;
	}
	.df-item code {
		color: inherit;
	}
	.df-more {
		font-style: italic;
	}
	.df-details {
		margin-top: 0.4rem;
	}
	.df-details summary {
		cursor: pointer;
		color: var(--ink-muted);
		font-family: var(--mono);
	}

	.grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
		gap: 2rem 3rem;
		margin-top: 2.5rem;
	}

	article h2 {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		margin: 0 0 0.5rem;
		font-weight: 500;
		border-bottom: 1px solid var(--rule);
		padding-bottom: 0.2rem;
	}

	table {
		width: 100%;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.82rem;
		font-variant-numeric: tabular-nums;
	}

	th, td {
		padding: 0.25rem 0.6rem 0.25rem 0;
		text-align: left;
		vertical-align: baseline;
	}

	th {
		font-weight: 500;
		color: var(--ink-muted);
		font-size: 0.72rem;
		text-transform: lowercase;
	}

	tbody tr {
		border-top: 1px dotted var(--rule);
	}

	td {
		color: var(--ink);
	}

	.num {
		text-align: right;
		font-variant-numeric: tabular-nums;
	}
</style>
