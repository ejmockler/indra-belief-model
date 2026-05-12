<script lang="ts">
	import type { PageData } from './$types';
	import { invalidateAll } from '$app/navigation';
	import { onMount, onDestroy } from 'svelte';
	import BeliefPrimitive from '$lib/components/BeliefPrimitive.svelte';
	import HeuristicCoverage from '$lib/components/HeuristicCoverage.svelte';
	import Validity from '$lib/components/Validity.svelte';

	let { data }: { data: PageData } = $props();
	const o = $derived(data.overview);
	const focus = $derived(data.focus);
	const findings = $derived(data.findings);
	const residuals = $derived(data.residuals);
	const narratives = $derived(data.narratives);
	const coverage = $derived(data.coverage);
	const datasets = $derived(data.datasets);

	function fmtBytes(n: number): string {
		if (n < 1024) return `${n} B`;
		if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
		if (n < 1024 * 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
		return `${(n / (1024 * 1024 * 1024)).toFixed(2)} GB`;
	}

	/** Slugify a filename into a valid truth_set_id / source_dump_id. */
	function slugFromPath(path: string): string {
		const base = path.split('/').pop() ?? path;
		return base
			.replace(/\.(jsonl|json|json\.gz|gz)$/i, '')
			.replace(/[^A-Za-z0-9_-]+/g, '_')
			.replace(/^_+|_+$/g, '')
			.toLowerCase();
	}

	// Per-card action state: { phase: 'idle'|'confirming'|'running'|'done'|'error', message? }
	type ActionPhase = 'idle' | 'confirming' | 'running' | 'done' | 'error';
	type ActionState = { phase: ActionPhase; message?: string };
	let actionStates: Record<string, ActionState> = $state({});

	function setAction(path: string, state: ActionState) {
		actionStates = { ...actionStates, [path]: state };
	}

	function actionState(path: string): ActionState {
		return actionStates[path] ?? { phase: 'idle' };
	}

	async function registerAsTruthSet(d: { path: string; filename: string }) {
		const truth_set_id = slugFromPath(d.path);
		const truth_set_name = d.filename;
		setAction(d.path, { phase: 'running' });
		try {
			const res = await fetch('/api/truth-sets', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					path: d.path,
					truth_set_id,
					truth_set_name,
					target_kind: 'evidence',
					field: 'tag'
				})
			});
			const body = await res.json();
			if (!res.ok) {
				setAction(d.path, { phase: 'error', message: body?.stderr?.slice?.(0, 200) ?? 'register failed' });
				return;
			}
			const sum = body?.summary as { n_loaded?: number; n_missing_target?: number; duration_s?: number } | null;
			setAction(d.path, {
				phase: 'done',
				message: sum
					? `registered as truth_set_id=${truth_set_id} · ${sum.n_loaded ?? '?'} labels · ${sum.n_missing_target ?? 0} missing target · ${sum.duration_s ?? '?'}s`
					: 'registered'
			});
			await invalidateAll();
		} catch (err) {
			setAction(d.path, { phase: 'error', message: String(err).slice(0, 200) });
		}
	}

	// Cost preflight + score state per dataset path
	type CostEstimate = { model_id: string; cost_usd: number; n_stmts: number; n_evidences_est: number; n_llm_calls_est: number };
	type PreflightState =
		| { phase: 'idle' }
		| { phase: 'estimating' }
		| { phase: 'estimated'; estimates: CostEstimate[] }
		| { phase: 'scoring'; model: string; n_evidences_done: number; n_evidences_total: number | null; latest_stmt: string | null; t_started: number }
		| { phase: 'scored'; run_id: string; model: string; n_evidences: number; duration_s: number }
		| { phase: 'error'; message: string };
	let preflightStates: Record<string, PreflightState> = $state({});
	// AbortControllers for in-flight score runs, indexed by dataset path.
	// Kept outside $state because AbortController isn't a plain JSON value.
	const scoreControllers = new Map<string, AbortController>();
	function setPre(path: string, st: PreflightState) {
		preflightStates = { ...preflightStates, [path]: st };
	}
	function preState(path: string): PreflightState {
		return preflightStates[path] ?? { phase: 'idle' };
	}
	function cancelScore(path: string) {
		const ctrl = scoreControllers.get(path);
		if (ctrl) {
			ctrl.abort();
			scoreControllers.delete(path);
		}
		const cur = preState(path);
		if (cur.phase === 'scoring') {
			setPre(path, {
				phase: 'error',
				message: `canceled by user after ${cur.n_evidences_done} evidences (~${Math.round((Date.now() - cur.t_started) / 1000)}s)`
			});
		}
	}

	function fmtCost(c: number): string {
		if (c < 0.01) return '<$0.01';
		if (c < 1) return '$' + c.toFixed(2);
		if (c < 100) return '$' + c.toFixed(2);
		return '$' + c.toFixed(0);
	}

	async function estimateCost(d: { path: string }) {
		setPre(d.path, { phase: 'estimating' });
		try {
			const res = await fetch('/api/runs/estimate-cost', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ path: d.path })
			});
			const body = await res.json();
			if (!res.ok) {
				setPre(d.path, { phase: 'error', message: body?.stderr?.slice?.(0, 200) ?? 'estimate failed' });
				return;
			}
			const estimates = (body?.summary?.estimates as CostEstimate[]) ?? [];
			setPre(d.path, { phase: 'estimated', estimates });
		} catch (err) {
			setPre(d.path, { phase: 'error', message: String(err).slice(0, 200) });
		}
	}

	async function scoreCorpus(d: { path: string; filename: string }, model: string) {
		const source_dump_id = slugFromPath(d.path);
		const scorer_version = 'prod-v1';
		setPre(d.path, {
			phase: 'scoring',
			model,
			n_evidences_done: 0,
			n_evidences_total: null,
			latest_stmt: null,
			t_started: Date.now()
		});
		const ctrl = new AbortController();
		scoreControllers.set(d.path, ctrl);
		try {
			const res = await fetch('/api/runs/score', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({
					path: d.path,
					source_dump_id,
					model,
					scorer_version
				}),
				signal: ctrl.signal
			});
			if (!res.ok || !res.body) {
				const body = await res.text();
				setPre(d.path, { phase: 'error', message: body.slice(0, 300) });
				return;
			}
			// SSE-ish stream — server emits `data: <json>\n\n` per event
			const reader = res.body.getReader();
			const decoder = new TextDecoder();
			let buf = '';
			while (true) {
				const { value, done } = await reader.read();
				if (done) break;
				buf += decoder.decode(value, { stream: true });
				let nl: number;
				while ((nl = buf.indexOf('\n\n')) >= 0) {
					const block = buf.slice(0, nl);
					buf = buf.slice(nl + 2);
					const dataLine = block.split('\n').find((l) => l.startsWith('data: '));
					if (!dataLine) continue;
					let ev: Record<string, unknown>;
					try {
						ev = JSON.parse(dataLine.slice(6));
					} catch {
						continue;
					}
					const t = ev.event as string;
					if (t === 'loaded') {
						const cur = preState(d.path);
						if (cur.phase === 'scoring') {
							setPre(d.path, { ...cur, n_evidences_total: Number(ev.n_statements) * 2 });
						}
					} else if (t === 'progress') {
						const cur = preState(d.path);
						if (cur.phase === 'scoring') {
							setPre(d.path, {
								...cur,
								n_evidences_done: Number(ev.n_evidences_done),
								latest_stmt: (ev.latest_stmt_hash as string) ?? null
							});
						}
					} else if (t === 'done') {
						setPre(d.path, {
							phase: 'scored',
							run_id: String(ev.run_id),
							model,
							n_evidences: Number(ev.n_evidences_done),
							duration_s: Number(ev.duration_s)
						});
						await invalidateAll();
					} else if (t === 'error') {
						setPre(d.path, {
							phase: 'error',
							message: String(ev.stderr ?? ev.error ?? 'score failed').slice(0, 400)
						});
					}
				}
			}
		} catch (err) {
			// AbortError is the user-canceled path; cancelScore() set state
			const cur = preState(d.path);
			if ((err as Error).name === 'AbortError' && cur.phase === 'error') {
				// already handled in cancelScore()
			} else {
				setPre(d.path, { phase: 'error', message: String(err).slice(0, 200) });
			}
		} finally {
			scoreControllers.delete(d.path);
		}
	}

	async function ingestCorpus(d: { path: string; filename: string }) {
		const source_dump_id = slugFromPath(d.path);
		setAction(d.path, { phase: 'running' });
		try {
			const res = await fetch('/api/datasets/ingest', {
				method: 'POST',
				headers: { 'content-type': 'application/json' },
				body: JSON.stringify({ path: d.path, source_dump_id })
			});
			const body = await res.json();
			if (!res.ok) {
				setAction(d.path, { phase: 'error', message: body?.stderr?.slice?.(0, 200) ?? 'ingest failed' });
				return;
			}
			const sum = body?.summary as { n_statements?: number; duration_s?: number } | null;
			setAction(d.path, {
				phase: 'done',
				message: sum
					? `ingested ${sum.n_statements ?? '?'} statements as source_dump_id=${source_dump_id} · ${sum.duration_s ?? '?'}s`
					: 'ingested'
			});
			await invalidateAll();
		} catch (err) {
			setAction(d.path, { phase: 'error', message: String(err).slice(0, 200) });
		}
	}

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
		{ key: 'biggest_disagreement', title: 'we disagree most with INDRA on these', emptyMsg: 'no scored statements' },
		{ key: 'probe_split', title: 'the four probes disagreed among themselves', emptyMsg: 'no multi-probe statements' },
		{ key: 'low_confidence_high_stakes', title: 'mid-range belief, multi-evidence — worth a closer look', emptyMsg: 'no mid-range statements with multi-evidence' },
		{ key: 'verdict_regression', title: 'verdict moved correct → incorrect since prev run', emptyMsg: 'no prev run or no regressions' },
		{ key: 'verdict_recovery', title: 'verdict moved incorrect → correct since prev run', emptyMsg: 'no prev run or no recoveries' }
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
		<p class="dashboard-subtitle">
			INDRA Statement belief rescorer. Below: the statement that disagreed most with INDRA's prior in the latest run, what changed, and where we are weakest.
		</p>

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
			<section class="findings" aria-label="other notable statements from this run">
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

		{#if coverage}
			<HeuristicCoverage {coverage} />
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

		{#if datasets && datasets.length > 0}
			{@const corpora = datasets.filter((d) => d.kind === 'corpus')}
			{@const benchmarks = datasets.filter((d) => d.kind === 'benchmark')}
			<section class="datasets">
				<h2 class="ds-h">datasets on disk</h2>
				<p class="ds-intro">JSON / JSONL files in <code>data/corpora/</code> and <code>data/benchmark/</code>. Read-only for now — actions ship in U4 (register as truth_set) and U5 (ingest + score).</p>

				<div class="ds-group">
					<h3 class="ds-group-h">data/corpora/ <span class="muted">({corpora.length})</span></h3>
					{#if corpora.length === 0}
						<p class="ds-empty">no corpora yet — drop a JSON of INDRA statements in <code>data/corpora/</code> and refresh.</p>
					{:else}
						<ul class="ds-list">
							{#each corpora as d}
								{@const st = actionState(d.path)}
								{@const pre = preState(d.path)}
								{@const canIngest = d.shape.kind_detail === 'indra_json' && (d.ingest?.n_in_file ?? 0) > 0 && (d.ingest?.n_already_ingested ?? 0) < (d.ingest?.n_in_file ?? 0)}
								<li class="ds-row">
									<div class="ds-row-head">
										<code class="ds-name">{d.filename}</code>
										<span class="ds-meta">
											<span>{fmtBytes(d.size_bytes)}</span>
											<span class="muted">·</span>
											<span>{d.shape.n_records ?? '—'} {d.shape.kind_detail === 'jsonl_records' ? 'records' : d.shape.kind_detail === 'indra_json' ? 'statements' : ''}</span>
											{#if d.shape.source_apis.length > 0}
												<span class="muted">·</span>
												<span>sources: {d.shape.source_apis.join(', ')}</span>
											{/if}
										</span>
										{#if d.ingest}
											{@const ing = d.ingest}
											{#if ing.n_in_file === 0}
												<span class="ds-badge ds-badge-unknown" title={ing.notes.join(' · ')}>ingest status unknown</span>
											{:else if ing.n_already_ingested === 0}
												<span class="ds-badge ds-badge-fresh">not ingested</span>
											{:else if ing.n_already_ingested >= ing.n_in_file}
												<span class="ds-badge ds-badge-done">{ing.sampled ? `${ing.n_already_ingested}+ ingested` : 'fully ingested'}</span>
											{:else}
												<span class="ds-badge ds-badge-partial">partial · {ing.n_already_ingested}/{ing.n_in_file} ingested</span>
											{/if}
										{/if}
										{#if canIngest && st.phase === 'idle'}
											<button class="ds-action" onclick={() => ingestCorpus(d)}>ingest into corpus.duckdb →</button>
										{:else if st.phase === 'running'}
											<span class="ds-action ds-action-running">ingesting…</span>
										{:else if st.phase === 'done'}
											<span class="ds-action ds-action-done">✓ {st.message}</span>
										{:else if st.phase === 'error'}
											<span class="ds-action ds-action-error" title={st.message}>✗ ingest failed</span>
										{/if}
									</div>
									{#if d.shape.sample_lines.length > 0}
										<ul class="ds-samples">
											{#each d.shape.sample_lines as s}
												<li><span class="ds-sample">{s}</span></li>
											{/each}
										</ul>
									{/if}
									{#if d.shape.notes.length > 0}
										<p class="ds-notes">{d.shape.notes.join(' · ')}</p>
									{/if}
									{#if d.shape.kind_detail === 'indra_json'}
										<div class="ds-preflight">
											{#if pre.phase === 'idle'}
												<button class="ds-action" onclick={() => estimateCost(d)}>preview scoring cost →</button>
											{:else if pre.phase === 'estimating'}
												<span class="ds-action ds-action-running">estimating…</span>
											{:else if pre.phase === 'estimated'}
												<div class="ds-cost-panel">
													<p class="ds-cost-intro">
														Projected cost to score this corpus (assumes ~5 LLM calls per evidence; substrate short-circuits typically reduce this 30–60%):
													</p>
													<table class="ds-cost-table">
														<thead>
															<tr><th>model</th><th class="num">cost</th><th class="num">calls</th><th></th></tr>
														</thead>
														<tbody>
															{#each pre.estimates as e}
																<tr>
																	<td><code>{e.model_id}</code></td>
																	<td class="num">{fmtCost(e.cost_usd)}</td>
																	<td class="num">{e.n_llm_calls_est.toLocaleString()}</td>
																	<td><button class="ds-action ds-action-confirm" onclick={() => scoreCorpus(d, e.model_id)} title="ingests + scores · cost is best-estimate · cancel by closing the tab">score with this →</button></td>
																</tr>
															{/each}
														</tbody>
													</table>
													<p class="ds-cost-warn">
														Scoring is destructive of API budget. Closing the browser tab does not stop the worker — wait for it to finish or kill the python process by hand.
													</p>
													<button class="ds-action ds-action-cancel" onclick={() => setPre(d.path, { phase: 'idle' })}>cancel</button>
												</div>
											{:else if pre.phase === 'scoring'}
												<div class="ds-cost-panel ds-scoring">
													<p class="ds-scoring-line">
														<strong>scoring with {pre.model}</strong>
														{#if pre.n_evidences_total}· {pre.n_evidences_done} / {pre.n_evidences_total} evidences{:else}· {pre.n_evidences_done} evidences scored{/if}
														· elapsed {Math.round((Date.now() - pre.t_started) / 1000)}s
														{#if pre.latest_stmt}<span class="muted">· latest stmt {pre.latest_stmt.slice(0, 8)}</span>{/if}
													</p>
													<p class="ds-cost-warn muted">stream connected; partial state is persisted per evidence — cancel below to stop the worker (sends SIGTERM, then SIGKILL after 2s).</p>
													<button class="ds-action ds-action-cancel" onclick={() => cancelScore(d.path)}>cancel run →</button>
												</div>
											{:else if pre.phase === 'scored'}
												<span class="ds-action ds-action-done">✓ scored as run {pre.run_id.slice(0, 8)} · {pre.n_evidences} evidences · {pre.duration_s.toFixed(1)}s</span>
											{:else if pre.phase === 'error'}
												<span class="ds-action ds-action-error" title={pre.message}>✗ {pre.message.slice(0, 100)}</span>
												<button class="ds-action ds-action-cancel" onclick={() => setPre(d.path, { phase: 'idle' })}>reset</button>
											{/if}
										</div>
									{/if}
								</li>
							{/each}
						</ul>
					{/if}
				</div>

				<div class="ds-group">
					<h3 class="ds-group-h">data/benchmark/ <span class="muted">({benchmarks.length})</span></h3>
					{#if benchmarks.length === 0}
						<p class="ds-empty">no benchmark files found.</p>
					{:else}
						<ul class="ds-list">
							{#each benchmarks as d}
								{@const st = actionState(d.path)}
								{@const canRegister = d.shape.kind_detail === 'jsonl_records' && (d.shape.n_records ?? 0) > 0}
								<li class="ds-row">
									<div class="ds-row-head">
										<code class="ds-name">{d.filename}</code>
										<span class="ds-meta">
											<span>{fmtBytes(d.size_bytes)}</span>
											<span class="muted">·</span>
											<span>{d.shape.n_records ?? '—'} {d.shape.kind_detail === 'jsonl_records' ? 'records' : d.shape.kind_detail === 'indra_json' ? 'statements' : 'unparsed'}</span>
											{#if d.shape.source_apis.length > 0}
												<span class="muted">·</span>
												<span>sources: {d.shape.source_apis.join(', ')}</span>
											{/if}
										</span>
										{#if d.ingest && d.ingest.n_in_file > 0}
											{@const ing = d.ingest}
											{#if ing.n_already_ingested === 0}
												<span class="ds-badge ds-badge-fresh">not yet ingested</span>
											{:else if ing.n_already_ingested >= ing.n_in_file}
												<span class="ds-badge ds-badge-done">{ing.sampled ? `${ing.n_already_ingested}+ ingested` : 'fully ingested'}</span>
											{:else}
												<span class="ds-badge ds-badge-partial">partial · {ing.n_already_ingested}/{ing.n_in_file} ingested</span>
											{/if}
										{/if}
										{#if canRegister && st.phase === 'idle'}
											<button class="ds-action" onclick={() => registerAsTruthSet(d)} title="reads `tag` field on each record; registers as evidence-level truth_set; reruns validity for the latest scored run">register `tag` as truth_set →</button>
										{:else if st.phase === 'running'}
											<span class="ds-action ds-action-running">registering…</span>
										{:else if st.phase === 'done'}
											<span class="ds-action ds-action-done">✓ {st.message}</span>
										{:else if st.phase === 'error'}
											<span class="ds-action ds-action-error" title={st.message}>✗ register failed</span>
										{/if}
									</div>
									{#if d.shape.sample_lines.length > 0}
										<ul class="ds-samples">
											{#each d.shape.sample_lines as s}
												<li><span class="ds-sample">{s}</span></li>
											{/each}
										</ul>
									{/if}
									{#if d.shape.notes.length > 0}
										<p class="ds-notes">{d.shape.notes.join(' · ')}</p>
									{/if}
								</li>
							{/each}
						</ul>
					{/if}
				</div>
			</section>
		{/if}

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

	.dashboard-subtitle {
		font-family: var(--serif);
		font-size: 1rem;
		color: var(--ink-muted);
		margin: 0.3rem 0 1.6rem;
		line-height: 1.5;
		max-width: 60ch;
	}

	.focus {
		margin-top: 0.5rem;
		margin-bottom: 2.5rem;
	}

	.findings {
		margin: 0 0 2.5rem;
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

	.datasets {
		margin: 0 0 2.5rem;
	}
	.ds-h {
		font-family: var(--serif);
		font-size: 1.15rem;
		font-weight: 400;
		color: var(--ink);
		margin: 0 0 0.4rem;
	}
	.ds-intro {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.88rem;
		color: var(--ink-muted);
		margin: 0 0 1rem;
		line-height: 1.5;
	}
	.ds-group {
		margin-bottom: 1.4rem;
	}
	.ds-group-h {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.04em;
		font-weight: 500;
		margin: 0 0 0.4rem;
		border-bottom: 1px dotted var(--rule);
		padding-bottom: 0.2rem;
	}
	.ds-empty {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.86rem;
		color: var(--ink-faint);
		margin: 0;
	}
	.ds-list {
		list-style: none;
		padding: 0;
		margin: 0;
	}
	.ds-row {
		padding: 0.5rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.ds-row:last-child {
		border-bottom: none;
	}
	.ds-row-head {
		display: flex;
		gap: 0.8rem;
		align-items: baseline;
		flex-wrap: wrap;
		font-family: var(--mono);
		font-size: 0.82rem;
	}
	.ds-name {
		color: var(--ink);
		font-weight: 500;
	}
	.ds-meta {
		font-size: 0.74rem;
		color: var(--ink-muted);
		display: inline-flex;
		gap: 0.4rem;
		align-items: baseline;
		flex-wrap: wrap;
	}
	.ds-samples {
		list-style: none;
		padding: 0;
		margin: 0.3rem 0 0 1rem;
		border-left: 2px solid var(--rule);
	}
	.ds-samples li {
		padding: 0.15rem 0.6rem;
	}
	.ds-sample {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.88rem;
		color: var(--ink);
		line-height: 1.4;
	}
	.ds-badge {
		font-family: var(--mono);
		font-size: 0.7rem;
		padding: 0.05rem 0.4rem;
		border: 1px solid currentColor;
		text-transform: lowercase;
		letter-spacing: 0.04em;
	}
	.ds-badge-fresh { color: var(--ink-muted); }
	.ds-badge-partial { color: var(--accent); }
	.ds-badge-done { color: var(--ok-green); }
	.ds-badge-unknown { color: var(--ink-faint); }

	.ds-action {
		font-family: var(--mono);
		font-size: 0.74rem;
		padding: 0.15rem 0.5rem;
		border: 1px solid var(--accent);
		background: transparent;
		color: var(--accent);
		cursor: pointer;
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.ds-action:hover {
		background: var(--accent-wash);
	}
	.ds-action-running {
		border: 1px dashed var(--ink-muted);
		color: var(--ink-muted);
		cursor: progress;
	}
	.ds-action-done {
		border: 1px solid var(--ok-green);
		color: var(--ok-green);
		cursor: default;
	}
	.ds-action-error {
		border: 1px solid var(--accent);
		color: var(--accent);
		cursor: help;
	}
	.ds-action-confirm {
		font-weight: 500;
	}
	.ds-action-cancel {
		border: 1px solid var(--ink-faint);
		color: var(--ink-muted);
	}

	.ds-preflight {
		margin-top: 0.6rem;
	}
	.ds-cost-panel {
		margin-top: 0.4rem;
		padding: 0.8rem 1rem;
		border-left: 3px solid var(--accent);
		background: var(--accent-wash);
	}
	.ds-cost-intro {
		font-family: var(--serif);
		font-size: 0.86rem;
		color: var(--ink);
		margin: 0 0 0.6rem;
		line-height: 1.5;
	}
	.ds-cost-table {
		width: 100%;
		max-width: 520px;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.78rem;
		margin: 0.3rem 0;
	}
	.ds-cost-table th {
		text-align: left;
		font-weight: 500;
		color: var(--ink-muted);
		font-size: 0.7rem;
		padding: 0.2rem 0.6rem 0.2rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.ds-cost-table td {
		padding: 0.3rem 0.6rem 0.3rem 0;
		vertical-align: baseline;
	}
	.ds-cost-table td.num {
		text-align: right;
		font-variant-numeric: tabular-nums;
	}
	.ds-cost-warn {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.78rem;
		color: var(--accent);
		margin: 0.4rem 0 0.6rem;
		line-height: 1.45;
	}
	.ds-cost-warn.muted { color: var(--ink-muted); }

	.ds-scoring {
		border-left-color: var(--ok-green);
	}
	.ds-scoring-line {
		font-family: var(--serif);
		font-size: 0.95rem;
		color: var(--ink);
		margin: 0 0 0.3rem;
	}

	.ds-notes {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
		margin: 0.2rem 0 0;
		font-style: italic;
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
