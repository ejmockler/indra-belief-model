<script lang="ts">
	import type { PageData } from './$types';
	import type { TruthLabelRow } from '$lib/db';
	import BeliefPrimitive from '$lib/components/BeliefPrimitive.svelte';
	import { shortHash, fmtBelief, pluralS } from '$lib/format';

	let { data }: { data: PageData } = $props();
	const d = $derived(data.detail);
	const probes = $derived(data.probes);

	function parseDbRefs(json: string): Array<[string, string]> {
		try {
			const o = JSON.parse(json);
			return Object.entries(o).map(([k, v]) => [k, String(v)]);
		} catch {
			return [];
		}
	}

	function epistemicsRow(label: string, val: boolean | null): string {
		if (val == null) return '';
		return val ? '✓' : '✗';
	}

	// Group truth labels by truth_set_id and target_kind
	function truthByTarget(
		labels: TruthLabelRow[],
		kind: string,
		targetId: string
	): TruthLabelRow[] {
		return labels.filter((l) => l.target_kind === kind && l.target_id === targetId);
	}

	const stmtTruthLabels = $derived(
		truthByTarget(d.truth_labels, 'stmt', d.stmt_hash)
	);

	const presentSets = $derived(
		Array.from(new Set(d.truth_labels.map((l) => l.truth_set_id))).sort()
	);
	const absentSets = $derived(
		d.registered_truth_sets.filter((tid) => !presentSets.includes(tid))
	);

	function shortTruthLabel(tid: string): string {
		return tid
			.replace(/^indra_/, '')
			.replace(/^source_db_/, 'src:')
			.replace(/^gold_pool_/, 'gold:')
			.replace(/^u2_per_probe_/, 'u2:');
	}

	// Aggregate our score across this statement's scored evidences (mean —
	// matches Phase 6 default aggregator + Phase 4 calibration target)
	const scoredEvidences = $derived(
		d.scorer_steps.filter((s) => s.step_kind === 'aggregate')
	);
	const ourBelief = $derived.by(() => {
		const scores: number[] = [];
		for (const step of scoredEvidences) {
			try {
				const v = JSON.parse(step.output_json)?.score;
				if (typeof v === 'number') scores.push(v);
			} catch {}
		}
		return scores.length ? scores.reduce((a, b) => a + b, 0) / scores.length : null;
	});
	const verdictTally = $derived.by((): Record<string, number> => {
		// Permissive: count any string verdict, not just the known three.
		// If Python ever emits a new verdict, the tally stays honest about
		// totals (correct+incorrect+abstain+others = total scored evidences).
		const out: Record<string, number> = { correct: 0, incorrect: 0, abstain: 0 };
		for (const step of scoredEvidences) {
			try {
				const v = JSON.parse(step.output_json)?.verdict;
				if (typeof v === 'string') out[v] = (out[v] ?? 0) + 1;
			} catch {}
		}
		return out;
	});

	let hoveredEvidenceHash: string | null = $state(null);
	let expandedEvHash: string | null = $state(null);
	let copiedHash: string | null = $state(null);

	function toggleExpand(h: string) {
		expandedEvHash = expandedEvHash === h ? null : h;
	}

	const STEP_KINDS: Array<[string, string]> = [
		['parse_claim', 'parse_claim'],
		['build_context', 'build_context'],
		['substrate_route', 'route'],
		['subject_role_probe', 'subj_role'],
		['object_role_probe', 'obj_role'],
		['relation_axis_probe', 'axis'],
		['scope_probe', 'scope'],
		['grounding', 'grounding'],
		['adjudicate', 'adjudicate']
	];

	function fmtDelta(d: number): string {
		const sign = d >= 0 ? '+' : '−';
		return `${sign}${Math.abs(d).toFixed(2)}`;
	}
</script>

<svelte:head><title>{d.indra_type} · {shortHash(d.stmt_hash)} · INDRA Belief</title></svelte:head>

<header>
	<div class="crumb">
		<a href="/">corpus</a><span class="sep"> / </span><a href="/statements">statements</a><span class="sep"> / </span><strong>{shortHash(d.stmt_hash)}</strong>
	</div>
	<div class="meta">
		{#if scoredEvidences.length > 0}
			<span>{d.scorer_steps[0].scorer_version} · {d.scorer_steps[0].model_id ?? '—'}</span>
		{:else}
			<span class="muted">unscored</span>
		{/if}
	</div>
</header>

<main id="main">
	<section class="stmt-header">
		<BeliefPrimitive
			stmt={{ stmt_hash: d.stmt_hash, indra_type: d.indra_type, agents: d.agents }}
			our_score={ourBelief}
			indra_score={d.indra_belief}
			probes={probes}
			evidences={[]}
			mode="full"
		/>
		<div class="stmt-meta">
			{#if ourBelief != null}
				<span class="verdict-tally">
					{#each Object.entries(verdictTally).filter(([_, n]) => n > 0) as [v, n], i}
						{#if i > 0}<span class="dot">·</span>{/if}
						<span class="vt-{v}">{v} {n}</span>
					{/each}
					<span class="muted">of {scoredEvidences.length}</span>
				</span>
				<span class="dot">·</span>
			{:else}
				<span class="hint">unscored · INDRA {fmtBelief(d.indra_belief)}</span>
				<span class="dot">·</span>
			{/if}
			<span>supports {d.supports_count}/{d.supported_by_count}</span>
			<span class="dot">·</span>
			<span class="source">{d.source_dump_id ?? '<no source_dump>'}</span>
		</div>
	</section>

	<div class="cols">
		<!-- Center column: scorer trace + evidences -->
		<div class="trace">
			<h2>scorer trace
				{#if scoredEvidences.length > 0}
					<span class="counter">· {scoredEvidences.length} aggregate step{scoredEvidences.length === 1 ? '' : 's'}</span>
				{:else}
					<span class="counter">· not run</span>
				{/if}
			</h2>
			<!-- 9-step rail: horizontal tick axis. One tick lit per kind that
			     has rows. Brutalist iter-2: vertical list of muted labels read
			     as a TODO; tick axis is honest about what's wired and legible
			     proportional to the work that exists. -->
			<div class="rail-axis" aria-label="9-step scorer trace">
				<div class="rail-track">
					{#each STEP_KINDS as [kind, label], i}
						{@const stepRows = d.scorer_steps.filter((s) => s.step_kind === kind)}
						{@const aggForAdjudicate = kind === 'adjudicate' && scoredEvidences.length > 0}
						{@const lit = stepRows.length > 0 || aggForAdjudicate}
						<div class="rail-cell" class:lit title="{i + 1}. {label}{lit ? (aggForAdjudicate ? ` — via aggregate ×${scoredEvidences.length}` : ` — ${stepRows.length}×`) : ' — not run'}">
							<span class="rail-tick" aria-hidden="true">{lit ? '●' : '·'}</span>
							<span class="rail-cell-label">{label}</span>
						</div>
					{/each}
				</div>
				{#if scoredEvidences.length > 0}
					{@const litLabels = STEP_KINDS
						.filter(([k]) => d.scorer_steps.some((s) => s.step_kind === k))
						.map(([_, lab]) => lab)}
					{@const adjLitViaAgg = !litLabels.includes('adjudicate')}
					{@const totalLit = litLabels.length + (adjLitViaAgg ? 1 : 0)}
					<p class="rail-note">
						<span class="muted">{totalLit}/9 lit · </span>
						{litLabels.join(', ')}{adjLitViaAgg ? ', adjudicate (via aggregate)' : ''}.
					</p>
				{:else}
					<p class="rail-note hint">No run yet · ticks light when <code>score_corpus</code> writes scorer_step rows.</p>
				{/if}
			</div>
			{#if scoredEvidences.length === 0}
				<p class="hint">
					Per-step rail lights up when <code>score_corpus(con, [stmt], decompose=True)</code> writes per-step rows.
				</p>
			{/if}

			<h2>evidences <span class="counter">{d.evidences.length}</span></h2>
			{#each d.evidences as e}
				{@const evTruth = truthByTarget(d.truth_labels, 'evidence', e.evidence_hash)}
				{@const score = d.scorer_steps.find((s) => s.evidence_hash === e.evidence_hash && s.step_kind === 'aggregate')}
				{@const out = score ? JSON.parse(score.output_json) : null}
				<!-- D10 Bret Victor lever: hovering an evidence card highlights truth-set rows
				     that judge any of its labels (epistemics on this evidence) -->
				{@const isExpanded = expandedEvHash === e.evidence_hash}
				<!-- Whole-article click target per iter-3 brutalist BLOCKER #15:
				     body sentence used to be dead space. Now the entire <article> is
				     a button. Inner .ev-expanded panel has pointer-events:none so
				     clicking inside the expansion does NOT collapse the parent. -->
				<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
				<article class="evidence" class:ev-expanded-state={isExpanded}
					class:ev-clickable={!!score}
					data-evidence-hash={e.evidence_hash}
					role={score ? 'button' : undefined}
					tabindex={score ? 0 : undefined}
					aria-expanded={score ? isExpanded : undefined}
					onmouseenter={() => { hoveredEvidenceHash = e.evidence_hash; }}
					onmouseleave={() => { hoveredEvidenceHash = null; }}
					onclick={() => { if (score) toggleExpand(e.evidence_hash); }}
					onkeydown={(ev) => {
						if (score && (ev.key === 'Enter' || ev.key === ' ')) {
							ev.preventDefault();
							toggleExpand(e.evidence_hash);
						}
					}}>
					<div class="ev-meta">
						<code class="ev-hash">{shortHash(e.evidence_hash)}</code>
						<span class="ev-source">{e.source_api ?? '—'}</span>
						{#if e.pmid}
							<span class="ev-pmid">pmid:{e.pmid}</span>
						{/if}
						{#if out}
							<span class="verdict-stamp verdict-{out.verdict}">{out.verdict} / {out.confidence}</span>
							<span class="score-stamp">score {out.score?.toFixed?.(2) ?? '—'}</span>
							{#if score?.latency_ms != null && score.latency_ms > 0}
								<span class="latency">{score.latency_ms}ms</span>
							{/if}
						{/if}
						{#if score}
							<span class="ev-chevron" aria-hidden="true">{isExpanded ? '▾' : '▸'}</span>
						{/if}
					</div>
					<p class="ev-text">{e.text ?? ''}</p>
					<div class="ev-flags">
						<span class="flag">direct: <span class="flag-val">{epistemicsRow('direct', e.is_direct)}</span></span>
						<span class="flag">negated: <span class="flag-val">{epistemicsRow('negated', e.is_negated)}</span></span>
						<span class="flag">curated: <span class="flag-val">{epistemicsRow('curated', e.is_curated)}</span></span>
						{#if evTruth.length > 0}
							<span class="truth-stamp">{evTruth.length} truth label{pluralS(evTruth.length)}</span>
						{/if}
					</div>
					{#if out && out.reasons && out.reasons.length > 0}
						<div class="reasons">
							<span class="reasons-label">reasons</span>
							{#each out.reasons as r}
								<span class="reason-token">· {r}</span>
							{/each}
						</div>
					{/if}
					{#if isExpanded && out}
						<!-- svelte-ignore a11y_click_events_have_key_events -->
						<!-- svelte-ignore a11y_no_static_element_interactions -->
						<div class="ev-expanded"
							onclick={(ev) => ev.stopPropagation()}>
							{#if out.rationale}
								<div class="exp-row">
									<span class="exp-label">rationale</span>
									<span class="exp-val">{out.rationale}</span>
								</div>
							{/if}
							{#if out.call_log && out.call_log.length === 1}
								{@const c = out.call_log[0]}
								<div class="exp-row">
									<span class="exp-label">call_log</span>
									<span class="exp-val mono">
										{c.kind ?? '—'} · {(c.duration_s ?? 0).toFixed(2)}s · {c.prompt_tokens ?? '—'}→{c.out_tokens ?? '—'} · {c.finish_reason ?? '—'}
									</span>
								</div>
							{:else if out.call_log && out.call_log.length > 1}
								<div class="exp-row">
									<span class="exp-label">call_log</span>
									<table class="call-log">
										<thead>
											<tr><th>kind</th><th class="num">duration</th><th class="num">in→out</th><th>finish</th></tr>
										</thead>
										<tbody>
											{#each out.call_log as call}
												<tr>
													<td>{call.kind ?? '—'}</td>
													<td class="num">{(call.duration_s ?? 0).toFixed(2)}s</td>
													<td class="num">{call.prompt_tokens ?? '—'}→{call.out_tokens ?? '—'}</td>
													<td>{call.finish_reason ?? '—'}</td>
												</tr>
											{/each}
										</tbody>
									</table>
								</div>
							{:else}
								<div class="exp-row">
									<span class="exp-label muted">call_log empty (substrate-resolved or no LLM probes invoked)</span>
								</div>
							{/if}
							{#if out.error}
								<div class="exp-row exp-error">
									<span class="exp-label">error</span>
									<span class="exp-val">{out.error}</span>
								</div>
							{/if}
							{#if d.scorer_steps.filter((s) => s.evidence_hash === e.evidence_hash && s.step_kind !== 'aggregate').length > 0}
								{@const evSteps = d.scorer_steps.filter((s) => s.evidence_hash === e.evidence_hash && s.step_kind !== 'aggregate')}
								<div class="exp-row">
									<span class="exp-label">per-step</span>
									<div class="step-list">
										{#each evSteps as step}
											{@const stepOut = (() => { try { return JSON.parse(step.output_json); } catch { return null; }})()}
											<div class="step-item">
												<span class="step-kind">{step.step_kind}</span>
												{#if step.is_substrate_answered === true}<span class="step-source">substrate</span>{/if}
												{#if step.is_substrate_answered === false}<span class="step-source">LLM</span>{/if}
												{#if stepOut}
													{#if stepOut.answer != null}
														<span class="step-answer">{stepOut.answer}</span>
													{/if}
													{#if stepOut.confidence != null}
														<span class="muted">{stepOut.confidence}</span>
													{/if}
													{#if stepOut.span}
														<span class="step-span">"{stepOut.span}"</span>
													{/if}
													{#if stepOut.stmt_type != null}
														<span class="muted">{stepOut.stmt_type}</span>
													{/if}
													{#if stepOut.n_aliases != null}
														<span class="muted">{stepOut.n_aliases} aliases</span>
													{/if}
													{#if stepOut.n_detected_relations != null}
														<span class="muted">{stepOut.n_detected_relations} relations</span>
													{/if}
												{/if}
											</div>
										{/each}
									</div>
								</div>
							{/if}
							<div class="exp-row">
								<span class="exp-label muted">step_hash</span>
								<button type="button" class="hash-copy"
									onclick={(ev) => {
										ev.stopPropagation();
										navigator.clipboard?.writeText(score?.step_hash ?? '').then(() => {
											copiedHash = score?.step_hash ?? null;
											setTimeout(() => { copiedHash = null; }, 1200);
										});
									}}
									title="Copy step_hash to clipboard">
									<code class="exp-val muted">{score?.step_hash}</code>
									<span class="copy-mark">{copiedHash === score?.step_hash ? '✓ copied' : '⎘'}</span>
								</button>
							</div>
						</div>
					{/if}
				</article>
			{/each}
		</div>

		<!-- Right column: truth panel + agents + supports -->
		<aside class="truth-panel">
			<h2>truth_sets</h2>
			<dl class="truth-list">
				{#each presentSets as tsetId}
					{@const labels = d.truth_labels.filter((l) => l.truth_set_id === tsetId)}
					{@const isActive = hoveredEvidenceHash != null && labels.some((l) =>
						(l.target_kind === 'evidence' && l.target_id === hoveredEvidenceHash) ||
						l.target_kind === 'stmt' ||
						l.target_kind === 'agent'
					)}
					{@const isPassive = hoveredEvidenceHash != null && !isActive}
					<div class="truth-row" class:truth-active={isActive} class:truth-passive={isPassive}>
						<dt>{tsetId}</dt>
						<dd>
							<span class="truth-count">{labels.length}</span>
							<span class="muted">label{labels.length === 1 ? '' : 's'}</span>
						</dd>
					</div>
				{/each}
				{#if absentSets.length > 0}
					<div class="truth-row absent rollup" title={absentSets.join(', ')}>
						<dt>absent</dt>
						<dd>
							<span class="truth-count">{absentSets.length}</span>
							<span class="muted">truth_set{absentSets.length === 1 ? '' : 's'}:</span>
							<span class="absent-mark">{absentSets.map(shortTruthLabel).join(' · ')}</span>
						</dd>
					</div>
				{/if}
			</dl>

			<h2>agents <span class="counter">{d.agents.length}</span></h2>
			{#each d.agents as a}
				{@const agTruth = truthByTarget(d.truth_labels, 'agent', a.agent_hash)}
				<div class="agent">
					<div class="agent-line">
						<span class="agent-role">{a.role}</span>
						<span class="agent-name">{a.name}</span>
					</div>
					<div class="agent-refs">
						{#each parseDbRefs(a.db_refs_json) as [ns, id]}
							<span class="ref-chip"><span class="ref-ns">{ns}</span>:{id}</span>
						{/each}
					</div>
					{#if a.location}
						<div class="agent-loc">loc: {a.location}</div>
					{/if}
					{#if agTruth.length > 0}
						<div class="agent-truth">{agTruth.length} truth label{pluralS(agTruth.length)}</div>
					{/if}
				</div>
			{/each}

			{#if d.supports_edges.length > 0}
				<h2>supports edges <span class="counter">{d.supports_edges.length}</span></h2>
				<ul class="edges">
					{#each d.supports_edges as edge}
						<li>
							<span class="edge-kind">{edge.kind}</span>
							<code>{shortHash(edge.to_stmt_hash)}</code>
						</li>
					{/each}
				</ul>
			{/if}
		</aside>
	</div>
</main>

<style>
	:global(:root) {
		--ink: #1a1a1a;
		--ink-muted: #6a6a6a;
		--ink-faint: #a8a8a8;
		--paper: #fdfcf8;
		--rule: #e6e2d6;
		--accent: #7d2a1a;
		--mono: ui-monospace, 'SF Mono', 'JetBrains Mono', Menlo, monospace;
		--serif: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
	}
	:global(html, body) {
		background: var(--paper);
		color: var(--ink);
		font-family: var(--serif);
		font-size: 16px;
		line-height: 1.5;
		margin: 0;
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
		position: sticky;
		top: 0;
		background: var(--paper);
		z-index: 2;
	}
	.crumb a { color: var(--ink-muted); text-decoration: none; }
	.crumb a:hover { color: var(--ink); }
	.crumb strong { color: var(--ink); font-weight: 500; }
	.crumb .sep, .muted, .dot { color: var(--ink-faint); }
	.dot { margin: 0 0.4rem; }

	main {
		max-width: 1200px;
		margin: 0 auto;
		padding: 1.5rem 1.5rem 4rem;
	}

	.stmt-header {
		padding-bottom: 1.2rem;
		border-bottom: 1px solid var(--ink);
		margin-bottom: 1.5rem;
	}

	.stmt-glyph {
		font-family: var(--serif);
		font-size: 1.5rem;
		font-weight: 400;
		color: var(--ink);
		margin: 0 0 0.4rem;
		line-height: 1.3;
	}

	.indra-type {
		color: var(--accent);
	}

	.paren {
		color: var(--ink-faint);
	}

	.stmt-meta {
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
	}

	.belief-line {
		font-variant-numeric: tabular-nums;
	}

	.belief-num {
		color: var(--ink);
		font-weight: 500;
		font-variant-numeric: tabular-nums;
	}

	.belief-indra {
		color: var(--ink-muted);
		font-variant-numeric: tabular-nums;
	}

	.delta {
		font-family: var(--mono);
		font-variant-numeric: tabular-nums;
	}

	.delta.pos { color: #2a6f2a; }
	.delta.neg { color: var(--accent); }

	.verdict-tally {
		font-family: var(--mono);
		font-variant-numeric: tabular-nums;
	}

	.vt-correct { color: #2a6f2a; }
	.vt-incorrect { color: var(--accent); }
	.vt-abstain { color: var(--ink-muted); font-style: italic; }

	/* 9-step rail — horizontal tick axis */
	.rail-axis {
		margin: 0 0 1.2rem;
	}

	.rail-track {
		display: grid;
		grid-template-columns: repeat(9, 1fr);
		border-bottom: 1px solid var(--rule);
		padding-bottom: 0.3rem;
	}

	.rail-cell {
		display: flex;
		flex-direction: column;
		align-items: center;
		gap: 0.15rem;
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
		cursor: help;
	}

	.rail-tick {
		font-size: 0.9rem;
		line-height: 1;
		color: var(--ink-faint);
	}

	.rail-cell.lit .rail-tick {
		color: var(--accent);
	}

	.rail-cell.lit .rail-cell-label {
		color: var(--ink);
	}

	.rail-cell-label {
		text-align: center;
		font-size: 0.62rem;
		letter-spacing: 0.01em;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		max-width: 12ch;
	}

	.rail-note {
		font-family: var(--serif);
		font-size: 0.84rem;
		color: var(--ink-muted);
		margin: 0.4rem 0 0;
	}

	.rail-note.hint {
		font-style: italic;
	}

	.hint {
		color: var(--ink-faint);
		font-style: italic;
		font-size: 0.92em;
	}

	.cols {
		display: grid;
		grid-template-columns: 1fr 320px;
		gap: 2rem;
	}

	@media (max-width: 880px) {
		.cols { grid-template-columns: 1fr; }
	}

	h2 {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		margin: 1.5rem 0 0.6rem;
		font-weight: 500;
		border-bottom: 1px solid var(--rule);
		padding-bottom: 0.2rem;
	}

	h2:first-child {
		margin-top: 0;
	}

	.counter {
		color: var(--ink-faint);
		font-weight: 400;
	}

	.trace-placeholder {
		font-family: var(--mono);
		font-size: 0.82rem;
		padding: 0.4rem 0 0 0.6rem;
		border-left: 2px solid var(--rule);
	}

	.rail {
		padding: 0.15rem 0;
	}

	.step-num {
		display: inline-block;
		width: 3ch;
		color: var(--ink-faint);
	}

	.evidence {
		padding: 0.7rem 0;
		border-bottom: 1px dotted var(--rule);
	}

	.evidence:last-child {
		border-bottom: none;
	}

	.ev-meta {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-muted);
		display: flex;
		gap: 0.8rem;
		margin-bottom: 0.3rem;
	}

	.ev-hash {
		color: var(--ink-faint);
	}

	.ev-source {
		color: var(--accent);
	}

	.ev-pmid {
		color: var(--ink-muted);
	}

	.ev-text {
		font-family: var(--serif);
		font-size: 1.02rem;
		line-height: 1.5;
		margin: 0.2rem 0 0.4rem;
		color: var(--ink);
	}

	.ev-flags {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-muted);
		display: flex;
		gap: 1rem;
	}

	.flag-val {
		color: var(--ink);
	}

	.truth-stamp {
		color: var(--accent);
	}

	.verdict-stamp {
		font-family: var(--mono);
		font-size: 0.72rem;
		text-transform: lowercase;
		padding: 0 0.3rem;
	}

	.verdict-correct { color: #2a6f2a; }
	.verdict-incorrect { color: #7d2a1a; }
	.verdict-abstain { color: var(--ink-muted); font-style: italic; }

	.score-stamp {
		font-variant-numeric: tabular-nums;
		color: var(--ink);
	}

	.latency {
		color: var(--ink-faint);
	}

	.reasons {
		font-family: var(--mono);
		font-size: 0.74rem;
		margin-top: 0.35rem;
		color: var(--ink);
	}

	.reasons-label {
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.04em;
		margin-right: 0.4rem;
	}

	.reason-token {
		margin-right: 0.4rem;
	}

	/* Truth panel */
	.truth-panel {
		font-family: var(--mono);
		font-size: 0.82rem;
	}

	.truth-list {
		margin: 0;
	}

	.truth-row {
		display: flex;
		justify-content: space-between;
		padding: 0.18rem 0;
		border-bottom: 1px dotted var(--rule);
	}

	.truth-row dt {
		color: var(--ink);
		margin: 0;
	}

	.truth-row dd {
		color: var(--ink);
		margin: 0;
		font-variant-numeric: tabular-nums;
	}

	.truth-row.absent dt {
		color: var(--ink-faint);
	}

	.truth-row.absent dd {
		color: var(--ink-faint);
	}

	.truth-row.truth-active dt {
		color: var(--accent);
		font-weight: 500;
	}

	.truth-row.truth-active .truth-count {
		color: var(--accent);
		font-weight: 500;
	}

	.truth-row.truth-passive {
		opacity: 0.45;
	}

	.truth-count {
		color: var(--ink);
		font-variant-numeric: tabular-nums;
	}

	.evidence:hover {
		background: rgba(125, 42, 26, 0.025);
	}

	.ev-clickable {
		cursor: pointer;
	}

	.ev-clickable:hover .ev-chevron {
		color: var(--accent);
	}

	.ev-clickable:focus-visible {
		outline: 2px solid var(--accent);
		outline-offset: 2px;
	}

	/* Inner expansion content shouldn't trigger collapse on click */
	.ev-expanded {
		pointer-events: auto;  /* keeps copy button functional */
	}

	.ev-chevron {
		margin-left: auto;
		font-size: 0.7rem;
		color: var(--ink-faint);
		font-family: var(--mono);
	}

	.evidence.ev-expanded-state .ev-chevron {
		color: var(--accent);
	}

	.ev-expanded {
		margin-top: 0.5rem;
		padding: 0.5rem 0.6rem;
		background: rgba(0, 0, 0, 0.02);
		border-left: 2px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.74rem;
	}

	.exp-row {
		margin: 0.2rem 0;
		display: flex;
		gap: 0.6rem;
		align-items: baseline;
	}

	.exp-label {
		color: var(--ink-muted);
		min-width: 8ch;
		font-size: 0.7rem;
		text-transform: lowercase;
	}

	.exp-val {
		color: var(--ink);
		font-family: var(--serif);
		font-size: 0.92rem;
	}

	.exp-error .exp-val {
		color: var(--accent);
	}

	.exp-val.mono {
		font-family: var(--mono);
		font-size: 0.78rem;
	}

	.hash-copy {
		all: unset;
		cursor: pointer;
		display: inline-flex;
		align-items: baseline;
		gap: 0.3rem;
	}

	.hash-copy:hover .copy-mark {
		color: var(--accent);
	}

	.copy-mark {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-faint);
	}

	.step-list {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		font-family: var(--mono);
		font-size: 0.72rem;
	}

	.step-item {
		display: flex;
		gap: 0.5rem;
		align-items: baseline;
		flex-wrap: wrap;
	}

	.step-kind {
		color: var(--ink);
		font-weight: 500;
		min-width: 16ch;
	}

	.step-source {
		color: var(--accent);
		font-size: 0.66rem;
		text-transform: uppercase;
	}

	.step-answer {
		color: var(--ink);
	}

	.step-span {
		color: var(--ink-muted);
		font-style: italic;
	}

	.call-log {
		font-family: var(--mono);
		font-size: 0.72rem;
		border-collapse: collapse;
		flex: 1;
	}

	.call-log th {
		text-align: left;
		font-weight: 500;
		color: var(--ink-muted);
		font-size: 0.66rem;
		padding-right: 1rem;
		border-bottom: 1px dotted var(--rule);
	}

	.call-log td {
		padding: 0.1rem 1rem 0.1rem 0;
		font-variant-numeric: tabular-nums;
	}

	.call-log .num {
		text-align: right;
	}

	.absent-mark {
		font-style: italic;
	}

	/* Agent cards */
	.agent {
		padding: 0.5rem 0;
		border-bottom: 1px dotted var(--rule);
	}

	.agent-line {
		display: flex;
		gap: 0.6rem;
		font-size: 0.92rem;
	}

	.agent-role {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		text-transform: uppercase;
		min-width: 4ch;
	}

	.agent-name {
		font-family: var(--serif);
		font-weight: 500;
	}

	.agent-refs {
		display: flex;
		flex-wrap: wrap;
		gap: 0.3rem;
		margin-top: 0.25rem;
	}

	.ref-chip {
		font-family: var(--mono);
		font-size: 0.72rem;
		padding: 0.05rem 0.35rem;
		background: rgba(125, 42, 26, 0.06);
		color: var(--ink);
	}

	.ref-ns {
		color: var(--accent);
	}

	.agent-loc {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		margin-top: 0.25rem;
	}

	.agent-truth {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--accent);
		margin-top: 0.25rem;
	}

	.edges {
		list-style: none;
		padding: 0;
		margin: 0;
	}

	.edges li {
		padding: 0.15rem 0;
		font-family: var(--mono);
		font-size: 0.78rem;
	}

	.edge-kind {
		color: var(--ink-muted);
		display: inline-block;
		min-width: 12ch;
	}

	code {
		font-family: var(--mono);
		font-size: 0.88em;
	}
</style>
