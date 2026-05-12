<script lang="ts">
	import type { PageData } from './$types';
	import type { ScorerStepRow, TruthLabelRow } from '$lib/db';
	import BeliefPrimitive from '$lib/components/BeliefPrimitive.svelte';
	import {
		evidenceParts,
		extractProbeCue,
		fmtBelief,
		pluralS,
		shortHash,
		verdictDisplay
	} from '$lib/format';

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
	let expandedCall: string | null = $state(null);

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

	function safeParseJSON(s: string | null): Record<string, unknown> | null {
		if (!s) return null;
		try {
			return JSON.parse(s);
		} catch {
			return null;
		}
	}

	function firstStepOutput(kind: string, evidenceHash?: string | null): Record<string, unknown> | null {
		for (const s of d.scorer_steps) {
			if (s.step_kind !== kind) continue;
			if (evidenceHash !== undefined && s.evidence_hash !== evidenceHash) continue;
			const o = safeParseJSON(s.output_json);
			if (o) return o;
		}
		return null;
	}

	function probeOutputsForEvidence(evidenceHash: string): Array<{ kind: string; out: Record<string, unknown>; step: ScorerStepRow }> {
		const probeKinds = ['subject_role_probe', 'object_role_probe', 'relation_axis_probe', 'scope_probe'];
		const rows = d.scorer_steps.filter((s) => probeKinds.includes(s.step_kind) && s.evidence_hash === evidenceHash);
		return rows
			.map((step) => {
				const out = safeParseJSON(step.output_json);
				return out ? { kind: step.step_kind, out, step } : null;
			})
			.filter((x): x is { kind: string; out: Record<string, unknown>; step: ScorerStepRow } => x !== null);
	}

	function cueForEvidence(evidenceHash: string): string | null {
		const probes = probeOutputsForEvidence(evidenceHash);
		for (const { out } of probes) {
			const c = extractProbeCue((out.rationale as string | null) ?? null);
			if (c) return c;
		}
		return null;
	}

	const probeStepLabels: Record<string, string> = {
		subject_role_probe: 'subj-role',
		object_role_probe: 'obj-role',
		relation_axis_probe: 'relation-axis',
		scope_probe: 'scope'
	};

	/**
	 * Trace narrative for the whole statement. Returns an ordered list of
	 * sentence lines explaining the journey the pipeline took. Unrun steps
	 * collapse into a single explanatory line instead of N missing dots.
	 */
	const traceLines = $derived.by(() => {
		const lines: Array<{ key: string; prose: string; muted?: boolean }> = [];

		const pc = firstStepOutput('parse_claim');
		if (pc) {
			const stype = pc.stmt_type ?? d.indra_type;
			const agentList = d.agents.map((a) => a.name).join(' / ');
			lines.push({
				key: 'parse_claim',
				prose: `Parsed the claim as ${stype}${agentList ? ` (${agentList})` : ''}.`
			});
		}

		const bc = firstStepOutput('build_context');
		if (bc) {
			const aliasN = (bc.n_aliases as number | undefined) ?? 0;
			const relN = (bc.n_detected_relations as number | undefined) ?? 0;
			lines.push({
				key: 'build_context',
				prose: `Built context: ${aliasN} alias${aliasN === 1 ? '' : 'es'}, ${relN} relation${relN === 1 ? '' : 's'} detected.`
			});
		}

		const sr = d.scorer_steps.find((s) => s.step_kind === 'substrate_route');
		if (sr) {
			lines.push({
				key: 'substrate_route',
				prose: 'Routed to the deterministic substrate layer (regex / Gilda) before any LLM probe.'
			});
		}

		const probeOrder = ['subject_role_probe', 'object_role_probe', 'relation_axis_probe', 'scope_probe'];
		const ranProbeKinds: string[] = [];
		const skippedProbeKinds: string[] = [];
		for (const kind of probeOrder) {
			const out = firstStepOutput(kind);
			if (out && (out.answer ?? null) !== null && out.answer !== 'abstain') {
				ranProbeKinds.push(kind);
				const source = out.source ?? '?';
				const conf = out.confidence ?? '?';
				const cue = extractProbeCue((out.rationale as string | null) ?? null);
				const cueText = cue ? ` (cue: “${cue}”)` : '';
				lines.push({
					key: kind,
					prose: `${probeStepLabels[kind]}: ${out.answer} — ${source}, ${conf} confidence${cueText}.`
				});
			} else {
				skippedProbeKinds.push(probeStepLabels[kind]);
			}
		}
		if (skippedProbeKinds.length > 0 && ranProbeKinds.length > 0) {
			lines.push({
				key: 'probes_skipped',
				muted: true,
				prose: `${skippedProbeKinds.join(', ')} did not fire — substrate's earlier finding short-circuited the chain.`
			});
		} else if (skippedProbeKinds.length > 0 && ranProbeKinds.length === 0) {
			lines.push({
				key: 'probes_all_skipped',
				muted: true,
				prose: `No probes fired in this run.`
			});
		}

		const gr = firstStepOutput('grounding');
		if (gr) {
			lines.push({
				key: 'grounding',
				prose: `Checked entity grounding.`
			});
		}

		const agg = firstStepOutput('aggregate');
		if (agg) {
			const verdict = (agg.verdict as string | null) ?? '?';
			const conf = (agg.confidence as string | null) ?? '?';
			const score = agg.score as number | null;
			const scoreText = typeof score === 'number' ? score.toFixed(2) : '—';
			lines.push({
				key: 'aggregate',
				prose: `Aggregated to verdict: ${verdictDisplay(verdict)} (${conf} confidence, score ${scoreText}).`
			});
		}

		return lines;
	});

	/** epistemics flag → reader-facing phrase */
	function epistemicsLine(e: { is_direct: boolean | null; is_negated: boolean | null; is_curated: boolean | null }): string {
		const parts: string[] = [];
		if (e.is_direct === false) parts.push('indirect citation');
		else if (e.is_direct === true) parts.push('direct citation');
		if (e.is_negated === true) parts.push('explicitly negated');
		else if (e.is_negated === false) parts.push('not negated');
		if (e.is_curated === true) parts.push('human-curated');
		else if (e.is_curated === false) parts.push('not human-curated');
		return parts.length === 0 ? 'no epistemic flags set' : parts.join(' · ');
	}

	/** Reader-friendly description for the truth-set IDs we ship. */
	function truthSetDescription(tid: string): string {
		const known: Record<string, string> = {
			indra_published_belief: "INDRA's published belief score for the statement (prior)",
			indra_grounding: 'entity grounding mappings from INDRA (db_refs)',
			indra_epistemics: "INDRA's epistemic flags on the evidence (direct / negated / curated)",
			demo_gold: 'hand-labeled gold examples used for P/R/F1'
		};
		return known[tid] ?? tid;
	}

	let showDebug: boolean = $state(false);
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
			level="h1"
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
			<h2>how the pipeline scored this</h2>
			{#if traceLines.length === 0}
				<p class="hint">
					Not yet scored — invoke <code>score_corpus(con, [stmt], decompose=True)</code> to see the pipeline trace.
				</p>
			{:else}
				<ol class="trace-narrative">
					{#each traceLines as line, i}
						<li class="trace-line" class:trace-muted={line.muted}>
							<span class="trace-num">{i + 1}</span>
							<span class="trace-prose">{line.prose}</span>
						</li>
					{/each}
				</ol>
			{/if}

			<h2>evidences <span class="counter">{d.evidences.length}</span></h2>
			<p class="ev-section-note">
				Each evidence carries a verdict ∈ {'{'}supported, contradicted, abstained{'}'} and a confidence ∈ {'{'}high, medium, low{'}'}. The displayed score is a lookup from that (verdict, confidence) pair — only 7 distinct values are possible:
				<span class="ev-bucket-table" title="src/indra_belief/scorers/commitments.py::_VERDICT_SCORE">
					correct/high <code>0.95</code> · correct/medium <code>0.80</code> · correct/low <code>0.65</code> · abstain/* <code>0.50</code> · incorrect/low <code>0.35</code> · incorrect/medium <code>0.20</code> · incorrect/high <code>0.05</code>
				</span>
			</p>
			{#each d.evidences as e}
				{@const evTruth = truthByTarget(d.truth_labels, 'evidence', e.evidence_hash)}
				{@const score = d.scorer_steps.find((s) => s.evidence_hash === e.evidence_hash && s.step_kind === 'aggregate')}
				{@const out = score ? JSON.parse(score.output_json) : null}
				{@const cue = cueForEvidence(e.evidence_hash)}
				{@const parts = evidenceParts(e.text, cue)}
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
					{#if out}
						<div class="ev-verdict-line">
							<span class="ev-verdict ev-verdict-{out.verdict}">verdict: {verdictDisplay(out.verdict)}</span>
							<span class="ev-confidence">{out.confidence ?? '—'} confidence</span>
							<span class="ev-score">score <span class="ev-score-num">{out.score?.toFixed?.(2) ?? '—'}</span></span>
							{#if score}
								<span class="ev-chevron" aria-hidden="true">{isExpanded ? '▾ collapse' : '▸ expand'}</span>
							{/if}
						</div>
					{/if}
					<div class="ev-meta-secondary">
						<span class="ev-source">[{e.source_api ?? 'no source'}]</span>
						{#if e.pmid}
							<span class="ev-pmid">pmid:{e.pmid}</span>
						{/if}
						<code class="ev-hash" title={e.evidence_hash}>{shortHash(e.evidence_hash)}</code>
						{#if score?.latency_ms != null && score.latency_ms > 0}
							<span class="latency">{score.latency_ms}ms</span>
						{/if}
					</div>
					<p class="ev-text">{#each parts as part}{#if part.highlight}<mark class="ev-text-cue">{part.text}</mark>{:else}{part.text}{/if}{/each}</p>
					<div class="ev-flags">
						<span class="ev-flags-prose">{epistemicsLine(e)}</span>
						{#if evTruth.length > 0}
							<span class="truth-stamp">· {evTruth.length} truth label{pluralS(evTruth.length)}</span>
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
						{@const callLog = (out.call_log as Array<Record<string, unknown>> | undefined) ?? []}
						{@const evSteps = d.scorer_steps.filter((s) => s.evidence_hash === e.evidence_hash && s.step_kind !== 'aggregate')}
						<!-- svelte-ignore a11y_click_events_have_key_events -->
						<!-- svelte-ignore a11y_no_static_element_interactions -->
						<div class="ev-expanded" onclick={(ev) => ev.stopPropagation()}>

							<!-- Aggregate rationale (the scorer's prose summary for this evidence) -->
							{#if out.rationale}
								<div class="exp-block">
									<h4 class="exp-h">rationale</h4>
									<p class="exp-rationale">{out.rationale}</p>
								</div>
							{/if}

							<!-- LLM calls: each row is one model invocation; click to expand. -->
							<div class="exp-block">
								<h4 class="exp-h">LLM calls <span class="exp-h-count">({callLog.length})</span>
									<span class="exp-h-note">one row per call sent to the model during this aggregate step · click a row to see the prompt and the model's response</span>
								</h4>
								{#if callLog.length === 0}
									<p class="exp-empty">no LLM calls — every probe was answered by the substrate layer (regex / Gilda)</p>
								{:else}
									<div class="call-list">
										{#each callLog as call, ci}
											{@const callKey = `${e.evidence_hash}-${ci}`}
											{@const isCallOpen = expandedCall === callKey}
											{@const hasContent = typeof call.content === 'string' && call.content.length > 0}
											{@const hasReasoning = typeof call.reasoning === 'string' && call.reasoning.length > 0}
											{@const hasMessages = Array.isArray(call.messages)}
											{@const hasError = typeof call.error === 'string'}
											<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
											<!-- svelte-ignore a11y_click_events_have_key_events -->
											<div class="call-row" class:call-row-open={isCallOpen} class:call-row-error={hasError}
												role="button"
												tabindex="0"
												onclick={(ev) => { ev.stopPropagation(); expandedCall = isCallOpen ? null : callKey; }}>
												<span class="call-chev" aria-hidden="true">{isCallOpen ? '▾' : '▸'}</span>
												<span class="call-kind">{(call.kind as string | undefined) ?? '—'}</span>
												<span class="call-model muted">{(call.model_id as string | undefined) ?? ''}</span>
												<span class="call-duration">{((call.duration_s as number | undefined) ?? 0).toFixed(2)}s</span>
												<span class="call-tokens">{(call.prompt_tokens as number | null | undefined) ?? '—'}→{(call.out_tokens as number | null | undefined) ?? '—'}</span>
												<span class="call-finish muted">{(call.finish_reason as string | null | undefined) ?? '—'}</span>
												<span class="call-flags muted">
													{#if hasReasoning}reasoning {(call.reasoning as string).length}c · {/if}{#if hasContent}content {(call.content as string).length}c{:else}<span class="exp-not-captured-inline">no content captured</span>{/if}
												</span>
											</div>
											{#if isCallOpen}
												<div class="call-detail">
													{#if hasError}
														<div class="call-detail-block">
															<h5 class="call-detail-h">error</h5>
															<pre class="call-detail-pre call-detail-error">{call.error}</pre>
														</div>
													{/if}
													{#if (call.system as string | undefined)}
														<div class="call-detail-block">
															<h5 class="call-detail-h">system prompt</h5>
															<pre class="call-detail-pre">{call.system}</pre>
														</div>
													{/if}
													{#if hasMessages}
														<div class="call-detail-block">
															<h5 class="call-detail-h">messages ({(call.messages as unknown[]).length})</h5>
															{#each (call.messages as Array<{role: string; content: string}>) as msg}
																<div class="call-msg">
																	<span class="call-msg-role">{msg.role}</span>
																	<pre class="call-detail-pre call-detail-msg">{msg.content}</pre>
																</div>
															{/each}
														</div>
													{:else if !hasError}
														<div class="call-detail-block">
															<h5 class="call-detail-h">prompt</h5>
															<p class="exp-not-captured">not captured — pre-Layer-B runs don't persist input messages. Re-score to capture.</p>
														</div>
													{/if}
													{#if hasReasoning}
														<div class="call-detail-block">
															<h5 class="call-detail-h">reasoning <span class="muted">({(call.reasoning as string).length} chars)</span></h5>
															<pre class="call-detail-pre call-detail-reasoning">{call.reasoning}</pre>
														</div>
													{/if}
													{#if hasContent}
														<div class="call-detail-block">
															<h5 class="call-detail-h">response</h5>
															<pre class="call-detail-pre call-detail-content">{call.content}</pre>
														</div>
													{:else if !hasError}
														<div class="call-detail-block">
															<h5 class="call-detail-h">response</h5>
															<p class="exp-not-captured">not captured — pre-Layer-B runs don't persist response text. Re-score to capture.</p>
														</div>
													{/if}
												</div>
											{/if}
										{/each}
									</div>
								{/if}
							</div>

							<!-- Per-step records: one row per scorer_step DB row for this evidence -->
							{#if evSteps.length > 0}
								<div class="exp-block">
									<h4 class="exp-h">pipeline steps <span class="exp-h-count">({evSteps.length})</span>
										<span class="exp-h-note">one row per <code>scorer_step</code> persisted for this evidence (parse, probes, grounding, ...)</span>
									</h4>
									<table class="trace-table">
										<thead>
											<tr>
												<th>step</th>
												<th>source</th>
												<th>output</th>
												<th class="num">in→out tok</th>
												<th class="num">latency</th>
											</tr>
										</thead>
										<tbody>
											{#each evSteps as step}
												{@const stepOut = safeParseJSON(step.output_json)}
												{@const hasInputPayload = step.input_payload_json != null && step.input_payload_json !== 'null'}
												<tr>
													<td>{step.step_kind}</td>
													<td>
														{#if step.is_substrate_answered === true}<span class="step-source-tag step-source-substrate">substrate</span>
														{:else if step.is_substrate_answered === false}<span class="step-source-tag step-source-llm">LLM</span>
														{:else}<span class="muted">—</span>{/if}
													</td>
													<td class="step-output">
														{#if !stepOut}
															<span class="muted">—</span>
														{:else}
															{#if stepOut.answer != null}<span class="step-answer">{stepOut.answer}</span>{/if}
															{#if stepOut.confidence != null}<span class="muted">· {stepOut.confidence}</span>{/if}
															{#if stepOut.rationale}<span class="step-rationale">— {stepOut.rationale}</span>{/if}
															{#if stepOut.stmt_type != null}<span>{stepOut.stmt_type}</span>{/if}
															{#if stepOut.n_aliases != null}<span class="muted">{stepOut.n_aliases} aliases, {stepOut.n_detected_relations ?? 0} relations</span>{/if}
															{#if stepOut.span}<span class="step-span">“{stepOut.span}”</span>{/if}
														{/if}
													</td>
													<td class="num">
														{#if step.prompt_tokens != null || step.out_tokens != null}{step.prompt_tokens ?? '—'}→{step.out_tokens ?? '—'}
														{:else}<span class="muted">—</span>{/if}
													</td>
													<td class="num">
														{#if step.latency_ms != null && step.latency_ms > 0}{step.latency_ms}ms
														{:else}<span class="muted">—</span>{/if}
													</td>
												</tr>
												{#if showDebug}
													<tr class="exp-debug-row">
														<td colspan="5" class="exp-debug-cell">
															<span class="exp-debug-label">step_hash</span>
															<code>{step.step_hash}</code>
															<span class="exp-debug-label">input_payload</span>
															<span>{hasInputPayload ? 'present' : 'not captured'}</span>
															<span class="exp-debug-label">model</span>
															<span>{step.model_id ?? '—'}</span>
															{#if step.error}<span class="exp-debug-label">error</span><span class="exp-error-inline">{step.error}</span>{/if}
														</td>
													</tr>
												{/if}
											{/each}
										</tbody>
									</table>
								</div>
							{/if}

							{#if out.error}
								<div class="exp-block exp-error-block">
									<h4 class="exp-h">aggregate error</h4>
									<p class="exp-error">{out.error}</p>
								</div>
							{/if}

							<!-- Developer detail: hashes, model id, raw payload pointers -->
							<div class="exp-block">
								<label class="exp-debug-toggle">
									<input type="checkbox" bind:checked={showDebug} onclick={(ev) => ev.stopPropagation()}/>
									<span>show developer detail (step_hash, model_id, payload state)</span>
								</label>
								{#if showDebug}
									<div class="exp-debug-block">
										<span class="exp-debug-label">aggregate step_hash</span>
										<button type="button" class="hash-copy"
											onclick={(ev) => {
												ev.stopPropagation();
												navigator.clipboard?.writeText(score?.step_hash ?? '').then(() => {
													copiedHash = score?.step_hash ?? null;
													setTimeout(() => { copiedHash = null; }, 1200);
												});
											}}
											title="Copy step_hash to clipboard">
											<code>{score?.step_hash}</code>
											<span class="copy-mark">{copiedHash === score?.step_hash ? '✓ copied' : '⎘'}</span>
										</button>
									</div>
								{/if}
							</div>
						</div>
					{/if}
				</article>
			{/each}
		</div>

		<!-- Right column: truth panel + agents + supports -->
		<aside class="truth-panel">
			<h2>what other systems say about this</h2>
			<ul class="truth-list">
				{#each presentSets as tsetId}
					{@const labels = d.truth_labels.filter((l) => l.truth_set_id === tsetId)}
					{@const isActive = hoveredEvidenceHash != null && labels.some((l) =>
						(l.target_kind === 'evidence' && l.target_id === hoveredEvidenceHash) ||
						l.target_kind === 'stmt' ||
						l.target_kind === 'agent'
					)}
					{@const isPassive = hoveredEvidenceHash != null && !isActive}
					<li class="truth-row" class:truth-active={isActive} class:truth-passive={isPassive}>
						<div class="truth-row-head">
							<span class="truth-name"><code>{tsetId}</code></span>
							<span class="truth-count">{labels.length} label{labels.length === 1 ? '' : 's'}</span>
						</div>
						<p class="truth-desc">{truthSetDescription(tsetId)}</p>
					</li>
				{/each}
				{#if absentSets.length > 0}
					<li class="truth-row truth-row-absent" title={absentSets.join(', ')}>
						<div class="truth-row-head">
							<span class="truth-name muted">no labels yet from</span>
							<span class="truth-count muted">{absentSets.length} set{absentSets.length === 1 ? '' : 's'}</span>
						</div>
						<p class="truth-desc muted">{absentSets.map(shortTruthLabel).join(' · ')}</p>
					</li>
				{/if}
			</ul>

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
		--ink-faint: #727272;
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


	.evidence {
		padding: 0.7rem 0;
		border-bottom: 1px dotted var(--rule);
	}

	.evidence:last-child {
		border-bottom: none;
	}

	/* Trace narrative — sentence-flow replacement of the old 9-dot rail */
	.trace-narrative {
		list-style: none;
		counter-reset: trace;
		padding: 0;
		margin: 0 0 1.6rem;
	}
	.trace-line {
		display: grid;
		grid-template-columns: 1.6rem 1fr;
		gap: 0.5rem;
		align-items: baseline;
		padding: 0.3rem 0;
		font-family: var(--serif);
		font-size: 0.98rem;
		line-height: 1.45;
	}
	.trace-num {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-faint);
		text-align: right;
	}
	.trace-prose {
		color: var(--ink);
	}
	.trace-line.trace-muted .trace-prose {
		color: var(--ink-faint);
		font-style: italic;
	}

	/* Verdict line — top of each evidence card */
	.ev-verdict-line {
		display: flex;
		gap: 0.8rem;
		align-items: baseline;
		flex-wrap: wrap;
		font-family: var(--mono);
		font-size: 0.92rem;
		margin: 0.4rem 0 0.2rem;
	}
	.ev-verdict {
		font-weight: 500;
	}
	.ev-verdict-correct { color: var(--ok-green); }
	.ev-verdict-incorrect { color: var(--accent); }
	.ev-verdict-abstain { color: var(--ink-muted); font-style: italic; }
	.ev-confidence { color: var(--ink); }
	.ev-score { color: var(--ink); }
	.ev-score-num {
		font-variant-numeric: tabular-nums;
		font-weight: 500;
	}
	.ev-chevron {
		margin-left: auto;
		color: var(--ink-faint);
		font-family: var(--mono);
		font-size: 0.72rem;
	}
	.ev-clickable:hover .ev-chevron {
		color: var(--accent);
	}
	.evidence.ev-expanded-state .ev-chevron {
		color: var(--accent);
	}

	.ev-meta-secondary {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
		display: flex;
		gap: 0.8rem;
		margin-bottom: 0.5rem;
	}
	.ev-source { color: var(--accent); }
	.ev-pmid { color: var(--ink-muted); }
	.ev-hash { color: var(--ink-faint); }
	.latency { color: var(--ink-faint); }

	.ev-text {
		font-family: var(--serif);
		font-size: 1.04rem;
		line-height: 1.5;
		margin: 0.2rem 0 0.4rem;
		color: var(--ink);
	}
	.ev-text-cue {
		background: var(--accent-wash);
		color: var(--accent);
		padding: 0 0.15em;
		font-style: italic;
		font-weight: 500;
		border-bottom: 1px solid var(--accent);
	}

	.ev-flags {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.86rem;
		color: var(--ink-muted);
		margin: 0.2rem 0;
	}
	.ev-flags-prose { color: var(--ink-muted); }

	.ev-section-note {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.84rem;
		color: var(--ink-muted);
		margin: 0 0 1rem;
		line-height: 1.5;
		max-width: 65ch;
	}
	.ev-bucket-table {
		display: block;
		font-family: var(--mono);
		font-style: normal;
		font-size: 0.74rem;
		color: var(--ink-faint);
		margin-top: 0.3rem;
		line-height: 1.6;
	}
	.ev-bucket-table code {
		color: var(--ink);
		font-variant-numeric: tabular-nums;
	}
	.truth-stamp {
		color: var(--accent);
		font-family: var(--mono);
		font-style: normal;
		font-size: 0.74rem;
		margin-left: 0.4rem;
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

	.truth-row {
		display: block;
		padding: 0.5rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.truth-row:last-child {
		border-bottom: none;
	}
	.truth-row-head {
		display: flex;
		justify-content: space-between;
		align-items: baseline;
		gap: 0.6rem;
		font-family: var(--mono);
		font-size: 0.82rem;
	}
	.truth-name {
		color: var(--ink);
	}
	.truth-count {
		color: var(--ink);
		font-variant-numeric: tabular-nums;
		font-size: 0.76rem;
	}
	.truth-desc {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.86rem;
		color: var(--ink-muted);
		margin: 0.2rem 0 0;
		line-height: 1.4;
	}
	.truth-row-absent .truth-name,
	.truth-row-absent .truth-count {
		color: var(--ink-faint);
	}
	.truth-row.truth-active .truth-name {
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
		margin-top: 0.6rem;
		padding: 0.6rem 0.8rem;
		background: rgba(0, 0, 0, 0.02);
		border-left: 2px solid var(--rule);
		font-family: var(--mono);
		font-size: 0.78rem;
	}

	/* Each expanded section: rationale, LLM calls, pipeline steps, debug */
	.exp-block {
		margin-bottom: 0.9rem;
	}
	.exp-block:last-child {
		margin-bottom: 0;
	}
	.exp-h {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.04em;
		margin: 0 0 0.4rem;
		font-weight: 500;
	}
	.exp-h-count {
		color: var(--ink-faint);
		font-weight: 400;
		margin-left: 0.2rem;
	}
	.exp-h-note {
		display: block;
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.78rem;
		color: var(--ink-faint);
		text-transform: none;
		letter-spacing: 0;
		margin-top: 0.2rem;
	}
	.exp-rationale {
		font-family: var(--serif);
		font-size: 0.92rem;
		color: var(--ink);
		margin: 0;
		line-height: 1.45;
	}
	.exp-empty {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.86rem;
		color: var(--ink-muted);
		margin: 0;
	}
	.exp-not-captured {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.78rem;
		color: var(--ink-faint);
		margin: 0.4rem 0 0;
	}
	.exp-error,
	.exp-error-block .exp-error {
		color: var(--accent);
		font-family: var(--serif);
		font-size: 0.9rem;
		margin: 0;
	}
	.exp-error-inline {
		color: var(--accent);
	}

	/* Tables inside the expansion (LLM calls + pipeline steps) */
	.trace-table {
		width: 100%;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.78rem;
	}
	.trace-table th {
		text-align: left;
		font-weight: 500;
		color: var(--ink-muted);
		font-size: 0.7rem;
		padding: 0.2rem 0.6rem 0.2rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.trace-table td {
		padding: 0.25rem 0.6rem 0.25rem 0;
		vertical-align: baseline;
		border-bottom: 1px dotted var(--rule);
	}
	.trace-table .num {
		text-align: right;
		font-variant-numeric: tabular-nums;
	}

	.step-output {
		font-family: var(--mono);
		font-size: 0.78rem;
		display: flex;
		flex-wrap: wrap;
		gap: 0.35rem;
		align-items: baseline;
	}
	.step-answer { color: var(--ink); font-weight: 500; }
	.step-rationale { color: var(--ink-muted); font-family: var(--serif); font-style: italic; font-size: 0.86rem; flex-basis: 100%; }
	.step-span { color: var(--ink-muted); font-style: italic; }
	.step-source-tag {
		font-family: var(--mono);
		font-size: 0.66rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		padding: 0 0.3rem;
	}
	.step-source-substrate { color: var(--accent); }
	.step-source-llm { color: var(--ok-green); }

	/* LLM call expandable rows */
	.call-list {
		display: flex;
		flex-direction: column;
		gap: 0;
		font-family: var(--mono);
		font-size: 0.78rem;
	}
	.call-row {
		display: grid;
		grid-template-columns: 1.5rem minmax(0, 1fr) minmax(0, 1.2fr) 4ch 7ch 6ch minmax(0, 1.4fr);
		gap: 0.6rem;
		align-items: baseline;
		padding: 0.3rem 0.4rem;
		border-bottom: 1px dotted var(--rule);
		cursor: pointer;
		font-variant-numeric: tabular-nums;
	}
	.call-row:hover {
		background: var(--accent-wash);
	}
	.call-row-open {
		background: var(--accent-wash);
	}
	.call-row-error {
		color: var(--accent);
	}
	.call-chev {
		color: var(--ink-faint);
	}
	.call-kind {
		color: var(--ink);
		font-weight: 500;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.call-model {
		font-size: 0.7rem;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}
	.call-duration, .call-tokens, .call-finish {
		color: var(--ink);
		font-size: 0.74rem;
	}
	.call-flags {
		font-size: 0.7rem;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.exp-not-captured-inline {
		font-style: italic;
		color: var(--ink-faint);
	}

	.call-detail {
		padding: 0.6rem 0.8rem;
		margin-bottom: 0.4rem;
		border-left: 2px solid var(--accent);
		background: rgba(0, 0, 0, 0.015);
	}
	.call-detail-block {
		margin-bottom: 0.8rem;
	}
	.call-detail-block:last-child {
		margin-bottom: 0;
	}
	.call-detail-h {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.04em;
		margin: 0 0 0.3rem;
		font-weight: 500;
	}
	.call-detail-pre {
		font-family: var(--mono);
		font-size: 0.78rem;
		background: var(--paper);
		border: 1px solid var(--rule);
		padding: 0.5rem 0.7rem;
		margin: 0;
		white-space: pre-wrap;
		word-break: break-word;
		max-height: 320px;
		overflow-y: auto;
		line-height: 1.45;
		color: var(--ink);
	}
	.call-detail-error {
		color: var(--accent);
	}
	.call-detail-reasoning {
		font-style: italic;
		color: var(--ink-muted);
	}
	.call-detail-content {
		color: var(--ink);
	}
	.call-msg {
		margin-bottom: 0.4rem;
	}
	.call-msg-role {
		display: inline-block;
		font-family: var(--mono);
		font-size: 0.66rem;
		text-transform: uppercase;
		letter-spacing: 0.04em;
		color: var(--ink-muted);
		margin-bottom: 0.15rem;
	}
	.call-detail-msg {
		margin-top: 0.1rem;
	}

	/* Developer detail block — collapsed by default behind a checkbox */
	.exp-debug-toggle {
		display: inline-flex;
		gap: 0.4rem;
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-muted);
		cursor: pointer;
	}
	.exp-debug-block,
	.exp-debug-row {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
	}
	.exp-debug-block {
		padding: 0.4rem 0 0;
		display: flex;
		gap: 0.6rem;
		flex-wrap: wrap;
		align-items: baseline;
	}
	.exp-debug-cell {
		padding-left: 0;
	}
	.exp-debug-cell .exp-debug-label {
		margin-left: 1rem;
	}
	.exp-debug-cell .exp-debug-label:first-child {
		margin-left: 0;
	}
	.exp-debug-label {
		color: var(--ink-faint);
		text-transform: lowercase;
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
