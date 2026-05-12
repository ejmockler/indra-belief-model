<script lang="ts">
	import type { HeuristicCoverage, ProbeCoverageRow } from '$lib/db';

	let { coverage }: { coverage: HeuristicCoverage } = $props();

	const totalProbeSlots = $derived(coverage.per_probe.reduce((s, p) => s + p.total, 0));
	const totalSubstrate = $derived(coverage.per_probe.reduce((s, p) => s + p.substrate_n, 0));
	const totalLlm = $derived(coverage.per_probe.reduce((s, p) => s + p.llm_n, 0));
	const totalAbstain = $derived(coverage.per_probe.reduce((s, p) => s + p.abstain_n, 0));
	const totalNotrun = $derived(coverage.per_probe.reduce((s, p) => s + p.notrun_n, 0));

	function pctOf(n: number, total: number): number {
		return total > 0 ? (n / total) * 100 : 0;
	}

	function pctStr(p: number): string {
		if (p === 0) return '0%';
		if (p < 1) return '<1%';
		if (p > 99 && p < 100) return '>99%';
		return `${Math.round(p)}%`;
	}

	function probeLabel(p: string): string {
		return (
			{
				subject_role: 'subj-role',
				object_role: 'obj-role',
				relation_axis: 'relation-axis',
				scope: 'scope'
			} as Record<string, string>
		)[p] ?? p;
	}

	const summary = $derived.by(() => {
		const slots = totalProbeSlots;
		if (slots === 0) return 'no probes recorded for this run';
		const sPct = pctStr(pctOf(totalSubstrate, slots));
		const lPct = pctStr(pctOf(totalLlm, slots));
		const nPct = pctStr(pctOf(totalNotrun, slots));
		const llmEvidence =
			totalLlm > 0
				? `LLM fired for ${lPct} of probe slots`
				: `the LLM did not fire in this run`;
		return `Substrate (regex / Gilda / catalog) resolved ${sPct} of probe slots; ${llmEvidence}; ${nPct} short-circuited (probe didn't run for that evidence).`;
	});

	function probeOrder(p: ProbeCoverageRow): number {
		return (
			{ subject_role: 0, object_role: 1, relation_axis: 2, scope: 3 } as Record<string, number>
		)[p.probe] ?? 99;
	}
	const orderedProbes = $derived([...coverage.per_probe].sort((a, b) => probeOrder(a) - probeOrder(b)));
</script>

<section class="coverage">
	<h2 class="cov-h">
		what the system is doing in this run
		<span class="cov-run-id" title="run_id">{coverage.run_id.slice(0, 8)}</span>
	</h2>

	<p class="cov-summary">{summary}</p>

	{#if coverage.n_evidences > 0}
		<table class="cov-table" aria-label="per-probe substrate / LLM / abstain / not-run breakdown">
			<thead>
				<tr>
					<th>probe</th>
					<th class="cov-bar-head">coverage</th>
					<th class="num">substrate</th>
					<th class="num">llm</th>
					<th class="num">abstain</th>
					<th class="num">not run</th>
				</tr>
			</thead>
			<tbody>
				{#each orderedProbes as p}
					{@const sP = pctOf(p.substrate_n, p.total)}
					{@const lP = pctOf(p.llm_n, p.total)}
					{@const aP = pctOf(p.abstain_n, p.total)}
					{@const nP = pctOf(p.notrun_n, p.total)}
					<tr>
						<td class="cov-name">{probeLabel(p.probe)}</td>
						<td class="cov-bar-cell">
							<div class="cov-bar" role="img" aria-label="{probeLabel(p.probe)} coverage: substrate {sP.toFixed(0)}%, llm {lP.toFixed(0)}%, abstain {aP.toFixed(0)}%, not run {nP.toFixed(0)}%">
								{#if sP > 0}<span class="cov-seg cov-seg-substrate" style:width="{sP}%" title="substrate {p.substrate_n}/{p.total}"></span>{/if}
								{#if lP > 0}<span class="cov-seg cov-seg-llm" style:width="{lP}%" title="llm {p.llm_n}/{p.total}"></span>{/if}
								{#if aP > 0}<span class="cov-seg cov-seg-abstain" style:width="{aP}%" title="abstain {p.abstain_n}/{p.total}"></span>{/if}
								{#if nP > 0}<span class="cov-seg cov-seg-notrun" style:width="{nP}%" title="not run {p.notrun_n}/{p.total}"></span>{/if}
							</div>
						</td>
						<td class="num cov-c-substrate">{pctStr(sP)}</td>
						<td class="num cov-c-llm">{pctStr(lP)}</td>
						<td class="num cov-c-abstain">{pctStr(aP)}</td>
						<td class="num cov-c-notrun">{pctStr(nP)}</td>
					</tr>
				{/each}
			</tbody>
		</table>

		<div class="cov-aux">
			<span class="cov-aux-row">
				<span class="cov-aux-label">all-substrate evidences</span>
				<span class="cov-aux-val">{pctStr(coverage.all_substrate_rate * 100)}</span>
				<span class="muted">— evidences where every probe was substrate-resolved (zero LLM calls)</span>
			</span>
			<span class="cov-aux-row">
				<span class="cov-aux-label">short-circuited evidences</span>
				<span class="cov-aux-val">{pctStr(coverage.short_circuited_rate * 100)}</span>
				<span class="muted">— at least one probe did not fire (earlier probe decided the outcome)</span>
			</span>
			<span class="cov-aux-row">
				<span class="cov-aux-label">mean probes invoked per evidence</span>
				<span class="cov-aux-val">{coverage.mean_invoked_probes.toFixed(2)}</span>
				<span class="muted">of 4</span>
			</span>
		</div>
	{:else}
		<p class="cov-empty">no probe records persisted for this run yet</p>
	{/if}
</section>

<style>
	.coverage {
		margin: 0 0 2.5rem;
	}
	.cov-h {
		font-family: var(--serif);
		font-size: 1.15rem;
		font-weight: 400;
		color: var(--ink);
		margin: 0 0 0.6rem;
		display: flex;
		align-items: baseline;
		gap: 0.6rem;
	}
	.cov-run-id {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
	}
	.cov-summary {
		font-family: var(--serif);
		font-size: 0.98rem;
		color: var(--ink);
		margin: 0 0 1rem;
		line-height: 1.5;
		font-variant-numeric: tabular-nums;
	}
	.cov-table {
		width: 100%;
		max-width: 720px;
		border-collapse: collapse;
		font-family: var(--mono);
		font-size: 0.82rem;
		font-variant-numeric: tabular-nums;
	}
	.cov-table th {
		text-align: left;
		font-weight: 500;
		color: var(--ink-muted);
		font-size: 0.72rem;
		text-transform: lowercase;
		letter-spacing: 0.02em;
		padding: 0.3rem 0.6rem 0.3rem 0;
		border-bottom: 1px dotted var(--rule);
	}
	.cov-table td {
		padding: 0.4rem 0.6rem 0.4rem 0;
		vertical-align: middle;
		border-bottom: 1px dotted var(--rule);
	}
	.cov-table tr:last-child td {
		border-bottom: none;
	}
	.cov-table th.num,
	.cov-table td.num {
		text-align: right;
	}
	.cov-name {
		color: var(--ink);
	}
	.cov-bar-head {
		min-width: 16rem;
	}
	.cov-bar-cell {
		min-width: 12rem;
		width: 100%;
	}
	.cov-bar {
		display: flex;
		width: 100%;
		min-width: 12rem;
		max-width: 30rem;
		height: 14px;
		border: 1px solid var(--rule);
		overflow: hidden;
	}
	.cov-seg {
		display: block;
		min-width: 1px;
	}
	.cov-seg-substrate { background: var(--ok-green); }
	.cov-seg-llm { background: var(--accent); }
	.cov-seg-abstain { background: var(--ink-muted); }
	.cov-seg-notrun {
		background: repeating-linear-gradient(
			135deg,
			var(--ink-faint),
			var(--ink-faint) 2px,
			var(--paper) 2px,
			var(--paper) 4px
		);
	}
	.cov-c-substrate { color: var(--ok-green); }
	.cov-c-llm { color: var(--accent); }
	.cov-c-abstain { color: var(--ink-muted); }
	.cov-c-notrun { color: var(--ink-faint); }

	.cov-aux {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		margin-top: 1rem;
		font-family: var(--mono);
		font-size: 0.82rem;
	}
	.cov-aux-row {
		display: flex;
		gap: 0.6rem;
		align-items: baseline;
	}
	.cov-aux-label {
		color: var(--ink-muted);
		font-size: 0.74rem;
		min-width: 16rem;
	}
	.cov-aux-val {
		color: var(--ink);
		font-weight: 500;
		font-variant-numeric: tabular-nums;
	}
	.cov-empty {
		font-family: var(--serif);
		font-style: italic;
		color: var(--ink-muted);
		margin: 0;
	}
	.muted { color: var(--ink-faint); }
</style>
