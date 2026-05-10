<script lang="ts">
	import type { ProbeAttribution } from '$lib/probeAttribution';
	import { beliefSemantic, fmtBelief, fmtDelta, pluralS, sentenceFromStatement, shortHash, type SentenceAgent } from '$lib/format';

	export interface BeliefPrimitiveProps {
		stmt: {
			stmt_hash: string;
			indra_type: string;
			agents: SentenceAgent[];
		};
		our_score: number | null;
		indra_score: number | null;
		probes?: ProbeAttribution[];
		evidences?: Array<{
			evidence_hash: string;
			source_api: string | null;
			text: string | null;
		}>;
		why_this_one?: string;
		mode?: 'full' | 'compact';
		/** When true, the card is clickable and behaves as <a href={href}>. */
		href?: string;
	}

	let {
		stmt,
		our_score,
		indra_score,
		probes = [],
		evidences = [],
		why_this_one,
		mode = 'full',
		href
	}: BeliefPrimitiveProps = $props();

	const sentence = $derived(sentenceFromStatement(stmt.indra_type, stmt.agents));
	const delta = $derived(
		our_score != null && indra_score != null ? our_score - indra_score : null
	);
	const semantic = $derived(beliefSemantic(our_score));
	const probeOrder = $derived(
		(['subject_role', 'object_role', 'relation_axis', 'scope'] as const).map((kind) =>
			probes.find((p) => p.probe === kind) ?? null
		)
	);
	const evShown = $derived(evidences.slice(0, 3));
	const evMore = $derived(Math.max(0, evidences.length - evShown.length));

	function probeLabel(kind: string): string {
		return ({
			subject_role: 'subj-role',
			object_role: 'obj-role',
			relation_axis: 'relation',
			scope: 'scope'
		})[kind] ?? kind;
	}

	function probeBar(a: ProbeAttribution): string {
		const filled = Math.round(a.normalized_width * 10);
		const empty = 10 - filled;
		const ch = a.contribution >= 0 ? '█' : '▓';
		return ch.repeat(filled) + '░'.repeat(empty);
	}

	function compactProbeCell(a: ProbeAttribution | null): string {
		if (!a) return '·';
		const c = a.contribution;
		if (c > 0.6) return '█';
		if (c > 0.2) return '▆';
		if (c > -0.2) return '▃';
		if (c > -0.6) return '▂';
		return '▁';
	}
</script>

{#if mode === 'compact'}
	{#snippet compactBody()}
		<span class="b-delta b-delta-{delta != null ? (delta >= 0 ? 'pos' : 'neg') : 'na'}">
			{delta == null ? '—' : `Δ${fmtDelta(delta)}`}
		</span>
		<span class="b-sentence">{sentence}</span>
		{#if probes.length > 0}
			<span class="b-probes" title={probeOrder.map((p, i) => `${['subj','obj','rel','scope'][i]}: ${p?.answer ?? '—'} (${p?.contribution.toFixed(2) ?? '—'})`).join(' · ')}>
				{#each probeOrder as p, i}<span class="b-pcell b-{p == null ? 'na' : (p.contribution >= 0 ? 'pos' : 'neg')}">{compactProbeCell(p)}</span>{/each}
			</span>
		{/if}
		<span class="b-num-pair">
			<span class="b-score-compact b-{semantic}">{fmtBelief(our_score)}</span>
			<span class="b-num-sep">/</span>
			<span class="b-num-mid-compact">{fmtBelief(indra_score)}</span>
		</span>
		{#if evidences.length > 0}
			<span class="b-ev-count" title="number of evidences for this statement">n<sub>ev</sub>={evidences.length}</span>
		{/if}
		<span class="b-hash">{shortHash(stmt.stmt_hash)}</span>
		{#if why_this_one}
			<span class="b-why-compact">{why_this_one}</span>
		{/if}
	{/snippet}

	{#if href}
		<a class="b-card b-card-compact" {href}>{@render compactBody()}</a>
	{:else}
		<div class="b-card b-card-compact">{@render compactBody()}</div>
	{/if}
{:else}
	<article class="b-card b-card-full">
		<h3 class="b-sentence b-sentence-full">{sentence}</h3>
		<div class="b-meta">
			<span class="b-type">{stmt.indra_type}</span>
			<span class="b-hash">{shortHash(stmt.stmt_hash)}</span>
		</div>

		<div class="b-grid">
			<div class="b-numbers">
				{#if delta != null}
					<div class="b-row">
						<dt>Δ vs INDRA</dt>
						<dd class="b-score-big b-delta-{delta >= 0 ? 'pos' : 'neg'}">{fmtDelta(delta)}</dd>
					</div>
				{:else}
					<div class="b-row">
						<dt>belief</dt>
						<dd class="b-score-big b-{semantic}">{fmtBelief(our_score)}</dd>
					</div>
				{/if}
				<div class="b-row b-row-secondary">
					<dt>belief</dt>
					<dd class="b-num-mid b-{semantic}">{fmtBelief(our_score)}</dd>
				</div>
				<div class="b-row b-row-secondary">
					<dt>indra</dt>
					<dd class="b-num-mid">{fmtBelief(indra_score)}</dd>
				</div>
			</div>

			<div class="b-probes-full">
				{#each probeOrder as p, i}
					{#if p == null}
						<div class="b-prow b-prow-absent">
							<span class="b-plabel">{probeLabel(['subject_role', 'object_role', 'relation_axis', 'scope'][i])}</span>
							<span class="b-pbar">·</span>
							<span class="b-pval">—</span>
						</div>
					{:else}
						<div class="b-prow" class:b-prow-decisive={p.decisive} title={p.rationale ?? ''}>
							<span class="b-plabel">{probeLabel(p.probe)}</span>
							<span class="b-pbar b-pbar-{p.contribution >= 0 ? 'pos' : 'neg'}">{probeBar(p)}</span>
							<span class="b-pval b-pval-{p.contribution >= 0 ? 'pos' : 'neg'}">{p.contribution >= 0 ? '+' : '−'}{Math.abs(p.contribution).toFixed(2)}</span>
							<span class="b-panswer">{p.answer ?? '—'}</span>
							<span class="b-pconf">·{p.confidence ?? '—'}{p.source === 'substrate' ? ' (substrate)' : p.source === 'llm' ? ' (llm)' : ''}</span>
						</div>
					{/if}
				{/each}
			</div>
		</div>

		{#if evidences.length > 0}
			<div class="b-ev">
				<div class="b-ev-head">
					evidence ({evidences.length}){#if evMore > 0}, showing {evShown.length}{/if}
				</div>
				{#each evShown as e}
					<div class="b-ev-row">
						{#if e.source_api}<span class="b-ev-src">[{e.source_api}]</span>{/if}
						<span class="b-ev-text">{e.text ?? '(no text)'}</span>
					</div>
				{/each}
			</div>
		{/if}

		{#if why_this_one}
			<div class="b-why">why this one: {why_this_one}</div>
		{/if}
	</article>
{/if}

<style>
	.b-card {
		font-family: var(--serif);
	}
	.b-card-full {
		padding: 1.2rem 1.4rem;
		border-left: 3px solid var(--accent);
		background: transparent;
		margin: 0 0 1.6rem;
	}
	.b-card-compact {
		display: grid;
		grid-template-columns: 6ch 1fr auto auto auto 9ch auto;
		gap: 0.8rem;
		align-items: baseline;
		padding: 0.35rem 0.4rem;
		border-bottom: 1px dotted var(--rule);
		font-size: 0.84rem;
		text-decoration: none;
		color: var(--ink);
	}
	a.b-card-compact:hover {
		background: var(--accent-wash);
	}

	.b-sentence {
		font-family: var(--serif);
		font-style: italic;
		color: var(--ink);
	}
	.b-sentence-full {
		font-size: 1.25rem;
		font-weight: 400;
		margin: 0 0 0.2rem;
		line-height: 1.3;
	}
	.b-meta {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
		display: flex;
		gap: 0.8rem;
		margin-bottom: 1rem;
	}
	.b-type {
		text-transform: lowercase;
		letter-spacing: 0.04em;
	}

	.b-grid {
		display: grid;
		grid-template-columns: 9rem 1fr;
		gap: 1.4rem;
		align-items: start;
	}
	.b-numbers {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
		font-family: var(--mono);
	}
	.b-row {
		display: grid;
		grid-template-columns: 4rem 1fr;
		align-items: baseline;
	}
	.b-row dt {
		font-size: 0.72rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
	}
	.b-row dd {
		margin: 0;
		font-variant-numeric: tabular-nums;
	}
	.b-row-secondary dt {
		font-size: 0.66rem;
	}
	.b-score-big {
		font-family: var(--mono);
		font-size: 1.8rem;
		line-height: 1;
		font-weight: 500;
		font-variant-numeric: tabular-nums;
	}
	.b-num-mid {
		font-family: var(--mono);
		font-size: 0.86rem;
		color: var(--ink-muted);
	}
	.b-high { color: #2a6f2a; }
	.b-low { color: var(--accent); }
	.b-mid { color: var(--ink); }
	.b-absent { color: var(--ink-faint); }

	.b-delta-pos { color: #2a6f2a; }
	.b-delta-neg { color: var(--accent); }
	.b-delta-na { color: var(--ink-faint); }

	.b-probes-full {
		font-family: var(--mono);
		font-size: 0.78rem;
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
	}
	.b-prow {
		display: grid;
		grid-template-columns: 5rem 11ch 5ch auto 1fr auto;
		gap: 0.6rem;
		align-items: baseline;
	}
	.b-prow-absent {
		color: var(--ink-faint);
	}
	.b-prow-decisive {
		background: var(--accent-wash);
		border-left: 2px solid var(--accent);
		padding-left: 0.3rem;
		margin-left: -0.5rem;
	}
	.b-plabel {
		color: var(--ink-muted);
	}
	.b-pbar {
		letter-spacing: -0.05em;
		font-variant-numeric: tabular-nums;
	}
	.b-pbar-pos { color: #2a6f2a; }
	.b-pbar-neg { color: var(--accent); }
	.b-pval {
		text-align: right;
		font-variant-numeric: tabular-nums;
	}
	.b-pval-pos { color: #2a6f2a; }
	.b-pval-neg { color: var(--accent); }
	.b-panswer {
		color: var(--ink);
		font-size: 0.74rem;
	}
	.b-pconf {
		color: var(--ink-faint);
		font-size: 0.72rem;
	}

	.b-ev {
		margin-top: 1.2rem;
		padding-top: 0.8rem;
		border-top: 1px dotted var(--rule);
	}
	.b-ev-head {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		margin-bottom: 0.4rem;
	}
	.b-ev-row {
		display: grid;
		grid-template-columns: auto 1fr;
		gap: 0.6rem;
		align-items: baseline;
		margin: 0.15rem 0;
	}
	.b-ev-src {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
	}
	.b-ev-text {
		font-family: var(--serif);
		font-style: italic;
		color: var(--ink);
		line-height: 1.4;
	}

	.b-why {
		margin-top: 0.8rem;
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
	}

	/* Compact-mode cells */
	.b-card-compact .b-sentence {
		font-style: italic;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}
	.b-card-compact .b-delta {
		font-family: var(--mono);
		font-size: 0.92rem;
		font-weight: 500;
		font-variant-numeric: tabular-nums;
		text-align: right;
	}
	.b-num-pair {
		font-family: var(--mono);
		font-size: 0.78rem;
		font-variant-numeric: tabular-nums;
		text-align: right;
		color: var(--ink-muted);
	}
	.b-score-compact { color: var(--ink); font-weight: 500; }
	.b-num-sep { color: var(--ink-faint); margin: 0 0.1em; }
	.b-num-mid-compact { color: var(--ink-faint); }
	.b-why-compact {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
		grid-column: 2 / -1;
		font-style: italic;
	}
	.b-probes {
		font-family: var(--mono);
		font-size: 0.92rem;
		letter-spacing: -0.05em;
		display: inline-flex;
		gap: 0.05rem;
	}
	.b-pcell.b-pos { color: #2a6f2a; }
	.b-pcell.b-neg { color: var(--accent); }
	.b-pcell.b-na { color: var(--ink-faint); }
	.b-ev-count {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-muted);
	}
	.b-hash {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
	}
</style>
