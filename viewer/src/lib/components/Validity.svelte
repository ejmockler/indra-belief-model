<script lang="ts">
	import type { LatestValidity, ResidualDistribution } from '$lib/db';
	import { residualBraille, residualPath } from '$lib/residuals';

	let {
		v,
		residuals
	}: {
		v: LatestValidity;
		residuals: ResidualDistribution | null;
	} = $props();

	const verdictTotal = $derived(v.verdicts.reduce((s, vd) => s + vd.n, 0));
	const verdictPct = (n: number) => (verdictTotal === 0 ? 0 : (n / verdictTotal) * 100);

	function verdictDisplayName(name: string): string {
		return name === 'correct' ? 'supported' : name === 'incorrect' ? 'contradicted' : name;
	}

	/** "Did our system match the hand-labeled gold?" One row per truth_set. */
	const goldLines = $derived(
		v.truthPresent.map((t) => {
			const tp = t.tp;
			const fp = t.fp;
			const fn = t.fn;
			const judged = tp + fp + fn;
			const perfect = t.precision === 1.0 && t.recall === 1.0;
			const setLabel = t.truth_set_id.replace(/^indra_/, '').replace(/^source_db_/, '').replace(/_/g, ' ');
			let headline: string;
			let detail: string;
			if (perfect) {
				headline = `matched ${tp}/${judged}`;
				const abstained = t.n_compared - judged;
				detail = abstained > 0 ? `${abstained} abstain excluded · n_compared=${t.n_compared}` : `n_compared=${t.n_compared}`;
			} else {
				headline = `${tp}/${judged} correct · F1 ${t.f1.toFixed(2)} (P ${t.precision.toFixed(2)} R ${t.recall.toFixed(2)})`;
				detail = `${fp} false-positive · ${fn} false-negative · n_compared=${t.n_compared}`;
			}
			return {
				set_label: setLabel,
				step_kind: t.step_kind,
				perfect,
				headline,
				detail
			};
		})
	);

	/** "How do we compare to INDRA's priors?" */
	const indraLine = $derived.by(() => {
		const c = v.calibration;
		if (c.mae == null) {
			return { glyph: '—', direction: 'na' as const, headline: 'no INDRA-comparable beliefs in this run', detail: '' };
		}
		const bias = c.bias ?? 0;
		const oneSided = c.mae > 0 && Math.abs(Math.abs(bias) - c.mae) < 1e-3;
		const n = c.n_stmts ?? 0;
		if (Math.abs(bias) < 0.05) {
			return {
				glyph: '≈',
				direction: 'eq' as const,
				headline: `closely matched`,
				detail: `MAE ${c.mae.toFixed(2)} · n=${n}`
			};
		}
		const dir = bias > 0 ? 'over' : 'under';
		const arrow = bias > 0 ? '▲' : '▼';
		const oneSidedNote =
			oneSided && c.mae > 0.001
				? `every score landed ${bias < 0 ? 'below' : 'above'} INDRA`
				: `MAE ${c.mae.toFixed(2)}`;
		return {
			glyph: arrow,
			direction: bias > 0 ? ('up' as const) : ('down' as const),
			headline: `${dir}-confident by ${Math.abs(bias).toFixed(2)}`,
			detail: `${oneSidedNote} · n=${n}`
		};
	});

	/** Stratification: only worth surfacing when the slice has > 1 stratum,
	 *  otherwise "top 1" is the whole thing and tells you nothing. */
	const byType = $derived(v.byIndraType.length > 1 ? v.byIndraType.slice(0, 5) : []);
	const bySource = $derived(v.bySourceApi.length > 1 ? v.bySourceApi.slice(0, 5) : []);
	const lonelyType = $derived(v.byIndraType.length === 1 ? v.byIndraType[0] : null);
	const lonelySource = $derived(v.bySourceApi.length === 1 ? v.bySourceApi[0] : null);

	const consistencySentence = $derived.by(() => {
		const c = v.inter_evidence_consistency;
		if (c.mean_stdev == null) return null;
		return {
			headline: `per-evidence scores agreed within stdev ${c.mean_stdev.toFixed(2)}`,
			detail: `${c.n_multi_ev ?? 0} statement${c.n_multi_ev === 1 ? '' : 's'} had multiple evidences`
		};
	});

	/** Show the histogram only when there are enough points for the shape to mean
	 *  something. Below ~30 the bars are misleading — the prose already carries
	 *  the bias direction. */
	const SHOW_HISTOGRAM_THRESHOLD = 30;
	const showHistogram = $derived(residuals != null && residuals.n_total >= SHOW_HISTOGRAM_THRESHOLD);
</script>

<section class="validity">
	<h2 class="v-h">how is the system doing in this run? <span class="v-run-id" title="run_id">{v.run_id.slice(0, 8)}</span></h2>

	{#each goldLines as g}
		<div class="v-line">
			<span class="v-line-label">vs gold <span class="muted">({g.set_label}{g.step_kind === 'aggregate' ? '' : ` / ${g.step_kind}`})</span></span>
			<span class="v-glyph v-glyph-{g.perfect ? 'ok' : 'fail'}">{g.perfect ? '✓' : '✗'}</span>
			<span class="v-headline">{g.headline}</span>
			<span class="muted v-detail">· {g.detail}</span>
		</div>
	{/each}

	<div class="v-line">
		<span class="v-line-label">vs INDRA's priors</span>
		<span class="v-glyph v-glyph-{indraLine.direction}">{indraLine.glyph}</span>
		<span class="v-headline">{indraLine.headline}</span>
		{#if indraLine.detail}<span class="muted v-detail">· {indraLine.detail}</span>{/if}
	</div>

	{#if verdictTotal > 0}
		<div class="v-line v-line-block">
			<span class="v-line-label" title="our scorer's classification of each evidence">per-evidence verdicts</span>
			<div class="v-pillbar-wrap">
				<div class="v-pillbar" role="img" aria-label="verdict distribution">
					{#each v.verdicts as vd}
						{@const pct = verdictPct(vd.n)}
						{#if pct > 0}
							<span class="v-pill v-pill-{vd.verdict}" style:width="{pct}%" title="{verdictDisplayName(vd.verdict)}: {vd.n} of {verdictTotal} ({pct.toFixed(1)}%)"></span>
						{/if}
					{/each}
				</div>
				<div class="v-pill-caption">
					{#each v.verdicts as vd, i}{#if i > 0}<span class="muted"> · </span>{/if}<span class="v-pill-tag v-pill-tag-{vd.verdict}">{verdictDisplayName(vd.verdict)}</span> <span class="v-pill-num">{vd.n}</span>{/each}
					<span class="muted"> · n={verdictTotal}</span>
				</div>
				<p class="v-explain">each evidence sentence is judged separately — a statement with multiple evidences contributes multiple counts</p>
			</div>
		</div>
	{/if}

	{#if byType.length > 0 || bySource.length > 0 || lonelyType || lonelySource}
		<div class="v-line v-line-block">
			<span class="v-line-label">weakest by slice</span>
			<div class="v-weakest-list">
				{#if byType.length > 0}
					<div class="v-weak-group">
						<span class="muted">by indra_type:</span>
						{#each byType as s, i}
							{#if i > 0}<span class="muted">·</span>{/if}
							<span class="v-weak-row">
								<span class="v-weak-name">{s.value}</span>
								<span class="v-weak-mae">MAE {s.mae.toFixed(2)}</span>
								<span class="v-weak-bias" class:b-pos={s.bias >= 0.05} class:b-neg={s.bias <= -0.05}>{s.bias >= 0 ? '▲' : '▼'}{Math.abs(s.bias).toFixed(2)}</span>
								<span class="muted">n={s.n}</span>
							</span>
						{/each}
					</div>
				{:else if lonelyType}
					<div class="v-weak-group">
						<span class="muted">by indra_type: only one type in this run —</span>
						<span class="v-weak-row">
							<span class="v-weak-name">{lonelyType.value}</span>
							<span class="v-weak-mae">MAE {lonelyType.mae.toFixed(2)}</span>
							<span class="muted">n={lonelyType.n}</span>
						</span>
					</div>
				{/if}
				{#if bySource.length > 0}
					<div class="v-weak-group">
						<span class="muted">by source_api:</span>
						{#each bySource as s, i}
							{#if i > 0}<span class="muted">·</span>{/if}
							<span class="v-weak-row">
								<span class="v-weak-name">{s.value}</span>
								<span class="v-weak-mae">MAE {s.mae.toFixed(2)}</span>
								<span class="v-weak-bias" class:b-pos={s.bias >= 0.05} class:b-neg={s.bias <= -0.05}>{s.bias >= 0 ? '▲' : '▼'}{Math.abs(s.bias).toFixed(2)}</span>
								<span class="muted">n={s.n}</span>
							</span>
						{/each}
					</div>
				{:else if lonelySource}
					<div class="v-weak-group">
						<span class="muted">by source_api: only one source in this run —</span>
						<span class="v-weak-row">
							<span class="v-weak-name">{lonelySource.value}</span>
							<span class="v-weak-mae">MAE {lonelySource.mae.toFixed(2)}</span>
							<span class="muted">n={lonelySource.n}</span>
						</span>
					</div>
				{/if}
			</div>
		</div>
	{/if}

	{#if consistencySentence}
		<div class="v-line">
			<span class="v-line-label" title="when a statement has multiple evidences, do their per-evidence scores agree?">multi-evidence agreement</span>
			<span class="v-headline">{consistencySentence.headline}</span>
			<span class="muted v-detail">· {consistencySentence.detail}</span>
		</div>
	{/if}

	{#if v.supports_graph_delta != null}
		<div class="v-line">
			<span class="v-line-label" title="how much our scores stratify across the supports_edge graph">supports-graph plausibility</span>
			<span class="v-headline">Δ {v.supports_graph_delta >= 0 ? '+' : ''}{v.supports_graph_delta.toFixed(2)}</span>
		</div>
	{/if}

	{#if showHistogram}
		<div class="v-residuals">
			<p class="v-residuals-label">residual distribution (our − INDRA), n={residuals!.n_total}</p>
			<svg viewBox="0 0 240 40" preserveAspectRatio="none" class="v-res-svg" aria-label="residual histogram (our − INDRA), n={residuals!.n_total}">
				<line x1="120" y1="0" x2="120" y2="40" stroke="var(--ink-faint)" stroke-width="0.5" stroke-dasharray="2,2"/>
				<path d={residualPath(residuals!.bins, 240, 40)} fill="var(--ink)" opacity="0.85"/>
			</svg>
			<div class="v-res-axis">
				<span>−1</span><span>0</span><span>+1</span>
			</div>
			<span class="v-braille muted" title="braille fallback">{residualBraille(residuals!.bins)}</span>
		</div>
	{/if}
</section>

<style>
	.validity {
		margin: 0 0 2.5rem;
	}
	.v-h {
		font-family: var(--serif);
		font-size: 1.15rem;
		font-weight: 400;
		color: var(--ink);
		margin: 0 0 1rem;
		display: flex;
		align-items: baseline;
		gap: 0.6rem;
	}
	.v-run-id {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
	}

	.v-line {
		display: flex;
		flex-wrap: wrap;
		gap: 0.6rem;
		align-items: baseline;
		padding: 0.4rem 0;
		border-bottom: 1px dotted var(--rule);
		font-family: var(--mono);
		font-size: 0.86rem;
	}
	.v-line-block {
		flex-direction: column;
		align-items: flex-start;
		gap: 0.3rem;
	}
	.v-line:last-child {
		border-bottom: none;
	}

	.v-line-label {
		flex-basis: 14rem;
		flex-shrink: 0;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.02em;
		font-size: 0.78rem;
	}
	/* In block (column-flex) lines, flex-basis: 14rem becomes 14rem of *height*
	   rather than width — which produced a giant vertical gap. Reset to auto. */
	.v-line-block .v-line-label {
		flex-basis: auto;
	}

	.v-glyph {
		font-family: var(--mono);
		font-weight: 500;
		min-width: 1.2rem;
		text-align: center;
	}
	.v-glyph-ok { color: var(--ok-green); }
	.v-glyph-fail { color: var(--accent); }
	.v-glyph-up { color: var(--ok-green); }
	.v-glyph-down { color: var(--accent); }
	.v-glyph-eq { color: var(--ink-muted); }
	.v-glyph-na { color: var(--ink-faint); }

	.v-headline {
		color: var(--ink);
		font-variant-numeric: tabular-nums;
	}
	.v-detail {
		font-size: 0.78rem;
	}

	/* Verdict pillbar */
	.v-pillbar-wrap {
		display: flex;
		flex-direction: column;
		gap: 0.2rem;
		width: 100%;
		max-width: 480px;
	}
	.v-pillbar {
		display: flex;
		width: 100%;
		height: 16px;
		border: 1px solid var(--rule);
		overflow: hidden;
	}
	.v-pill {
		display: block;
		min-width: 2px;
	}
	.v-pill-correct { background: var(--ok-green); }
	.v-pill-incorrect { background: var(--accent); }
	.v-pill-abstain { background: var(--ink-faint); }
	.v-pill-caption {
		font-family: var(--mono);
		font-size: 0.78rem;
		font-variant-numeric: tabular-nums;
	}
	.v-pill-tag-correct { color: var(--ok-green); }
	.v-pill-tag-incorrect { color: var(--accent); }
	.v-pill-tag-abstain { color: var(--ink-muted); }
	.v-pill-num { color: var(--ink); font-weight: 500; }
	.v-explain {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.78rem;
		color: var(--ink-faint);
		margin: 0.2rem 0 0;
	}

	/* Weakest by slice */
	.v-weakest-list {
		display: flex;
		flex-direction: column;
		gap: 0.3rem;
		font-family: var(--mono);
		font-size: 0.8rem;
	}
	.v-weak-group {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
		align-items: baseline;
	}
	.v-weak-row {
		display: inline-flex;
		gap: 0.4rem;
		align-items: baseline;
	}
	.v-weak-name { color: var(--ink); }
	.v-weak-mae { color: var(--ink); font-variant-numeric: tabular-nums; }
	.v-weak-bias { color: var(--ink-muted); font-variant-numeric: tabular-nums; }
	.v-weak-bias.b-pos { color: var(--ok-green); }
	.v-weak-bias.b-neg { color: var(--accent); }

	/* Residual histogram (only at n ≥ 30) */
	.v-residuals {
		margin-top: 1rem;
		padding-top: 0.6rem;
		border-top: 1px dotted var(--rule);
		max-width: 280px;
	}
	.v-residuals-label {
		font-family: var(--mono);
		font-size: 0.74rem;
		color: var(--ink-muted);
		margin: 0 0 0.2rem;
	}
	.v-res-svg {
		width: 100%;
		height: 40px;
		display: block;
	}
	.v-res-axis {
		display: flex;
		justify-content: space-between;
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
	}
	.v-braille {
		font-family: var(--mono);
		letter-spacing: -0.05em;
		display: block;
		margin-top: 0.1rem;
	}

	.muted { color: var(--ink-faint); }
</style>
