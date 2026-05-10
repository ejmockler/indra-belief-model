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

	const calibrationSentence = $derived.by(() => {
		const c = v.calibration;
		if (c.mae == null) return 'calibration unavailable (no INDRA-comparable beliefs in this run)';
		const bias = c.bias ?? 0;
		const dir = bias > 0.05 ? 'over-believing vs INDRA' : bias < -0.05 ? 'under-believing vs INDRA' : 'centered on INDRA';
		// When |bias| ≈ MAE, every residual lies on the same side of zero —
		// no absolute-value cancellation occurred. That's a sharper diagnostic
		// than two adjacent numbers.
		const oneSided = c.mae > 0 && Math.abs(Math.abs(bias) - c.mae) < 1e-3;
		const oneSidedNote = oneSided && c.mae > 0.001 ? ' · every residual on the same side' : '';
		return `MAE ${c.mae.toFixed(3)} · ${dir} (bias ${bias >= 0 ? '+' : '−'}${Math.abs(bias).toFixed(3)})${oneSidedNote} · n=${c.n_stmts ?? '—'}`;
	});

	const residualMaxBin = $derived(residuals ? residuals.bins.reduce((m, n) => Math.max(m, n), 0) : 0);

	// Top-5 worst strata by MAE.
	const byType = $derived([...v.byIndraType].slice(0, 5));
	const bySource = $derived([...v.bySourceApi].slice(0, 5));
</script>

<section class="validity">
	<h2 class="v-h">validity <span class="v-run-id" title="latest succeeded run_id">{v.run_id.slice(0, 8)}</span></h2>

	<p class="v-summary">{calibrationSentence}</p>

	{#if verdictTotal > 0}
		<div class="v-pillbar-wrap">
			<div class="v-pillbar" role="img" aria-label="verdict shares per scored evidence">
				{#each v.verdicts as vd}
					{@const pct = verdictPct(vd.n)}
					{#if pct > 0}
						<span class="v-pill v-pill-{vd.verdict}" style:width="{pct}%" title="{vd.verdict}: {vd.n} of {verdictTotal} ({pct.toFixed(1)}%)">
							<span class="v-pill-label">{vd.verdict} {vd.n}</span>
						</span>
					{/if}
				{/each}
			</div>
			<span class="v-pillbar-denom">verdict per evidence · n={verdictTotal} aggregate steps</span>
		</div>
	{/if}

	{#if residuals && residuals.n_total > 0}
		<div class="v-residuals">
			<svg viewBox="0 0 220 36" preserveAspectRatio="none" class="v-res-svg" aria-label="residual histogram (our − INDRA), n={residuals.n_total}">
				<line x1="110" y1="0" x2="110" y2="36" stroke="var(--ink-faint)" stroke-width="0.5" stroke-dasharray="2,2"/>
				<path d={residualPath(residuals.bins, 220, 36)} fill="var(--ink)" opacity="0.85"/>
			</svg>
			<div class="v-res-axis">
				<span>−1</span><span>0</span><span>+1</span>
			</div>
			<div class="v-res-meta">
				<span class="v-braille" title="residual histogram (our − INDRA)">{residualBraille(residuals.bins)}</span>
				<span class="muted">mean {residuals.mean_residual?.toFixed(3) ?? '—'} · n={residuals.n_total}</span>
			</div>
		</div>
	{/if}

	<div class="v-sub-grid">
		{#if byType.length > 0}
			<div class="v-strata">
				<h3 class="v-strata-h">worst by indra_type (top {byType.length})</h3>
				<ul class="v-strata-list">
					{#each byType as s}
						<li class="v-strata-row">
							<span class="v-strata-label">{s.value}</span>
							<span class="v-strata-mae">MAE {s.mae.toFixed(3)}</span>
							<span class="v-strata-bias" class:b-pos={s.bias >= 0.05} class:b-neg={s.bias <= -0.05}>{s.bias >= 0 ? '▲' : '▼'}{Math.abs(s.bias).toFixed(2)}</span>
							<span class="muted">n={s.n}</span>
						</li>
					{/each}
				</ul>
			</div>
		{/if}
		{#if bySource.length > 0}
			<div class="v-strata">
				<h3 class="v-strata-h">worst by source_api (top {bySource.length})</h3>
				<ul class="v-strata-list">
					{#each bySource as s}
						<li class="v-strata-row">
							<span class="v-strata-label">{s.value}</span>
							<span class="v-strata-mae">MAE {s.mae.toFixed(3)}</span>
							<span class="v-strata-bias" class:b-pos={s.bias >= 0.05} class:b-neg={s.bias <= -0.05}>{s.bias >= 0 ? '▲' : '▼'}{Math.abs(s.bias).toFixed(2)}</span>
							<span class="muted">n={s.n}</span>
						</li>
					{/each}
				</ul>
			</div>
		{/if}
	</div>

	{#if v.truthPresent.length > 0}
		<div class="v-truth">
			<h3 class="v-strata-h">P/R/F1 vs truth_set</h3>
			<ul class="v-strata-list">
				{#each v.truthPresent as t}
					<li class="v-strata-row">
						<span class="v-strata-label">{t.truth_set_id}{t.step_kind === 'aggregate' ? '' : ` / ${t.step_kind}`}</span>
						<span>P {t.precision.toFixed(2)}</span>
						<span>R {t.recall.toFixed(2)}</span>
						<span class="v-strata-mae">F1 {t.f1.toFixed(2)}</span>
						<span class="muted">TP={t.tp} FP={t.fp} FN={t.fn}</span>
						<span class="muted">n={t.n_compared}</span>
					</li>
				{/each}
			</ul>
		</div>
	{/if}

	{#if v.inter_evidence_consistency.mean_stdev != null || v.supports_graph_delta != null}
		<div class="v-aux">
			{#if v.inter_evidence_consistency.mean_stdev != null}
				<span>inter-evidence stdev <span class="v-num">{v.inter_evidence_consistency.mean_stdev.toFixed(3)}</span> <span class="muted">n_multi={v.inter_evidence_consistency.n_multi_ev}</span></span>
			{/if}
			{#if v.supports_graph_delta != null}
				<span class="v-sep">·</span>
				<span>supports-graph delta <span class="v-num">{v.supports_graph_delta >= 0 ? '+' : ''}{v.supports_graph_delta.toFixed(3)}</span></span>
			{/if}
		</div>
	{/if}
</section>

<style>
	.validity {
		margin: 0 0 2.5rem;
	}
	.v-h {
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.04em;
		margin: 0 0 0.4rem;
		font-weight: 500;
		display: flex;
		align-items: baseline;
		gap: 0.6rem;
	}
	.v-run-id {
		font-size: 0.72rem;
		color: var(--ink-faint);
	}
	.v-summary {
		font-family: var(--serif);
		font-size: 1.05rem;
		color: var(--ink);
		margin: 0.2rem 0 0.8rem;
		font-variant-numeric: tabular-nums;
	}
	.v-pillbar-wrap {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		max-width: 480px;
		margin: 0.4rem 0 1rem;
	}
	.v-pillbar {
		display: flex;
		width: 100%;
		height: 22px;
		border: 1px solid var(--rule);
		overflow: hidden;
		font-family: var(--mono);
		font-size: 0.7rem;
	}
	.v-pillbar-denom {
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
		font-style: italic;
	}
	.v-pill {
		display: inline-flex;
		align-items: center;
		justify-content: center;
		color: var(--paper);
		overflow: hidden;
		white-space: nowrap;
	}
	.v-pill-label {
		padding: 0 0.3rem;
		font-variant-numeric: tabular-nums;
	}
	.v-pill-correct { background: #2a6f2a; }
	.v-pill-incorrect { background: var(--accent); }
	.v-pill-abstain { background: var(--ink-faint); color: var(--ink); }

	.v-residuals {
		display: flex;
		flex-direction: column;
		gap: 0.15rem;
		margin: 0.8rem 0 1.2rem;
		max-width: 280px;
	}
	.v-res-svg {
		width: 100%;
		height: 36px;
		display: block;
	}
	.v-res-axis {
		display: flex;
		justify-content: space-between;
		font-family: var(--mono);
		font-size: 0.66rem;
		color: var(--ink-faint);
	}
	.v-res-meta {
		display: flex;
		gap: 0.6rem;
		align-items: baseline;
		font-family: var(--mono);
		font-size: 0.72rem;
	}
	.v-braille {
		letter-spacing: -0.05em;
		color: var(--ink);
	}

	.v-sub-grid {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
		gap: 1rem 2rem;
		margin: 0.4rem 0 0.8rem;
	}
	.v-strata-h {
		font-family: var(--mono);
		font-size: 0.7rem;
		text-transform: lowercase;
		letter-spacing: 0.02em;
		color: var(--ink-muted);
		font-weight: 400;
		margin: 0 0 0.3rem;
	}
	.v-strata-list {
		list-style: none;
		padding: 0;
		margin: 0;
		font-family: var(--mono);
		font-size: 0.78rem;
	}
	.v-strata-row {
		display: grid;
		grid-template-columns: 1fr auto auto auto;
		gap: 0.6rem;
		padding: 0.15rem 0;
		border-bottom: 1px dotted var(--rule);
		align-items: baseline;
	}
	.v-strata-row:last-child {
		border-bottom: none;
	}
	.v-strata-label {
		color: var(--ink);
	}
	.v-strata-mae {
		font-variant-numeric: tabular-nums;
		color: var(--ink);
	}
	.v-strata-bias {
		font-variant-numeric: tabular-nums;
		color: var(--ink-muted);
	}
	.v-strata-bias.b-pos { color: #2a6f2a; }
	.v-strata-bias.b-neg { color: var(--accent); }

	.v-truth {
		margin: 0.4rem 0 0.8rem;
	}
	.v-truth .v-strata-row {
		grid-template-columns: 1fr auto auto auto auto auto;
	}

	.v-aux {
		font-family: var(--mono);
		font-size: 0.78rem;
		color: var(--ink-muted);
		margin-top: 0.6rem;
		padding-top: 0.6rem;
		border-top: 1px dotted var(--rule);
	}
	.v-num {
		color: var(--ink);
		font-variant-numeric: tabular-nums;
	}
	.v-sep {
		color: var(--ink-faint);
		margin: 0 0.4rem;
	}
	.muted {
		color: var(--ink-faint);
	}
</style>
