<script lang="ts">
	import type { ProbeAttribution, ProbeKind } from '$lib/probeAttribution';
	import {
		beliefSemantic,
		evidenceParts,
		extractProbeCue,
		fmtBelief,
		fmtDelta,
		pluralS,
		sentenceFromStatement,
		shortHash,
		type SentenceAgent
	} from '$lib/format';

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

	function compactProbeCell(a: ProbeAttribution | null): string {
		if (!a) return '·';
		const c = a.contribution;
		if (c > 0.6) return '█';
		if (c > 0.2) return '▆';
		if (c > -0.2) return '▃';
		if (c > -0.6) return '▂';
		return '▁';
	}

	/**
	 * Probe answer → valence direction, qualitatively.
	 * The scorer doesn't emit numerical weights per probe; it routes the
	 * categorical answer through the adjudicator's decision table. Valence
	 * here is just a *color cue* for the reader — green = the answer that
	 * supports the claim's surface form; accent = the answer that pushes
	 * against it. No magnitude implied.
	 */
	function valence(probe: ProbeKind, answer: string | null): 'pro' | 'con' | 'neutral' {
		if (!answer || answer === 'abstain') return 'neutral';
		if (probe === 'scope') {
			if (answer === 'asserted') return 'pro';
			if (answer === 'negated') return 'con';
			return 'neutral';
		}
		if (probe === 'subject_role') {
			if (answer === 'present_as_subject') return 'pro';
			if (answer === 'present_as_object' || answer === 'absent' || answer === 'present_as_decoy') return 'con';
			return 'neutral';
		}
		if (probe === 'object_role') {
			if (answer === 'present_as_object') return 'pro';
			if (answer === 'present_as_subject' || answer === 'absent' || answer === 'present_as_decoy') return 'con';
			return 'neutral';
		}
		if (probe === 'relation_axis') {
			if (answer === 'direct_sign_match' || answer === 'via_mediator') return 'pro';
			if (
				answer === 'direct_sign_mismatch' ||
				answer === 'direct_axis_mismatch' ||
				answer === 'direct_partner_mismatch' ||
				answer === 'no_relation'
			)
				return 'con';
			return 'neutral';
		}
		return 'neutral';
	}

	/**
	 * The seven verdict-confidence buckets the scorer can emit, defined in
	 * `src/indra_belief/scorers/commitments.py::_VERDICT_SCORE`. Showing
	 * them as ladder ticks on the score axis surfaces that the score is
	 * categorical, not continuous.
	 */
	const SCORE_BUCKETS = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95];

	function beliefPhrase(b: number | null): string {
		if (b == null) return 'unscored';
		if (b >= 0.85) return 'near-certain';
		if (b >= 0.7) return 'confident';
		if (b >= 0.5) return 'moderate';
		if (b >= 0.3) return 'doubtful';
		return 'low';
	}

	const verdictLine = $derived.by(() => {
		if (our_score == null && indra_score == null) {
			return 'Not yet scored. No INDRA prior.';
		}
		if (our_score == null) {
			return `Not yet scored · INDRA prior ${indra_score!.toFixed(2)} (${beliefPhrase(indra_score)}).`;
		}
		const ourP = beliefPhrase(our_score);
		if (indra_score == null) {
			return `We score this ${our_score.toFixed(2)} (${ourP}) · no INDRA prior to compare.`;
		}
		const d = our_score - indra_score;
		let comparison: string;
		if (Math.abs(d) < 0.1) {
			comparison = `matches INDRA's ${indra_score.toFixed(2)} (${beliefPhrase(indra_score)})`;
		} else if (d < -0.4) {
			comparison = `we doubt it strongly · INDRA was ${indra_score.toFixed(2)} (${beliefPhrase(indra_score)})`;
		} else if (d < 0) {
			comparison = `less confident than INDRA's ${indra_score.toFixed(2)} (${beliefPhrase(indra_score)})`;
		} else if (d > 0.4) {
			comparison = `we believe it more strongly than INDRA's ${indra_score.toFixed(2)} (${beliefPhrase(indra_score)})`;
		} else {
			comparison = `more confident than INDRA's ${indra_score.toFixed(2)} (${beliefPhrase(indra_score)})`;
		}
		return `We score this ${our_score.toFixed(2)} (${ourP}) · ${comparison}.`;
	});

	const invokedProbes = $derived(probes.filter((p) => p.answer != null && p.answer !== 'abstain'));
	const notInvokedKinds = $derived(
		(['subject_role', 'object_role', 'relation_axis', 'scope'] as const)
			.filter((k) => !invokedProbes.some((p) => p.probe === k))
			.map(probeLabel)
	);

	/**
	 * Decisive criterion — kept honest now: when exactly one probe fires
	 * (others didn't run or abstained), that single probe carries the entire
	 * categorical signal so it deserves the visual emphasis. With ≥2 firing
	 * probes we cannot say which was decisive from the persisted output
	 * alone (the adjudicator's decision table would have to be re-evaluated
	 * per row), so we don't claim.
	 */
	const decisiveProbe = $derived(invokedProbes.length === 1 ? invokedProbes[0] : null);

	/** Cue from the first invoked probe that has one — used to highlight inside the evidence text. */
	const primaryCue = $derived.by(() => {
		for (const p of invokedProbes) {
			const c = extractProbeCue(p.rationale);
			if (c) return c;
		}
		return null;
	});

	/** Score-axis geometry: viewBox 0..320 wide, ticks land at x = margin + score * track. */
	const AXIS_W = 320;
	const AXIS_MARGIN = 14;
	const AXIS_TRACK = AXIS_W - 2 * AXIS_MARGIN;
	function tickX(score: number): number {
		return AXIS_MARGIN + Math.max(0, Math.min(1, score)) * AXIS_TRACK;
	}

	/**
	 * Adaptive label anchor: when a tick sits near an SVG edge, centered-text
	 * would clip past the viewBox. Hug the edge instead.
	 *  - tick > AXIS_W − labelHalf   → anchor: end at the tick (text extends leftward)
	 *  - tick < labelHalf            → anchor: start at the tick (text extends rightward)
	 *  - else                        → anchor: middle at the tick
	 * labelHalf is a conservative pixel estimate at font-size 8 for "indra 1.00".
	 */
	function labelAnchor(tx: number): { anchor: 'middle' | 'start' | 'end'; x: number } {
		const labelHalf = 26;
		if (tx + labelHalf > AXIS_W) return { anchor: 'end', x: tx };
		if (tx - labelHalf < 0) return { anchor: 'start', x: tx };
		return { anchor: 'middle', x: tx };
	}
</script>

{#if mode === 'compact'}
	{#snippet compactBody()}
		<span class="b-sentence">{sentence}</span>
		<span class="b-num-pair" title={delta == null ? '' : `Δ ${fmtDelta(delta)} (we ${delta < 0 ? 'doubt' : delta > 0 ? 'support' : 'match'} more than INDRA)`}>
			<span class="b-num-pair-label">we</span>
			<span class="b-score-compact b-{semantic}">{fmtBelief(our_score)}</span>
			<span class="b-num-pair-label">indra</span>
			<span class="b-num-mid-compact">{fmtBelief(indra_score)}</span>
		</span>
		{#if evidences.length > 0}
			<span class="b-ev-count" title="number of evidences for this statement">{evidences.length} ev</span>
		{/if}
		<span class="b-hash">{shortHash(stmt.stmt_hash)}</span>
	{/snippet}

	{#if href}
		<a class="b-card b-card-compact" {href}>{@render compactBody()}</a>
	{:else}
		<div class="b-card b-card-compact">{@render compactBody()}</div>
	{/if}
{:else}
	<article class="b-card b-card-full">
		<h3 class="b-sentence b-sentence-full">{sentence}</h3>
		<p class="b-verdict-line">{verdictLine}</p>
		<div class="b-meta">
			<span class="b-type">{stmt.indra_type}</span>
			<span class="b-hash">{shortHash(stmt.stmt_hash)}</span>
		</div>

		{#if our_score != null || indra_score != null}
			{@const anyNearZero = (our_score != null && our_score <= 0.05) || (indra_score != null && indra_score <= 0.05)}
			{@const anyNearOne = (our_score != null && our_score >= 0.95) || (indra_score != null && indra_score >= 0.95)}
			<div class="b-axis-wrap" role="img" aria-label="belief scale 0 to 1; ours above the axis, INDRA below">
				<svg viewBox="0 0 {AXIS_W} 60" class="b-axis-svg" preserveAspectRatio="xMidYMid meet">
					<!-- axis track -->
					<line x1={AXIS_MARGIN} y1="28" x2={AXIS_W - AXIS_MARGIN} y2="28" stroke="var(--ink)" stroke-width="1"/>
					<!-- endpoint tick marks -->
					<line x1={AXIS_MARGIN} y1="24" x2={AXIS_MARGIN} y2="32" stroke="var(--ink-faint)" stroke-width="1"/>
					<line x1={AXIS_W - AXIS_MARGIN} y1="24" x2={AXIS_W - AXIS_MARGIN} y2="32" stroke="var(--ink-faint)" stroke-width="1"/>
					<!-- 7 verdict-bucket ticks (categorical rungs the prod scorer can hit) -->
					{#each SCORE_BUCKETS as b}
						<line x1={tickX(b)} y1="26" x2={tickX(b)} y2="30" stroke="var(--ink-faint)" stroke-width="0.6" opacity="0.7"/>
					{/each}
					<!-- 0 / 1 scale anchors: hidden when a tick value already says the same thing -->
					{#if !anyNearZero}
						<text x={AXIS_MARGIN} y="56" text-anchor="middle" class="b-axis-endlabel">0</text>
					{/if}
					{#if !anyNearOne}
						<text x={AXIS_W - AXIS_MARGIN} y="56" text-anchor="middle" class="b-axis-endlabel">1</text>
					{/if}
					<!-- INDRA lives below the axis: outlined circle + label at y=44 -->
					{#if indra_score != null}
						{@const tx = tickX(indra_score)}
						{@const la = labelAnchor(tx)}
						<circle cx={tx} cy="28" r="4.5" fill="var(--paper)" stroke="var(--ink)" stroke-width="1.5"/>
						<text x={la.x} y="44" text-anchor={la.anchor} class="b-axis-label b-axis-label-indra">indra {indra_score.toFixed(2)}</text>
					{/if}
					<!-- Ours lives above the axis: filled circle + label at y=14 -->
					{#if our_score != null}
						{@const tx = tickX(our_score)}
						{@const la = labelAnchor(tx)}
						<circle cx={tx} cy="28" r="5" fill="var(--accent)"/>
						<text x={la.x} y="14" text-anchor={la.anchor} class="b-axis-label b-axis-label-ours">ours {our_score.toFixed(2)}</text>
					{/if}
				</svg>
				<p class="b-axis-footnote">
					ours snaps to one of 7 categorical buckets (verdict × confidence); INDRA's prior is continuous
				</p>
			</div>
		{/if}

		{#if invokedProbes.length > 0 || evidences.length > 0}
			<div class="b-cause">
				<h4 class="b-cause-h">{decisiveProbe ? 'Why we doubt' : invokedProbes.length > 0 ? 'How we read it' : 'Evidence'}</h4>
				{#if invokedProbes.length > 0}
					<p class="b-cause-legend">
						probes produce categorical answers — the adjudicator combines them via a 7-bucket decision table, not by summing weights
					</p>
				{/if}

				{#each evShown as e}
					{@const parts = evidenceParts(e.text, primaryCue)}
					<div class="b-cause-ev">
						<div class="b-cause-ev-line">
							<span class="b-ev-src">[{e.source_api ?? 'no source'}]</span>
							<span class="b-ev-text">{#each parts as part}{#if part.highlight}<mark class="b-ev-cue">{part.text}</mark>{:else}{part.text}{/if}{/each}</span>
						</div>
					</div>
				{/each}
				{#if evMore > 0}
					<p class="b-cause-evmore">…and {evMore} more evidence{evMore === 1 ? '' : 's'}</p>
				{/if}

				{#each invokedProbes as p}
					{@const v = valence(p.probe, p.answer)}
					<div class="b-probe-line" class:b-probe-decisive={p === decisiveProbe}>
						<span class="b-probe-arrow">↳</span>
						<span class="b-probe-name">{probeLabel(p.probe)}</span>
						<span class="b-probe-answer b-probe-valence-{v}">{p.answer ?? '—'}</span>
						<span class="b-probe-meta">·{p.confidence ?? '—'}{p.source === 'substrate' ? ' · substrate' : p.source === 'llm' ? ' · llm' : ''}</span>
						{#if p.rationale}
							<span class="b-probe-rationale">— {p.rationale}</span>
						{/if}
						{#if p === decisiveProbe}<span class="b-probe-decisive-tag">sole probe that fired</span>{/if}
					</div>
				{/each}

				{#if notInvokedKinds.length > 0 && invokedProbes.length > 0}
					<p class="b-cause-footer">
						<span class="b-cause-footer-label">other probes:</span>
						{notInvokedKinds.join(', ')} <span class="muted">— did not run (short-circuit on this evidence)</span>
					</p>
				{/if}
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
		grid-template-columns: minmax(0, 1fr) auto auto auto;
		gap: 1rem;
		align-items: baseline;
		padding: 0.4rem 0.4rem;
		border-bottom: 1px dotted var(--rule);
		font-size: 0.92rem;
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
	.b-verdict-line {
		font-family: var(--serif);
		font-size: 1rem;
		color: var(--ink);
		margin: 0.1rem 0 0.6rem;
		line-height: 1.45;
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

	.b-high { color: #2a6f2a; }
	.b-low { color: var(--accent); }
	.b-mid { color: var(--ink); }
	.b-absent { color: var(--ink-faint); }

	.b-delta-pos { color: #2a6f2a; }
	.b-delta-neg { color: var(--accent); }
	.b-delta-na { color: var(--ink-faint); }

	/* Score axis — comparison as position */
	.b-axis-wrap {
		margin: 0.6rem 0 1.2rem;
		max-width: 360px;
	}
	.b-axis-svg {
		width: 100%;
		height: auto;
		display: block;
	}
	.b-axis-footnote {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.78rem;
		color: var(--ink-faint);
		margin: 0.2rem 0 0;
		line-height: 1.4;
	}
	:global(.b-axis-endlabel) {
		font-family: var(--mono);
		font-size: 7px;
		fill: var(--ink-faint);
	}
	:global(.b-axis-label) {
		font-family: var(--mono);
		font-size: 8px;
		font-variant-numeric: tabular-nums;
	}
	:global(.b-axis-label-ours) { fill: var(--accent); font-weight: 500; }
	:global(.b-axis-label-indra) { fill: var(--ink-muted); }

	/* Cause section — what drove the score */
	.b-cause {
		margin-top: 0.4rem;
	}
	.b-cause-h {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
		text-transform: lowercase;
		letter-spacing: 0.04em;
		margin: 0 0 0.4rem;
		font-weight: 500;
	}
	.b-cause-legend {
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.84rem;
		color: var(--ink-faint);
		margin: 0 0 0.6rem;
		line-height: 1.45;
	}
	.b-cause-ev {
		margin: 0.2rem 0;
	}
	.b-cause-ev-line {
		display: grid;
		grid-template-columns: auto 1fr;
		gap: 0.6rem;
		align-items: baseline;
	}
	.b-ev-src {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
	}
	.b-ev-text {
		font-family: var(--serif);
		font-style: italic;
		color: var(--ink);
		font-size: 1rem;
		line-height: 1.45;
	}
	.b-ev-cue {
		background: var(--accent-wash);
		color: var(--accent);
		padding: 0 0.15em;
		font-style: italic;
		font-weight: 500;
		border-bottom: 1px solid var(--accent);
	}
	.b-cause-evmore {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
		margin: 0.2rem 0;
		font-style: italic;
	}

	/* Probe lines (paired with evidence above) */
	.b-probe-line {
		display: flex;
		flex-wrap: wrap;
		gap: 0.5rem;
		align-items: baseline;
		margin: 0.2rem 0 0.2rem 1.4rem;
		font-family: var(--mono);
		font-size: 0.78rem;
		padding-left: 0.4rem;
		border-left: 2px solid var(--rule);
	}
	.b-probe-decisive {
		border-left-color: var(--accent);
	}
	.b-probe-arrow {
		color: var(--ink-faint);
		margin-left: -1.2rem;
		font-family: var(--mono);
	}
	.b-probe-name {
		color: var(--ink-muted);
		min-width: 5rem;
	}
	.b-probe-answer {
		font-weight: 500;
	}
	.b-probe-valence-pro { color: var(--ok-green); }
	.b-probe-valence-con { color: var(--accent); }
	.b-probe-valence-neutral { color: var(--ink); }
	.b-probe-meta {
		color: var(--ink-faint);
		font-size: 0.72rem;
	}
	.b-probe-rationale {
		color: var(--ink-muted);
		font-family: var(--serif);
		font-style: italic;
		font-size: 0.88rem;
		flex-basis: 100%;
		margin-left: 0.4rem;
	}
	.b-probe-decisive-tag {
		color: var(--ink-faint);
		font-family: var(--mono);
		font-size: 0.7rem;
		font-style: italic;
	}

	.b-cause-footer {
		margin: 0.8rem 0 0;
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-muted);
	}
	.b-cause-footer-label {
		color: var(--ink-faint);
		margin-right: 0.3rem;
	}

	.b-why {
		margin-top: 1rem;
		padding-top: 0.6rem;
		border-top: 1px dotted var(--rule);
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
		font-size: 1rem;
	}
	.b-num-pair {
		font-family: var(--mono);
		font-size: 0.82rem;
		font-variant-numeric: tabular-nums;
		text-align: right;
		display: inline-flex;
		gap: 0.3rem;
		align-items: baseline;
	}
	.b-num-pair-label {
		color: var(--ink-faint);
		font-size: 0.7rem;
		text-transform: lowercase;
	}
	.b-score-compact { color: var(--ink); font-weight: 500; }
	.b-num-mid-compact { color: var(--ink-muted); }
	.b-ev-count {
		font-family: var(--mono);
		font-size: 0.72rem;
		color: var(--ink-faint);
	}
	.b-hash {
		font-family: var(--mono);
		font-size: 0.7rem;
		color: var(--ink-faint);
	}
</style>
