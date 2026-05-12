/**
 * Probe attribution model (T1.4 of the belief instrument hypergraph).
 *
 * The four scorer probes — subject_role, object_role, relation_axis, scope —
 * each emit an enum answer with a confidence. The aggregate decision table
 * combines them into a verdict and score. To make the pipeline's reasoning
 * legible we attribute the final score to its probes: each gets a signed
 * contribution in [-1, +1], a normalized bar width for rendering, and a
 * `decisive` flag (true if removing this probe alone would flip the sign).
 *
 * Pure logic only. DB I/O lives in db.ts.
 */
export type ProbeKind = 'subject_role' | 'object_role' | 'relation_axis' | 'scope';
export type ProbeConfidence = 'high' | 'medium' | 'low' | null;
export type ProbeSource = 'substrate' | 'llm' | 'abstain' | null;

export interface ProbeOutput {
	probe: ProbeKind;
	evidence_hash: string | null;
	answer: string | null;
	confidence: ProbeConfidence;
	source: ProbeSource;
	rationale: string | null;
}

export interface ProbeAttribution {
	probe: ProbeKind;
	answer: string | null;
	confidence: ProbeConfidence;
	source: ProbeSource;
	rationale: string | null;
	/** Evidence the probe ran on, when aggregating from per-evidence outputs. */
	evidence_hash: string | null;
	/** Signed magnitude in roughly [-1, +1]. Positive = pushes toward correct. */
	contribution: number;
	/** [0, 1]; `abs(contribution) / max(abs(contribution))` across the probe set. */
	normalized_width: number;
	/** True iff this probe alone determines the sign of the sum. */
	decisive: boolean;
}

const ROLE_VOTES: Record<string, number> = {
	present_as_subject: 1.0,
	present_as_object: 1.0,
	present_as_mediator: 0.3,
	present_as_decoy: -0.5,
	absent: -1.0
};

const RELATION_AXIS_VOTES: Record<string, number> = {
	direct_sign_match: 1.0,
	via_mediator: 0.5,
	via_mediator_partial: 0.2,
	direct_sign_mismatch: -1.0,
	direct_axis_mismatch: -0.7,
	direct_partner_mismatch: -0.7,
	no_relation: -1.0
};

const SCOPE_VOTES: Record<string, number> = {
	asserted: 1.0,
	hedged: -0.3,
	negated: -1.0
};

const CONFIDENCE_WEIGHT: Record<string, number> = {
	high: 1.0,
	medium: 0.7,
	low: 0.4
};

/**
 * Map a probe answer to a signed vote in [-1, +1].
 *
 * subject_role and object_role share the "named in some role" answer space.
 * Voting differs: for subject_role the "right" answer is `present_as_subject`
 * (correct role), and `present_as_object` is a role-swap (wrong role); for
 * object_role the polarity is flipped. We model role-swap as -1 (strong
 * evidence against) by negating the same-role hit; `present_as_mediator` and
 * `present_as_decoy` are weaker signals that apply symmetrically.
 */
export function voteForAnswer(probe: ProbeKind, answer: string | null): number {
	if (!answer || answer === 'abstain') return 0;

	if (probe === 'subject_role') {
		if (answer === 'present_as_subject') return 1.0;
		if (answer === 'present_as_object') return -1.0; // role swap
		return ROLE_VOTES[answer] ?? 0;
	}
	if (probe === 'object_role') {
		if (answer === 'present_as_object') return 1.0;
		if (answer === 'present_as_subject') return -1.0; // role swap
		return ROLE_VOTES[answer] ?? 0;
	}
	if (probe === 'relation_axis') return RELATION_AXIS_VOTES[answer] ?? 0;
	if (probe === 'scope') return SCOPE_VOTES[answer] ?? 0;
	return 0;
}

export function confidenceWeight(c: ProbeConfidence | string | null): number {
	if (!c) return 0;
	return CONFIDENCE_WEIGHT[c] ?? 0;
}

export function computeAttributions(probes: ProbeOutput[]): ProbeAttribution[] {
	const raw = probes.map((p) => {
		const vote = voteForAnswer(p.probe, p.answer);
		const weight = p.source === 'abstain' ? 0 : confidenceWeight(p.confidence);
		return { p, contribution: vote * weight };
	});

	const sum = raw.reduce((s, x) => s + x.contribution, 0);
	const sumSign = sum >= 0;
	const maxAbs = raw.reduce((m, x) => Math.max(m, Math.abs(x.contribution)), 0);

	return raw.map(({ p, contribution }) => {
		const without = sum - contribution;
		const withoutSign = without >= 0;
		const decisive = Math.abs(contribution) > 1e-6 && sumSign !== withoutSign;
		return {
			probe: p.probe,
			answer: p.answer,
			confidence: p.confidence,
			source: p.source,
			rationale: p.rationale,
			evidence_hash: p.evidence_hash,
			contribution,
			normalized_width: maxAbs > 0 ? Math.abs(contribution) / maxAbs : 0,
			decisive
		};
	});
}

/**
 * Reduce a list of per-evidence probe outputs to a single "summary"
 * per probe. Picks the highest-confidence answer per probe; for ties,
 * prefers `substrate` over `llm` (deterministic substrate is the truth
 * floor — see memory: substrate-vs-llm lever).
 */
export function summarizeAcrossEvidences(probes: ProbeOutput[]): ProbeOutput[] {
	const byProbe = new Map<ProbeKind, ProbeOutput[]>();
	for (const p of probes) {
		if (!byProbe.has(p.probe)) byProbe.set(p.probe, []);
		byProbe.get(p.probe)!.push(p);
	}
	const order: ProbeKind[] = ['subject_role', 'object_role', 'relation_axis', 'scope'];
	const out: ProbeOutput[] = [];
	for (const probe of order) {
		const list = byProbe.get(probe);
		if (!list || list.length === 0) continue;
		const sorted = [...list].sort((a, b) => {
			const cw = confidenceWeight(b.confidence) - confidenceWeight(a.confidence);
			if (cw !== 0) return cw;
			const sa = a.source === 'substrate' ? 0 : 1;
			const sb = b.source === 'substrate' ? 0 : 1;
			return sa - sb;
		});
		out.push(sorted[0]);
	}
	return out;
}

/** Render a signed contribution as a monospace bar block (10 cells). */
export function attributionBar(contribution: number, width = 10): string {
	const filled = Math.round(Math.min(1, Math.max(0, Math.abs(contribution))) * width);
	const empty = width - filled;
	const fill = contribution >= 0 ? '█' : '▓';
	return fill.repeat(filled) + '░'.repeat(empty);
}
