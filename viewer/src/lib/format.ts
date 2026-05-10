/**
 * Shared formatting primitives for the viewer.
 *
 * Phase 5b primitives extraction: when a function is duplicated across 2+
 * routes/components, hoist it here. Tufte-pure, no styling — just data
 * shape. Components live in `$lib/components/`.
 */

/** First 8 chars of a hash; canonical short form for the UI. */
export function shortHash(h: string): string {
	return h.slice(0, 8);
}

/** Belief value to 2 decimals; em-dash for null. */
export function fmtBelief(b: number | null | undefined): string {
	return b == null ? '—' : b.toFixed(2);
}

/** Signed delta with proper minus glyph; em-dash for null. */
export function fmtDelta(d: number | null | undefined): string {
	if (d == null) return '—';
	const sign = d >= 0 ? '+' : '−';
	return `${sign}${Math.abs(d).toFixed(2)}`;
}

/** Plural-aware label suffix. `n=1 → ''`, else `'s'`. */
export function pluralS(n: number): string {
	return n === 1 ? '' : 's';
}

export interface SentenceAgent {
	role: string;
	name: string;
}

/**
 * Render an INDRA statement as a readable English sentence. Falls back to
 * `Type(name1, name2)` for indra_types we don't have a verb form for.
 * Order of preference: subj/obj, then enz/sub, then members.
 */
export function sentenceFromStatement(
	indra_type: string,
	agents: SentenceAgent[]
): string {
	if (!agents || agents.length === 0) return indra_type;

	const by_role = (roles: string[]): SentenceAgent | undefined =>
		agents.find((a) => roles.includes(a.role));
	const members = agents.filter((a) => a.role === 'member');

	const verbs: Record<string, [string[], string[], string]> = {
		Activation: [['subj'], ['obj'], 'activates'],
		Inhibition: [['subj'], ['obj'], 'inhibits'],
		Phosphorylation: [['enz'], ['sub'], 'phosphorylates'],
		Dephosphorylation: [['enz'], ['sub'], 'dephosphorylates'],
		Ubiquitination: [['enz'], ['sub'], 'ubiquitinates'],
		Deubiquitination: [['enz'], ['sub'], 'deubiquitinates'],
		Methylation: [['enz'], ['sub'], 'methylates'],
		Demethylation: [['enz'], ['sub'], 'demethylates'],
		Acetylation: [['enz'], ['sub'], 'acetylates'],
		Deacetylation: [['enz'], ['sub'], 'deacetylates'],
		Sumoylation: [['enz'], ['sub'], 'sumoylates'],
		Desumoylation: [['enz'], ['sub'], 'desumoylates'],
		Hydroxylation: [['enz'], ['sub'], 'hydroxylates'],
		Dehydroxylation: [['enz'], ['sub'], 'dehydroxylates'],
		IncreaseAmount: [['subj'], ['obj'], 'increases the amount of'],
		DecreaseAmount: [['subj'], ['obj'], 'decreases the amount of'],
		Gef: [['gef'], ['ras'], 'activates'],
		Gap: [['gap'], ['ras'], 'inactivates'],
		Conversion: [['subj'], ['obj_from', 'obj'], 'converts'],
		RegulateActivity: [['subj'], ['obj'], 'regulates'],
		RegulateAmount: [['subj'], ['obj'], 'regulates the amount of']
	};

	const v = verbs[indra_type];
	if (v) {
		const [subjRoles, objRoles, verb] = v;
		const s = by_role(subjRoles);
		const o = by_role(objRoles);
		if (s && o) return `${s.name} ${verb} ${o.name}`;
	}

	if (indra_type === 'Complex' && members.length >= 2) {
		return `${members.map((m) => m.name).join(' · ')} (complex)`;
	}
	if (indra_type === 'Translocation' && agents.length > 0) {
		return `${agents[0].name} translocates`;
	}
	if (indra_type === 'Autophosphorylation' && agents.length > 0) {
		return `${agents[0].name} autophosphorylates`;
	}

	return `${indra_type}(${agents.map((a) => a.name).join(', ')})`;
}

/**
 * Map a belief score to a 3-class semantic role.
 * Used for color application on the score itself.
 */
export function beliefSemantic(b: number | null | undefined): 'high' | 'low' | 'mid' | 'absent' {
	if (b == null || Number.isNaN(b)) return 'absent';
	if (b >= 0.7) return 'high';
	if (b <= 0.3) return 'low';
	return 'mid';
}
