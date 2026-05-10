/**
 * Residual distribution (T1.5 of the belief instrument hypergraph).
 *
 * Calibration as shape, not a scalar. The histogram bins (our − indra)
 * over [-1, +1]. Two render modes share the same bin array:
 *   - braille block string for inline use ("▁▁▂▃▆█▆▃▂▁▁")
 *   - SVG path for the validity main view
 */
export const RESIDUAL_BINS = 11;
export const RESIDUAL_RANGE = [-1, 1] as const;

const BRAILLE = ['▁', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/** Render an 11-bin histogram as a Unicode braille block string. */
export function residualBraille(bins: readonly number[]): string {
	const max = bins.reduce((m, n) => Math.max(m, n), 0);
	if (max === 0) return '·'.repeat(bins.length);
	return bins
		.map((n) => {
			if (n === 0) return ' ';
			const idx = Math.min(BRAILLE.length - 1, Math.round((n / max) * (BRAILLE.length - 1)));
			return BRAILLE[idx];
		})
		.join('');
}

/**
 * Render bins as an SVG `path` d-attribute. Domain x ∈ [-1, +1],
 * y is bin count normalized to [0, 1] then flipped.
 * Caller supplies (w, h) and applies them.
 */
export function residualPath(bins: readonly number[], w: number, h: number): string {
	const max = bins.reduce((m, n) => Math.max(m, n), 0);
	if (max === 0) return '';
	const step = w / bins.length;
	const points: string[] = [];
	bins.forEach((n, i) => {
		const x0 = i * step;
		const x1 = (i + 1) * step;
		const y = h - (n / max) * h;
		points.push(`M ${x0.toFixed(1)} ${h} L ${x0.toFixed(1)} ${y.toFixed(1)} L ${x1.toFixed(1)} ${y.toFixed(1)} L ${x1.toFixed(1)} ${h}`);
	});
	return points.join(' ');
}
