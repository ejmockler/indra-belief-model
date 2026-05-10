import {
	getCorpusOverview,
	getFindings,
	getFocusStatement,
	getResidualDistribution,
	getRunNarrative,
	type RunNarrative
} from '$lib/db';
import type { PageServerLoad } from './$types';

const STMT_HASH_RE = /^[a-f0-9]{16}$/i;

export const load: PageServerLoad = async ({ url }) => {
	const focusParam = url.searchParams.get('focus');
	const focusHash = focusParam && STMT_HASH_RE.test(focusParam) ? focusParam : undefined;

	const [overview, focus, findings, residuals] = await Promise.all([
		getCorpusOverview(),
		getFocusStatement(focusHash),
		getFindings(),
		getResidualDistribution()
	]);

	const narratives: Record<string, RunNarrative> = {};
	const narrativeRows = await Promise.all(
		overview.scorerRuns
			.filter((r) => r.status === 'succeeded')
			.map(async (r) => [r.run_id, await getRunNarrative(r.run_id)] as const)
	);
	for (const [rid, n] of narrativeRows) {
		if (n) narratives[rid] = n;
	}

	return { overview, focus, findings, residuals, narratives };
};
