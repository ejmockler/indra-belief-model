import {
	getCorpusOverview,
	getFindings,
	getFocusStatement,
	getHeuristicCoverage,
	getResidualDistribution,
	getRunNarrative,
	type HeuristicCoverage,
	type RunNarrative
} from '$lib/db';
import {
	datasetShape,
	getDatasets,
	getDatasetIngestStatus,
	type DatasetDescriptor,
	type DatasetShape,
	type IngestStatus
} from '$lib/datasets';
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

	let coverage: HeuristicCoverage | null = null;
	if (overview.latestValidity?.run_id) {
		coverage = await getHeuristicCoverage(overview.latestValidity.run_id);
	}

	// U2: filesystem-discovered datasets + lazy shape preview + ingest status.
	const baseDescriptors = getDatasets();
	const datasets: Array<DatasetDescriptor & { shape: DatasetShape; ingest: IngestStatus | null }> = await Promise.all(
		baseDescriptors.map(async (d) => ({
			...d,
			shape: datasetShape(d),
			ingest: await getDatasetIngestStatus(d)
		}))
	);

	return { overview, focus, findings, residuals, narratives, coverage, datasets };
};
