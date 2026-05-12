import { error } from '@sveltejs/kit';
import {
	connect,
	dbExists,
	getHeuristicCoverage,
	getRunNarrative,
	type HeuristicCoverage,
	type RunNarrative
} from '$lib/db';
import type { PageServerLoad } from './$types';

const RUN_ID_RE = /^[a-f0-9]{32}$/i;

export interface RunMeta {
	run_id: string;
	scorer_version: string;
	indra_version: string | null;
	model_id_default: string | null;
	started_at: string;
	status: string;
	n_stmts: number | null;
	n_evidences: number | null;
	cost_estimate_usd: number | null;
	cost_actual_usd: number | null;
}

export interface AllRunsRow {
	run_id: string;
	scorer_version: string;
	started_at: string;
	status: string;
}

async function getRunMeta(run_id: string): Promise<RunMeta | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		// `score_run` doesn't carry n_evidences directly — count from scorer_step.
		const qr = run_id.replace(/'/g, "''");
		const reader = await con.runAndReadAll(
			`SELECT
			   sr.run_id, sr.scorer_version, sr.indra_version, sr.model_id_default,
			   sr.started_at::VARCHAR AS started_at, sr.status, sr.n_stmts,
			   (SELECT COUNT(DISTINCT evidence_hash)
			    FROM scorer_step WHERE run_id = sr.run_id) AS n_evidences,
			   sr.cost_estimate_usd, sr.cost_actual_usd
			 FROM score_run sr WHERE sr.run_id='${qr}'`
		);
		const rs = reader.getRowObjects();
		if (rs.length === 0) return null;
		const r = rs[0];
		return {
			run_id: r.run_id as string,
			scorer_version: r.scorer_version as string,
			indra_version: (r.indra_version as string | null) ?? null,
			model_id_default: (r.model_id_default as string | null) ?? null,
			started_at: r.started_at as string,
			status: r.status as string,
			n_stmts: r.n_stmts != null ? Number(r.n_stmts) : null,
			n_evidences: r.n_evidences != null ? Number(r.n_evidences) : null,
			cost_estimate_usd: r.cost_estimate_usd != null ? Number(r.cost_estimate_usd) : null,
			cost_actual_usd: r.cost_actual_usd != null ? Number(r.cost_actual_usd) : null
		};
	} finally {
		con.disconnectSync?.();
	}
}

async function listSucceededRuns(): Promise<AllRunsRow[]> {
	if (!dbExists()) return [];
	const con = await connect();
	try {
		const reader = await con.runAndReadAll(
			`SELECT run_id, scorer_version, started_at::VARCHAR AS started_at, status
			 FROM score_run
			 WHERE status='succeeded'
			 ORDER BY started_at DESC`
		);
		return reader.getRowObjects().map((r) => ({
			run_id: r.run_id as string,
			scorer_version: r.scorer_version as string,
			started_at: r.started_at as string,
			status: r.status as string
		}));
	} finally {
		con.disconnectSync?.();
	}
}

export const load: PageServerLoad = async ({ params, url }) => {
	if (!RUN_ID_RE.test(params.run_id)) {
		throw error(400, 'invalid run_id: must be 32 hex chars (UUID hex)');
	}
	const meta = await getRunMeta(params.run_id);
	if (!meta) throw error(404, `run_id ${params.run_id} not found`);

	// Optional ?compare_to=<run_id> overrides the auto-found predecessor.
	const compareToParam = url.searchParams.get('compare_to');
	const explicitPrev =
		compareToParam && RUN_ID_RE.test(compareToParam) ? compareToParam : undefined;

	const [narrative, coverage, allRuns] = await Promise.all([
		getRunNarrative(params.run_id, explicitPrev) as Promise<RunNarrative | null>,
		getHeuristicCoverage(params.run_id) as Promise<HeuristicCoverage | null>,
		listSucceededRuns()
	]);

	return {
		meta,
		narrative,
		coverage,
		allRuns,
		compareToParam: explicitPrev ?? null
	};
};
