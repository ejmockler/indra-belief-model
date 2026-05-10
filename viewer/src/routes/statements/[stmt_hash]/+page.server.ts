import { error } from '@sveltejs/kit';
import { connect, dbExists, getProbeAttribution, getStatementDetail } from '$lib/db';
import type { PageServerLoad } from './$types';

// stmt_hash is INDRA's `Statement.get_hash(shallow=True)` — 16 hex nibbles
// (see corpus/loader.py::_hex). Reject anything else at the gate as a
// defense-in-depth (SQL escaping in db.ts is the inner layer).
const STMT_HASH_RE = /^[a-f0-9]{16}$/i;

export const load: PageServerLoad = async ({ params }) => {
	if (!STMT_HASH_RE.test(params.stmt_hash)) {
		throw error(400, `invalid stmt_hash: must be 16 hex chars`);
	}
	const detail = await getStatementDetail(params.stmt_hash);
	if (!detail) {
		throw error(404, `statement ${params.stmt_hash} not found`);
	}

	let probes: Awaited<ReturnType<typeof getProbeAttribution>> = [];
	if (dbExists()) {
		const con = await connect();
		try {
			const reader = await con.runAndReadAll(
				`SELECT run_id FROM score_run
				 WHERE status='succeeded' AND run_id IN (
				   SELECT DISTINCT run_id FROM scorer_step
				   WHERE stmt_hash='${params.stmt_hash.replace(/'/g, "''")}'
				 )
				 ORDER BY started_at DESC LIMIT 1`
			);
			const r = reader.getRowObjects();
			if (r.length > 0) {
				const runId = r[0].run_id as string;
				probes = await getProbeAttribution(runId, params.stmt_hash);
			}
		} finally {
			con.disconnectSync?.();
		}
	}

	return { detail, probes };
};
