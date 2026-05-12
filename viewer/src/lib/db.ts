/**
 * DuckDB connection helper.
 *
 * The viewer reads from a single .duckdb file produced by the corpus loader.
 * Path resolves from the VIEWER_DUCKDB_PATH env var, falling back to a default
 * inside the project's data/ folder.
 */

import { DuckDBInstance, type DuckDBConnection } from '@duckdb/node-api';
import { existsSync, statSync } from 'node:fs';
import { resolve } from 'node:path';
import {
	computeAttributions,
	summarizeAcrossEvidences,
	type ProbeAttribution,
	type ProbeKind,
	type ProbeOutput,
	type ProbeConfidence,
	type ProbeSource
} from './probeAttribution';

export type { ProbeAttribution, ProbeKind, ProbeOutput } from './probeAttribution';

const DEFAULT_DB_PATH = resolve(
	process.cwd(),
	'..',
	'data',
	'corpus.duckdb'
);

let _instance: DuckDBInstance | null = null;
let _resolvedPath = '';
let _instanceMtimeMs = 0;

export function dbPath(): string {
	if (_resolvedPath) return _resolvedPath;
	const env = process.env.VIEWER_DUCKDB_PATH;
	_resolvedPath = env ? resolve(env) : DEFAULT_DB_PATH;
	return _resolvedPath;
}

export function dbExists(): boolean {
	return existsSync(dbPath());
}

export async function connect(): Promise<DuckDBConnection> {
	const path = dbPath();
	// File-mtime invalidation: when the Python loader replaces / rewrites
	// the .duckdb file, our cached READ_ONLY instance was mmap'd against
	// the old contents and serves stale data. Re-instantiate when mtime
	// changes (or the file was created since the cached instance opened).
	let currentMtime = 0;
	try {
		currentMtime = statSync(path).mtimeMs;
	} catch {
		// File may not exist yet
	}
	if (_instance && currentMtime > _instanceMtimeMs) {
		try {
			(_instance as unknown as { closeSync?: () => void }).closeSync?.();
		} catch {
			// best-effort
		}
		_instance = null;
	}
	if (!_instance) {
		// READ_ONLY so the Python loader can hold the writer lock concurrently.
		// DuckDB allows many concurrent readers + one writer.
		_instance = await DuckDBInstance.create(path, { access_mode: 'READ_ONLY' });
		_instanceMtimeMs = currentMtime;
	}
	return _instance.connect();
}

export interface CorpusOverview {
	dbPath: string;
	dbExists: boolean;
	statementCount: number;
	evidenceCount: number;
	agentCount: number;
	supportsEdgeCount: number;
	truthLabelCount: number;
	truthSets: Array<{ id: string; name: string; rowCount: number }>;
	sourceDumps: Array<{ source_dump_id: string | null; n: number }>;
	indraTypes: Array<{ indra_type: string; n: number }>;
	scorerRuns: Array<{
		run_id: string;
		scorer_version: string;
		started_at: string;
		status: string;
		n_stmts: number | null;
		cost_estimate_usd: number | null;
		mae: number | null;
		bias: number | null;
		hasIndraExport: boolean;
		hasCardExport: boolean;
	}>;
	latestValidity: LatestValidity | null;
}

export interface LatestValidity {
	run_id: string;
	scorer_version: string;
	verdicts: Array<{ verdict: string; n: number }>;
	calibration: { mae: number | null; rmse: number | null; bias: number | null; n_stmts: number | null };
	inter_evidence_consistency: { mean_stdev: number | null; n_multi_ev: number | null };
	supports_graph_delta: number | null;
	byIndraType: StratumRow[];
	bySourceApi: StratumRow[];
	truthPresent: TruthPresentRow[];
}

export interface TruthPresentRow {
	truth_set_id: string;
	step_kind: string;
	precision: number;
	recall: number;
	f1: number;
	n_compared: number;
	tp: number;
	fp: number;
	fn: number;
}

export interface StratumRow {
	value: string;
	n: number;
	mae: number;
	bias: number;
}

const EMPTY_OVERVIEW: Omit<CorpusOverview, 'dbPath' | 'dbExists'> = {
	statementCount: 0,
	evidenceCount: 0,
	agentCount: 0,
	supportsEdgeCount: 0,
	truthLabelCount: 0,
	truthSets: [],
	sourceDumps: [],
	indraTypes: [],
	scorerRuns: [],
	latestValidity: null
};

async function getLatestValidity(con: DuckDBConnection): Promise<LatestValidity | null> {
	const runs = await rows<{ run_id: string; scorer_version: string }>(
		con,
		`SELECT run_id, scorer_version FROM score_run
		 WHERE status = 'succeeded'
		 ORDER BY started_at DESC LIMIT 1`
	);
	if (runs.length === 0) return null;
	const { run_id, scorer_version } = runs[0];

	const [verdicts, mae, rmse, bias, calN, consistency, supports, byIndraType, bySourceApi, truthPresent] = await Promise.all([
		rows<{ verdict: string; n: number }>(
			con,
			`SELECT replace(replace(metric_name, 'verdict_share.', ''), '"', '') AS verdict,
			        CAST(json_extract(slice_json, '$.n') AS BIGINT) AS n
			 FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}' AND metric_name LIKE 'verdict_share.%'
			 ORDER BY n DESC`
		),
		rows<{ value: number }>(
			con,
			`SELECT value FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'indra_belief_calibration.mae' AND truth_set_id = 'indra_published_belief' LIMIT 1`
		),
		rows<{ value: number }>(
			con,
			`SELECT value FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'indra_belief_calibration.rmse' AND truth_set_id = 'indra_published_belief' LIMIT 1`
		),
		rows<{ value: number }>(
			con,
			`SELECT value FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'indra_belief_calibration.bias' AND truth_set_id = 'indra_published_belief' LIMIT 1`
		),
		rows<{ n: number }>(
			con,
			`SELECT CAST(json_extract(slice_json, '$.n_stmts') AS BIGINT) AS n
			 FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'indra_belief_calibration.mae' AND truth_set_id = 'indra_published_belief' LIMIT 1`
		),
		rows<{ value: number; n: number }>(
			con,
			`SELECT value, CAST(json_extract(slice_json, '$.n_multi_evidence_stmts') AS BIGINT) AS n
			 FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'inter_evidence_consistency.mean_stdev' LIMIT 1`
		),
		rows<{ value: number }>(
			con,
			`SELECT value FROM metric
			 WHERE run_id = '${run_id.replace(/'/g, "''")}'
			   AND metric_name = 'supports_graph_plausibility.delta' LIMIT 1`
		),
		rows<StratumRow>(
			con,
			`WITH mae_rows AS (
				SELECT json_extract(slice_json, '$.value')::VARCHAR AS value,
				       CAST(json_extract(slice_json, '$.n') AS BIGINT) AS n,
				       value AS mae
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name = 'indra_belief_calibration_by_type.mae' AND truth_set_id = 'indra_published_belief'
			),
			bias_rows AS (
				SELECT json_extract(slice_json, '$.value')::VARCHAR AS value,
				       value AS bias
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name = 'indra_belief_calibration_by_type.bias' AND truth_set_id = 'indra_published_belief'
			)
			SELECT replace(m.value, '"', '') AS value, m.n, m.mae, COALESCE(b.bias, 0) AS bias
			FROM mae_rows m LEFT JOIN bias_rows b USING(value)
			ORDER BY m.mae DESC`
		),
		rows<StratumRow>(
			con,
			`WITH mae_rows AS (
				SELECT json_extract(slice_json, '$.value')::VARCHAR AS value,
				       CAST(json_extract(slice_json, '$.n') AS BIGINT) AS n,
				       value AS mae
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name = 'indra_belief_calibration_by_source.mae' AND truth_set_id = 'indra_published_belief'
			),
			bias_rows AS (
				SELECT json_extract(slice_json, '$.value')::VARCHAR AS value,
				       value AS bias
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name = 'indra_belief_calibration_by_source.bias' AND truth_set_id = 'indra_published_belief'
			)
			SELECT replace(m.value, '"', '') AS value, m.n, m.mae, COALESCE(b.bias, 0) AS bias
			FROM mae_rows m LEFT JOIN bias_rows b USING(value)
			ORDER BY m.mae DESC`
		),
		rows<TruthPresentRow>(
			con,
			`WITH p AS (
				SELECT truth_set_id,
				       replace(json_extract(slice_json, '$.step_kind')::VARCHAR, '"', '') AS step_kind,
				       value AS precision,
				       CAST(json_extract(slice_json, '$.n_compared') AS BIGINT) AS n_compared,
				       CAST(json_extract(slice_json, '$.tp') AS BIGINT) AS tp,
				       CAST(json_extract(slice_json, '$.fp') AS BIGINT) AS fp,
				       CAST(json_extract(slice_json, '$.fn') AS BIGINT) AS fn
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name LIKE 'truth_present.%.precision'
			),
			r AS (
				SELECT truth_set_id,
				       replace(json_extract(slice_json, '$.step_kind')::VARCHAR, '"', '') AS step_kind,
				       value AS recall
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name LIKE 'truth_present.%.recall'
			),
			f AS (
				SELECT truth_set_id,
				       replace(json_extract(slice_json, '$.step_kind')::VARCHAR, '"', '') AS step_kind,
				       value AS f1
				FROM metric
				WHERE run_id = '${run_id.replace(/'/g, "''")}'
				  AND metric_name LIKE 'truth_present.%.f1'
			)
			SELECT p.truth_set_id, p.step_kind,
			       p.precision, COALESCE(r.recall, 0) AS recall, COALESCE(f.f1, 0) AS f1,
			       p.n_compared, p.tp, p.fp, p.fn
			FROM p
			LEFT JOIN r USING (truth_set_id, step_kind)
			LEFT JOIN f USING (truth_set_id, step_kind)
			ORDER BY p.truth_set_id, p.step_kind`
		)
	]);

	const cleanVal = (rs: { value: number }[]) => {
		if (rs.length === 0) return null;
		const v = rs[0].value;
		return typeof v === 'number' && !Number.isNaN(v) ? v : null;
	};

	return {
		run_id,
		scorer_version,
		verdicts,
		calibration: {
			mae: cleanVal(mae),
			rmse: cleanVal(rmse),
			bias: cleanVal(bias),
			n_stmts: calN[0]?.n ?? null
		},
		inter_evidence_consistency: {
			mean_stdev: consistency.length && !Number.isNaN(consistency[0].value)
				? consistency[0].value
				: null,
			n_multi_ev: consistency[0]?.n ?? null
		},
		supports_graph_delta: cleanVal(supports),
		byIndraType,
		bySourceApi,
		truthPresent
	};
}

async function scalar(con: DuckDBConnection, sql: string): Promise<number> {
	try {
		const reader = await con.runAndReadAll(sql);
		const rows = reader.getRowObjects();
		if (rows.length === 0) return 0;
		const v = Object.values(rows[0])[0];
		return typeof v === 'bigint' ? Number(v) : Number(v ?? 0);
	} catch {
		return 0;
	}
}

/**
 * Normalize DuckDB result values:
 *  - bigint → Number (UI works in JS numbers)
 *  - LIST of STRUCT → plain array of plain objects
 *    DuckDB wraps `list(struct(...))` as `{items: [{entries: {...}}]}` via
 *    @duckdb/node-api. Strip the wrapper here so call sites can treat the
 *    column as a plain array of objects.
 *  - STRUCT (non-list) → plain object (strip `entries`)
 */
function normalizeDuckValue(v: unknown): unknown {
	if (typeof v === 'bigint') return Number(v);
	if (v && typeof v === 'object') {
		const obj = v as Record<string, unknown>;
		if ('items' in obj && Array.isArray(obj.items) && Object.keys(obj).length === 1) {
			return obj.items.map((it) => normalizeDuckValue(it));
		}
		if ('entries' in obj && obj.entries && typeof obj.entries === 'object' && Object.keys(obj).length === 1) {
			const e = obj.entries as Record<string, unknown>;
			const out: Record<string, unknown> = {};
			for (const [k, val] of Object.entries(e)) out[k] = normalizeDuckValue(val);
			return out;
		}
	}
	return v;
}

async function rows<T = Record<string, unknown>>(
	con: DuckDBConnection,
	sql: string
): Promise<T[]> {
	try {
		const reader = await con.runAndReadAll(sql);
		return reader.getRowObjects().map((row) => {
			const out: Record<string, unknown> = {};
			for (const [k, v] of Object.entries(row)) {
				out[k] = normalizeDuckValue(v);
			}
			return out as T;
		});
	} catch {
		return [];
	}
}

export interface StatementMatrixRow {
	stmt_hash: string;
	indra_type: string;
	indra_belief: number | null;
	agent_names: string;
	n_evidences: number;
	supports_count: number;
	supported_by_count: number;
	source_apis: string;
	source_dump_id: string | null;
	is_curated_any: number;
	our_belief: number | null;
	belief_delta: number | null;
}

export async function getStatementMatrix(): Promise<StatementMatrixRow[]> {
	if (!dbExists()) return [];
	const con = await connect();
	try {
		// Latest succeeded run, used to surface our_belief + belief_delta
		const latestRun = await rows<{ run_id: string }>(
			con,
			`SELECT run_id FROM score_run
			 WHERE status = 'succeeded' ORDER BY started_at DESC LIMIT 1`
		);
		const runId = latestRun[0]?.run_id ?? null;
		const runJoin = runId
			? `LEFT JOIN (
					SELECT stmt_hash,
					       AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS our_belief
					FROM scorer_step
					WHERE run_id = '${runId.replace(/'/g, "''")}'
					  AND step_kind = 'aggregate'
					  AND json_extract(output_json, '$.score') IS NOT NULL
					GROUP BY stmt_hash
				) ours ON ours.stmt_hash = s.stmt_hash`
			: 'LEFT JOIN (SELECT NULL::VARCHAR AS stmt_hash, NULL::DOUBLE AS our_belief WHERE FALSE) ours ON FALSE';

		const MATRIX_LIMIT = 50_000;
		const matrixRows = await rows<StatementMatrixRow>(
			con,
			`SELECT
			   s.stmt_hash,
			   s.indra_type,
			   s.indra_belief,
			   COALESCE(string_agg(DISTINCT a.name, ', ' ORDER BY a.name), '') AS agent_names,
			   COUNT(DISTINCT e.evidence_hash) AS n_evidences,
			   s.supports_count,
			   s.supported_by_count,
			   COALESCE(string_agg(DISTINCT e.source_api, ',' ORDER BY e.source_api), '') AS source_apis,
			   s.source_dump_id,
			   MAX(CASE WHEN e.is_curated THEN 1 ELSE 0 END) AS is_curated_any,
			   ours.our_belief AS our_belief,
			   CASE WHEN ours.our_belief IS NOT NULL AND s.indra_belief IS NOT NULL
			        THEN ours.our_belief - s.indra_belief
			        ELSE NULL END AS belief_delta
			 FROM statement s
			 LEFT JOIN evidence e ON e.stmt_hash = s.stmt_hash
			 LEFT JOIN agent a ON a.stmt_hash = s.stmt_hash
			 ${runJoin}
			 GROUP BY s.stmt_hash, s.indra_type, s.indra_belief,
			          s.supports_count, s.supported_by_count, s.source_dump_id,
			          ours.our_belief
			 ORDER BY ABS(COALESCE(ours.our_belief - s.indra_belief, 0)) DESC, s.stmt_hash
			 LIMIT ${MATRIX_LIMIT}`
		);
		// Warn at the ceiling — silent truncation lies about coverage. At
		// 50K, a corpus larger than rasmachine has been loaded and the
		// matrix should grow server-side pagination rather than truncate.
		if (matrixRows.length >= MATRIX_LIMIT) {
			console.warn(
				`getStatementMatrix: hit LIMIT ${MATRIX_LIMIT} — corpus ` +
				`exceeds matrix ceiling, results truncated. Add server-side ` +
				`pagination if this becomes routine.`
			);
		}
		return matrixRows;
	} finally {
		con.disconnectSync?.();
	}
}

export interface StatementDetail {
	stmt_hash: string;
	indra_type: string;
	indra_belief: number | null;
	supports_count: number;
	supported_by_count: number;
	source_dump_id: string | null;
	raw_json: string;
	agents: AgentRow[];
	evidences: EvidenceRow[];
	truth_labels: TruthLabelRow[];
	registered_truth_sets: string[];
	supports_edges: SupportsEdgeRow[];
	scorer_steps: ScorerStepRow[];
}

export interface ScorerStepRow {
	step_hash: string;
	evidence_hash: string | null;
	scorer_version: string;
	model_id: string | null;
	step_kind: string;
	is_substrate_answered: boolean | null;
	input_payload_json: string | null;
	output_json: string;
	latency_ms: number | null;
	prompt_tokens: number | null;
	out_tokens: number | null;
	finish_reason: string | null;
	error: string | null;
}

export interface AgentRow {
	agent_hash: string;
	role: string;
	role_index: number;
	name: string;
	db_refs_json: string;
	mods_json: string | null;
	location: string | null;
}

export interface EvidenceRow {
	evidence_hash: string;
	source_api: string | null;
	source_id: string | null;
	pmid: string | null;
	text: string | null;
	is_direct: boolean | null;
	is_negated: boolean | null;
	is_curated: boolean | null;
	epistemics_json: string | null;
}

export interface TruthLabelRow {
	truth_set_id: string;
	target_kind: string;
	target_id: string;
	field: string;
	value_text: string | null;
	value_json: string | null;
	provenance: string | null;
}

export interface SupportsEdgeRow {
	from_stmt_hash: string;
	to_stmt_hash: string;
	kind: string;
}

export interface RunNarrative {
	run_id: string;
	prev_run_id: string | null;
	summary_sentence: string;
	mae_delta: number | null;
	bias_delta: number | null;
	verdicts_moved_total: number;
	verdicts_moved_to_correct: number;
	verdicts_moved_to_incorrect: number;
	verdict_crossings: Array<{ stmt_hash: string; prev_verdict: string; curr_verdict: string }>;
}

/**
 * Run-over-run narrative for the dashboard's runs feed and the /runs/[id]
 * page. Compares the named run to its predecessor by started_at. When there
 * is no predecessor, returns a self-summary ("first run, MAE 0.187").
 */
export async function getRunNarrative(run_id: string): Promise<RunNarrative | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const qr = run_id.replace(/'/g, "''");
		const prevRows = await rows<{ run_id: string }>(
			con,
			`SELECT run_id FROM score_run
			 WHERE status='succeeded'
			   AND started_at < (SELECT started_at FROM score_run WHERE run_id='${qr}')
			 ORDER BY started_at DESC LIMIT 1`
		);
		const prev_run_id = prevRows[0]?.run_id ?? null;

		const calRow = await rows<{ mae: number | null; bias: number | null }>(
			con,
			`SELECT
			   (SELECT value FROM metric WHERE run_id='${qr}' AND metric_name='indra_belief_calibration.mae' AND truth_set_id='indra_published_belief' LIMIT 1) AS mae,
			   (SELECT value FROM metric WHERE run_id='${qr}' AND metric_name='indra_belief_calibration.bias' AND truth_set_id='indra_published_belief' LIMIT 1) AS bias`
		);
		const mae = calRow[0]?.mae ?? null;
		const bias = calRow[0]?.bias ?? null;

		let mae_delta: number | null = null;
		let bias_delta: number | null = null;
		const crossings: Array<{ stmt_hash: string; prev_verdict: string; curr_verdict: string }> = [];

		if (prev_run_id) {
			const qp = prev_run_id.replace(/'/g, "''");
			const prevCalRow = await rows<{ mae: number | null; bias: number | null }>(
				con,
				`SELECT
				   (SELECT value FROM metric WHERE run_id='${qp}' AND metric_name='indra_belief_calibration.mae' AND truth_set_id='indra_published_belief' LIMIT 1) AS mae,
				   (SELECT value FROM metric WHERE run_id='${qp}' AND metric_name='indra_belief_calibration.bias' AND truth_set_id='indra_published_belief' LIMIT 1) AS bias`
			);
			const prev_mae = prevCalRow[0]?.mae ?? null;
			const prev_bias = prevCalRow[0]?.bias ?? null;
			if (mae != null && prev_mae != null) mae_delta = mae - prev_mae;
			if (bias != null && prev_bias != null) bias_delta = bias - prev_bias;

			const verdictRows = await rows<{ stmt_hash: string; curr_verdict: string; prev_verdict: string }>(
				con,
				`WITH curr AS (
					SELECT stmt_hash,
					       MAX(replace(json_extract_string(output_json, '$.verdict'), '"', '')) AS verdict
					FROM scorer_step WHERE run_id='${qr}' AND step_kind='aggregate' GROUP BY stmt_hash
				),
				prev AS (
					SELECT stmt_hash,
					       MAX(replace(json_extract_string(output_json, '$.verdict'), '"', '')) AS verdict
					FROM scorer_step WHERE run_id='${qp}' AND step_kind='aggregate' GROUP BY stmt_hash
				)
				SELECT curr.stmt_hash AS stmt_hash,
				       curr.verdict AS curr_verdict,
				       prev.verdict AS prev_verdict
				FROM curr JOIN prev USING(stmt_hash)
				WHERE curr.verdict <> prev.verdict`
			);
			for (const r of verdictRows) {
				crossings.push({ stmt_hash: r.stmt_hash, prev_verdict: r.prev_verdict, curr_verdict: r.curr_verdict });
			}
		}

		const moved_to_correct = crossings.filter((c) => c.curr_verdict === 'correct').length;
		const moved_to_incorrect = crossings.filter((c) => c.curr_verdict === 'incorrect').length;

		const fmtSigned3 = (n: number) => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(3)}`;
		let summary_sentence: string;
		if (!prev_run_id) {
			summary_sentence =
				mae != null
					? `first comparable run · MAE ${mae.toFixed(3)} · bias ${bias != null ? fmtSigned3(bias) : '—'}`
					: 'first run, no calibration data';
		} else {
			const parts: string[] = [];
			if (mae_delta != null) parts.push(`MAE ${fmtSigned3(mae_delta)}`);
			if (bias_delta != null) parts.push(`bias ${fmtSigned3(bias_delta)}`);
			if (crossings.length > 0) {
				parts.push(
					`${crossings.length} verdict${crossings.length === 1 ? '' : 's'} moved (` +
						`${moved_to_correct} to correct, ${moved_to_incorrect} to incorrect)`
				);
			} else {
				parts.push('no verdicts moved');
			}
			summary_sentence = parts.join(' · ');
		}

		return {
			run_id,
			prev_run_id,
			summary_sentence,
			mae_delta,
			bias_delta,
			verdicts_moved_total: crossings.length,
			verdicts_moved_to_correct: moved_to_correct,
			verdicts_moved_to_incorrect: moved_to_incorrect,
			verdict_crossings: crossings
		};
	} finally {
		con.disconnectSync?.();
	}
}

export interface ProbeCoverageRow {
	probe: ProbeKind;
	total: number;
	substrate_n: number;
	llm_n: number;
	abstain_n: number;
	notrun_n: number;
}

export interface HeuristicCoverage {
	run_id: string;
	n_evidences: number;
	per_probe: ProbeCoverageRow[];
	/** % of evidences where every invoked probe was substrate-answered. */
	all_substrate_rate: number;
	/** % of evidences where at least one probe didn't run (short-circuit). */
	short_circuited_rate: number;
	/** Mean count of probes that fired (substrate + LLM, not "not run") per evidence. */
	mean_invoked_probes: number;
}

/**
 * Per-probe substrate/LLM/abstain coverage for a single run.
 *
 * Reads two sources and unions:
 *   1. substrate_route rows expose per-probe slot results.
 *   2. Individual {probe}_probe rows are written when substrate answers
 *      OR when LLM fires its own probe step.
 *
 * Coverage = the "final" source per (evidence, probe), with substrate-route's
 * slot taking precedence (it's authoritative for what substrate decided).
 * Missing data on both sides means "probe did not run for this evidence" —
 * counted in `notrun_n`, not silently dropped.
 */
export async function getHeuristicCoverage(run_id: string): Promise<HeuristicCoverage | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const qr = run_id.replace(/'/g, "''");
		const probes: ProbeKind[] = ['subject_role', 'object_role', 'relation_axis', 'scope'];
		const slotUnions = probes
			.map(
				(p) => `
				SELECT evidence_hash, '${p}' AS probe,
				       json_extract_string(output_json, '$.${p}.source') AS substrate_source,
				       json_extract_string(output_json, '$.${p}.answer') AS substrate_answer
				FROM scorer_step
				WHERE run_id='${qr}' AND step_kind='substrate_route'`
			)
			.join(' UNION ALL ');
		const sql = `
			WITH substrate_slots AS (${slotUnions}),
			individual_probes AS (
				SELECT evidence_hash,
				       replace(step_kind, '_probe', '') AS probe,
				       json_extract_string(output_json, '$.source') AS llm_source,
				       json_extract_string(output_json, '$.answer') AS llm_answer
				FROM scorer_step
				WHERE run_id='${qr}'
				  AND step_kind IN ('subject_role_probe','object_role_probe','relation_axis_probe','scope_probe')
			),
			joined AS (
				SELECT s.evidence_hash, s.probe,
				       CASE
				         WHEN s.substrate_source = 'substrate' THEN 'substrate'
				         WHEN ip.llm_source = 'llm' THEN 'llm'
				         WHEN ip.llm_source = 'abstain' OR ip.llm_answer = 'abstain' THEN 'abstain'
				         WHEN ip.llm_source IS NOT NULL THEN ip.llm_source
				         ELSE NULL
				       END AS final_source
				FROM substrate_slots s
				LEFT JOIN individual_probes ip
				  ON ip.evidence_hash = s.evidence_hash AND ip.probe = s.probe
			),
			per_evidence AS (
				SELECT evidence_hash,
				       SUM(CASE WHEN final_source = 'substrate' THEN 1 ELSE 0 END) AS substrate_count,
				       SUM(CASE WHEN final_source = 'llm' THEN 1 ELSE 0 END) AS llm_count,
				       SUM(CASE WHEN final_source = 'abstain' THEN 1 ELSE 0 END) AS abstain_count,
				       SUM(CASE WHEN final_source IS NULL THEN 1 ELSE 0 END) AS notrun_count
				FROM joined
				GROUP BY evidence_hash
			)
			SELECT
				probe,
				COUNT(*) AS total,
				SUM(CASE WHEN final_source='substrate' THEN 1 ELSE 0 END) AS substrate_n,
				SUM(CASE WHEN final_source='llm' THEN 1 ELSE 0 END) AS llm_n,
				SUM(CASE WHEN final_source='abstain' THEN 1 ELSE 0 END) AS abstain_n,
				SUM(CASE WHEN final_source IS NULL THEN 1 ELSE 0 END) AS notrun_n
			FROM joined
			GROUP BY probe
			ORDER BY probe`;
		const perProbeRaw = await rows<{
			probe: string;
			total: number;
			substrate_n: number;
			llm_n: number;
			abstain_n: number;
			notrun_n: number;
		}>(con, sql);
		const per_probe: ProbeCoverageRow[] = perProbeRaw.map((r) => ({
			probe: r.probe as ProbeKind,
			total: r.total,
			substrate_n: r.substrate_n,
			llm_n: r.llm_n,
			abstain_n: r.abstain_n,
			notrun_n: r.notrun_n
		}));

		const aggRows = await rows<{
			n_evidences: number;
			all_substrate: number;
			short_circuited: number;
			mean_invoked: number;
		}>(
			con,
			`WITH substrate_slots AS (${slotUnions}),
			individual_probes AS (
				SELECT evidence_hash,
				       replace(step_kind, '_probe', '') AS probe,
				       json_extract_string(output_json, '$.source') AS llm_source,
				       json_extract_string(output_json, '$.answer') AS llm_answer
				FROM scorer_step
				WHERE run_id='${qr}'
				  AND step_kind IN ('subject_role_probe','object_role_probe','relation_axis_probe','scope_probe')
			),
			joined AS (
				SELECT s.evidence_hash, s.probe,
				       CASE
				         WHEN s.substrate_source = 'substrate' THEN 'substrate'
				         WHEN ip.llm_source = 'llm' THEN 'llm'
				         WHEN ip.llm_source = 'abstain' OR ip.llm_answer = 'abstain' THEN 'abstain'
				         WHEN ip.llm_source IS NOT NULL THEN ip.llm_source
				         ELSE NULL
				       END AS final_source
				FROM substrate_slots s
				LEFT JOIN individual_probes ip
				  ON ip.evidence_hash = s.evidence_hash AND ip.probe = s.probe
			),
			per_evidence AS (
				SELECT evidence_hash,
				       SUM(CASE WHEN final_source='substrate' THEN 1 ELSE 0 END) AS substrate_count,
				       SUM(CASE WHEN final_source='llm' THEN 1 ELSE 0 END) AS llm_count,
				       SUM(CASE WHEN final_source IS NULL THEN 1 ELSE 0 END) AS notrun_count
				FROM joined
				GROUP BY evidence_hash
			)
			SELECT
				COUNT(*) AS n_evidences,
				SUM(CASE WHEN notrun_count=0 AND llm_count=0 THEN 1 ELSE 0 END) AS all_substrate,
				SUM(CASE WHEN notrun_count > 0 THEN 1 ELSE 0 END) AS short_circuited,
				AVG(substrate_count + llm_count) AS mean_invoked
			FROM per_evidence`
		);

		const agg = aggRows[0] ?? { n_evidences: 0, all_substrate: 0, short_circuited: 0, mean_invoked: 0 };
		const n_evidences = agg.n_evidences;
		return {
			run_id,
			n_evidences,
			per_probe,
			all_substrate_rate: n_evidences > 0 ? agg.all_substrate / n_evidences : 0,
			short_circuited_rate: n_evidences > 0 ? agg.short_circuited / n_evidences : 0,
			mean_invoked_probes: agg.mean_invoked ?? 0
		};
	} finally {
		con.disconnectSync?.();
	}
}

export interface ResidualDistribution {
	run_id: string;
	bins: number[];
	n_total: number;
	mean_residual: number | null;
}

/**
 * Histogram of (our_belief − indra_belief) for the latest succeeded run.
 * Always 11 bins on [-1, +1]; bin 5 (index 5) is the zero-centered bucket.
 */
export async function getResidualDistribution(run_id?: string): Promise<ResidualDistribution | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		let resolved = run_id ?? null;
		if (!resolved) {
			const r = await rows<{ run_id: string }>(
				con,
				`SELECT run_id FROM score_run WHERE status='succeeded' ORDER BY started_at DESC LIMIT 1`
			);
			resolved = r[0]?.run_id ?? null;
		}
		if (!resolved) return null;
		const qr = resolved.replace(/'/g, "''");
		const residualRows = await rows<{ residual: number }>(
			con,
			`WITH ours AS (
				SELECT stmt_hash,
				       AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS our
				FROM scorer_step
				WHERE run_id='${qr}' AND step_kind='aggregate'
				  AND json_extract(output_json, '$.score') IS NOT NULL
				GROUP BY stmt_hash
			)
			SELECT (ours.our - s.indra_belief) AS residual
			FROM statement s
			JOIN ours ON ours.stmt_hash = s.stmt_hash
			WHERE s.indra_belief IS NOT NULL AND ours.our IS NOT NULL`
		);
		const bins = new Array(11).fill(0);
		let sum = 0;
		for (const r of residualRows) {
			const v = r.residual;
			if (typeof v !== 'number' || Number.isNaN(v)) continue;
			sum += v;
			const clamped = Math.max(-1, Math.min(1, v));
			const idx = Math.min(10, Math.max(0, Math.floor((clamped + 1) * 5.5)));
			bins[idx] += 1;
		}
		return {
			run_id: resolved,
			bins,
			n_total: residualRows.length,
			mean_residual: residualRows.length > 0 ? sum / residualRows.length : null
		};
	} finally {
		con.disconnectSync?.();
	}
}

export type FindingKind =
	| 'biggest_disagreement'
	| 'probe_split'
	| 'verdict_regression'
	| 'verdict_recovery'
	| 'low_confidence_high_stakes';

export interface FindingRow {
	kind: FindingKind;
	stmt_hash: string;
	indra_type: string;
	agents: Array<{ role: string; name: string }>;
	our_score: number | null;
	indra_score: number | null;
	n_evidences: number;
	why_text: string;
	/** Sort key used to pick the top-K. Always > 0 for ranking purposes. */
	rank_value: number;
}

export interface Findings {
	run_id: string;
	prev_run_id: string | null;
	biggest_disagreement: FindingRow[];
	probe_split: FindingRow[];
	verdict_regression: FindingRow[];
	verdict_recovery: FindingRow[];
	low_confidence_high_stakes: FindingRow[];
}

const FINDING_K = 5;

/**
 * Rank the latest run's statements along five "interesting" axes. Each lane
 * returns up to K rows with the same shape; the UI renders them as compact
 * BeliefPrimitive cards. Empty lanes are normal — e.g. no prev_run means no
 * regressions can be computed.
 */
export async function getFindings(): Promise<Findings | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const runs = await rows<{ run_id: string }>(
			con,
			`SELECT run_id FROM score_run WHERE status='succeeded' ORDER BY started_at DESC LIMIT 2`
		);
		if (runs.length === 0) return null;
		const run_id = runs[0].run_id;
		const prev_run_id = runs.length > 1 ? runs[1].run_id : null;
		const qr = run_id.replace(/'/g, "''");
		const qp = prev_run_id ? prev_run_id.replace(/'/g, "''") : null;

		const baseSelect = (whereExtra: string, orderBy: string) => `
			WITH ours AS (
				SELECT stmt_hash,
				       AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS our,
				       MAX(replace(json_extract_string(output_json, '$.verdict'), '"', '')) AS verdict
				FROM scorer_step
				WHERE run_id = '${qr}'
				  AND step_kind = 'aggregate'
				GROUP BY stmt_hash
			),
			ev_n AS (
				SELECT stmt_hash, COUNT(*) AS n_evidences FROM evidence GROUP BY stmt_hash
			),
			ags AS (
				SELECT stmt_hash,
				       list({role: role, name: name}
				            ORDER BY CASE role
				              WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				              WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				              WHEN 'member' THEN 2 ELSE 3 END,
				            role_index) AS agents
				FROM agent GROUP BY stmt_hash
			)
			SELECT s.stmt_hash, s.indra_type, s.indra_belief AS indra_score,
			       ours.our AS our_score, ours.verdict AS verdict,
			       COALESCE(ev_n.n_evidences, 0) AS n_evidences,
			       ags.agents AS agents
			FROM statement s
			JOIN ours ON ours.stmt_hash = s.stmt_hash
			LEFT JOIN ev_n ON ev_n.stmt_hash = s.stmt_hash
			LEFT JOIN ags ON ags.stmt_hash = s.stmt_hash
			${whereExtra}
			ORDER BY ${orderBy}
			LIMIT ${FINDING_K}`;

		type Row = {
			stmt_hash: string;
			indra_type: string;
			indra_score: number | null;
			our_score: number | null;
			verdict: string | null;
			n_evidences: number;
			agents: Array<{ role: string; name: string }> | null;
		};

		const [disagreeRows, probeSplitRows, lowConfRows, prevAggRows] = await Promise.all([
			rows<Row>(
				con,
				baseSelect(
					`WHERE s.indra_belief IS NOT NULL AND ours.our IS NOT NULL`,
					`ABS(ours.our - s.indra_belief) DESC, s.stmt_hash`
				)
			),
			rows<Row & { probe_stdev: number }>(
				con,
				`WITH probe_votes AS (
					SELECT stmt_hash,
					       CASE
					         WHEN step_kind = 'subject_role_probe' AND json_extract_string(output_json, '$.answer') = 'present_as_subject' THEN 1.0
					         WHEN step_kind = 'object_role_probe'  AND json_extract_string(output_json, '$.answer') = 'present_as_object'  THEN 1.0
					         WHEN step_kind = 'relation_axis_probe' AND json_extract_string(output_json, '$.answer') = 'direct_sign_match' THEN 1.0
					         WHEN step_kind = 'scope_probe' AND json_extract_string(output_json, '$.answer') = 'asserted' THEN 1.0
					         WHEN step_kind = 'scope_probe' AND json_extract_string(output_json, '$.answer') = 'negated'  THEN -1.0
					         WHEN json_extract_string(output_json, '$.answer') IN ('absent','present_as_decoy','direct_sign_mismatch','direct_axis_mismatch','no_relation') THEN -1.0
					         ELSE 0
					       END AS vote
					FROM scorer_step
					WHERE run_id = '${qr}'
					  AND step_kind IN ('subject_role_probe','object_role_probe','relation_axis_probe','scope_probe')
				),
				stdevs AS (
					SELECT stmt_hash, STDDEV_POP(vote) AS probe_stdev
					FROM probe_votes GROUP BY stmt_hash
				)
				SELECT s.stmt_hash, s.indra_type, s.indra_belief AS indra_score,
				       (SELECT AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE))
				        FROM scorer_step WHERE run_id='${qr}'
				          AND step_kind='aggregate' AND stmt_hash=s.stmt_hash) AS our_score,
				       (SELECT MAX(replace(json_extract_string(output_json, '$.verdict'), '"', ''))
				        FROM scorer_step WHERE run_id='${qr}'
				          AND step_kind='aggregate' AND stmt_hash=s.stmt_hash) AS verdict,
				       (SELECT COUNT(*) FROM evidence WHERE stmt_hash=s.stmt_hash) AS n_evidences,
				       (SELECT list({role: role, name: name}
				                    ORDER BY CASE role
				                      WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				                      WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				                      WHEN 'member' THEN 2 ELSE 3 END,
				                    role_index)
				        FROM agent WHERE stmt_hash=s.stmt_hash) AS agents,
				       stdevs.probe_stdev AS probe_stdev
				FROM statement s
				JOIN stdevs ON stdevs.stmt_hash = s.stmt_hash
				WHERE stdevs.probe_stdev > 0
				ORDER BY stdevs.probe_stdev DESC, s.stmt_hash
				LIMIT ${FINDING_K}`
			),
			rows<Row>(
				con,
				baseSelect(
					`WHERE ours.our BETWEEN 0.4 AND 0.6
					  AND COALESCE(ev_n.n_evidences, 0) >= 3`,
					`ev_n.n_evidences DESC, ABS(ours.our - 0.5) ASC, s.stmt_hash`
				)
			),
			prev_run_id
				? rows<{ stmt_hash: string; verdict: string }>(
						con,
						`SELECT stmt_hash,
						        MAX(replace(json_extract_string(output_json, '$.verdict'), '"', '')) AS verdict
						 FROM scorer_step WHERE run_id='${qp}' AND step_kind='aggregate'
						 GROUP BY stmt_hash`
					)
				: Promise.resolve([])
		]);

		const prevVerdict = new Map(prevAggRows.map((r) => [r.stmt_hash, r.verdict]));
		const allCurrent = await rows<Row>(
			con,
			baseSelect('WHERE 1=1', 's.stmt_hash')
		);
		const verdictMoved = (curr: string | null, prev: string | undefined, dir: 'down' | 'up') => {
			if (!curr || !prev) return false;
			if (dir === 'down') return prev === 'correct' && curr === 'incorrect';
			return prev === 'incorrect' && curr === 'correct';
		};
		const regressions: Row[] = [];
		const recoveries: Row[] = [];
		for (const r of allCurrent) {
			const pv = prevVerdict.get(r.stmt_hash);
			if (verdictMoved(r.verdict, pv, 'down')) regressions.push(r);
			if (verdictMoved(r.verdict, pv, 'up')) recoveries.push(r);
		}

		const toFinding = (
			kind: FindingKind,
			r: Row,
			rank: number,
			why: string
		): FindingRow => ({
			kind,
			stmt_hash: r.stmt_hash,
			indra_type: r.indra_type,
			agents: r.agents ?? [],
			our_score: r.our_score,
			indra_score: r.indra_score,
			n_evidences: r.n_evidences,
			rank_value: rank,
			why_text: why
		});

		const fmtSigned = (n: number) => `${n >= 0 ? '+' : '−'}${Math.abs(n).toFixed(2)}`;

		return {
			run_id,
			prev_run_id,
			biggest_disagreement: disagreeRows.map((r) => {
				const d = (r.our_score ?? 0) - (r.indra_score ?? 0);
				return toFinding(
					'biggest_disagreement',
					r,
					Math.abs(d),
					`Δ ${fmtSigned(d)} vs INDRA · n_ev=${r.n_evidences}`
				);
			}),
			probe_split: probeSplitRows.map((r) =>
				toFinding(
					'probe_split',
					r,
					r.probe_stdev,
					`probe stdev ${r.probe_stdev.toFixed(2)} · disagreement across the four probes`
				)
			),
			verdict_regression: regressions
				.slice(0, FINDING_K)
				.map((r) =>
					toFinding(
						'verdict_regression',
						r,
						1,
						`verdict moved correct → incorrect since prev run`
					)
				),
			verdict_recovery: recoveries
				.slice(0, FINDING_K)
				.map((r) =>
					toFinding(
						'verdict_recovery',
						r,
						1,
						`verdict moved incorrect → correct since prev run`
					)
				),
			low_confidence_high_stakes: lowConfRows.map((r) =>
				toFinding(
					'low_confidence_high_stakes',
					r,
					r.n_evidences,
					`belief ${(r.our_score ?? 0).toFixed(2)} (mid-range) · n_ev=${r.n_evidences} ≥ 3`
				)
			)
		};
	} finally {
		con.disconnectSync?.();
	}
}

export interface FocusStatement {
	run_id: string;
	stmt: {
		stmt_hash: string;
		indra_type: string;
		agents: Array<{ role: string; name: string }>;
	};
	our_score: number | null;
	indra_score: number | null;
	probes: ProbeAttribution[];
	evidences: Array<{ evidence_hash: string; source_api: string | null; text: string | null }>;
	n_evidences: number;
	why_this_one: string;
}

/**
 * Pick a focus statement to lead the dashboard with. Defaults to the
 * highest-|Δ vs INDRA| in the latest succeeded run; can be deep-linked to a
 * specific stmt_hash. Returns null if there's no scoring data yet.
 */
export async function getFocusStatement(
	focus_hash?: string,
	run_id?: string
): Promise<FocusStatement | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		let resolvedRun = run_id ?? null;
		if (!resolvedRun) {
			const r = await rows<{ run_id: string }>(
				con,
				`SELECT run_id FROM score_run WHERE status='succeeded' ORDER BY started_at DESC LIMIT 1`
			);
			resolvedRun = r[0]?.run_id ?? null;
		}
		if (!resolvedRun) return null;

		let resolvedHash = focus_hash ?? null;
		let whyKind: 'biggest_delta' | 'requested' = 'biggest_delta';
		let biggestDelta: number | null = null;
		if (!resolvedHash) {
			const r = await rows<{ stmt_hash: string; delta: number }>(
				con,
				`WITH ours AS (
					SELECT stmt_hash,
					       AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS our
					FROM scorer_step
					WHERE run_id = '${resolvedRun.replace(/'/g, "''")}'
					  AND step_kind = 'aggregate'
					  AND json_extract(output_json, '$.score') IS NOT NULL
					GROUP BY stmt_hash
				)
				SELECT s.stmt_hash AS stmt_hash,
				       ours.our - s.indra_belief AS delta
				FROM statement s
				JOIN ours ON ours.stmt_hash = s.stmt_hash
				WHERE s.indra_belief IS NOT NULL
				ORDER BY ABS(ours.our - s.indra_belief) DESC, s.stmt_hash
				LIMIT 1`
			);
			if (r.length === 0) return null;
			resolvedHash = r[0].stmt_hash;
			biggestDelta = r[0].delta;
		} else {
			whyKind = 'requested';
		}

		const stmtRows = await rows<{
			stmt_hash: string;
			indra_type: string;
			indra_belief: number | null;
			our_score: number | null;
			n_evidences: number;
		}>(
			con,
			`WITH ours AS (
				SELECT stmt_hash,
				       AVG(CAST(json_extract(output_json, '$.score') AS DOUBLE)) AS our
				FROM scorer_step
				WHERE run_id = '${resolvedRun.replace(/'/g, "''")}'
				  AND step_kind = 'aggregate'
				  AND json_extract(output_json, '$.score') IS NOT NULL
				GROUP BY stmt_hash
			)
			SELECT s.stmt_hash, s.indra_type, s.indra_belief,
			       ours.our AS our_score,
			       (SELECT COUNT(*) FROM evidence WHERE stmt_hash = s.stmt_hash) AS n_evidences
			FROM statement s
			LEFT JOIN ours ON ours.stmt_hash = s.stmt_hash
			WHERE s.stmt_hash = '${resolvedHash.replace(/'/g, "''")}'`
		);
		if (stmtRows.length === 0) return null;
		const s = stmtRows[0];

		const [agents, evidences, probes] = await Promise.all([
			rows<{ role: string; name: string }>(
				con,
				`SELECT role, name FROM agent
				 WHERE stmt_hash = '${resolvedHash.replace(/'/g, "''")}'
				 ORDER BY CASE role
				   WHEN 'subj' THEN 0 WHEN 'enz' THEN 0
				   WHEN 'obj' THEN 1 WHEN 'sub' THEN 1
				   WHEN 'member' THEN 2 ELSE 3 END, role_index`
			),
			rows<{ evidence_hash: string; source_api: string | null; text: string | null }>(
				con,
				`SELECT evidence_hash, source_api, text
				 FROM evidence WHERE stmt_hash = '${resolvedHash.replace(/'/g, "''")}'
				 ORDER BY (text IS NULL), length(text) DESC LIMIT 3`
			),
			getProbeAttribution(resolvedRun, resolvedHash)
		]);

		const evPlural = s.n_evidences === 1 ? '' : 's';
		const evText = `${s.n_evidences} evidence${evPlural}`;
		// `why_this_one` only renders when the system chose the focus editorially
		// (largest |Δ|). When the user deep-linked to a stmt_hash, the URL
		// already explains why they're here — silencing avoids self-evident
		// bookkeeping ("opened via deep-link").
		const whyText = whyKind === 'biggest_delta'
			? `the largest disagreement with INDRA in this run · ${evText}`
			: '';

		return {
			run_id: resolvedRun,
			stmt: { stmt_hash: s.stmt_hash, indra_type: s.indra_type, agents },
			our_score: s.our_score,
			indra_score: s.indra_belief,
			probes,
			evidences,
			n_evidences: s.n_evidences,
			why_this_one: whyText
		};
	} finally {
		con.disconnectSync?.();
	}
}

const PROBE_KINDS: Record<string, ProbeKind> = {
	subject_role_probe: 'subject_role',
	object_role_probe: 'object_role',
	relation_axis_probe: 'relation_axis',
	scope_probe: 'scope'
};

/**
 * For a (run_id, stmt_hash), return the four probes' contributions to the
 * final score. When `evidence_hash` is supplied, returns the per-evidence
 * view; otherwise picks the highest-confidence answer per probe across all
 * evidences for that statement. Pure logic lives in probeAttribution.ts.
 */
export async function getProbeAttribution(
	run_id: string,
	stmt_hash: string,
	evidence_hash?: string
): Promise<ProbeAttribution[]> {
	if (!dbExists()) return [];
	const con = await connect();
	try {
		const evClause = evidence_hash
			? `AND evidence_hash = '${evidence_hash.replace(/'/g, "''")}'`
			: '';
		const [stepRows, substrateRows] = await Promise.all([
			rows<{
				step_kind: string;
				evidence_hash: string | null;
				answer: string | null;
				confidence: string | null;
				source: string | null;
				rationale: string | null;
			}>(
				con,
				`SELECT
				   step_kind,
				   evidence_hash,
				   json_extract_string(output_json, '$.answer') AS answer,
				   json_extract_string(output_json, '$.confidence') AS confidence,
				   json_extract_string(output_json, '$.source') AS source,
				   json_extract_string(output_json, '$.rationale') AS rationale
				 FROM scorer_step
				 WHERE run_id = '${run_id.replace(/'/g, "''")}'
				   AND stmt_hash = '${stmt_hash.replace(/'/g, "''")}'
				   AND step_kind IN ('subject_role_probe', 'object_role_probe', 'relation_axis_probe', 'scope_probe')
				   ${evClause}`
			),
			rows<{ evidence_hash: string | null; output_json: string }>(
				con,
				`SELECT evidence_hash, output_json::VARCHAR AS output_json
				 FROM scorer_step
				 WHERE run_id = '${run_id.replace(/'/g, "''")}'
				   AND stmt_hash = '${stmt_hash.replace(/'/g, "''")}'
				   AND step_kind = 'substrate_route'
				   ${evClause}`
			)
		]);

		const probeOutputs: ProbeOutput[] = [];
		const seen = new Set<string>();
		for (const r of stepRows) {
			const probe = PROBE_KINDS[r.step_kind];
			if (!probe) continue;
			probeOutputs.push({
				probe,
				evidence_hash: r.evidence_hash,
				answer: r.answer,
				confidence: (r.confidence as ProbeConfidence) ?? null,
				source: (r.source as ProbeSource) ?? null,
				rationale: r.rationale
			});
			seen.add(`${r.evidence_hash ?? ''}|${probe}`);
		}

		for (const sr of substrateRows) {
			let parsed: Record<string, { source?: string; answer?: string; confidence?: string }> | null = null;
			try {
				parsed = JSON.parse(sr.output_json);
			} catch {
				continue;
			}
			if (!parsed) continue;
			const probesInRow: ProbeKind[] = ['subject_role', 'object_role', 'relation_axis', 'scope'];
			for (const probe of probesInRow) {
				if (seen.has(`${sr.evidence_hash ?? ''}|${probe}`)) continue;
				const slot = parsed[probe];
				if (!slot || !slot.answer) continue;
				probeOutputs.push({
					probe,
					evidence_hash: sr.evidence_hash,
					answer: slot.answer ?? null,
					confidence: (slot.confidence as ProbeConfidence) ?? null,
					source: (slot.source as ProbeSource) ?? 'substrate',
					rationale: `substrate-resolved (no LLM call)`
				});
				seen.add(`${sr.evidence_hash ?? ''}|${probe}`);
			}
		}

		if (evidence_hash) {
			return computeAttributions(probeOutputs);
		}
		return computeAttributions(summarizeAcrossEvidences(probeOutputs));
	} finally {
		con.disconnectSync?.();
	}
}

export async function getStatementDetail(stmt_hash: string): Promise<StatementDetail | null> {
	if (!dbExists()) return null;
	const con = await connect();
	try {
		const stmtRows = await rows<{
			stmt_hash: string;
			indra_type: string;
			indra_belief: number | null;
			supports_count: number;
			supported_by_count: number;
			source_dump_id: string | null;
			raw_json: string;
		}>(
			con,
			`SELECT stmt_hash, indra_type, indra_belief, supports_count,
			        supported_by_count, source_dump_id, raw_json::VARCHAR AS raw_json
			 FROM statement WHERE stmt_hash = '${stmt_hash.replace(/'/g, "''")}'`
		);
		if (stmtRows.length === 0) return null;
		const s = stmtRows[0];

		const [agents, evidences, truth_labels, registered_truth_sets_rows, supports_edges, scorer_steps] = await Promise.all([
			rows<AgentRow>(
				con,
				`SELECT agent_hash, role, role_index, name,
				        db_refs_json::VARCHAR AS db_refs_json,
				        mods_json::VARCHAR AS mods_json,
				        location
				 FROM agent
				 WHERE stmt_hash = '${stmt_hash.replace(/'/g, "''")}'
				 ORDER BY
				   CASE role
				     WHEN 'subj' THEN 0
				     WHEN 'enz'  THEN 0
				     WHEN 'obj'  THEN 1
				     WHEN 'sub'  THEN 1
				     WHEN 'member' THEN 2
				     ELSE 3
				   END,
				   role_index`
			),
			rows<EvidenceRow>(
				con,
				`SELECT evidence_hash, source_api, source_id, pmid, text,
				        is_direct, is_negated, is_curated,
				        epistemics_json::VARCHAR AS epistemics_json
				 FROM evidence
				 WHERE stmt_hash = '${stmt_hash.replace(/'/g, "''")}'
				 ORDER BY source_api, evidence_hash`
			),
			rows<TruthLabelRow>(
				con,
				`SELECT truth_set_id, target_kind, target_id, field, value_text,
				        value_json::VARCHAR AS value_json, provenance
				 FROM truth_label
				 WHERE (target_kind = 'stmt' AND target_id = '${stmt_hash.replace(/'/g, "''")}')
				    OR (target_kind = 'evidence' AND target_id IN (
				          SELECT evidence_hash FROM evidence WHERE stmt_hash = '${stmt_hash.replace(/'/g, "''")}'))
				    OR (target_kind = 'agent' AND target_id IN (
				          SELECT agent_hash FROM agent WHERE stmt_hash = '${stmt_hash.replace(/'/g, "''")}'))
				 ORDER BY truth_set_id, field`
			),
			rows<{ id: string }>(
				con,
				'SELECT id FROM truth_set ORDER BY id'
			),
			rows<SupportsEdgeRow>(
				con,
				`SELECT from_stmt_hash, to_stmt_hash, kind
				 FROM supports_edge
				 WHERE from_stmt_hash = '${stmt_hash.replace(/'/g, "''")}'
				 ORDER BY kind, to_stmt_hash`
			),
			rows<ScorerStepRow>(
				con,
				`SELECT step_hash, evidence_hash, scorer_version, model_id,
				        step_kind, is_substrate_answered,
				        input_payload_json::VARCHAR AS input_payload_json,
				        output_json::VARCHAR AS output_json,
				        latency_ms, prompt_tokens, out_tokens, finish_reason, error
				 FROM scorer_step
				 WHERE stmt_hash = '${stmt_hash.replace(/'/g, "''")}'
				 ORDER BY scorer_version DESC, evidence_hash, step_kind`
			)
		]);

		return {
			...s,
			agents,
			evidences,
			truth_labels,
			registered_truth_sets: registered_truth_sets_rows.map((r) => r.id),
			supports_edges,
			scorer_steps
		};
	} finally {
		con.disconnectSync?.();
	}
}

export async function getCorpusOverview(): Promise<CorpusOverview> {
	const path = dbPath();
	if (!dbExists()) {
		return { dbPath: path, dbExists: false, ...EMPTY_OVERVIEW };
	}

	const con = await connect();
	try {
		const [
			statementCount,
			evidenceCount,
			agentCount,
			supportsEdgeCount,
			truthLabelCount,
			truthSets,
			sourceDumps,
			indraTypes,
			scorerRuns,
			latestValidity
		] = await Promise.all([
			scalar(con, 'SELECT COUNT(*) FROM statement'),
			scalar(con, 'SELECT COUNT(*) FROM evidence'),
			scalar(con, 'SELECT COUNT(*) FROM agent'),
			scalar(con, 'SELECT COUNT(*) FROM supports_edge'),
			scalar(con, 'SELECT COUNT(*) FROM truth_label'),
			rows<{ id: string; name: string; rowCount: number }>(
				con,
				`SELECT ts.id AS id, ts.name AS name,
				        COUNT(tl.label_id) AS "rowCount"
				 FROM truth_set ts
				 LEFT JOIN truth_label tl ON tl.truth_set_id = ts.id
				 GROUP BY ts.id, ts.name
				 ORDER BY "rowCount" DESC, ts.id`
			),
			rows<{ source_dump_id: string | null; n: number }>(
				con,
				`SELECT source_dump_id, COUNT(*) AS n
				 FROM statement
				 GROUP BY source_dump_id
				 ORDER BY n DESC
				 LIMIT 16`
			),
			rows<{ indra_type: string; n: number }>(
				con,
				`SELECT indra_type, COUNT(*) AS n
				 FROM statement
				 GROUP BY indra_type
				 ORDER BY n DESC
				 LIMIT 32`
			),
			rows<{
				run_id: string;
				scorer_version: string;
				started_at: string;
				status: string;
				n_stmts: number | null;
				cost_estimate_usd: number | null;
				mae: number | null;
				bias: number | null;
			}>(
				con,
				`SELECT
				   sr.run_id,
				   sr.scorer_version,
				   sr.started_at::VARCHAR AS started_at,
				   sr.status,
				   sr.n_stmts,
				   sr.cost_estimate_usd,
				   (SELECT value FROM metric WHERE run_id = sr.run_id
				    AND metric_name = 'indra_belief_calibration.mae' AND truth_set_id = 'indra_published_belief' LIMIT 1) AS mae,
				   (SELECT value FROM metric WHERE run_id = sr.run_id
				    AND metric_name = 'indra_belief_calibration.bias' AND truth_set_id = 'indra_published_belief' LIMIT 1) AS bias
				 FROM score_run sr
				 ORDER BY sr.started_at DESC
				 LIMIT 8`
			),
			getLatestValidity(con)
		]);

		// Pre-flight check: which exports already exist on disk? Avoids
		// dashboard offering ↓ links that 404 because export wasn't run.
		const exportDir = resolve(process.cwd(), '..', 'data', 'exports');
		const enrichedRuns = scorerRuns.map((r) => ({
			...r,
			hasIndraExport: existsSync(resolve(exportDir, `${r.run_id}_indra.json`)),
			hasCardExport: existsSync(resolve(exportDir, `${r.run_id}_card.json`)),
		}));

		return {
			dbPath: path,
			dbExists: true,
			statementCount,
			evidenceCount,
			agentCount,
			supportsEdgeCount,
			truthLabelCount,
			truthSets,
			sourceDumps,
			indraTypes,
			scorerRuns: enrichedRuns,
			latestValidity
		};
	} finally {
		// connections are short-lived; the DuckDBInstance is reused
		con.disconnectSync?.();
	}
}
