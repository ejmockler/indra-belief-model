/**
 * POST /api/truth-sets — register a JSONL of tagged records as a truth_set.
 *
 * Spawns `python -m indra_belief.worker register-truth-set ...` via
 * node:child_process. Currently synchronous: reads the worker's stdout
 * to completion, returns final event as JSON. SSE streaming is a U3.3
 * follow-up.
 *
 * Body:
 *   {
 *     path: string,             // absolute path to JSONL (we trust the
 *                                  // datasets discovery layer)
 *     truth_set_id: string,     // e.g. "demo_holdout_v5"
 *     truth_set_name: string,   // human label
 *     target_kind: 'stmt'|'evidence',
 *     field: string,            // record field whose value becomes the label
 *     target_hash_field?: string  // override default hash field
 *   }
 */
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { closeInstance, dbPath } from '$lib/db';
import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const TRUTH_SET_ID_RE = /^[a-z][a-z0-9_]{2,63}$/i;

function pythonBin(): string {
	if (process.env.PYTHON_BIN) return process.env.PYTHON_BIN;
	const repoRoot = resolve(dbPath(), '..', '..');
	const venv = resolve(repoRoot, '.venv', 'bin', 'python');
	if (existsSync(venv)) return venv;
	return 'python3';
}

function repoRoot(): string {
	return resolve(dbPath(), '..', '..');
}

export const POST: RequestHandler = async ({ request }) => {
	const body = (await request.json()) as Record<string, unknown>;
	const path = body.path as string | undefined;
	const truth_set_id = body.truth_set_id as string | undefined;
	const truth_set_name = body.truth_set_name as string | undefined;
	const target_kind = body.target_kind as string | undefined;
	const field = body.field as string | undefined;
	const target_hash_field = body.target_hash_field as string | undefined;

	if (!path || typeof path !== 'string') throw error(400, 'path required');
	if (!existsSync(path)) throw error(404, `file not found: ${path}`);
	if (!truth_set_id || !TRUTH_SET_ID_RE.test(truth_set_id))
		throw error(400, 'truth_set_id must match /^[a-z][a-z0-9_]{2,63}$/i');
	if (!truth_set_name) throw error(400, 'truth_set_name required');
	if (target_kind !== 'stmt' && target_kind !== 'evidence')
		throw error(400, 'target_kind must be "stmt" or "evidence"');
	if (!field) throw error(400, 'field required');

	const args = [
		'-m', 'indra_belief.worker',
		'register-truth-set',
		'--db', dbPath(),
		'--path', path,
		'--truth-set-id', truth_set_id,
		'--truth-set-name', truth_set_name,
		'--target-kind', target_kind,
		'--field', field
	];
	if (target_hash_field) args.push('--target-hash-field', target_hash_field);
	// Default behavior: re-run compute_validity so the viewer's validity
	// panel grows a P/R/F1 row for this truth_set immediately.
	args.push('--recompute-latest-validity');

	const py = pythonBin();
	const events: Array<Record<string, unknown>> = [];
	let stderrBuf = '';

	// Release the viewer's cached READ_ONLY DuckDB instance so the worker can
	// acquire the file lock. Next dashboard read will lazy-reopen.
	closeInstance();

	const exitCode: number = await new Promise((resolveP) => {
		const child = spawn(py, args, {
			cwd: repoRoot(),
			env: { ...process.env, PYTHONPATH: resolve(repoRoot(), 'src') }
		});

		// Parse stdout line-by-line for JSON events
		let stdoutBuf = '';
		child.stdout.on('data', (chunk: Buffer) => {
			stdoutBuf += chunk.toString('utf-8');
			let nl: number;
			while ((nl = stdoutBuf.indexOf('\n')) >= 0) {
				const line = stdoutBuf.slice(0, nl).trim();
				stdoutBuf = stdoutBuf.slice(nl + 1);
				if (!line) continue;
				try {
					events.push(JSON.parse(line));
				} catch {
					// Non-JSON stdout (shouldn't happen) — capture as a note event
					events.push({ event: 'stdout_raw', line });
				}
			}
		});
		child.stderr.on('data', (chunk: Buffer) => {
			stderrBuf += chunk.toString('utf-8');
		});
		child.on('exit', (code) => resolveP(code ?? -1));
		child.on('error', (err) => {
			stderrBuf += `\nspawn error: ${err.message}`;
			resolveP(-1);
		});
	});

	if (exitCode !== 0) {
		return json(
			{
				ok: false,
				exit_code: exitCode,
				events,
				stderr: stderrBuf.slice(0, 4096)
			},
			{ status: 500 }
		);
	}

	const done = events.find((e) => e.event === 'done') ?? null;
	return json({ ok: true, events, summary: done });
};
