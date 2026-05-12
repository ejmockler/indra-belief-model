/**
 * POST /api/datasets/ingest — ingest an INDRA Statement JSON into corpus.duckdb.
 *
 * Spawns `python -m indra_belief.worker ingest ...` (the same subprocess
 * channel U3.1 decided on). Currently synchronous — reads the worker's
 * stdout to completion, returns final event as JSON. SSE streaming is a
 * U3.3 follow-up.
 *
 * Body:
 *   {
 *     path: string,            // absolute path to JSON (from the datasets surface)
 *     source_dump_id: string   // e.g. "rasmachine_2026-05-11"
 *   }
 */
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { dbPath } from '$lib/db';
import { error, json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const SOURCE_DUMP_RE = /^[a-z][a-z0-9_-]{1,63}$/i;

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
	const source_dump_id = body.source_dump_id as string | undefined;

	if (!path || typeof path !== 'string') throw error(400, 'path required');
	if (!existsSync(path)) throw error(404, `file not found: ${path}`);
	if (!source_dump_id || !SOURCE_DUMP_RE.test(source_dump_id))
		throw error(400, 'source_dump_id must match /^[a-z][a-z0-9_-]{1,63}$/i');

	const args = [
		'-m', 'indra_belief.worker',
		'ingest',
		'--db', dbPath(),
		'--path', path,
		'--source-dump-id', source_dump_id
	];
	const py = pythonBin();
	const events: Array<Record<string, unknown>> = [];
	let stderrBuf = '';

	const exitCode: number = await new Promise((resolveP) => {
		const child = spawn(py, args, {
			cwd: repoRoot(),
			env: { ...process.env, PYTHONPATH: resolve(repoRoot(), 'src') }
		});
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
			{ ok: false, exit_code: exitCode, events, stderr: stderrBuf.slice(0, 4096) },
			{ status: 500 }
		);
	}
	const done = events.find((e) => e.event === 'done') ?? null;
	return json({ ok: true, events, summary: done });
};
