/**
 * POST /api/datasets/ingest — ingest an INDRA Statement JSON (or .json.gz)
 * into corpus.duckdb.
 *
 * U7.1: was synchronous (drain stdout, return JSON) — broke down on the
 * 460MB benchmark .gz where parse + DB-write takes minutes with no signal
 * to the user. Now streams SSE the same way /api/runs/score does, with
 * AbortController-driven SIGTERM/SIGKILL so closing the tab kills the
 * worker.
 *
 * Body:
 *   {
 *     path: string,            // absolute path to JSON or .json.gz
 *     source_dump_id: string   // e.g. "rasmachine_2026-05-11"
 *   }
 *
 * Returns: SSE stream. Events:
 *   data: {"event": "started", ...}
 *   data: {"event": "loaded", "n_statements": N}
 *   data: {"event": "progress", "n_statements_done": N, "n_statements_total": M}
 *   data: {"event": "done", "n_statements": N, "duration_s": ...}
 *   data: {"event": "error", ...}
 * Terminated by `data: {"event": "channel_closed"}`.
 */
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { closeInstance, dbPath } from '$lib/db';
import { error } from '@sveltejs/kit';
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

export const POST: RequestHandler = async (event) => {
	const request = event.request;
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

	// Release the viewer's cached READ_ONLY DuckDB instance so the Python
	// writer can acquire the file lock. Next dashboard read will lazy-reopen.
	closeInstance();

	const stream = new ReadableStream<Uint8Array>({
		start(controller) {
			const encoder = new TextEncoder();
			const writeEvent = (obj: unknown) => {
				controller.enqueue(encoder.encode(`data: ${JSON.stringify(obj)}\n\n`));
			};

			const child = spawn(py, args, {
				cwd: repoRoot(),
				env: { ...process.env, PYTHONPATH: resolve(repoRoot(), 'src') }
			});

			const onAbort = () => {
				try {
					if (!child.killed) {
						child.kill('SIGTERM');
						setTimeout(() => {
							if (!child.killed) child.kill('SIGKILL');
						}, 2000);
					}
				} catch {
					/* already dead */
				}
				writeEvent({ event: 'canceled', reason: 'client_disconnected' });
				try { controller.close(); } catch { /* already closed */ }
			};
			event.request.signal.addEventListener('abort', onAbort);

			let stdoutBuf = '';
			let stderrBuf = '';
			// Bound stderr accumulation. A chatty worker (verbose warnings on a
			// multi-minute ingest) would otherwise grow this unbounded.
			const STDERR_CAP = 64 * 1024;

			child.stdout.on('data', (chunk: Buffer) => {
				stdoutBuf += chunk.toString('utf-8');
				let nl: number;
				while ((nl = stdoutBuf.indexOf('\n')) >= 0) {
					const line = stdoutBuf.slice(0, nl).trim();
					stdoutBuf = stdoutBuf.slice(nl + 1);
					if (!line) continue;
					try {
						writeEvent(JSON.parse(line));
					} catch {
						writeEvent({ event: 'stdout_raw', line });
					}
				}
			});
			child.stderr.on('data', (chunk: Buffer) => {
				if (stderrBuf.length >= STDERR_CAP) return;
				stderrBuf += chunk.toString('utf-8');
				if (stderrBuf.length > STDERR_CAP) {
					stderrBuf = stderrBuf.slice(0, STDERR_CAP) + '\n…[stderr truncated]';
				}
			});
			child.on('exit', (code, signal) => {
				event.request.signal.removeEventListener('abort', onAbort);
				if (code !== 0 && signal !== 'SIGTERM' && signal !== 'SIGKILL') {
					writeEvent({
						event: 'error',
						exit_code: code,
						signal,
						stderr: stderrBuf.slice(0, 4096)
					});
				}
				writeEvent({ event: 'channel_closed', exit_code: code ?? -1, signal });
				try { controller.close(); } catch { /* already closed */ }
			});
			child.on('error', (err) => {
				writeEvent({ event: 'spawn_error', error: err.message });
				controller.close();
			});
		}
	});

	return new Response(stream, {
		headers: {
			'content-type': 'text/event-stream',
			'cache-control': 'no-cache, no-transform',
			'x-accel-buffering': 'no'
		}
	});
};
