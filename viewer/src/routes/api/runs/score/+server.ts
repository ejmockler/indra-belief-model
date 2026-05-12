/**
 * POST /api/runs/score — ingest (idempotent) + score a corpus end-to-end.
 *
 * Spawns `python -m indra_belief.worker score ...` and streams stdout events
 * as text/event-stream (SSE) so the viewer can render per-evidence progress.
 *
 * Body:
 *   {
 *     path: string,
 *     source_dump_id: string,
 *     model: string,                  // e.g. "claude-sonnet-4-6"
 *     scorer_version: string,         // e.g. "prod-v1"
 *     cost_threshold_usd?: number     // hard cap; worker aborts above this
 *   }
 *
 * Returns: SSE stream. Each event is one of:
 *   data: {"event": "started", ...}
 *   data: {"event": "loaded", "n_statements": N}
 *   data: {"event": "ingested"}
 *   data: {"event": "progress", "n_evidences_done": N, "latest_stmt_hash": "..."}
 *   data: {"event": "done", "run_id": "...", ...}
 *   data: {"event": "error", ...}
 * Terminated by `data: {"event": "channel_closed"}` and an end-of-stream.
 */
import { spawn } from 'node:child_process';
import { resolve } from 'node:path';
import { existsSync } from 'node:fs';
import { dbPath } from '$lib/db';
import { error } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const SOURCE_DUMP_RE = /^[a-z][a-z0-9_-]{1,63}$/i;
const MODEL_RE = /^[a-z0-9][a-z0-9_.\-/:]{1,63}$/i;
const SCORER_VERSION_RE = /^[a-z0-9][a-z0-9_.\-]{1,63}$/i;

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
	const model = body.model as string | undefined;
	const scorer_version = body.scorer_version as string | undefined;
	const cost_threshold_usd = body.cost_threshold_usd as number | undefined;

	if (!path || typeof path !== 'string') throw error(400, 'path required');
	if (!existsSync(path)) throw error(404, `file not found: ${path}`);
	if (!source_dump_id || !SOURCE_DUMP_RE.test(source_dump_id))
		throw error(400, 'source_dump_id must match /^[a-z][a-z0-9_-]{1,63}$/i');
	if (!model || !MODEL_RE.test(model))
		throw error(400, 'model required (and must be safe shell-token)');
	if (!scorer_version || !SCORER_VERSION_RE.test(scorer_version))
		throw error(400, 'scorer_version required');
	if (cost_threshold_usd != null && (typeof cost_threshold_usd !== 'number' || cost_threshold_usd <= 0))
		throw error(400, 'cost_threshold_usd must be positive number');

	const args = [
		'-m', 'indra_belief.worker',
		'score',
		'--db', dbPath(),
		'--path', path,
		'--source-dump-id', source_dump_id,
		'--model', model,
		'--scorer-version', scorer_version
	];
	if (cost_threshold_usd != null) {
		args.push('--cost-threshold-usd', cost_threshold_usd.toString());
	}

	const py = pythonBin();

	// SSE stream. Each line of worker stdout is already JSON; we wrap as SSE events.
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

			// U5.5: if the client disconnects (closed tab, browser-side
			// AbortController.abort()), kill the worker so we don't keep
			// spending API budget on a run nobody is watching. SIGTERM first
			// for graceful shutdown, then SIGKILL after a short grace window.
			const onAbort = () => {
				try {
					if (!child.killed) {
						child.kill('SIGTERM');
						setTimeout(() => {
							if (!child.killed) child.kill('SIGKILL');
						}, 2000);
					}
				} catch {
					// already dead — fine
				}
				writeEvent({ event: 'canceled', reason: 'client_disconnected' });
				try { controller.close(); } catch { /* already closed */ }
			};
			event.request.signal.addEventListener('abort', onAbort);

			let stdoutBuf = '';
			let stderrBuf = '';

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
				stderrBuf += chunk.toString('utf-8');
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
