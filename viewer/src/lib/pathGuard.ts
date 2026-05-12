/**
 * Path validation for worker-spawning endpoints.
 *
 * Security assumption: the viewer is a local-only dev tool — there is no
 * auth, and any client that can POST to /api/datasets/ingest can drive
 * the spawned python worker. To narrow the blast radius of misuse (a
 * stray fetch, an accidental copy/paste of the wrong path), all `path`
 * arguments must resolve under `<repoRoot>/data/`. Paths outside this
 * subtree fail closed with HTTP 400.
 *
 * This is defense-in-depth — not a substitute for actual auth in a
 * multi-user deployment. The README documents the dev-only assumption.
 */
import { existsSync, realpathSync } from 'node:fs';
import { resolve } from 'node:path';
import { error } from '@sveltejs/kit';
import { dbPath } from './db';

function dataRoot(): string {
	// dbPath() = <repoRoot>/data/corpus.duckdb → <repoRoot>/data
	return resolve(dbPath(), '..');
}

/**
 * Resolve a user-supplied path, asserting it exists and lives under
 * `<repoRoot>/data/`. Throws a SvelteKit error() on any failure so
 * the route's RequestHandler can let it propagate.
 *
 * Returns the *real* path (symlinks followed) so the spawned worker
 * receives a canonical filesystem location.
 */
export function assertPathUnderData(path: unknown): string {
	if (!path || typeof path !== 'string') throw error(400, 'path required (string)');
	if (!existsSync(path)) throw error(404, `file not found: ${path}`);

	const root = dataRoot();
	let real: string;
	let realRoot: string;
	try {
		real = realpathSync(path);
		realRoot = realpathSync(root);
	} catch (e) {
		throw error(400, `path resolution failed: ${(e as Error).message}`);
	}

	// Canonical comparison: real must start with realRoot + sep.
	const rootWithSep = realRoot.endsWith('/') ? realRoot : realRoot + '/';
	if (real !== realRoot && !real.startsWith(rootWithSep)) {
		throw error(
			400,
			`path must live under <repoRoot>/data/ — got ${real}. This is a local-only dev tool; ingesting paths outside the data/ subtree is disallowed.`
		);
	}
	return real;
}
