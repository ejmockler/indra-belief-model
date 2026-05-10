import { error } from '@sveltejs/kit';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import type { RequestHandler } from './$types';

const EXPORT_DIR = resolve(process.cwd(), '..', 'data', 'exports');

const SUFFIX: Record<string, string> = {
	indra: '_indra.json',
	card: '_card.json'
};

export const GET: RequestHandler = ({ params }) => {
	const { run_id, kind } = params;
	const suffix = SUFFIX[kind];
	if (!suffix) throw error(404, `unknown export kind: ${kind}`);

	// Defensive: run_id is `uuid.uuid4().hex` from corpus.scoring → 32 hex chars.
	// Tighter than `[a-f0-9]+` so a 1-char ?run_id=a still 400s at the gate.
	if (!/^[a-f0-9]{32}$/i.test(run_id)) throw error(400, 'invalid run_id');

	const filePath = resolve(EXPORT_DIR, `${run_id}${suffix}`);
	if (!filePath.startsWith(EXPORT_DIR)) throw error(400, 'path escape');
	if (!existsSync(filePath)) {
		throw error(404, `export not generated yet · run \`from indra_belief.corpus import export_beliefs, model_card\` in Python`);
	}

	const body = readFileSync(filePath, 'utf-8');
	const filename = `${run_id.slice(0, 8)}_${kind}.json`;
	return new Response(body, {
		headers: {
			'content-type': 'application/json; charset=utf-8',
			'content-disposition': `attachment; filename="${filename}"`
		}
	});
};
