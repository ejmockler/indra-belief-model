/**
 * Filesystem discovery for datasets on disk (U2 of the pipeline-in-viewer
 * hypergraph). Walks `data/corpora/` and `data/benchmark/` for JSON / JSONL
 * files and produces lazy descriptors the viewer can render without parsing.
 *
 * Shape extraction (statement counts, source_apis, sample biology) lives in
 * `datasetShape()` and is invoked per-descriptor on demand — never on the
 * dashboard's hot path.
 */
import { readdirSync, statSync, existsSync, openSync, readSync, closeSync } from 'node:fs';
import { resolve, basename, relative, join } from 'node:path';
import { dbPath } from './db';

export type DatasetKind = 'corpus' | 'benchmark';

export interface DatasetDescriptor {
	path: string;
	rel_path: string;
	filename: string;
	kind: DatasetKind;
	size_bytes: number;
	file_mtime_iso: string;
	ext: string;
}

export interface DatasetShape {
	kind_detail: 'indra_json' | 'jsonl_records' | 'json_gz' | 'unknown';
	n_records: number | null;
	source_apis: string[];
	sample_lines: string[];
	notes: string[];
}

/**
 * Repo-root resolution: db.ts already resolves the duckdb path relative to
 * the viewer's cwd; we go one level up for the corpora / benchmark dirs.
 */
function repoRoot(): string {
	const dbAbsolute = dbPath();
	// data/corpus.duckdb → data → repo
	return resolve(dbAbsolute, '..', '..');
}

function listDir(dir: string, kind: DatasetKind): DatasetDescriptor[] {
	if (!existsSync(dir)) return [];
	const out: DatasetDescriptor[] = [];
	for (const name of readdirSync(dir)) {
		if (name.startsWith('.')) continue;
		const full = join(dir, name);
		let st;
		try {
			st = statSync(full);
		} catch {
			continue;
		}
		if (!st.isFile()) continue;
		// Skip uninteresting files
		const lower = name.toLowerCase();
		if (!(lower.endsWith('.json') || lower.endsWith('.jsonl') || lower.endsWith('.json.gz'))) continue;
		const ext = lower.endsWith('.json.gz') ? 'json.gz' : lower.endsWith('.jsonl') ? 'jsonl' : 'json';
		out.push({
			path: full,
			rel_path: relative(repoRoot(), full),
			filename: basename(full),
			kind,
			size_bytes: st.size,
			file_mtime_iso: st.mtime.toISOString(),
			ext
		});
	}
	out.sort((a, b) => a.filename.localeCompare(b.filename));
	return out;
}

export function getDatasets(): DatasetDescriptor[] {
	const root = repoRoot();
	return [
		...listDir(join(root, 'data', 'corpora'), 'corpus'),
		...listDir(join(root, 'data', 'benchmark'), 'benchmark')
	];
}

/**
 * Cheap line-count for JSONL via streaming read.
 * Caps at maxBytes to avoid pegging the dashboard load on a multi-GB file.
 */
function countLinesUpTo(path: string, maxBytes: number): { lines: number; truncated: boolean } {
	const fd = openSync(path, 'r');
	try {
		const bufSize = 64 * 1024;
		const buf = Buffer.alloc(bufSize);
		let pos = 0;
		let lines = 0;
		while (pos < maxBytes) {
			const n = readSync(fd, buf, 0, bufSize, pos);
			if (n === 0) break;
			for (let i = 0; i < n; i++) if (buf[i] === 0x0a) lines++;
			pos += n;
		}
		const truncated = pos >= maxBytes;
		return { lines, truncated };
	} finally {
		closeSync(fd);
	}
}

function readFirstChunk(path: string, maxBytes: number): string {
	const fd = openSync(path, 'r');
	try {
		const buf = Buffer.alloc(maxBytes);
		const n = readSync(fd, buf, 0, maxBytes, 0);
		return buf.subarray(0, n).toString('utf-8');
	} finally {
		closeSync(fd);
	}
}

/**
 * Lazy shape preview. JSON: tries to parse as INDRA statement list / object.
 * JSONL: line-counts + samples first 3 records. json.gz: reports gzipped size only
 * (gunzipping a benchmark corpus on every dashboard load is too expensive — Layer B
 * could add a cached side index).
 */
export function datasetShape(d: DatasetDescriptor): DatasetShape {
	const notes: string[] = [];
	if (d.ext === 'json.gz') {
		return {
			kind_detail: 'json_gz',
			n_records: null,
			source_apis: [],
			sample_lines: [],
			notes: ['gzipped; expand to count records (deferred to a cached side index)']
		};
	}
	if (d.ext === 'jsonl') {
		const SAMPLE_BYTES = 8 * 1024;
		const LINE_COUNT_CAP = 16 * 1024 * 1024; // 16MB cap on line counting
		const head = readFirstChunk(d.path, SAMPLE_BYTES);
		const sampleLines = head.split('\n').slice(0, 3).filter((l) => l.trim().length > 0);
		const parsedLines = sampleLines.map(safeParse);
		const { lines, truncated } = countLinesUpTo(d.path, LINE_COUNT_CAP);
		if (truncated) notes.push(`record count truncated at ${LINE_COUNT_CAP} bytes`);
		const source_apis = collectSourceApis(parsedLines.filter((p): p is Record<string, unknown> => p !== null));
		return {
			kind_detail: 'jsonl_records',
			n_records: lines,
			source_apis,
			sample_lines: parsedLines.map((p) => extractSamplePreview(p)),
			notes
		};
	}
	if (d.ext === 'json') {
		// Try to parse the whole thing — JSON corpora are typically a list of
		// INDRA statements; we slurp + count.
		try {
			const txt = readFirstChunk(d.path, d.size_bytes);
			const parsed: unknown = JSON.parse(txt);
			let stmts: unknown[] = [];
			if (Array.isArray(parsed)) stmts = parsed;
			else if (parsed && typeof parsed === 'object' && Array.isArray((parsed as { statements?: unknown }).statements)) {
				stmts = (parsed as { statements: unknown[] }).statements;
			} else {
				notes.push('JSON root is not a list and has no `statements` key');
			}
			const samples = stmts.slice(0, 3).map(extractSamplePreviewFromStatement);
			const source_apis = collectSourceApis(stmts.slice(0, 200));
			return {
				kind_detail: 'indra_json',
				n_records: stmts.length,
				source_apis,
				sample_lines: samples,
				notes
			};
		} catch (e) {
			notes.push(`parse failed: ${String((e as Error).message ?? e).slice(0, 120)}`);
			return { kind_detail: 'unknown', n_records: null, source_apis: [], sample_lines: [], notes };
		}
	}
	return { kind_detail: 'unknown', n_records: null, source_apis: [], sample_lines: [], notes };
}

function safeParse(s: string): Record<string, unknown> | null {
	try {
		return JSON.parse(s);
	} catch {
		return null;
	}
}

function collectSourceApis(records: unknown[]): string[] {
	const set = new Set<string>();
	for (const r of records) {
		if (!r || typeof r !== 'object') continue;
		const rec = r as Record<string, unknown>;
		// INDRA Statement: evidence[*].source_api
		const ev = (rec.evidence as Array<Record<string, unknown>> | undefined) ?? null;
		if (Array.isArray(ev)) {
			for (const e of ev) {
				const sa = e?.source_api;
				if (typeof sa === 'string') set.add(sa);
			}
		}
		// Benchmark record: source_api on the record directly
		const top = rec.source_api;
		if (typeof top === 'string') set.add(top);
	}
	return [...set].sort();
}

function extractSamplePreviewFromStatement(s: unknown): string {
	if (!s || typeof s !== 'object') return '(unparseable)';
	const rec = s as Record<string, unknown>;
	const type = (rec.type as string) ?? 'Statement';
	const agents: string[] = [];
	for (const role of ['enz', 'subj', 'sub', 'obj', 'gef', 'gap', 'ras']) {
		const a = rec[role] as Record<string, unknown> | undefined;
		if (a && typeof a.name === 'string') agents.push(a.name);
	}
	if (Array.isArray(rec.members)) {
		for (const m of rec.members as Array<Record<string, unknown>>) {
			if (m && typeof m.name === 'string') agents.push(m.name);
		}
	}
	const first_text = ((rec.evidence as Array<Record<string, unknown>> | undefined) ?? [])[0]?.text;
	const text_preview =
		typeof first_text === 'string' ? ` — "${first_text.slice(0, 80)}${first_text.length > 80 ? '…' : ''}"` : '';
	return `${type}(${agents.join(', ')})${text_preview}`;
}

function extractSamplePreview(o: Record<string, unknown> | null): string {
	if (!o) return '(unparseable line)';
	// Benchmark record: stmt_type + subj/obj + first evidence sentence + tag
	const stmt_type = typeof o.stmt_type === 'string' ? o.stmt_type : null;
	const subj = typeof o.subject === 'string' ? (o.subject as string) : null;
	const obj = typeof o.object === 'string' ? (o.object as string) : null;
	const ev_text =
		typeof o.evidence_text === 'string'
			? (o.evidence_text as string)
			: typeof o.text === 'string'
				? (o.text as string)
				: typeof o.evidence === 'object' && o.evidence && typeof (o.evidence as Record<string, unknown>).text === 'string'
					? ((o.evidence as Record<string, unknown>).text as string)
					: null;
	const tag = typeof o.tag === 'string' ? (o.tag as string) : null;
	const parts: string[] = [];
	if (stmt_type && subj && obj) parts.push(`${stmt_type}(${subj}, ${obj})`);
	else if (stmt_type) parts.push(stmt_type);
	if (ev_text) parts.push(`— "${ev_text.slice(0, 90)}${ev_text.length > 90 ? '…' : ''}"`);
	if (tag) parts.push(`[tag: ${tag}]`);
	if (parts.length > 0) return parts.join(' ');
	return Object.keys(o).slice(0, 5).join(', ');
}
