<script lang="ts">
	import { page } from '$app/state';
	const status = $derived(page.status);
	const msg = $derived(page.error?.message ?? '');
	const isWriterLock = $derived(msg.startsWith('writer_in_progress:'));
</script>

<svelte:head>
	<title>{status} · INDRA Belief</title>
</svelte:head>

<main id="main">
	{#if isWriterLock}
		<section class="lock-state">
			<h1>writer in progress</h1>
			<p class="lock-line">
				An ingest or score worker is holding the DuckDB write lock — the
				dashboard pauses until it finishes. This is the documented trade-off
				of the single-file DuckDB architecture (db.ts:closeInstance).
			</p>
			<p class="lock-hint">
				Watch the ingest panel in the originating tab. Reload this page once
				the worker emits <code>done</code>.
			</p>
			<button type="button" onclick={() => location.reload()}>reload now</button>
		</section>
	{:else}
		<section class="generic-err">
			<h1>{status}</h1>
			<p>{msg || 'Something went wrong.'}</p>
			<p><a href="/">← back to dashboard</a></p>
		</section>
	{/if}
</main>

<style>
	:global(html, body) {
		background: #fdfcf8;
		color: #1a1a1a;
		font-family: 'Iowan Old Style', 'Source Serif Pro', Georgia, serif;
		font-size: 16px;
		line-height: 1.5;
		margin: 0;
	}
	main {
		max-width: 640px;
		margin: 4rem auto;
		padding: 0 1.5rem;
	}
	h1 {
		font-weight: 400;
		font-size: 1.6rem;
		margin: 0 0 1rem;
	}
	.lock-state h1 {
		color: #7d2a1a;
	}
	.lock-line {
		font-size: 1.05rem;
		margin: 0 0 1rem;
	}
	.lock-hint {
		font-size: 0.95rem;
		color: #6a6a6a;
		margin: 0 0 1.4rem;
	}
	code {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 0.86rem;
		background: rgba(125, 42, 26, 0.04);
		padding: 0 0.3rem;
	}
	button {
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 0.86rem;
		border: 1px solid #7d2a1a;
		background: transparent;
		color: #7d2a1a;
		padding: 0.3rem 0.8rem;
		cursor: pointer;
	}
	button:hover {
		background: rgba(125, 42, 26, 0.04);
	}
	a {
		color: #7d2a1a;
	}
</style>
