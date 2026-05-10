<script lang="ts">
	import favicon from '$lib/assets/favicon.svg';

	let { children } = $props();
</script>

<svelte:head>
	<link rel="icon" href={favicon} />
</svelte:head>

<a href="#main" class="skip-link">Skip to main content</a>

{@render children()}

<style>
	.skip-link {
		position: absolute;
		left: -9999px;
		top: 0;
		font-family: ui-monospace, 'SF Mono', Menlo, monospace;
		font-size: 0.82rem;
		background: #1a1a1a;
		color: #fdfcf8;
		padding: 0.5rem 0.8rem;
		text-decoration: none;
		z-index: 100;
	}

	.skip-link:focus {
		left: 1rem;
		top: 1rem;
	}

	:global(.visually-hidden) {
		position: absolute;
		width: 1px;
		height: 1px;
		padding: 0;
		margin: -1px;
		overflow: hidden;
		clip: rect(0, 0, 0, 0);
		white-space: nowrap;
		border: 0;
	}
</style>

