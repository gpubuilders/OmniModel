import { mdsvex } from 'mdsvex';
import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: [vitePreprocess(), mdsvex()],

	kit: {
		paths: {
			base: ''
		},
		router: { type: 'pathname' },
		adapter: adapter({
			pages: '../temp_dist',
			assets: '../temp_dist',
			fallback: 'index.html',
			precompress: false,
			strict: true
		}),
		output: {
			bundleStrategy: 'inline'
		},
		alias: {
			$styles: 'src/styles'
		}
	},

	extensions: ['.svelte', '.svx']
};

export default config;
