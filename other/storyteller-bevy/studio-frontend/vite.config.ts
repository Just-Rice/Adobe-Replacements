
/// <reference types='vitest' />
import { defineConfig } from "vite";

import { nxViteTsPaths } from "@nx/vite/plugins/nx-tsconfig-paths.plugin";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";
import mergeAssetsDirs from "../vite-multiple-assets";
import { hmrPlugin, presets} from "vite-plugin-web-components-hmr";

export default defineConfig({
	cacheDir: "../node_modules/.vite/studio-frontend",
	server:{
		port: 4200,
		host: "localhost",
		hmr: true,
	},
	preview:{
		port: 4300,
		host: "localhost",
	},
	plugins: [
		nxViteTsPaths(),
		wasm(),
		topLevelAwait(),
		mergeAssetsDirs([
			"studio-frontend/public",
			"dist/studio/public",
		]),
		hmrPlugin({
			include: ['./src/**/*.ts'],
			presets: [presets.lit]
		})
	],
});
