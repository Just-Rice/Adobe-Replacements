import fs from "node:fs";
import url from "node:url";
import path from "node:path";
import { rimraf } from "rimraf";

import copyAssets from "./copy-assets.mjs";
import handleError from "./handle-error.mjs";
import run, { wasmBindgenCliVersion } from "./run.mjs";

const WASM_BINDGEN_CLI_VERSION = "0.2.92";

async function main() {
	try {
		const __dirname = url.fileURLToPath(new URL(".", import.meta.url));
		const workspaceRoot = path.resolve(__dirname, "../..");

		const extraCargoArgs = process.argv.slice(2);
		console.log(`Extra cargo args:`);
		for (let arg of extraCargoArgs) {
			console.log(`   ${arg}`);
		}

		/** @type import("node:child_process").StdioOptions */
		const stdio = ["inherit", "inherit", "inherit"];

		const outDir = path.join(workspaceRoot, "dist/studio");

		if (fs.existsSync(outDir))
			await rimraf(outDir);

		await run("cargo", [
			"build",
			"-p", "studio",
			"--lib",
			"--release",
			"--target", "wasm32-unknown-unknown",
		].concat(extraCargoArgs), {
			shell: true,
			cwd: workspaceRoot,
			stdio,
		});

		const wbVersion = await wasmBindgenCliVersion();
		if (!wbVersion || !wbVersion.includes(WASM_BINDGEN_CLI_VERSION)) {
			await run("cargo", [
				"install", "-f", "wasm-bindgen-cli",
				"--version", WASM_BINDGEN_CLI_VERSION,
			], {
				shell: true,
				cwd: workspaceRoot,
				stdio,
			});
		}

		await run("wasm-bindgen", [
			"--out-dir", "./dist/studio",
			"--target", "bundler",
			"./target/wasm32-unknown-unknown/release/studio.wasm",
		], {
			shell: true,
			cwd: workspaceRoot,
			stdio,
		});

		await copyAssets();

		await fs.promises.copyFile(
			path.join(workspaceRoot, "studio/package.json"),
			path.join(outDir, "package.json"),
		);
	}
	catch (err) {
		handleError(err);
	}
}

main();
