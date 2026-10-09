import chalk from "chalk";
import cp from "node:child_process";
import which from "which";

/**
 * @param {string} command
 * @param {readonly string[]} args
 * @param {import("node:child_process").SpawnOptions} options
 *
 * @returns {Promise<void>}
 */
export default function run(command, args, options) {
	console.log("");
	console.log(chalk.dim(`> ${command} ${args.join(" ")}`));

	return new Promise((resolve, reject) => {
		cp.spawn(command, args, options)
			.on("exit", code => {
				if (code) reject(code)
				else resolve();
			})
			.on("close", code => {
				if (code) reject(code)
				else resolve();
			})
			.on("error", reject);
	});
}

/**
 * @returns {Promise<string|null>}
 */
export async function wasmBindgenCliVersion() {
	const exePath = await which("wasm-bindgen", { nothrow: true });
	if (!exePath) return null;

	return new Promise((resolve, reject) => {
		let stdout = "";
		let stderr = "";

		const finish = code => {
			if (code) reject(new Error(stderr));
			else resolve(stdout);
		}

		const proc = cp.spawn(exePath, ["--version"], {
			shell: true,
			stdio: ["ignore", "pipe", "pipe"],
		})
		.on("exit", finish)
		.on("close", finish)
		.on("error", reject);

		proc.stdout.setEncoding("utf-8");
		proc.stdout.on("data", chunk => stdout += chunk);

		proc.stderr.setEncoding("utf-8");
		proc.stderr.on("data", chunk => stderr += chunk);
	});
}
