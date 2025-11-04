import * as vscode from "vscode";
import * as cp from "child_process";
import * as fs from "fs";
import * as path from "path";
// import { getCfg, workspaceFolderFor, resolvePathSetting } from './util';
import { runCuThermo } from "./runner";
import {
	workspaceFolderFor,
	resolveWorkDir,
	resolvePathSetting,
	activeEditorInfo,
} from "./util";

function runSync(cmd: string, args: string[] = [], cwd?: string) {
	try {
		const r = cp.spawnSync(cmd, args, { cwd, encoding: "utf8" });
		return {
			ok: r.status === 0,
			stdout: r.stdout?.trim() || "",
			stderr: r.stderr?.trim() || "",
		};
	} catch (e: any) {
		return { ok: false, stdout: "", stderr: String(e?.message || e) };
	}
}

function asBundledSo(context: vscode.ExtensionContext): string | undefined {
	// Adjust this path if you target multiple platforms later
	const p = context.asAbsolutePath(
		path.join("assets", "linux-x64", "cuThermostat.so")
	);
	return fs.existsSync(p) ? p : undefined;
}

async function ensureDir(p: string) {
	await fs.promises.mkdir(p, { recursive: true });
}

export function activate(context: vscode.ExtensionContext) {
	const out = vscode.window.createOutputChannel("cuThermo");

	context.subscriptions.push(
		vscode.commands.registerCommand("cuthermo.checkEnv", async () => {
			const folder = workspaceFolderFor();
			if (!folder) {
				vscode.window.showWarningMessage("Open a folder/workspace first.");
				return;
			}

			out.clear();
			out.appendLine("cuThermo — Environment Check\n");

			// 1) GPU/driver via nvidia-smi
			const smi = runSync("nvidia-smi", [
				"--query-gpu=name,driver_version",
				"--format=csv,noheader",
			]);
			if (!smi.ok) {
				out.appendLine("[!] nvidia-smi not found or failed.");
				out.appendLine(smi.stderr || "No details.");
				vscode.window.showWarningMessage(
					"nvidia-smi not found. Is this a CUDA-capable machine (local or Remote-SSH)?"
				);
			} else {
				out.appendLine("[OK] NVIDIA driver + GPU(s) detected:");
				out.appendLine(
					smi.stdout
						.split("\n")
						.map((l) => "  - " + l)
						.join("\n")
				);
			}

			// 2) nvcc (informational)
			const nvcc = runSync("nvcc", ["--version"]);
			if (nvcc.ok) {
				out.appendLine("\n[OK] nvcc detected.");
				const line =
					nvcc.stdout
						.split("\n")
						.find((l) => l.toLowerCase().includes("release")) ||
					nvcc.stdout.split("\n")[0];
				out.appendLine("  " + line);
			} else {
				out.appendLine("\n[•] nvcc not found (that is OK for Run-only mode).");
			}

			// 3) workDir writable?
			const workDir = resolveWorkDir(folder)
			try {
				await ensureDir(workDir);
				await fs.promises.access(workDir, fs.constants.W_OK);
				out.appendLine(`\n[OK] Work dir writable: ${workDir}`);
			} catch {
				out.appendLine(`\n[!] Work dir not writable: ${workDir}`);
			}

			// 4) .so presence
			const configuredSo = resolvePathSetting('cuthermo.soPath', folder);
			const bundledSo = asBundledSo(context);
			const configuredExists = configuredSo ? fs.existsSync(configuredSo) : false;
			const bundledExists = bundledSo ? fs.existsSync(bundledSo) : false;

			if (configuredExists) {
				out.appendLine(`\n[OK] cuThermostat.so (configured): ${configuredSo}`);
			} else if (bundledExists) {
				out.appendLine(`\n[OK] cuThermostat.so (bundled in extension): ${bundledSo}`);
				out.appendLine('    Tip: Run “cuThermo: Install Bundled cuThermostat.so to Workspace”.');
			} else {
				out.appendLine('\n[!] cuThermostat.so not found.');
				out.appendLine('    - Run the installer command, or');
				out.appendLine('    - Set cuthermo.soPath (e.g., ${workspaceFolder}/cuThermostat.so).');
			}

			out.appendLine('\n[Debug] Effective settings (expanded):');
			out.appendLine(`  soPath    = ${configuredSo}`);
			out.appendLine(`  execPath  = ${resolvePathSetting('cuthermo.execPath', folder)}`);
			out.appendLine(`  workDir   = ${workDir}`);
			out.appendLine(`  outputGlob= ${vscode.workspace.getConfiguration('cuthermo').get('outputGlob')}`);

			out.show(true);
			vscode.window.showInformationMessage(
				"cuThermo: Environment check finished. See output."
			);
		})
	);

	context.subscriptions.push(
		vscode.commands.registerCommand('cuthermo.installBundledSo', async () => {
			const folder = workspaceFolderFor();
			if (!folder) {
				vscode.window.showWarningMessage('Open a folder/workspace first.');
				return;
			}
			const bundled = asBundledSo(context);
			if (!bundled) {
				vscode.window.showErrorMessage('No bundled cuThermostat.so found in this extension.');
				return;
			}

			const destDir = folder.uri.fsPath;
			const dest = path.join(destDir, 'cuThermostat.so');

			try {
				await ensureDir(destDir);
				await fs.promises.copyFile(bundled, dest);
				await fs.promises.chmod(dest, 0o755);

				const cfg = vscode.workspace.getConfiguration('cuthermo');
				const scope = vscode.ConfigurationTarget.Workspace;

				// Workspace-scoped defaults
				await cfg.update('soPath', '${workspaceFolder}/cuThermostat.so', scope);
				await cfg.update('workDir', '${workspaceFolder}', scope);

				// For execPath, we don’t guess a filename; default to a common a.out.
				// Users can change this in Settings if they name their binary differently.
				if (!cfg.get('execPath')) {
					await cfg.update('execPath', '${workspaceFolder}/a.out', scope);
				}

				vscode.window.showInformationMessage(`Installed cuThermostat.so → ${dest}`);
			} catch (e: any) {
				vscode.window.showErrorMessage(`Failed to install .so: ${e?.message || e}`);
			}
		})

	);

	context.subscriptions.push(
		vscode.commands.registerCommand("cuthermo.setWorkDir", async () => {
			const folder = workspaceFolderFor();
			if (!folder) {
				vscode.window.showWarningMessage("Open a folder/workspace first.");
				return;
			}
			const pick = await vscode.window.showOpenDialog({
				canSelectFiles: false,
				canSelectFolders: true,
				canSelectMany: false,
				openLabel: "Select Work Directory",
				defaultUri: folder.uri,
			});
			if (!pick || !pick[0]) return;

			const dest = pick[0].fsPath;
			await vscode.workspace
				.getConfiguration("cuthermo")
				.update(
					"workDir",
					dest.replace(folder.uri.fsPath, "${workspaceFolder}"),
					vscode.ConfigurationTarget.Workspace
				);
			vscode.window.showInformationMessage(`cuThermo workDir set to: ${dest}`);
		})
	);

	const runCmd = vscode.commands.registerCommand(
		"cuthermo.runProfile",
		async (uri?: vscode.Uri) => {
			try {
				const task = vscode.window.withProgress(
					{
						location: vscode.ProgressLocation.Notification,
						title: "cuThermo",
						cancellable: false,
					},
					async (progress) => {
						progress.report({ message: "Running target under cuThermo…" });

						// 🔹 Add this block: preview the exact command to be executed
						const folder = workspaceFolderFor(uri);
						if (folder) {
							const soPath = resolvePathSetting("cuthermo.soPath", folder);
							const execPath = resolvePathSetting("cuthermo.execPath", folder);
							const workDir = resolveWorkDir(folder);
							const args =
								vscode.workspace
									.getConfiguration("cuthermo")
									.get<string[]>("args") || [];

							const fullCmd = `LD_PRELOAD=${soPath} ${execPath} ${args.join(" ")}`;
							const ch = vscode.window.createOutputChannel("cuThermo Run");
							ch.appendLine(`[Run Preview] ${fullCmd}`);
							ch.appendLine(`[cwd] ${workDir}`);
							ch.show(true);
							vscode.window.showInformationMessage(`Running: ${fullCmd}`);
						}




						const res = await runCuThermo(uri);

						if (res.outputFile) {
							await vscode.commands.executeCommand(
								"vscode.open",
								vscode.Uri.file(res.outputFile)
							);
						} else {
							vscode.window.showWarningMessage(
								"Run finished, but no output_*.txt was detected. Check workDir/outputGlob."
							);
						}

						if (res.stderr.trim()) {
							const ch = vscode.window.createOutputChannel("cuThermo Run");
							ch.appendLine("[stderr]");
							ch.appendLine(res.stderr);
							ch.show(true);
						}
					}
				);

				await task;
			} catch (e: any) {
				vscode.window.showErrorMessage(
					`cuThermo run failed: ${e?.message || e}`
				);
			}
		}
	);

	context.subscriptions.push(runCmd);

	const runInTerminal = vscode.commands.registerCommand("cuthermo.runInTerminal", async () => {
		const folder = workspaceFolderFor();
		if (!folder) return vscode.window.showWarningMessage("Open a folder/workspace first.");

		const soPath = resolvePathSetting("cuthermo.soPath", folder);
		const execPath = resolvePathSetting("cuthermo.execPath", folder);
		const workDir = resolveWorkDir(folder);
		const args = vscode.workspace.getConfiguration("cuthermo").get<string[]>("args") || [];

		// Prefer the active terminal; fall back to any existing; create only if none exist.
		let term = vscode.window.activeTerminal ?? vscode.window.terminals[0];
		if (!term) term = vscode.window.createTerminal({ name: "cuThermo" });
		term.show(true);

		// Safe quoting for paths with spaces
		const q = (s: string) => `"${s.replace(/"/g, '\\"')}"`;

		// Move to the desired working directory in that same terminal session
		term.sendText(`cd ${q(workDir)}`);

		// One-shot env assignment so LD_PRELOAD only applies to this process
		const cmd = `LD_PRELOAD=${q(soPath)} ${q(execPath)} ${args.map(q).join(" ")}`;
		term.sendText(cmd);
	});
	context.subscriptions.push(runInTerminal);

	// // in src/extension.ts
	// const runAsTask = vscode.commands.registerCommand("cuthermo.runAsTask", async () => {
	// 	const folder = workspaceFolderFor();
	// 	if (!folder) return vscode.window.showWarningMessage("Open a folder/workspace first.");

	// 	const soPath = resolvePathSetting("cuthermo.soPath", folder);
	// 	const execPath = resolvePathSetting("cuthermo.execPath", folder);
	// 	const workDir = resolveWorkDir(folder);
	// 	const args = vscode.workspace.getConfiguration("cuthermo").get<string[]>("args") || [];

	// 	const cmd = `LD_PRELOAD='${soPath}' '${execPath}' ${args.map(a => `'${a}'`).join(" ")}`;
	// 	const exec = new vscode.ShellExecution(cmd, { cwd: workDir });

	// 	const taskDef = { type: "shell", task: "cuThermo" };
	// 	const task = new vscode.Task(taskDef, folder, "Run Current Target", "cuThermo", exec);

	// 	const sub = vscode.tasks.onDidEndTaskProcess(async (e) => {
	// 		if (e.execution === execution) {
	// 			sub.dispose();
	// 			// optional: scan outputs here and open newest file
	// 			// const files = await glob(...); openTextDocument(...)
	// 		}
	// 	});

	// 	const execution = await vscode.tasks.executeTask(task);
	// });
	// context.subscriptions.push(runAsTask);


}

export function deactivate() { }
