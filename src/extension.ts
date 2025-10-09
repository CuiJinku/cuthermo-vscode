import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as fs from 'fs';
import * as path from 'path';
import { getCfg, workspaceFolderFor, resolvePathSetting } from './util';
import { runCuThermo } from './runner';

function runSync(cmd: string, args: string[] = [], cwd?: string) {
	try {
		const r = cp.spawnSync(cmd, args, { cwd, encoding: 'utf8' });
		return { ok: r.status === 0, stdout: r.stdout?.trim() || '', stderr: r.stderr?.trim() || '' };
	} catch (e: any) {
		return { ok: false, stdout: '', stderr: String(e?.message || e) };
	}
}

function asBundledSo(context: vscode.ExtensionContext): string | undefined {
	// Adjust this path if you target multiple platforms later
	const p = context.asAbsolutePath(path.join('assets', 'linux-x64', 'cuThermostat.so'));
	return fs.existsSync(p) ? p : undefined;
}

async function ensureDir(p: string) {
	await fs.promises.mkdir(p, { recursive: true });
}

export function activate(context: vscode.ExtensionContext) {
	const out = vscode.window.createOutputChannel('cuThermo');

	context.subscriptions.push(
		vscode.commands.registerCommand('cuthermo.checkEnv', async () => {
			const folder = workspaceFolderFor();
			if (!folder) {
				vscode.window.showWarningMessage('Open a folder/workspace first.');
				return;
			}

			out.clear();
			out.appendLine('cuThermo — Environment Check\n');

			// 1) GPU/driver via nvidia-smi
			const smi = runSync('nvidia-smi', ['--query-gpu=name,driver_version', '--format=csv,noheader']);
			if (!smi.ok) {
				out.appendLine('[!] nvidia-smi not found or failed.');
				out.appendLine(smi.stderr || 'No details.');
				vscode.window.showWarningMessage('nvidia-smi not found. Is this a CUDA-capable machine (local or Remote-SSH)?');
			} else {
				out.appendLine('[OK] NVIDIA driver + GPU(s) detected:');
				out.appendLine(smi.stdout.split('\n').map(l => '  - ' + l).join('\n'));
			}

			// 2) nvcc (informational)
			const nvcc = runSync('nvcc', ['--version']);
			if (nvcc.ok) {
				out.appendLine('\n[OK] nvcc detected.');
				const line = nvcc.stdout.split('\n').find(l => l.toLowerCase().includes('release')) || nvcc.stdout.split('\n')[0];
				out.appendLine('  ' + line);
			} else {
				out.appendLine('\n[•] nvcc not found (that is OK for Run-only mode).');
			}

			// 3) workDir writable?
			const workDir = resolvePathSetting('cuthermo.workDir', folder);
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
				out.appendLine('    Tip: Run “cuThermo: Install Bundled cuThermostat.so to Workspace” to copy it locally and update settings.');
			} else {
				out.appendLine('\n[!] cuThermostat.so not found.');
				out.appendLine('    - Place it at ${workspaceFolder}/tools/cuThermostat/cuThermostat.so, or');
				out.appendLine('    - Put it under assets/linux-x64 inside this extension, or');
				out.appendLine('    - Update setting: cuthermo.soPath');
			}

			out.show(true);
			vscode.window.showInformationMessage('cuThermo: Environment check finished. See the “cuThermo” output channel.');
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

			const destDir = path.join(folder.uri.fsPath, 'tools', 'cuThermostat');
			const dest = path.join(destDir, 'cuThermostat.so');
			try {
				await ensureDir(destDir);
				await fs.promises.copyFile(bundled, dest);
				// Make sure it’s readable
				await fs.promises.chmod(dest, 0o755);

				// Update setting to point to the installed file
				await vscode.workspace.getConfiguration('cuthermo').update(
					'soPath',
					dest.replace(folder.uri.fsPath, '${workspaceFolder}'),
					vscode.ConfigurationTarget.Workspace
				);

				vscode.window.showInformationMessage(`Installed cuThermostat.so to ${dest}`);
			} catch (e: any) {
				vscode.window.showErrorMessage(`Failed to install .so: ${e?.message || e}`);
			}
		})
	);

	const runCmd = vscode.commands.registerCommand('cuthermo.runProfile', async (uri?: vscode.Uri) => {
		try {
			const task = vscode.window.withProgress({
				location: vscode.ProgressLocation.Notification,
				title: 'cuThermo',
				cancellable: false
			}, async (progress) => {
				progress.report({ message: 'Running target under cuThermo…' });
				const res = await runCuThermo(uri);

				if (res.outputFile) {
					await vscode.commands.executeCommand('vscode.open', vscode.Uri.file(res.outputFile));
				} else {
					vscode.window.showWarningMessage('Run finished, but no output_*.txt was detected. Check workDir/outputGlob.');
				}

				if (res.stderr.trim()) {
					const ch = vscode.window.createOutputChannel('cuThermo Run');
					ch.appendLine('[stderr]');
					ch.appendLine(res.stderr);
					ch.show(true);
				}
			});

			await task;
		} catch (e: any) {
			vscode.window.showErrorMessage(`cuThermo run failed: ${e?.message || e}`);
		}
	});

	context.subscriptions.push(runCmd);
}

export function deactivate() { }
