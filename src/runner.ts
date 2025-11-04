import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { glob } from 'glob';
import { getCfg, resolvePathSetting, workspaceFolderFor, resolveWorkDir } from './util';
import { getUserShellEnv } from "./shellEnv";


export interface RunResult {
    outputFile?: string;
    stdout: string;
    stderr: string;
}



function newest(files: string[]): string | undefined {
    if (!files.length) return undefined;
    const stats = files.map(f => ({ f, t: fs.statSync(f).mtimeMs }));
    stats.sort((a, b) => b.t - a.t);
    return stats[0].f;
}

export async function runCuThermo(targetUri?: vscode.Uri): Promise<RunResult> {
    const folder = workspaceFolderFor(targetUri);
    if (!folder) throw new Error('Open a workspace/folder first.');

    const soPath = resolvePathSetting('cuthermo.soPath', folder);
    const workDir = resolveWorkDir(folder);          // ← auto-detect here
    const execPath = resolvePathSetting('cuthermo.execPath', folder);
    const args = getCfg<string[]>('cuthermo.args', []);
    const outputGlob = getCfg<string>('cuthermo.outputGlob', 'output_*.txt');
    const timeoutSec = getCfg<number>('cuthermo.timeoutSec', 180);

    // Snapshot outputs before run
    const before = await glob(outputGlob, { cwd: workDir, absolute: true });

    // Prepare env (inject LD_PRELOAD)
    const userEnv = getUserShellEnv();
    // const env = { ...process.env, LD_PRELOAD: soPath };
    const env: Record<string, string> = {
        ...userEnv,
        ...process.env,
        LD_PRELOAD: soPath,
        NVBIT_VERBOSE: '1', // temp: more logging from NVBit (remove later if too chatty)
    };

    // env.CUDA_INJECTION64_PATH = env.CUDA_INJECTION64_PATH || soPath; // optional
    if (!env.PATH && process.env.PATH) env.PATH = process.env.PATH;
    if (!env.LD_LIBRARY_PATH && process.env.LD_LIBRARY_PATH) env.LD_LIBRARY_PATH = process.env.LD_LIBRARY_PATH;

    // --- NEW: print a clear preview ---
    // --- diagnostics channel ---
    const ch = vscode.window.createOutputChannel('cuThermo Run');
    ch.clear();

    // Preflight checks
    const missing: string[] = [];
    if (!fs.existsSync(execPath)) missing.push(`execPath not found: ${execPath}`);
    if (!fs.existsSync(soPath)) missing.push(`cuThermostat.so not found: ${soPath}`);
    if (!fs.existsSync(workDir)) missing.push(`workDir not found: ${workDir}`);

    if (missing.length) {
        ch.appendLine('[Preflight errors]');
        missing.forEach(m => ch.appendLine('  - ' + m));
        ch.show(true);
        throw new Error(missing.join('; '));
    }

    // Preview
    ch.appendLine(`[Run Preview] LD_PRELOAD=${soPath} ${execPath} ${args.join(' ')}`.trim());
    ch.appendLine(`[cwd] ${workDir}`);

    // Show a few key envs
    const showEnv = (k: string) => { if (env[k]) ch.appendLine(`[env] ${k}=${env[k]}`); };
    showEnv('PATH');
    showEnv('LD_LIBRARY_PATH');
    showEnv('CUDA_VISIBLE_DEVICES');
    showEnv('CUDA_INJECTION64_PATH');
    showEnv('NVBIT_TOOL');
    showEnv('NVBIT_VERBOSE');

    ch.appendLine('[stdout]');
    ch.show(true);

    // Spawn
    const child = cp.spawn(execPath, args, { cwd: workDir, env });

    let stdout = '', stderr = '';
    child.stdout?.on('data', d => stdout += d.toString());
    child.stderr?.on('data', d => stderr += d.toString());

    // Extra lifecycle hooks for visibility
    child.on('spawn', () => ch.appendLine('\n[spawn] process started'));
    child.on('error', (err) => ch.appendLine(`\n[error] ${String(err?.message || err)}`));
    child.on('exit', (code, signal) => ch.appendLine(`\n[exit] code=${code} signal=${signal}`));
    child.on('close', (code, signal) => ch.appendLine(`\n[close] code=${code} signal=${signal}`));

    // Handle timeout
    const timedOut = await new Promise<boolean>((resolve) => {
        const timer = setTimeout(() => {
            try { child.kill('SIGKILL'); } catch { }
            ch.appendLine(`\n[timeout] ${timeoutSec}s exceeded → SIGKILL sent`);
            resolve(true);
        }, timeoutSec * 1000);

        child.on('exit', () => { clearTimeout(timer); resolve(false); });
        child.on('error', () => { clearTimeout(timer); resolve(false); });
    });

    if (timedOut) {
        throw new Error(`Run exceeded ${timeoutSec}s and was terminated.`);
    }

    // Find new (or newest) output
    const after = await glob(outputGlob, { cwd: workDir, absolute: true });
    const created = after.filter(a => !before.includes(a));
    const latest = newest(created.length ? created : after);

    return { outputFile: latest, stdout, stderr };
}
