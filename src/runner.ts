import * as vscode from 'vscode';
import * as cp from 'child_process';
import * as path from 'path';
import * as fs from 'fs';
import { glob } from 'glob';
import { getCfg, resolvePathSetting, workspaceFolderFor } from './util';

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
    const workDir = resolvePathSetting('cuthermo.workDir', folder);
    const execPath = resolvePathSetting('cuthermo.execPath', folder);
    const args = getCfg<string[]>('cuthermo.args', []);
    const outputGlob = getCfg<string>('cuthermo.outputGlob', 'output_*.txt');
    const timeoutSec = getCfg<number>('cuthermo.timeoutSec', 180);

    // Snapshot outputs before run
    const before = await glob(outputGlob, { cwd: workDir, absolute: true });

    // Prepare env (inject LD_PRELOAD)
    const env = { ...process.env, LD_PRELOAD: soPath };

    // Spawn
    const child = cp.spawn(execPath, args, { cwd: workDir, env });

    let stdout = '', stderr = '';
    child.stdout?.on('data', d => stdout += d.toString());
    child.stderr?.on('data', d => stderr += d.toString());

    // Handle timeout
    const timedOut = await new Promise<boolean>((resolve) => {
        const timer = setTimeout(() => {
            try { child.kill('SIGKILL'); } catch { }
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
