import * as vscode from "vscode";
import * as path from "path";
import * as fs from 'fs';

export function getCfg<T>(key: string, fallback?: T): T {
    return vscode.workspace
        .getConfiguration("cuthermo")
        .get<T>(key, fallback as T);
}

export function workspaceFolderFor(
    uri?: vscode.Uri
): vscode.WorkspaceFolder | undefined {
    if (uri) return vscode.workspace.getWorkspaceFolder(uri);
    const folders = vscode.workspace.workspaceFolders;
    return folders && folders.length > 0 ? folders[0] : undefined;
}

export function resolveVars(
    input: string,
    folder?: vscode.WorkspaceFolder
): string {
    const root =
        folder?.uri.fsPath ??
        vscode.workspace.workspaceFolders?.[0]?.uri.fsPath ??
        "";
    return input.replace(/\$\{workspaceFolder\}/g, root);
}

export function resolvePathSetting(
    key: string,
    folder?: vscode.WorkspaceFolder
): string {
    return path.resolve(resolveVars(getCfg<string>(key) || "", folder));
}

export function resolveWorkDir(folder?: vscode.WorkspaceFolder): string {
  const configured = resolvePathSetting('cuthermo.workDir', folder);
  if (configured && fs.existsSync(configured)) return configured;

  const execPath = resolvePathSetting('cuthermo.execPath', folder);
  if (execPath && fs.existsSync(execPath)) return path.dirname(execPath);

  if (folder) return folder.uri.fsPath;

  // No workspace open and no settings to infer from:
  throw new Error('No workspace folder open and no workDir/execPath set.');
}