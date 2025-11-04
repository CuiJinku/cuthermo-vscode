import * as vscode from "vscode";
import * as path from "path";
import * as fs from 'fs';

export function getCfg<T>(key: string, fallback?: T): T {
    return vscode.workspace
        .getConfiguration("cuthermo")
        .get<T>(key, fallback as T);
}

export function workspaceFolderFor(uri?: vscode.Uri) {
  if (uri) return vscode.workspace.getWorkspaceFolder(uri);
  return vscode.workspace.workspaceFolders?.[0];
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


export function resolvePathSetting(key: string, ws?: vscode.WorkspaceFolder): string {
  const raw = vscode.workspace.getConfiguration().get<string>(key) || '';
  const home = process.env.HOME || process.env.USERPROFILE || '';
  const wsPath = ws?.uri.fsPath ?? '';

  return raw
    .replace(/\$\{workspaceFolder\}/g, wsPath)
    .replace(/\$\{home\}/g, home);
}

export function resolveWorkDir(ws?: vscode.WorkspaceFolder): string {
  const configured = resolvePathSetting('cuthermo.workDir', ws);
  return configured || ws?.uri.fsPath || process.cwd();
}

export function activeEditorInfo() {
  const ed = vscode.window.activeTextEditor;
  const ws = ed
    ? vscode.workspace.getWorkspaceFolder(ed.document.uri) ?? vscode.workspace.workspaceFolders?.[0]
    : vscode.workspace.workspaceFolders?.[0];

  const filePath = ed?.document?.uri.fsPath;
  const fileDir = filePath ? path.dirname(filePath) : undefined;
  const fileBase = filePath ? path.basename(filePath) : undefined;
  const fileBaseNoExt = fileBase ? fileBase.replace(/\.[^.]+$/, '') : undefined;

  return { ws, ed, filePath, fileDir, fileBase, fileBaseNoExt };
}
