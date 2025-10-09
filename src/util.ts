import * as vscode from "vscode";
import * as path from "path";

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
