import * as cp from "child_process";
import * as os from "os";

/**
 * Capture the user's interactive login shell environment.
 * Returns a plain object mapping VAR -> VALUE.
 * Linux/macOS supported. Windows returns process.env clone.
 */
export function getUserShellEnv(): Record<string, string> {
    const env: Record<string, string> = {};

    // Windows: simplest fallback (NVBit/CUDA are Linux-oriented anyway)
    if (os.platform() === "win32") {
        return { ...process.env } as Record<string, string>;
    }

    // Determine shell; default to /bin/bash if unknown
    const shell = process.env.SHELL || "/bin/bash";

    // Build args based on common shells
    // -l: login, -i: interactive, -c: run command
    const args = shell.includes("zsh")
        ? ["-l", "-i", "-c", "env -0"]
        : ["-l", "-i", "-c", "env -0"];

    try {
        const res = cp.spawnSync(shell, args, {
            encoding: "utf8",
            // Inherit minimal env so the shell reads rc files itself
            env: { HOME: process.env.HOME || "", LOGNAME: process.env.LOGNAME || "", LANG: process.env.LANG || "C" }
        });

        if (res.status !== 0) {
            // Fallback to current env if shell failed
            return { ...process.env } as Record<string, string>;
        }

        // Parse NUL-separated env (env -0)
        const raw = res.stdout || "";
        const pairs = raw.split("\u0000").filter(Boolean);
        for (const line of pairs) {
            const idx = line.indexOf("=");
            if (idx > 0) {
                const k = line.slice(0, idx);
                const v = line.slice(idx + 1);
                env[k] = v;
            }
        }

        // Ensure we at least have PATH
        if (!env.PATH && process.env.PATH) env.PATH = process.env.PATH;
        return env;
    } catch {
        return { ...process.env } as Record<string, string>;
    }
}
