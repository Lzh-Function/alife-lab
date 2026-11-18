// Shared helpers: start/stop an isolated `uv run alife serve` and common assertions.
import { expect, type Page } from "@playwright/test";
import { spawn, type ChildProcess } from "node:child_process";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const PORT = Number(process.env.ALIFE_E2E_PORT ?? 8765);
export const BASE = `http://127.0.0.1:${PORT}`;
const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
export const DATA = process.env.ALIFE_E2E_DATA ?? mkdtempSync(path.join(tmpdir(), "alife-e2e-"));

export function startServer(): ChildProcess {
  const proc = spawn("uv", ["run", "alife", "serve", "--host", "127.0.0.1", "--port", String(PORT), "--log-level", "warning"], {
    cwd: ROOT,
    env: { ...process.env, ALIFE_HOME: path.join(DATA, "home"), ALIFE_RUNS_DIR: path.join(DATA, "runs") },
    stdio: ["ignore", "inherit", "inherit"],
  });
  return proc;
}

export async function waitHealthy(): Promise<void> {
  for (let i = 0; i < 200; i++) {
    try {
      const r = await fetch(`${BASE}/api/v1/health`);
      if (r.ok) return;
    } catch {
      /* not up yet */
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error("server did not become healthy");
}

export async function stopServer(proc: ChildProcess): Promise<void> {
  const exited = new Promise<void>((resolve) => proc.once("exit", () => resolve()));
  proc.kill("SIGINT"); // graceful: workers checkpoint at the next tick boundary
  await exited;
}

export async function tick(page: Page): Promise<number> {
  return Number(await page.getByTestId("tick").textContent());
}

export async function expectState(page: Page, state: string) {
  await expect(page.getByTestId("lab-head").getByTestId("run-state")).toHaveAttribute("data-state", state, { timeout: 20_000 });
}

