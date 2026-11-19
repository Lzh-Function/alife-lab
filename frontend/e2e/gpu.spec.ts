// GPU backend (CuPy) from the browser: Flow-Lenia on the local_gpu profile with GPU memory shown.
// Skipped when the local target offers no cupy backend (CPU-only machines and CI).
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy } from "./helpers";

test("gpu backend: flow lenia on local_gpu reports backend and GPU memory", async ({ page, request }) => {
  const server = startServer();
  await waitHealthy();
  try {
    const targets = await (await request.get(`${BASE}/api/v1/compute-targets`)).json();
    test.skip(!targets.data.targets[0].capabilities.backends.includes("cupy"), "no cupy backend on this machine");
    await page.goto(`${BASE}/#/catalog`);
    await page.getByTestId("preset-flow_lenia_example").click();
    await page.getByTestId("input-profile").selectOption("local_gpu");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "running");
    await expect.poll(() => tick(page), { timeout: 60_000 }).toBeGreaterThan(300);
    await expect(page.getByTestId("gpu-memory")).toContainText("MiB", { timeout: 15_000 });
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");
    await expect(page.getByText("cupy @ local")).toBeVisible();
    await page.screenshot({ path: path.join(DATA, "95_gpu_flow_lenia.png") });
  } finally {
    await stopServer(server);
  }
});
