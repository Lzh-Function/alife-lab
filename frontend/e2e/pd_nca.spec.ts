// Phase 4: Petri Dish NCA — competing NCAs with online learning. Skipped without JAX.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy } from "./helpers";

test("pd-nca: territories over epochs, seeding an NCA, interaction matrix", async ({ page, request }) => {
  const server = startServer();
  await waitHealthy();
  try {
    const targets = await (await request.get(`${BASE}/api/v1/compute-targets`)).json();
    test.skip(!targets.data.targets[0].capabilities.backends.includes("jax"), "JAX not installed");
    await page.goto(`${BASE}/#/catalog`);
    await page.getByTestId("preset-pd_nca_dish").click();
    await page.getByTestId("input-profile").selectOption("local_interactive");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "running");
    await expect.poll(() => tick(page), { timeout: 120_000 }).toBeGreaterThan(60);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");
    await expect(page.getByTestId("species-chart")).toContainText("nca_0");
    expect(Number(await page.getByTestId("metric-alive_ncas").textContent())).toBeGreaterThan(0);

    await page.getByTestId("brush-seed").check();
    await page.getByTestId("tool-channel").selectOption("1");
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.2, box.y + box.height * 0.2);
    await expect(page.getByTestId("intervention-list")).toContainText("seed", { timeout: 10_000 });
    await page.getByTestId("brush-inspect").check();
    await page.mouse.click(box.x + box.width * 0.2, box.y + box.height * 0.2);
    await expect(page.getByTestId("inspector-interaction")).toBeVisible({ timeout: 10_000 });
    await expect(page.getByTestId("inspector-pdnca")).toContainText("NCA 1");
    await page.screenshot({ path: path.join(DATA, "140_pd_nca.png") });
  } finally {
    await stopServer(server);
  }
});
