// Embodied creature: create the wave-gait quadruped, see it in 3D, push it, inspect its legs.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, openPreset, startServer, stopServer, tick, waitHealthy } from "./helpers";

test("creature: 3D view, push intervention and leg inspection", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "creature_quadruped_wave");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expect(page.locator("div.scene3d canvas")).toBeVisible({ timeout: 20_000 });
    await expect.poll(() => tick(page), { timeout: 20_000 }).toBeGreaterThan(100);
    await page.getByTestId("push-+x").click();
    await expect.poll(async () => Number(await page.getByTestId("metric-displacement").textContent())).toBeGreaterThan(0.1);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");
    await page.getByTestId("brush-inspect").check();
    await page.locator("div.scene3d").click();
    await expect(page.getByTestId("inspector-creature").locator("tbody tr")).toHaveCount(4);
    await page.screenshot({ path: path.join(DATA, "creature.png") });
    // The push is logged as an intervention.
    await page.getByTestId("brush-off").check();
  } finally {
    await stopServer(server);
  }
});

test("creature evolution: MAP-Elites search, lineage, best opened in 3D", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "search_creature_qd_map_elites");
    await page.getByTestId("field-dynamics.population").fill("8");
    await page.getByTestId("field-dynamics.eval_steps").fill("100");
    await page.getByTestId("field-runner.max_steps").fill("4");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "completed");
    await page.getByTestId("btn-lineage").click();
    await expect(page.locator("[data-testid^=node-]")).toHaveCount(32);
    await page.getByTestId("lineage-axis").selectOption("mass");
    await page.getByTestId("lineage-candidate-open").click();
    await expect(page.locator("div.scene3d canvas")).toBeVisible({ timeout: 20_000 });
    await expect(page.getByTestId("run-name")).toContainText("cand");
  } finally {
    await stopServer(server);
  }
});
