// Phase 2 slice D: third substrate (Particle Life) through the GUI, with the inspector.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy, openPreset } from "./helpers";

test("particle life: run, spawn typed particles, inspect matrix", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "particle_life_self_sorting");
    await page.getByTestId("field-initialization.count").fill("400");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-tps").fill("60");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "running");
    await expect(page.getByTestId("run-mode")).toContainText("exploratory");
    await expect.poll(() => tick(page), { timeout: 30_000 }).toBeGreaterThan(200);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");
    await expect(page.getByTestId("metric-population")).toHaveText("400");
    const seg = Number(await page.getByTestId("metric-segregation_index").textContent());
    expect(seg).toBeGreaterThan(0.3); // above the well-mixed baseline (~0.25)

    await page.getByTestId("brush-spawn").check();
    await page.getByTestId("tool-count").fill("20");
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await expect(page.getByTestId("metric-population")).toHaveText("420", { timeout: 10_000 });

    await page.getByTestId("brush-inspect").check();
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await expect(page.getByTestId("inspector-matrix").locator("tr")).toHaveCount(4, { timeout: 10_000 });
    await expect(page.getByTestId("inspector-particle")).toBeVisible();
    await page.screenshot({ path: path.join(DATA, "50_particle_life.png") });
  } finally {
    await stopServer(server);
  }
});
