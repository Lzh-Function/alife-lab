// Presentation views: 3D terrain and space-time volume for fields, glow for particles.
import { expect, test, type Page } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, openPreset, startServer, stopServer, tick, waitHealthy } from "./helpers";

async function create(page: Page, preset: string, tps = "30") {
  await page.goto(`${BASE}/#/catalog`);
  await openPreset(page, preset);
  await page.getByTestId("input-profile").selectOption("cpu_dev");
  await page.getByTestId("input-tps").fill(tps);
  await expect(page.getByTestId("validation")).toContainText("OK");
  await page.getByTestId("btn-create").click();
  await expect.poll(() => tick(page), { timeout: 20_000 }).toBeGreaterThan(5);
}

test("field views: 2D, 3D terrain and time stack", async ({ page }) => {
  test.setTimeout(120_000);
  await page.setViewportSize({ width: 1500, height: 1100 });
  const server = startServer();
  await waitHealthy();
  try {
    await create(page, "lenia_mc_swarm");
    await page.getByTestId("view-terrain").click();
    await expect(page.getByTestId("world-canvas-3d").locator("canvas")).toBeVisible({ timeout: 20_000 });
    await page.waitForTimeout(2500);
    await page.screenshot({ path: path.join(DATA, "view_terrain.png") });
    await page.getByTestId("view-spacetime").click();
    await expect(page.getByTestId("spacetime-ticks")).toBeVisible({ timeout: 20_000 });
    await expect.poll(async () => Number(await page.getByTestId("world-canvas-3d").getAttribute("data-slices")), { timeout: 30_000 }).toBeGreaterThan(5);
    // Replay jumps back in time: the stack starts again instead of mixing past and present.
    await page.getByTestId("btn-pause").click();
    await page.getByTestId("replay-toggle").check();
    await page.getByTestId("replay-tick-input").fill("0");
    await expect(page.getByTestId("replay-source").locator("strong")).toHaveText("t=0", { timeout: 30_000 });
    await expect.poll(async () => Number(await page.getByTestId("world-canvas-3d").getAttribute("data-slices")), { timeout: 30_000 }).toBe(1);
    await expect(page.getByTestId("spacetime-ticks")).toContainText("t=0…0");
    await page.getByTestId("replay-toggle").uncheck();
    await page.getByTestId("btn-play").click();
    // Back to the live run: the stack restarts from live frames only (no replayed t=0 left).
    await expect.poll(async () => (await page.getByTestId("spacetime-ticks").textContent()) ?? "", { timeout: 30_000 }).not.toContain("t=0…");
    await page.getByTestId("spacetime-every").selectOption("4");
    await page.waitForTimeout(6000);
    await page.screenshot({ path: path.join(DATA, "view_spacetime.png") });
    // Back to the measurement view: the brush works again and the canvas shows the field.
    await page.getByTestId("view-2d").click();
    await expect(page.getByTestId("world-canvas-3d")).toHaveCount(0);
    await expect(page.locator("canvas[data-testid=world-canvas]")).toBeVisible();
    await expect(page.getByTestId("brush-paint")).toBeEnabled();
    await page.screenshot({ path: path.join(DATA, "view_2d.png") });
  } finally {
    await stopServer(server);
  }
});

test("particle glow view", async ({ page }) => {
  await page.setViewportSize({ width: 1500, height: 1100 });
  const server = startServer();
  await waitHealthy();
  try {
    await create(page, "particle_life_clusters");
    await page.getByTestId("view-glow").click();
    await page.waitForTimeout(3000);
    await page.screenshot({ path: path.join(DATA, "view_glow.png") });
    await create(page, "lenia_orbium");
    await page.getByTestId("view-spacetime").click();
    await page.waitForTimeout(6000);
    await page.screenshot({ path: path.join(DATA, "view_orbium_spacetime.png") });
  } finally {
    await stopServer(server);
  }
});
