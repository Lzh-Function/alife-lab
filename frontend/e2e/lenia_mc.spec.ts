// Multichannel Lenia: create the 5N7KKM swarm, observe, inspect kernels.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, openPreset, startServer, stopServer, tick, waitHealthy } from "./helpers";

test("lenia_mc: swarm, run and kernel inspection", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "lenia_mc_swarm");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-start-paused").check();
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "paused");
    await page.getByTestId("btn-step").click();
    await expect.poll(() => tick(page)).toBe(1);
    const mass = Number(await page.getByTestId("metric-mass").textContent());
    expect(mass).toBeGreaterThan(300);
    await page.getByTestId("btn-play").click();
    await expect.poll(() => tick(page), { timeout: 20_000 }).toBeGreaterThan(20);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");
    await page.screenshot({ path: path.join(DATA, "lenia_mc.png") });

    await page.getByTestId("brush-inspect").check();
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    await page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    await expect(page.getByTestId("inspector-lenia-mc")).toBeVisible();
    await expect(page.getByTestId("inspector-lenia-mc").locator("tr")).toHaveCount(15);
  } finally {
    await stopServer(server);
  }
});
