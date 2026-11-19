// Phase 4: Physarum-inspired stigmergy (Jones 2010) in the GUI.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy, openPreset } from "./helpers";

test("physarum: network forms, attractant intervention, agent sensors in inspector", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "physarum_network");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "running");
    await expect(page.getByTestId("run-mode")).toContainText("reference");
    await expect.poll(() => tick(page), { timeout: 60_000 }).toBeGreaterThan(400);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");
    await expect.poll(async () => Number(await page.getByTestId("metric-trail_contrast").textContent())).toBeGreaterThan(0.8);

    await page.getByTestId("brush-attract").check();
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await expect(page.getByTestId("intervention-list")).toContainText("attract", { timeout: 10_000 });
    await page.getByTestId("brush-inspect").check();
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await expect(page.getByTestId("inspector-decision")).toBeVisible({ timeout: 10_000 });
    await page.screenshot({ path: path.join(DATA, "110_physarum.png") });
  } finally {
    await stopServer(server);
  }
});
