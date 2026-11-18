// Phase 3: Flow-Lenia (multi-channel, mass conserving) in the GUI.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy } from "./helpers";

test("flow lenia: run, mass conserved, paint one channel, inspect channels", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await page.getByTestId("preset-flow_lenia_example").click();
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "running");
    await expect(page.getByTestId("run-mode")).toContainText("reference");
    await expect.poll(() => tick(page), { timeout: 60_000 }).toBeGreaterThan(150);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");
    const drift = Math.abs(Number(await page.getByTestId("metric-mass_drift").textContent()));
    expect(drift).toBeLessThan(1e-9);
    await page.screenshot({ path: path.join(DATA, "80_flow_lenia.png") });

    await page.getByTestId("brush-paint").check();
    await page.getByTestId("tool-channel").selectOption("1");
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.1, box.y + box.height * 0.1);
    await expect(page.getByTestId("intervention-list")).toContainText("paint", { timeout: 10_000 });
    await expect(page.getByTestId("run-mode")).toContainText("exploratory");

    await page.getByTestId("brush-inspect").check();
    await page.mouse.click(box.x + box.width * 0.1, box.y + box.height * 0.1);
    const rows = page.getByTestId("inspector-channels").locator("tbody tr");
    await expect(rows).toHaveCount(3, { timeout: 10_000 });
    await expect(rows.nth(1).locator("td").nth(1)).toHaveText("1.000");
    await expect(rows.nth(0).locator("td").nth(1)).toHaveText("0.000");
  } finally {
    await stopServer(server);
  }
});
