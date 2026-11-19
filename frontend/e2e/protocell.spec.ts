// Phase 4: abstract protocells — growth, division, exact ledger, interventions, lineage.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy } from "./helpers";

test("protocell: cells divide, ledger stays exact, food and lineage in the inspector", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await page.getByTestId("preset-protocell_default").click();
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "running");
    await expect.poll(() => tick(page), { timeout: 60_000 }).toBeGreaterThan(400);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");
    expect(Number(await page.getByTestId("metric-divisions").textContent())).toBeGreaterThan(0);
    await expect(page.getByTestId("metric-ledger_residual")).toHaveText("0");

    await page.getByTestId("brush-add_food").check();
    await page.getByTestId("tool-count").fill("2000");
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await expect(page.getByTestId("intervention-list")).toContainText("add_food", { timeout: 10_000 });
    await page.getByTestId("btn-step").click();
    await expect(page.getByTestId("metric-ledger_residual")).toHaveText("0");

    await page.getByTestId("brush-inspect").check();
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await expect(page.getByTestId("inspector-ledger")).toContainText("residual 0", { timeout: 10_000 });
    await expect(page.getByTestId("inspector-ancestors")).toBeVisible();
    await page.screenshot({ path: path.join(DATA, "150_protocell.png") });
  } finally {
    await stopServer(server);
  }
});
