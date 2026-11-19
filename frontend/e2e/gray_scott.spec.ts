// Phase 4: Gray–Scott reaction–diffusion in the GUI (Pearson coral regime).
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy, openPreset } from "./helpers";

test("gray-scott: pattern grows, seed tool, inspector rates", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "gray_scott_coral");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "running");
    await expect(page.getByTestId("run-mode")).toContainText("reference");
    await expect.poll(() => tick(page), { timeout: 60_000 }).toBeGreaterThan(4000);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");
    await expect.poll(async () => Number(await page.getByTestId("metric-v_area_fraction").textContent())).toBeGreaterThan(0.05);

    await page.getByTestId("brush-seed").check();
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.15, box.y + box.height * 0.15);
    await expect(page.getByTestId("intervention-list")).toContainText("seed", { timeout: 10_000 });
    await expect(page.getByTestId("run-mode")).toContainText("exploratory");
    await page.getByTestId("brush-inspect").check();
    await page.mouse.click(box.x + box.width * 0.15, box.y + box.height * 0.15);
    await expect(page.getByTestId("inspector-rates")).toBeVisible({ timeout: 10_000 });
    await page.screenshot({ path: path.join(DATA, "100_gray_scott.png") });
  } finally {
    await stopServer(server);
  }
});
