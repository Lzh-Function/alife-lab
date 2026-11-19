// Phase 3: Lenia (continuous field) in the GUI: run Orbium, paint, inspect, checkpoint,
// restart the service and restore the float64 state.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy, openPreset } from "./helpers";

test("lenia orbium: run, paint, inspect, checkpoint and restore after restart", async ({ page }) => {
  let server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "lenia_orbium");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-tps").fill("60");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "running");
    await expect(page.getByTestId("run-mode")).toContainText("reference");
    await expect.poll(() => tick(page), { timeout: 30_000 }).toBeGreaterThan(60);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");
    const mass0 = Number(await page.getByTestId("metric-mass").textContent());
    expect(mass0).toBeGreaterThan(65);
    expect(mass0).toBeLessThan(85);
    const runId = page.url().split("/").pop()!;

    // Paint a disc of value 1 in an empty corner: mass grows by ~ the disc area.
    await page.getByTestId("brush-paint").check();
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.1, box.y + box.height * 0.1);
    await expect(page.getByTestId("intervention-list")).toContainText("paint", { timeout: 10_000 });
    await expect(page.getByTestId("run-mode")).toContainText("exploratory");
    await page.getByTestId("btn-step").click();
    await expect.poll(async () => Number(await page.getByTestId("metric-mass").textContent())).toBeGreaterThan(mass0 + 40);

    await page.getByTestId("brush-inspect").check();
    await page.mouse.click(box.x + box.width * 0.1, box.y + box.height * 0.1);
    await expect(page.getByTestId("inspector-field")).toBeVisible({ timeout: 10_000 });
    await page.screenshot({ path: path.join(DATA, "70_lenia.png") });

    await page.getByTestId("btn-checkpoint").click();
    await expect(page.getByTestId("checkpoint-list")).toContainText("manual", { timeout: 10_000 });
    const checkpointTick = await tick(page);
    const massAtCheckpoint = await page.getByTestId("metric-mass").textContent();

    await stopServer(server);
    server = startServer();
    await waitHealthy();
    await page.goto(`${BASE}/#/library`);
    await page.getByTestId(`open-${runId}`).click();
    await expectState(page, "interrupted");
    await page.getByTestId("btn-restore").click();
    await expectState(page, "paused");
    expect(await tick(page)).toBe(checkpointTick);
    await expect(page.getByTestId("metric-mass")).toHaveText(massAtCheckpoint!);
    await page.getByTestId("btn-step").click();
    await expect.poll(() => tick(page)).toBe(checkpointTick + 1);
  } finally {
    await stopServer(server);
  }
});
