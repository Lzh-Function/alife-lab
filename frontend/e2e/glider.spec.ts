// Phase 1 acceptance: launch, observe, pause, step, brush, checkpoint, restart server, restore.
// Starts and restarts `uv run alife serve` itself with an isolated ALIFE_HOME / ALIFE_RUNS_DIR.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy } from "./helpers";

test("conway glider vertical slice survives a server restart", async ({ page }) => {
  let server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await page.getByTestId("preset-conway_glider").click();
    await page.getByTestId("input-name").fill("e2e_glider");
    await page.getByTestId("input-max-steps").fill("100000");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-tps").fill("20");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();

    // Launch + observe: the run runs and the tick advances live over WebSocket.
    await expect(page).toHaveURL(/#\/lab\/run_[0-9a-f]{16}/);
    const runId = page.url().split("/").pop()!;
    await expectState(page, "running");
    await expect.poll(() => tick(page), { timeout: 15_000 }).toBeGreaterThan(8);
    await expect(page.getByTestId("metric-alive_cell_count")).toHaveText("5");
    await page.screenshot({ path: path.join(DATA, "01_running.png") });

    // Pause.
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");
    const pausedAt = await tick(page);
    await page.waitForTimeout(600);
    expect(await tick(page)).toBe(pausedAt);

    // Step exactly one tick.
    await page.getByTestId("btn-step").click();
    await expect.poll(() => tick(page)).toBe(pausedAt + 1);

    // Brush: draw a stroke; it is logged as an intervention and flips the mode to exploratory.
    await page.getByTestId("brush-draw").check();
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    await page.mouse.move(box.x + box.width * 0.1, box.y + box.height * 0.1);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.1, { steps: 8 });
    await page.mouse.up();
    await expect(page.getByTestId("intervention-list")).toContainText("applied", { timeout: 10_000 });
    await expect(page.getByTestId("run-mode")).toContainText("exploratory");
    await expect.poll(async () => Number(await page.getByTestId("metric-alive_cell_count").textContent())).toBeGreaterThan(5);
    const aliveAfterBrush = Number(await page.getByTestId("metric-alive_cell_count").textContent());
    await page.getByTestId("brush-off").check();
    await page.screenshot({ path: path.join(DATA, "02_brushed.png") });

    // Manual checkpoint.
    await page.getByTestId("btn-checkpoint").click();
    await expect(page.getByTestId("checkpoint-list")).toContainText("manual", { timeout: 10_000 });
    const checkpointTick = await tick(page);

    // Restart the service (graceful stop -> shutdown checkpoint -> new controller).
    await stopServer(server);
    server = startServer();
    await waitHealthy();

    await page.goto(`${BASE}/#/library`);
    await page.getByTestId(`open-${runId}`).click();
    await expectState(page, "interrupted");
    await expect.poll(() => tick(page)).toBe(checkpointTick);
    await page.getByTestId("btn-restore").click();
    await expectState(page, "paused");
    expect(await tick(page)).toBe(checkpointTick);
    await expect(page.getByTestId("metric-alive_cell_count")).toHaveText(String(aliveAfterBrush));
    await page.getByTestId("btn-step").click();
    await expect.poll(() => tick(page)).toBe(checkpointTick + 1);
    await expect(page.getByTestId("event-list")).toContainText("job_started");
    await page.screenshot({ path: path.join(DATA, "03_restored.png") });

    // Reload the browser: reconnects to the live run without duplicating the simulation.
    await page.reload();
    await expectState(page, "paused");
    expect(await tick(page)).toBe(checkpointTick + 1);

    const verify = await page.evaluate(async (id) => {
      const token = document.querySelector('meta[name="alife-token"]')!.getAttribute("content")!;
      const r = await fetch(`/api/v1/experiments/${id}/verify`, { method: "POST", headers: { "X-ALife-Token": token } });
      return (await r.json()).data.ok as boolean;
    }, runId);
    expect(verify).toBe(true);
    console.log(`E2E data dir: ${DATA}`);
  } finally {
    await stopServer(server);
  }
});
