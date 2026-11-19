// Phase 4: Growing NCA — draw a target, train (optimisation run), run the trained model, damage.
// Skipped when the local target offers no JAX backend.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy } from "./helpers";

test("nca: draw target, train, run the model and damage it", async ({ page, request }) => {
  const server = startServer();
  await waitHealthy();
  try {
    const targets = await (await request.get(`${BASE}/api/v1/compute-targets`)).json();
    test.skip(!targets.data.targets[0].capabilities.backends.includes("jax"), "JAX not installed");
    await page.goto(`${BASE}/#/catalog`);
    await page.getByTestId("preset-nca_train_regenerating").click();
    await page.getByTestId("field-target.size").fill("12");
    await page.getByTestId("field-dynamics.batch_size").fill("4");
    await page.getByTestId("field-dynamics.steps_min").fill("16");
    await page.getByTestId("field-dynamics.steps_max").fill("24");
    await page.getByTestId("target-custom").check();
    const pad = (await page.getByTestId("target-canvas").boundingBox())!;
    await page.mouse.move(pad.x + pad.width * 0.3, pad.y + pad.height * 0.5);
    await page.mouse.down();
    await page.mouse.move(pad.x + pad.width * 0.7, pad.y + pad.height * 0.5, { steps: 6 });
    await page.mouse.up();
    await page.getByTestId("input-profile").selectOption("local_interactive");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "running");
    await expect(page.locator(".stats dt").first()).toHaveText(/学習ステップ|optimizer step/);
    await expect.poll(() => tick(page), { timeout: 120_000 }).toBeGreaterThan(15);
    await expect(page.getByTestId("metric-loss")).not.toHaveText("—");
    await page.screenshot({ path: path.join(DATA, "130_nca_train.png") });

    await page.getByTestId("btn-run-model").click();
    await expect(page.getByTestId("run-name")).toContainText("_model_s", { timeout: 30_000 });
    await expectState(page, "running");
    await expect.poll(() => tick(page), { timeout: 60_000 }).toBeGreaterThan(20);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");
    await page.getByTestId("brush-damage").check();
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await expect(page.getByTestId("intervention-list")).toContainText("damage", { timeout: 10_000 });
    await page.getByTestId("brush-inspect").check();
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await expect(page.getByTestId("inspector-nca")).toBeVisible({ timeout: 10_000 });
  } finally {
    await stopServer(server);
  }
});
