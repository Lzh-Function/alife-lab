// Phase 2 slice 3: a second, continuous substrate (Boids) through the same GUI paths.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy } from "./helpers";

test("boids run: observe, spawn intervention, paired fork, replay verification", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await page.getByTestId("preset-boids_flocking").click();
    await page.getByTestId("input-name").fill("e2e_boids");
    await page.getByTestId("field-initialization.count").fill("200");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-tps").fill("60");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expect(page).toHaveURL(/#\/lab\/run_[0-9a-f]{16}/);
    const runId = page.url().split("/").pop()!;
    await expectState(page, "running");
    await expect(page.getByTestId("run-mode")).toContainText("exploratory");
    await expect.poll(() => tick(page), { timeout: 20_000 }).toBeGreaterThan(120);
    await expect(page.getByTestId("metric-population")).toHaveText("200");
    const pol = Number(await page.getByTestId("metric-polarization").textContent());
    expect(pol).toBeGreaterThan(0);
    expect(pol).toBeLessThanOrEqual(1);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");

    // Spawn intervention: click inside the world.
    await page.getByTestId("brush-spawn").check();
    await page.getByTestId("tool-count").fill("25");
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await expect(page.getByTestId("metric-population")).toHaveText("225", { timeout: 10_000 });
    await expect(page.getByTestId("intervention-list")).toContainText("spawn");
    // Inspector on the live boids state (read-only).
    await page.getByTestId("brush-inspect").check();
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await expect(page.getByTestId("inspector-boid")).toBeVisible({ timeout: 10_000 });
    await expect(page.getByTestId("inspector-alignment")).toContainText("|");
    await expect(page.getByTestId("metric-population")).toHaveText("225");
    await page.getByTestId("brush-off").check();
    await page.screenshot({ path: path.join(DATA, "30_boids.png") });

    // Replay: seek to a recomputed tick, then verify all checkpoint segments.
    await page.getByTestId("btn-checkpoint").click();
    await expect(page.getByTestId("checkpoint-list")).toContainText("manual", { timeout: 10_000 });
    await page.getByTestId("replay-toggle").check();
    const source = page.getByTestId("replay-source");
    await page.getByTestId("replay-tick-input").fill("77");
    await expect(source).toHaveAttribute("data-pending", "false", { timeout: 10_000 });
    await expect(source.locator("strong")).toHaveText("t=77");
    await expect(source).toContainText("200");
    await page.getByTestId("replay-verify").click();
    await expect(page.getByTestId("replay-verify-result")).toHaveClass(/ok-text/, { timeout: 30_000 });
    await page.getByTestId("replay-toggle").uncheck();

    // Paired fork: spawn into the treatment arm only, step both, compare.
    await page.getByTestId("btn-paired-fork").click();
    await expect(page).toHaveURL(/#\/compare\/cmp_[0-9a-f]{16}/, { timeout: 15_000 });
    for (const arm of ["control", "treatment"]) {
      await expect(page.getByTestId(`pane-${arm}`).getByTestId("run-state")).toHaveAttribute("data-state", "paused", { timeout: 20_000 });
    }
    await page.getByTestId("brush-arm-treatment").check();
    await page.getByTestId("brush-spawn").check();
    await page.getByTestId("tool-count").fill("25");
    const tbox = (await page.getByTestId("canvas-treatment").boundingBox())!;
    await page.mouse.click(tbox.x + tbox.width * 0.3, tbox.y + tbox.height * 0.3);
    await expect(page.getByTestId("pane-treatment").locator(".mode")).toHaveText("exploratory");
    await page.getByRole("spinbutton", { name: /tick/i }).fill("40");
    await page.getByTestId("btn-step-all").click();
    await page.getByTestId("btn-checkpoint-all").click();
    const table = page.getByTestId("divergence-table");
    await expect(table.locator("thead")).toContainText(/RMS/);
    // Fork origin + manual checkpoint (+ a periodic one if the 40 ticks cross a multiple of 200).
    await expect.poll(() => table.locator("tbody tr").count(), { timeout: 15_000 }).toBeGreaterThanOrEqual(2);
    const rows = await table.locator("tbody tr").allInnerTexts();
    const parse = (r: string) => r.trim().split(/\s+/).map(Number);
    const latest = parse(rows[0]);
    const origin = parse(rows[rows.length - 1]);
    expect(origin[1]).toBe(0); // identical fork-origin states
    expect(latest[1]).toBeGreaterThan(0); // treatment diverged
    expect(latest[2]).toBe(225); // control population
    expect(latest[3]).toBe(250); // treatment population (+25)
    await page.screenshot({ path: path.join(DATA, "31_boids_compare.png"), fullPage: true });
    console.log(`boids run ${runId}`);
  } finally {
    await stopServer(server);
  }
});
