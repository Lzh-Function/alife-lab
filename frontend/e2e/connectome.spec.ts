// Phase F2 acceptance gate: connectome create -> search -> stimulate/silence -> observe 2D & 3D
// -> deterministic rewind/replay -> fork.
import { expect, test } from "@playwright/test";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { BASE, DATA, expectState, openPreset, startServer, stopServer, tick, waitHealthy } from "./helpers";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const HAS_FLYWIRE = fs.existsSync(path.join(ROOT, ".alife/datasets/flywire_783/raw/Completeness_783.csv"));

test("connectome F2: create, search, stimulate/silence, 2D & 3D brain view, rewind/replay and fork", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    // 1. Create built-in 4-neuron reflex circuit from World Catalog.
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "connectome_tiny_circuit");
    await page.getByTestId("input-name").fill("e2e_connectome_f2");
    await page.getByTestId("field-runner.max_steps").fill("500");
    await page.getByTestId("field-runner.checkpoint_interval").fill("25");
    await page.getByTestId("input-profile").selectOption("local_cpu");
    await page.getByTestId("input-tps").fill("80");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expect(page).toHaveURL(/#\/lab\/run_[0-9a-f]{16}/);
    const runId = page.url().split("/").pop()!;

    // 2. Observe 2D spike raster / region canvas and MN9 firing response.
    await expect(page.getByTestId("world-canvas")).toBeVisible({ timeout: 20_000 });
    await expect(page.getByTestId("connectome-framing")).toBeVisible();
    await expect.poll(() => tick(page), { timeout: 20_000 }).toBeGreaterThan(45);
    await expect
      .poll(async () => Number(await page.getByTestId("metric-mn9_rate_hz").textContent()), { timeout: 20_000 })
      .toBeGreaterThan(0);

    // 3. Search cells by cell_type and inspect MN9 (CB0701).
    await page.getByTestId("connectome-search-q").fill("CB0701");
    await expect(page.getByTestId("connectome-search-count")).toContainText("1", { timeout: 10_000 });
    await page.getByTestId("btn-inspect-cell-2").click();
    await expect(page.getByTestId("inspector-connectome")).toContainText("CB0701", { timeout: 10_000 });

    // 4. Intervene live from the browser: stimulate bitter_grn and silence sugar_grn.
    await page.getByTestId("connectome-rate-input").fill("150");
    await page.getByTestId("btn-stim-bitter_grn").click();
    await page.getByTestId("btn-silence-sugar_grn").click();
    await expect(page.getByTestId("intervention-list")).toContainText("stimulate", { timeout: 10_000 });
    await expect(page.getByTestId("intervention-list")).toContainText("silence", { timeout: 10_000 });
    await expect(page.getByTestId("run-mode")).toHaveClass(/mode-exploratory/);

    // 5. Switch to 3D glowing brain view and confirm Three.js canvas renders.
    await page.getByTestId("view-brain3d").click();
    await expect(page.getByTestId("brain3d-wrap")).toBeVisible({ timeout: 15_000 });
    await expect(page.locator("[data-testid=brain3d-wrap] canvas")).toBeVisible({ timeout: 15_000 });
    await page.getByTestId("view-2d").click();
    await expect(page.getByTestId("world-canvas")).toBeVisible();

    // 6. Pause, rewind via Timeline deterministic replay, verify, and fork from a past tick.
    await expect.poll(() => tick(page), { timeout: 20_000 }).toBeGreaterThan(60);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");

    await page.getByTestId("replay-toggle").check();
    await expect(page.getByTestId("replay-banner")).toBeVisible();
    const source = page.getByTestId("replay-source");
    await page.getByTestId("replay-tick-input").fill("20");
    await expect(source).toHaveAttribute("data-pending", "false", { timeout: 15_000 });
    await expect(source.locator("strong")).toHaveText("t=20");

    await page.getByTestId("replay-verify").click();
    await expect(page.getByTestId("replay-verify-result")).toHaveClass(/ok-text/, { timeout: 20_000 });

    await page.getByTestId("replay-fork").click();
    await expect(page).not.toHaveURL(new RegExp(runId), { timeout: 20_000 });
    await expectState(page, "paused");
    expect(await tick(page)).toBe(20);
    await expect(page.getByTestId("lineage")).toContainText(runId);

    // 7. Beginner guide includes connectome entry.
    await page.goto(`${BASE}/#/guide/connectome`);
    await expect(page.locator("#guide-connectome")).toBeVisible();
  } finally {
    await stopServer(server);
  }
});

test("connectome F2 FlyWire v783: 138,639-neuron search, sugar->MN9 and 3D glowing brain", async ({ page }) => {
  test.skip(!HAS_FLYWIRE, "flywire_783 dataset not installed locally");
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "connectome_fly_sugar");
    await page.getByTestId("field-runner.max_steps").fill("200");
    await page.getByTestId("field-runner.checkpoint_interval").fill("100");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expect(page).toHaveURL(/#\/lab\/run_[0-9a-f]{16}/);

    // Wait for frames and MN9 response on the real 138,639-neuron brain.
    await expect(page.getByTestId("world-canvas")).toBeVisible({ timeout: 30_000 });
    await expect.poll(() => tick(page), { timeout: 40_000 }).toBeGreaterThan(20);

    // Search for MN9 (CB0701) across all 138,639 neurons.
    await page.getByTestId("connectome-search-q").fill("CB0701");
    await expect(page.getByTestId("connectome-search-count")).toContainText("2", { timeout: 20_000 });
    await page.getByTestId("btn-inspect-port-mn9_proboscis_extension").click();
    await expect(page.getByTestId("inspector-connectome")).toContainText("mn9_proboscis_extension", { timeout: 15_000 });
    await page.screenshot({ path: path.join(DATA, "connectome_flywire_2d.png") });

    // Switch to 3D glowing brain view (138,639 points in Three.js).
    await page.getByTestId("view-brain3d").click();
    await expect(page.getByTestId("brain3d-wrap")).toContainText("138,639", { timeout: 25_000 });
    await expect(page.locator("[data-testid=brain3d-wrap] canvas")).toBeVisible({ timeout: 15_000 });
    await page.waitForTimeout(1000);
    await page.screenshot({ path: path.join(DATA, "connectome_flywire_3d.png") });
  } finally {
    await stopServer(server);
  }
});

test("connectome F4: Mushroom Body KC->MBON plasticity displays changed synapses and forks cleanly", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "connectome_mb_tiny_pavlov");
    await page.getByTestId("input-name").fill("e2e_connectome_f4");
    await page.getByTestId("field-runner.max_steps").fill("300");
    await page.getByTestId("field-runner.checkpoint_interval").fill("50");
    await page.getByTestId("input-profile").selectOption("local_cpu");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expect(page).toHaveURL(/#\/lab\/run_[0-9a-f]{16}/);
    const runId = page.url().split("/").pop()!;

    // Wait for paired Odor A + PPL101 conditioning to complete and depress KC -> MBON11 synapses.
    await expectState(page, "completed");
    await expect(page.getByTestId("connectome-plasticity")).toBeVisible({ timeout: 25_000 });
    await expect
      .poll(async () => Number(await page.getByTestId("metric-plastic_changed_edges").textContent()), {
        timeout: 25_000,
      })
      .toBeGreaterThanOrEqual(2);
    await expect(page.getByTestId("connectome-plasticity")).toContainText("KCg-m");
    await expect(page.getByTestId("connectome-plasticity")).toContainText("MBON11");

    // Verify Replay + Fork works cleanly on a plastic run (pre- vs post-training comparison).
    await page.getByTestId("replay-toggle").check();
    await page.getByTestId("replay-tick-input").fill("50");
    const source = page.getByTestId("replay-source");
    await expect(source).toHaveAttribute("data-pending", "false", { timeout: 15_000 });
    await page.getByTestId("replay-fork").click();
    await expect(page).not.toHaveURL(new RegExp(runId), { timeout: 20_000 });
    expect(await tick(page)).toBe(50);
    await expect(page.getByTestId("lineage")).toContainText(runId);
  } finally {
    await stopServer(server);
  }
});

test("connectome F5: hidden-truth system ID runs in browser, renders RGB diagnostics and reports recovery", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "connectome_sysid_olfactory_tiny");
    await page.getByTestId("input-name").fill("e2e_connectome_f5");
    await page.getByTestId("field-runner.max_steps").fill("80");
    await page.getByTestId("field-runner.checkpoint_interval").fill("40");
    await page.getByTestId("input-profile").selectOption("local_cpu");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expect(page).toHaveURL(/#\/lab\/run_[0-9a-f]{16}/);

    await expect(page.getByTestId("world-canvas")).toBeVisible({ timeout: 25_000 });
    await expectState(page, "completed");
    await expect
      .poll(async () => Number(await page.getByTestId("metric-train_obs_r2").textContent()), {
        timeout: 20_000,
      })
      .toBeGreaterThan(0.8);
    await expect
      .poll(async () => Number(await page.getByTestId("metric-heldout_all_r2").textContent()), {
        timeout: 20_000,
      })
      .toBeGreaterThan(0.5);
  } finally {
    await stopServer(server);
  }
});
