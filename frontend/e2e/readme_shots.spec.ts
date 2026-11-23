// README screenshots. Runs only with ALIFE_README_SHOTS=1 and writes JPEGs to docs/images/.
//   ALIFE_README_SHOTS=1 ALIFE_E2E_CHROMIUM=<chromium> npx playwright test e2e/readme_shots.spec.ts
// Uses a throwaway server and data directory like the other E2E tests.
import { expect, test, type Locator, type Page } from "@playwright/test";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { BASE, openPreset, startServer, stopServer, tick, waitHealthy } from "./helpers";

const OUT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..", "docs", "images");
const enabled = process.env.ALIFE_README_SHOTS === "1";

async function shot(target: Page | Locator, name: string) {
  await target.screenshot({ path: path.join(OUT, `${name}.jpg`), type: "jpeg", quality: 80 });
}

async function create(page: Page, preset: string, tps: string, minTick = 3, edit?: (p: Page) => Promise<void>, profile = "cpu_dev") {
  await page.goto(`${BASE}/#/catalog`);
  await openPreset(page, preset);
  await edit?.(page);
  await page.getByTestId("input-profile").selectOption(profile);
  await page.getByTestId("input-tps").fill(tps);
  await expect(page.getByTestId("validation")).toContainText("OK");
  await page.getByTestId("btn-create").click();
  await expect.poll(() => tick(page), { timeout: 120_000 }).toBeGreaterThan(minTick);
}

async function stop(page: Page) {
  if (await page.getByTestId("btn-cancel").isEnabled()) await page.getByTestId("btn-cancel").click();
}

async function view(page: Page, mode: string) {
  // Worlds with only the 2D view have no view-mode buttons.
  const button = page.getByTestId(`view-${mode}`);
  if (await button.count()) await button.click();
}

test.describe("README screenshots", () => {
  test.skip(!enabled, "set ALIFE_README_SHOTS=1 to regenerate docs/images");
  test.setTimeout(30 * 60_000);

  test("capture", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 });
    const server = startServer();
    await waitHealthy();
    try {
      // Pages without a run.
      await page.goto(`${BASE}/#/catalog`);
      await page.waitForTimeout(1500);
      await shot(page, "catalog");
      await page.goto(`${BASE}/#/guide`);
      await page.waitForTimeout(1500);
      await shot(page, "guide");

      // Live lab overview: multichannel Lenia in the 2D measurement view.
      await create(page, "lenia_mc_gyrating", "30", 60);
      await view(page, "2d");
      await page.waitForTimeout(4000);
      await shot(page, "lab_overview");
      await shot(page.getByTestId("world-canvas"), "grid_lenia_mc");
      await view(page, "terrain");
      await page.waitForTimeout(5000);
      await shot(page.getByTestId("world-canvas-3d"), "view_terrain_lenia_mc");
      await stop(page);

      // Physarum network as a 3D landscape.
      await create(page, "physarum_network", "0", 300);
      await view(page, "terrain");
      await page.waitForTimeout(6000);
      await shot(page.getByTestId("world-canvas-3d"), "view_terrain_physarum");
      await page.getByTestId("view3d-rotate").check(); // a slightly turned view for the hero image
      await page.waitForTimeout(2500);
      await page.getByTestId("view3d-rotate").uncheck();
      await shot(page, "lab_physarum_3d");
      await stop(page);

      // Flow-Lenia as a 3D landscape.
      await create(page, "flow_lenia_example", "30", 200);
      await view(page, "terrain");
      await page.waitForTimeout(6000);
      await shot(page.getByTestId("world-canvas-3d"), "view_terrain_flow_lenia");
      await stop(page);

      // Time stack: the Orbium glider leaves a diagonal tube.
      await create(page, "lenia_orbium", "3", 3);
      await view(page, "spacetime");
      await expect
        .poll(async () => Number(await page.getByTestId("world-canvas-3d").getAttribute("data-slices")), { timeout: 180_000 })
        .toBeGreaterThan(70);
      await page.waitForTimeout(1500);
      await shot(page.getByTestId("world-canvas-3d"), "view_timestack_orbium");
      await stop(page);

      // Particle Life in the glow view.
      await create(page, "particle_life_clusters", "30", 150);
      await view(page, "glow");
      await page.waitForTimeout(4000);
      await shot(page.getByTestId("world-canvas"), "particle_life_glow");
      await stop(page);

      // A small gallery of 2D worlds (measurement view).
      const grids: Array<[string, string, number]> = [
        ["conway_random_soup", "grid_conway", 60],
        ["gray_scott_coral", "grid_gray_scott", 6000],
        ["bff_reference", "grid_bff", 300],
        ["protocell_default", "grid_protocell", 400],
        ["agent_world_evolution", "grid_agent_world", 300],
        ["nca_heart24_pretrained", "grid_nca", 120],
      ];
      for (const [preset, name, minTick] of grids) {
        // NCA runs on the JAX backend, offered by the interactive profile.
        await create(page, preset, "0", minTick, undefined, preset.startsWith("nca") ? "local_interactive" : "cpu_dev");
        await view(page, "2d");
        await page.waitForTimeout(1500);
        await shot(page.getByTestId("world-canvas"), name);
        await stop(page);
      }
      await create(page, "boids_flocking", "30", 300);
      await view(page, "glow");
      await page.waitForTimeout(4000);
      await shot(page.getByTestId("world-canvas"), "grid_boids");
      await stop(page);

      // Embodied creature on rough ground with reflexes: 3D view and gait diagram.
      await create(page, "creature_rough_wave_reflex", "30", 150);
      await page.waitForTimeout(4000);
      // Crop to the 3D view and the gait panel below it.
      await page.locator(".canvas-wrap").scrollIntoViewIfNeeded();
      const top = (await page.locator(".canvas-wrap").boundingBox())!;
      const bottom = (await page.getByTestId("gait-panel").boundingBox())!;
      const scroll = await page.evaluate(() => window.scrollY);
      await page.screenshot({
        path: path.join(OUT, "creature_rough.jpg"),
        type: "jpeg",
        quality: 80,
        fullPage: true,
        clip: { x: top.x, y: top.y + scroll, width: top.width, height: bottom.y + bottom.height - top.y },
      });
      await stop(page);

      // Evolution: a small MAP-Elites search of creature bodies, then its lineage view.
      await page.goto(`${BASE}/#/catalog`);
      await openPreset(page, "search_creature_qd_map_elites");
      await page.getByTestId("field-dynamics.population").fill("12");
      await page.getByTestId("field-dynamics.eval_steps").fill("200");
      await page.getByTestId("field-runner.max_steps").fill("10");
      await page.getByTestId("input-profile").selectOption("cpu_dev");
      await page.getByTestId("input-tps").fill("0");
      await page.getByTestId("btn-create").click();
      await expect(page.getByTestId("lab-head").getByTestId("run-state")).toHaveAttribute("data-state", "completed", { timeout: 600_000 });
      await page.waitForTimeout(1500);
      await shot(page, "search_lab");
      await page.getByTestId("btn-lineage").click();
      await expect(page.getByTestId("lineage-svg")).toBeVisible();
      await page.waitForTimeout(2000);
      await shot(page, "lineage");
    } finally {
      await stopServer(server);
    }
  });

  test("capture connectome and fly worlds", async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 1080 });
    const server = startServer();
    await waitHealthy();
    try {
      // 1. FlyWire v783 full 138,639-cell 3D glowing brain (sugar feeding reflex).
      await create(page, "connectome_fly_sugar", "60", 120, undefined, "local_gpu");
      await view(page, "brain3d");
      await page.waitForTimeout(4000);
      await shot(page.getByTestId("world-canvas"), "connectome_brain3d");
      await stop(page);

      // 2. Phase F8 Level 1: Embodied FlyWire 2-pane (CT 3D MuJoCo fly + 138,639-cell glowing brain).
      await create(page, "fly_embodied_flywire", "60", 250, undefined, "local_gpu");
      await page.waitForTimeout(4000);
      await shot(page.getByTestId("world-canvas"), "fly_embodied_2pane");
      await stop(page);

      // 3. Phase F8 Level 2: MaleCNS whole-CNS VNC direct motor drive + 6-leg x 6-muscle MN matrix.
      await create(page, "fly_embodied_malecns_vnc_learned", "60", 200, undefined, "local_gpu");
      await page.waitForTimeout(3500);
      await shot(page.getByTestId("world-canvas"), "fly_embodied_vnc_l2");
      await stop(page);

      // 4. Phase F11 Finale: Dual-Sex Whole-CNS Ecosystem (8 MaleCNS + 8 BANC = 2.58M neurons).
      await create(page, "fly_swarm_ecosystem_16", "60", 120, undefined, "local_gpu");
      await page.waitForTimeout(3000);
      await shot(page.getByTestId("fly-swarm-view"), "fly_swarm_ecosystem");
      await stop(page);

      // 5. Phase F12 + F12d: fly_world open-ended evolution + Signal Dictionary + F12d Replay panel.
      await create(page, "fly_world_flywire_evo", "0", 400, undefined, "local_gpu");
      await page.waitForTimeout(1500);
      const predBtn = page.getByTestId("fly-world-view").locator("button", { hasText: /🦅/ });
      if (await predBtn.isEnabled()) await predBtn.click();
      await page.waitForTimeout(1500);
      await shot(page.getByTestId("fly-world-view"), "fly_world_evolution");
      await stop(page);
    } finally {
      await stopServer(server);
    }
  });
});
