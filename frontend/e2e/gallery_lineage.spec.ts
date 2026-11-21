// Gallery (save what the canvas shows, list, remove) and the lineage view of a search run.
import { expect, test } from "@playwright/test";
import { BASE, expectState, openPreset, startServer, stopServer, waitHealthy } from "./helpers";

test("gallery: save the current picture of a run, see it, remove it", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "conway_glider");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-start-paused").check();
    await page.getByTestId("btn-create").click();
    await expectState(page, "paused");
    await page.getByTestId("btn-gallery").click();
    await page.getByTestId("gallery-title").fill("my glider");
    await page.getByTestId("gallery-note").fill("moves diagonally");
    await page.getByTestId("gallery-confirm").click();
    await expect(page.getByTestId("gallery-msg")).toBeVisible();
    await page.goto(`${BASE}/#/gallery`);
    const card = page.locator("[data-testid^=gallery-gal_]");
    await expect(card).toHaveCount(1);
    await expect(card).toContainText("my glider");
    await expect(card.locator("img")).toHaveJSProperty("complete", true);
    expect(await card.locator("img").evaluate((img: HTMLImageElement) => img.naturalWidth)).toBeGreaterThan(0);
    page.once("dialog", (d) => d.accept());
    await card.getByRole("button").click();
    await expect(page.getByTestId("gallery-empty")).toBeVisible();
  } finally {
    await stopServer(server);
  }
});

test("lineage: MAP-Elites run family tree with ancestry selection", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "search_lenia_qd_map_elites");
    await page.getByTestId("field-dynamics.population").fill("8");
    await page.getByTestId("field-dynamics.eval_steps").fill("120");
    await page.getByTestId("field-runner.max_steps").fill("6");
    await page.getByTestId("input-tps").fill("0");
    await page.getByTestId("btn-create").click();
    await expectState(page, "completed");
    await page.getByTestId("btn-lineage").click();
    await expect(page.getByTestId("lineage-page")).toBeVisible();
    await expect(page.locator("[data-testid^=node-]")).toHaveCount(48);
    await expect(page.getByTestId("lineage-candidate")).toBeVisible(); // the best by default
    await page.getByTestId("node-47").click();
    await expect(page.getByTestId("lineage-candidate")).toContainText("#47");
    await page.getByTestId("lineage-axis").selectOption("score");
    await page.getByTestId("lineage-candidate-open").click();
    await expect(page).toHaveURL(/#\/lab\/run_[0-9a-f]{16}/);
    await expect(page.getByTestId("run-name")).toContainText("cand47");
  } finally {
    await stopServer(server);
  }
});
