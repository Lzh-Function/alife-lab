// Search run: create from the catalog, watch generations, inspect the parameter map and open
// the best candidate as an ordinary Lenia run.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, openPreset, startServer, stopServer, tick, waitHealthy } from "./helpers";

test("search: CMA-ES over Lenia parameters, inspect and open a candidate", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "search_lenia_orbium_cma");
    await page.getByTestId("field-dynamics.population").fill("6");
    await page.getByTestId("field-dynamics.eval_steps").fill("120");
    await page.getByTestId("field-runner.max_steps").fill("4");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();

    await expect(page).toHaveURL(/#\/lab\/run_[0-9a-f]{16}/);
    await expect.poll(() => tick(page), { timeout: 60_000 }).toBe(4);
    await expectState(page, "completed");
    await expect(page.getByTestId("metric-evaluations")).toHaveText("24");
    await page.screenshot({ path: path.join(DATA, "search_done.png") });

    // Inspect: any click returns the best candidate; it can be opened as a Lenia run.
    await page.getByTestId("brush-inspect").check();
    const canvas = page.getByTestId("world-canvas");
    const box = (await canvas.boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.75, box.y + box.height * 0.5);
    await expect(page.getByTestId("inspector-search")).toBeVisible();
    const best = Number(await page.getByTestId("search-best-score").textContent());
    expect(best).toBeGreaterThan(0);
    await page.getByTestId("search-best-open").click();
    await expect(page.getByTestId("run-name")).toContainText("cand", { timeout: 20_000 });
    await expectState(page, "paused");
    await expect(page.getByTestId("metric-mass")).toBeVisible();
  } finally {
    await stopServer(server);
  }
});
