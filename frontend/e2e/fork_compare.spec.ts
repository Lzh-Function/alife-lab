// Phase 2 slice: Pause -> Checkpoint -> paired Fork -> Intervention -> Compare from the browser.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy } from "./helpers";

test("paired fork with a treatment brush diverges from the control arm", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await page.getByTestId("preset-conway_glider").click();
    await page.getByTestId("input-name").fill("e2e_fork_parent");
    await page.getByTestId("input-max-steps").fill("100000");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-start-paused").check();
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expect(page).toHaveURL(/#\/lab\/run_[0-9a-f]{16}/);
    const parentId = page.url().split("/").pop()!;
    await expectState(page, "paused");
    for (let i = 0; i < 3; i++) {
      const before = await tick(page);
      await page.getByTestId("btn-step").click();
      await expect.poll(() => tick(page)).toBe(before + 1);
    }

    // One operation: checkpoint the parent, create control + treatment arms, open the comparison.
    await page.getByTestId("btn-paired-fork").click();
    await expect(page).toHaveURL(/#\/compare\/cmp_[0-9a-f]{16}/, { timeout: 15_000 });
    for (const arm of ["control", "treatment"]) {
      await expect(page.getByTestId(`pane-${arm}`).getByTestId("run-state")).toHaveAttribute("data-state", "paused", { timeout: 20_000 });
      await expect(page.getByTestId(`tick-${arm}`)).toHaveText("3");
    }

    // Intervene on the treatment arm only.
    await page.getByTestId("brush-arm-treatment").check();
    const box = (await page.getByTestId("canvas-treatment").boundingBox())!;
    await page.mouse.move(box.x + box.width * 0.2, box.y + box.height * 0.2);
    await page.mouse.down();
    await page.mouse.move(box.x + box.width * 0.24, box.y + box.height * 0.22, { steps: 5 });
    await page.mouse.up();
    await expect(page.getByTestId("pane-treatment").locator(".mode")).toHaveText("exploratory", { timeout: 10_000 });
    await expect(page.getByTestId("pane-control").locator(".mode")).toHaveText("reference");

    // Advance both arms identically and checkpoint them at the same tick.
    await page.getByRole("spinbutton", { name: /tick/i }).fill("12");
    await page.getByTestId("btn-step-all").click();
    for (const arm of ["control", "treatment"]) await expect(page.getByTestId(`tick-${arm}`)).toHaveText("15", { timeout: 10_000 });
    await page.getByTestId("btn-checkpoint-all").click();

    const table = page.getByTestId("divergence-table");
    await expect(table.locator("tbody tr")).toHaveCount(2, { timeout: 10_000 });
    const rows = await table.locator("tbody tr").allInnerTexts();
    const parse = (r: string) => r.trim().split(/\s+/).map(Number);
    const [latest, origin] = rows.map(parse);
    expect(origin[0]).toBe(3);
    expect(origin[1]).toBe(0); // identical fork-origin states (before the brush)
    expect(latest[0]).toBe(15);
    expect(latest[1]).toBeGreaterThan(0); // treatment diverged
    await page.screenshot({ path: path.join(DATA, "10_compare.png"), fullPage: true });

    // Lineage is visible from the parent.
    await page.goto(`${BASE}/#/lab/${parentId}`);
    await expect(page.getByTestId("lineage")).toContainText("cmp_");
  } finally {
    await stopServer(server);
  }
});
