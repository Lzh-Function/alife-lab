// Phase 2 slice B: seed-level sweep from the GUI (alignment ablation for Boids).
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, startServer, stopServer, waitHealthy } from "./helpers";

test("boids alignment ablation sweep: conditions x seeds with seed-level statistics", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/sweeps`);
    await page.getByTestId("sweep-name").fill("e2e_alignment_ablation");
    await page.getByTestId("sweep-preset").selectOption("boids_flocking");
    await page.getByTestId("sweep-seed-count").fill("4");
    await page.getByTestId("sweep-factor").selectOption("dynamics.alignment_weight");
    await page.getByTestId("sweep-values").fill("0.05, 0.0");
    await page.getByTestId("sweep-labels").fill("alignment, no_alignment");
    await page.getByTestId("sweep-metric").selectOption("polarization");
    await page.getByTestId("sweep-max-steps").fill("600");
    await expect(page.getByTestId("sweep-plan")).toContainText("8 runs", { timeout: 10_000 });
    await page.getByTestId("sweep-create").click();

    await expect(page).toHaveURL(/#\/sweeps\/swp_[0-9a-f]{16}/);
    await expect(page.getByTestId("sweep-state")).toHaveAttribute("data-state", "completed", { timeout: 60_000 });
    await expect(page.getByTestId("sweep-progress")).toContainText("8 / 8");
    const conditions = page.getByTestId("sweep-conditions");
    await expect(conditions).toContainText("alignment");
    await expect(conditions.locator("tbody tr")).toHaveCount(2);
    await expect(page.getByTestId("strip-plot").locator("circle")).toHaveCount(8);

    // 4 vs 4 seeds with complete separation: exact two-sided p = 2 / C(8,4) = 0.0286.
    const row = page.getByTestId("sweep-comparisons").locator("tbody tr").first();
    await expect(row).toContainText("alignment − no_alignment");
    await expect(row).toContainText("0.0286");
    await expect(row).toContainText("exact");
    const cells = await row.locator("td").allInnerTexts();
    expect(Number(cells[1])).toBeGreaterThan(0.3); // mean polarization difference
    expect(Number(cells[3])).toBe(1); // Cliff's delta
    await expect(page.getByTestId("sweep-notes").locator('[data-code="exploratory"]')).toContainText("探索的");
    await expect(page.getByTestId("sweep-runs").locator("tbody tr")).toHaveCount(8);
    await page.screenshot({ path: path.join(DATA, "40_sweep.png"), fullPage: true });
  } finally {
    await stopServer(server);
  }
});
