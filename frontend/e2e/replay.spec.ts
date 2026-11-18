// Phase 2 slice 2: timeline seek by deterministic replay, verification, fork from any tick.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy } from "./helpers";

test("timeline replay seeks, verifies and forks from an arbitrary tick", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await page.getByTestId("preset-conway_glider").click();
    await page.getByTestId("input-name").fill("e2e_replay");
    await page.getByTestId("field-runner.max_steps").fill("100000");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-tps").fill("40");
    await page.getByTestId("btn-create").click();
    await expect(page).toHaveURL(/#\/lab\/run_[0-9a-f]{16}/);
    const runId = page.url().split("/").pop()!;
    await expectState(page, "running");
    await expect.poll(() => tick(page), { timeout: 20_000 }).toBeGreaterThan(130);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");

    // Seek to a tick between checkpoints (interval 100): recomputed, not stored.
    await page.getByTestId("replay-toggle").check();
    await expect(page.getByTestId("replay-banner")).toBeVisible();
    const source = page.getByTestId("replay-source");
    const settled = async (t: number) => {
      await expect(source).toHaveAttribute("data-pending", "false", { timeout: 10_000 });
      await expect(source.locator("strong")).toHaveText(`t=${t}`);
    };
    await page.getByTestId("replay-tick-input").fill("57");
    await settled(57);
    await expect(source).toContainText("t=0 + 57 tick");
    await page.getByTestId("replay-forward").click();
    await settled(58);
    await page.getByTestId("replay-tick-input").fill("100");
    await settled(100);
    await expect(source).toContainText(/厳密|Exact/); // a checkpoint exists at tick 100
    await page.screenshot({ path: path.join(DATA, "20_replay.png"), fullPage: true });

    // Replay playback advances through ticks.
    await page.getByTestId("replay-tick-input").fill("10");
    await settled(10);
    await page.getByTestId("replay-play").click();
    await expect(source.locator("strong")).toHaveText(/t=(1[5-9]|[2-9]\d)/, { timeout: 10_000 });
    await page.getByTestId("replay-play").click();

    // Verification of every checkpoint segment.
    await page.getByTestId("replay-verify").click();
    await expect(page.getByTestId("replay-verify-result")).toHaveClass(/ok-text/, { timeout: 20_000 });

    // Fork from an arbitrary (non-checkpointed) tick.
    await page.getByTestId("replay-tick-input").fill("57");
    await settled(57);
    await page.getByTestId("replay-fork").click();
    await expect(page).not.toHaveURL(new RegExp(runId), { timeout: 15_000 });
    await expectState(page, "paused");
    expect(await tick(page)).toBe(57);
    await expect(page.getByTestId("metric-alive_cell_count")).toHaveText("5");
    await expect(page.getByTestId("lineage")).toContainText(runId);
  } finally {
    await stopServer(server);
  }
});
