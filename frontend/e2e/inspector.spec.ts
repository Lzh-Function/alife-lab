// Inspector: read-only queries on the live worker and on replayed states.
import { expect, test, type Page } from "@playwright/test";
import { BASE, expectState, startServer, stopServer, tick, waitHealthy, openPreset } from "./helpers";

async function clickCell(page: Page, x: number, y: number, world = 128) {
  const box = (await page.getByTestId("world-canvas").boundingBox())!;
  await page.mouse.click(box.x + ((x + 0.5) / world) * box.width, box.y + ((y + 0.5) / world) * box.height);
}

test("inspector explains conway cells live and in replay without changing the run", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "conway_glider");
    await page.getByTestId("field-runner.max_steps").fill("100000");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-start-paused").check();
    await page.getByTestId("btn-create").click();
    await expectState(page, "paused");
    await page.getByTestId("brush-inspect").check();

    // Glider at offset (62, 62): cell (63, 62) is alive with 1 neighbour -> dies.
    await clickCell(page, 63, 62);
    await expect(page.getByTestId("inspector-next")).toHaveText("死亡");
    await expect(page.getByTestId("inspector-source")).toContainText("実行中ワーカー");
    // (62, 63) is dead with 3 live neighbours -> born.
    await clickCell(page, 62, 63);
    await expect(page.getByTestId("inspector-next")).toHaveText("誕生");
    await expect(page.getByTestId("inspector-cell")).toContainText("3");

    // Advance, then inspect tick 0 through replay.
    await page.getByTestId("btn-step").click();
    await page.getByTestId("btn-step").click();
    await expect.poll(() => tick(page)).toBe(2);
    await page.getByTestId("replay-toggle").check();
    await page.getByTestId("replay-tick-input").fill("0");
    await expect(page.getByTestId("replay-source").locator("strong")).toHaveText("t=0");
    await clickCell(page, 63, 62);
    // The replay process may need a cold start (spawn + imports) on the first request.
    await expect(page.getByTestId("inspector-source")).toContainText("リプレイ", { timeout: 30_000 });
    await expect(page.getByTestId("inspector-source")).toContainText("t=0");
    await expect(page.getByTestId("inspector-next")).toHaveText("死亡");

    // Inspection is read-only: no interventions, mode stays reference, tick unchanged.
    await page.getByTestId("replay-toggle").uncheck();
    await expect(page.getByTestId("intervention-list").locator("li")).toHaveCount(0);
    await expect(page.getByTestId("run-mode")).toContainText("reference");
    expect(await tick(page)).toBe(2);
  } finally {
    await stopServer(server);
  }
});
