// Discovery ledger: register the best candidate of a search run, run the pre-declared assay
// plan and read the verdict, evidence level and history.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, openPreset, startServer, stopServer, waitHealthy } from "./helpers";

test("discovery: register a search candidate and assay it", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "search_lenia_orbium_cma");
    await page.getByTestId("field-dynamics.population").fill("8");
    await page.getByTestId("field-runner.max_steps").fill("6");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "completed");

    await page.getByTestId("brush-inspect").check();
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.75, box.y + box.height * 0.5);
    await page.getByTestId("search-best-claim").selectOption("faster_than_base");
    await page.getByTestId("search-best-register").click();

    await expect(page).toHaveURL(/#\/discoveries\/dsc_[0-9a-f]{16}/);
    await expect(page.getByTestId("discovery-status")).toHaveAttribute("data-status", "new");
    await expect(page.getByTestId("discovery-evidence")).toHaveText("E1");
    await expect(page.getByTestId("check-baseline")).toBeVisible();
    await page.getByTestId("btn-triage").click();
    await expect(page.getByTestId("discovery-status")).toHaveAttribute("data-status", "triaged");
    await page.getByTestId("btn-assay").click();
    await expect(page.getByTestId("discovery-status")).toHaveAttribute("data-status", /validated|rejected|inconclusive/, { timeout: 90_000 });
    await expect(page.getByTestId("verdict-reproduce")).not.toHaveText("—");
    await expect(page.getByTestId("discovery-history")).toContainText("試験が完了");
    await page.screenshot({ path: path.join(DATA, "discovery.png"), fullPage: true });

    await page.getByRole("link", { name: /台帳の一覧へ/ }).click();
    await expect(page.getByTestId("discovery-list")).toBeVisible();
  } finally {
    await stopServer(server);
  }
});
