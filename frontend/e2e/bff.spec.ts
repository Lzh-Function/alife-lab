// Phase 3: BFF primordial soup in the GUI (epoch clock, soup view, inspection, injection).
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy, openPreset } from "./helpers";

test("bff soup: epochs, inspect a program, inject into a block of tiles", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "bff_reference");
    await page.getByTestId("field-soup.num_programs").fill("1024");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-tps").fill("0");
    await page.getByTestId("input-start-paused").check();
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "paused");
    await expect(page.getByTestId("run-mode")).toContainText("reference");
    await expect(page.locator(".stats dt").first()).toHaveText("エポック");
    for (let i = 0; i < 3; i++) {
      const before = await tick(page);
      await page.getByTestId("btn-step").click();
      await expect.poll(() => tick(page)).toBe(before + 1);
    }
    await expect(page.getByTestId("metric-unique_programs")).not.toHaveText("null");

    // Inspect the first tile (program #0).
    await page.getByTestId("brush-inspect").check();
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    const tile = box.width / 32; // 1024 programs -> 32 columns
    await page.mouse.click(box.x + tile * 0.5, box.y + tile * 0.5);
    await expect(page.getByTestId("inspector-program")).toContainText("#0");
    await expect(page.getByTestId("inspector-disassembly")).toBeVisible();

    // Inject a program into a 3x3 block around tile (5, 5) -> 9 exact copies.
    await page.getByTestId("brush-inject").check();
    await page.getByTestId("tool-program").fill("[[{.>]-]++++");
    await page.getByRole("slider").fill("1");
    await page.mouse.click(box.x + tile * 5.5, box.y + tile * 5.5);
    await expect(page.getByTestId("intervention-list")).toContainText("inject_program", { timeout: 10_000 });
    await expect(page.getByTestId("run-mode")).toContainText("exploratory");
    await page.getByTestId("brush-inspect").check();
    await page.mouse.click(box.x + tile * 5.5, box.y + tile * 5.5);
    await expect(page.getByTestId("inspector-copies")).toHaveText("9");
    await expect(page.getByTestId("inspector-disassembly")).toContainText("[[{.>]-]++++");
    await page.screenshot({ path: path.join(DATA, "60_bff.png") });
  } finally {
    await stopServer(server);
  }
});
