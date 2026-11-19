// Phase 4: abstract reaction network (Gillespie direct method) in the GUI.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy, openPreset } from "./helpers";

test("reaction network: species chart, cycle candidate, knock-out and injection", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "reaction_network_cycle");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "running");
    await expect.poll(() => tick(page), { timeout: 60_000 }).toBeGreaterThan(300);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");
    await expect(page.getByTestId("species-chart")).toContainText("I4");

    await page.getByTestId("brush-inspect").check();
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.9, box.y + box.height * (1.5 / 5)); // row 1 = I1
    await expect(page.getByTestId("inspector-cycles")).toContainText("I1 → I2 → I3 → I4", { timeout: 10_000 });
    await page.getByTestId("toggle-rep1").click();
    await expect(page.getByTestId("intervention-list")).toContainText("disable_reaction", { timeout: 10_000 });

    await page.getByTestId("brush-add").check();
    await page.getByTestId("tool-count").fill("500");
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * (0.5 / 5)); // row 0 = F
    await expect(page.getByTestId("intervention-list")).toContainText("add", { timeout: 10_000 });
    await page.getByTestId("btn-step").click();
    await page.screenshot({ path: path.join(DATA, "120_reaction_network.png"), fullPage: true });
  } finally {
    await stopServer(server);
  }
});
