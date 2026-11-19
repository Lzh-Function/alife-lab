// Phase 4: AgentWorld — evolving agents and the live individual inspector.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy, openPreset } from "./helpers";

test("agent world: population runs, agent inspector shows sensors and action history", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    await openPreset(page, "agent_world_lifetime");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-tps").fill("0");
    await expect(page.getByTestId("validation")).toContainText("OK");
    await page.getByTestId("btn-create").click();
    await expectState(page, "running");
    await expect.poll(() => tick(page), { timeout: 60_000 }).toBeGreaterThan(300);
    await page.getByTestId("btn-pause").click();
    await expectState(page, "paused");
    expect(Number(await page.getByTestId("metric-population").textContent())).toBeGreaterThan(0);

    await page.getByTestId("brush-inspect").check();
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    await page.mouse.click(box.x + box.width * 0.5, box.y + box.height * 0.5);
    await expect(page.getByTestId("inspector-agent-probs")).toBeVisible({ timeout: 10_000 });
    await expect(page.getByTestId("inspector-agent-actions")).toContainText(/[lfre]{5,}/);

    await page.getByTestId("brush-add_food").check();
    await page.mouse.click(box.x + box.width * 0.3, box.y + box.height * 0.3);
    await expect(page.getByTestId("intervention-list")).toContainText("add_food", { timeout: 10_000 });
    await page.screenshot({ path: path.join(DATA, "160_agent_world.png") });
  } finally {
    await stopServer(server);
  }
});
