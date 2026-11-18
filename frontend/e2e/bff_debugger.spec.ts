// Phase 3: BFF VM debugger — stand-alone trace and next-epoch pair trace from a run.
import { expect, test } from "@playwright/test";
import path from "node:path";
import { BASE, DATA, expectState, startServer, stopServer, tick, waitHealthy } from "./helpers";

test("bff debugger: step through a copier and a next-epoch pair of a run", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/debugger`);
    await page.getByTestId("dbg-run").click();
    await expect(page.getByTestId("bff-debugger")).toBeVisible();
    await expect(page.getByTestId("dbg-step")).toHaveText("0/146");
    await page.getByTestId("dbg-next").click();
    await page.getByTestId("dbg-next").click();
    await expect(page.getByTestId("dbg-step")).toHaveText("2/146");
    // Step 2 executed '.' at pc 3: tape[head1 = 64] <- tape[head0 = 0] (0x80).
    await expect(page.getByTestId("debugger-current")).toContainText("[64] 0→128");
    await page.getByTestId("dbg-next-write").click();
    await expect(page.getByTestId("dbg-step")).toHaveText("6/146");
    await expect(page.getByTestId("debugger-current")).toContainText("[65] 0→192");
    await page.screenshot({ path: path.join(DATA, "90_bff_debugger.png"), fullPage: true });
    await page.getByTestId("dbg-last").click();
    await expect(page.getByTestId("debugger-halt")).toContainText(/末尾|end of the tape/);
    await expect(page.getByTestId("debugger-b-equals-a")).toHaveText("true");

    // From a run: inspect program #0 and trace the pair it will be evaluated in next epoch.
    await page.goto(`${BASE}/#/catalog`);
    await page.getByTestId("preset-bff_reference").click();
    await page.getByTestId("field-soup.num_programs").fill("1024");
    await page.getByTestId("input-profile").selectOption("cpu_dev");
    await page.getByTestId("input-start-paused").check();
    await page.getByTestId("btn-create").click();
    await expectState(page, "paused");
    await page.getByTestId("btn-step").click();
    await expect.poll(() => tick(page)).toBe(1);
    await page.getByTestId("brush-inspect").check();
    const box = (await page.getByTestId("world-canvas").boundingBox())!;
    const tileSize = box.width / 32;
    await page.mouse.click(box.x + tileSize * 0.5, box.y + tileSize * 0.5);
    await expect(page.getByTestId("inspector-program")).toContainText("#0");
    await page.getByTestId("btn-trace-next").click();
    await expect(page.getByTestId("debugger-context")).toContainText("#0");
    await page.getByTestId("dbg-last").click();
    await expect(page.getByTestId("debugger-summary")).toBeVisible();
    await page.screenshot({ path: path.join(DATA, "91_bff_debugger_run.png"), fullPage: true });
  } finally {
    await stopServer(server);
  }
});
