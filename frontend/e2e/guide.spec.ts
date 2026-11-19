// Catalog navigation, help popovers, the guide page and the top-level CPU/GPU switch.
import { expect, test } from "@playwright/test";
import { BASE, startServer, stopServer, waitHealthy } from "./helpers";

test("catalog panes, help, guide and compute switch", async ({ page }) => {
  const server = startServer();
  await waitHealthy();
  try {
    await page.goto(`${BASE}/#/catalog`);
    // Choosing a substrate shows only its presets and selects the first one in the form.
    await page.getByTestId("substrate-gray_scott").click();
    await expect(page.getByTestId("substrate-intro")).toContainText("反応");
    await expect(page.getByTestId("preset-gray_scott_coral")).toBeVisible();
    await expect(page.getByTestId("preset-conway_glider")).toHaveCount(0);
    await expect(page.getByTestId("field-dynamics.F")).toBeVisible();

    // Help popover for a parameter.
    await page.getByTestId("help-dynamics.F").click();
    await expect(page.getByTestId("help-dynamics.F")).toContainText("補給");

    // Search spans all substrates.
    await page.getByTestId("catalog-search").fill("glider");
    await expect(page.getByTestId("preset-conway_glider")).toBeVisible();

    // CPU mode uses the GPU-free profile; the GPU switch follows the target's capabilities.
    await page.getByTestId("compute-cpu").click();
    await page.getByTestId("preset-conway_glider").click();
    await expect(page.getByTestId("input-profile")).toHaveValue("local_cpu");
    await expect(page.getByTestId("validation")).toContainText("OK");
    const gpu = page.getByTestId("compute-gpu");
    if (await gpu.isEnabled()) {
      await gpu.click();
      await expect(page.getByTestId("compute-note")).toContainText("GPU 版がない");
      await expect(page.getByTestId("input-profile")).toHaveValue("local_cpu");
      await page.getByTestId("catalog-search").fill("");
      await page.getByTestId("substrate-lenia").click();
      await expect(page.getByTestId("input-profile")).toHaveValue("local_gpu");
      await expect(page.getByTestId("validation")).toContainText("OK");
      await page.getByTestId("compute-cpu").click();
    }

    // Guide page with per-substrate anchors, and the link back to the catalog.
    await page.goto(`${BASE}/#/guide/physarum`);
    await expect(page.getByTestId("guide-page")).toBeVisible();
    await expect(page.getByTestId("guide-physarum")).toBeInViewport();
    await page.getByTestId("guide-physarum").getByRole("link").click();
    await expect(page.getByTestId("substrate-physarum")).toHaveClass(/active/);
  } finally {
    await stopServer(server);
  }
});
