import { defineConfig } from "@playwright/test";

// E2E against a running `uv run alife serve` (see scripts/e2e.sh).
export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  workers: 1,
  use: {
    baseURL: process.env.ALIFE_E2E_URL ?? "http://127.0.0.1:8765",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
    // Optional: reuse an already-installed Chromium instead of `npx playwright install`.
    launchOptions: process.env.ALIFE_E2E_CHROMIUM ? { executablePath: process.env.ALIFE_E2E_CHROMIUM } : {},
  },
  outputDir: "../.alife/e2e-results",
});
