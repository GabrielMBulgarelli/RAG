import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 30_000,
  expect: { timeout: 5_000 },
  outputDir: "/tmp/rag-playwright-results",
  reporter: "line",
  use: {
    baseURL: "http://127.0.0.1:7860",
    browserName: "chromium",
    screenshot: "only-on-failure",
    trace: "retain-on-failure",
  },
  webServer: {
    command: "uv run python -m modules.run",
    cwd: "..",
    env: {
      RAG_SERVER_HOST: "127.0.0.1",
      RAG_SERVER_PORT: "7860",
    },
    url: "http://127.0.0.1:7860",
    reuseExistingServer: false,
    timeout: 30_000,
  },
});
