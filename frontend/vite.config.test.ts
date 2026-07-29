import { describe, expect, it } from "vitest";

import { createViteConfig, DEFAULT_API_TARGET } from "./vite.config";

describe("Vite development integration", () => {
  it("proxies same-origin API requests to the local FastAPI server", () => {
    const config = createViteConfig();
    const proxy = config.server?.proxy as Record<string, {
      target: string;
      changeOrigin: boolean;
    }>;

    expect(DEFAULT_API_TARGET).toBe("http://127.0.0.1:7860");
    expect(proxy["/api"]).toEqual({
      target: DEFAULT_API_TARGET,
      changeOrigin: true,
    });
  });

  it("accepts an explicit API target override", () => {
    const config = createViteConfig("http://127.0.0.1:9000");
    const proxy = config.server?.proxy as Record<string, { target: string }>;

    expect(proxy["/api"]?.target).toBe("http://127.0.0.1:9000");
  });
});
