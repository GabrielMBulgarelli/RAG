import react from "@vitejs/plugin-react";
import { loadEnv, type UserConfig } from "vite";
import { defineConfig } from "vitest/config";

export const DEFAULT_API_TARGET = "http://127.0.0.1:7860";

export function createViteConfig(apiTarget = DEFAULT_API_TARGET): UserConfig {
  return {
    plugins: [react()],
    server: {
      proxy: {
        "/api": {
          target: apiTarget,
          changeOrigin: true,
        },
      },
    },
    test: {
      environment: "jsdom",
      setupFiles: "./src/test/setup.ts",
      restoreMocks: true,
    },
  };
}

export default defineConfig(({ mode }) => {
  // VITE_API_TARGET may point development at another local FastAPI instance.
  const apiTarget = loadEnv(mode, ".", "VITE_").VITE_API_TARGET || DEFAULT_API_TARGET;
  return createViteConfig(apiTarget);
});
