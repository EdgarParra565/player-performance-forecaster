/// <reference types="vitest/config" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The FastAPI service runs on :8000 (uvicorn api.main:app). All `/api` calls
// from the frontend are proxied there so the browser only ever talks to Vite.
// Override the target with API_PROXY_TARGET (e.g. http://localhost:8010).
const apiTarget = process.env.API_PROXY_TARGET ?? "http://localhost:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
      },
    },
  },
  build: {
    // Never inline fonts as data: URIs — the production CSP is
    // `font-src 'self'`, so every font must be a same-origin file.
    assetsInlineLimit: (filePath: string) =>
      /\.(woff2?|ttf|otf)$/.test(filePath) ? false : undefined,
    rollupOptions: {
      output: {
        // Keep the heavy charting payload in its own vendor chunk so it only
        // loads behind the lazy chart boundary, not on the initial route.
        manualChunks: {
          echarts: ["echarts", "echarts-for-react"],
        },
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
