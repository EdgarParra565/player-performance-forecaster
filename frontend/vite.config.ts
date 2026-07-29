/// <reference types="vitest/config" />
import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The FastAPI service runs on :8000 (uvicorn api.main:app). All `/api` calls
// from the frontend are proxied there so the browser only ever talks to Vite.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
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
