import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  base: mode === "production" ? "/regulo/" : "/",
  build: {
    target: "es2020",
    sourcemap: false,
    cssMinify: true,
    minify: "esbuild",
    chunkSizeWarningLimit: 800,
  },
  server: {
    host: true,
    port: 5173,
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    globals: true,
  },
}));
