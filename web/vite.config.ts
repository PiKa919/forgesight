import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The API runs on a different port in development. Proxying keeps the browser
// on one origin, so the frontend never has to know a base URL or deal with CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/v1": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/healthz": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  build: { outDir: "dist", sourcemap: true },
});
