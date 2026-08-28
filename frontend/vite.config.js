import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In dev, /api requests are proxied straight to the backend so the app
// code never needs to know the backend's host/port. In production the
// same /api path is proxied by nginx (see nginx.conf) -- the frontend
// code is identical in both environments.
export default defineConfig({
  plugins: [react()],
  server: {
    allowedHosts: true,
    proxy: {
      "/api": {
        target: process.env.VITE_DEV_API_TARGET || "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
});
