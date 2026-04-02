import { defineConfig } from "vite";

export default defineConfig({
  build: {
    lib: {
      entry: "src/index.ts",
      name: "uqa",
      formats: ["es", "umd"],
      fileName: (format) => `uqa.${format}.js`,
    },
    sourcemap: true,
    rollupOptions: {
      external: [
        "fs",
        "node:fs",
        "bayesian-bm25",
        "libpg-query",
        "sql.js",
        "apache-arrow",
        "@duckdb/duckdb-wasm",
        "xterm",
        "highlight.js",
        "comlink",
      ],
      output: {
        globals: {
          "bayesian-bm25": "BayesianBM25",
          "libpg-query": "libpgQuery",
          "sql.js": "initSqlJs",
          "apache-arrow": "Arrow",
          "@duckdb/duckdb-wasm": "DuckDB",
          xterm: "Terminal",
          "highlight.js": "hljs",
          comlink: "Comlink",
        },
      },
    },
  },
});
