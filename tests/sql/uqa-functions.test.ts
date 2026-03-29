import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

async function makeEngine(): Promise<Engine> {
  const e = new Engine();
  await e.sql("CREATE TABLE docs (id INTEGER PRIMARY KEY, title TEXT, body TEXT)");
  await e.sql("CREATE INDEX idx_docs_fts ON docs USING gin (title, body)");
  e.insert("docs", 1, { title: "machine learning basics", body: "algorithms for ML research" });
  e.insert("docs", 2, { title: "deep learning neural networks", body: "CNN and RNN models" });
  e.insert("docs", 3, { title: "cooking recipes", body: "pasta and pizza" });
  return e;
}

describe("UQA SQL functions", () => {
  it("text_match", async () => {
    const e = await makeEngine();
    const r = await e.sql("SELECT id, _score FROM docs WHERE text_match(title, 'learning') ORDER BY _score DESC");
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
  });

  it("multi_field_match", async () => {
    const e = await makeEngine();
    const r = await e.sql("SELECT id, _score FROM docs WHERE multi_field_match(title, body, 'learning') ORDER BY _score DESC");
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
  });

  it("fuse_log_odds", async () => {
    const e = await makeEngine();
    const r = await e.sql("SELECT id, _score FROM docs WHERE fuse_log_odds(text_match(title, 'learning'), text_match(body, 'algorithms')) ORDER BY _score DESC");
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
  });

  it("fuse_prob_and", async () => {
    const e = await makeEngine();
    const r = await e.sql("SELECT id, _score FROM docs WHERE fuse_prob_and(text_match(title, 'learning'), text_match(body, 'algorithms')) ORDER BY _score DESC");
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
  });

  it("fuse_prob_or", async () => {
    const e = await makeEngine();
    const r = await e.sql("SELECT id, _score FROM docs WHERE fuse_prob_or(text_match(title, 'learning'), text_match(body, 'pasta')) ORDER BY _score DESC");
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
  });

  it("fuse_prob_not", async () => {
    const e = await makeEngine();
    const r = await e.sql("SELECT id, _score FROM docs WHERE fuse_prob_not(text_match(title, 'cooking')) ORDER BY _score DESC");
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
  });

  it("sparse_threshold", async () => {
    const e = await makeEngine();
    const r = await e.sql("SELECT id, _score FROM docs WHERE sparse_threshold(text_match(title, 'learning'), 0.1) ORDER BY _score DESC");
    expect(r).not.toBeNull();
  });

  it("create_analyzer and list_analyzers", async () => {
    const e = new Engine();
    const r1 = await e.sql("SELECT * FROM create_analyzer('my_ws', '{\"tokenizer\":{\"type\":\"whitespace\"},\"token_filters\":[{\"type\":\"lowercase\"}],\"char_filters\":[]}')");
    expect(r1).not.toBeNull();
    const r2 = await e.sql("SELECT * FROM list_analyzers()");
    expect(r2).not.toBeNull();
    expect(r2!.rows.some((r: Record<string, unknown>) => r["analyzer_name"] === "my_ws")).toBe(true);
  });

  it("create_graph and drop_graph", async () => {
    const e = new Engine();
    const r1 = await e.sql("SELECT * FROM create_graph('test_g')");
    expect(r1).not.toBeNull();
    expect(e.graphStore.hasGraph("test_g")).toBe(true);
    await e.sql("SELECT * FROM drop_graph('test_g')");
    expect(e.graphStore.hasGraph("test_g")).toBe(false);
  });
});

describe("set_table_analyzer via SQL", () => {
  it("applies CJK analyzer and enables prefix search", async () => {
    const e = new Engine();
    await e.sql("CREATE TABLE pages (id INTEGER PRIMARY KEY, title TEXT)");
    await e.sql("CREATE INDEX idx_pages_fts ON pages USING gin (title)");
    await e.sql("SELECT * FROM set_table_analyzer('pages', 'title', 'standard_cjk', 'both')");
    e.insert("pages", 1, { title: "bayesian inference" });
    const r = await e.sql("SELECT id, _score FROM pages WHERE text_match(title, 'bayesi')");
    expect(r).not.toBeNull();
    expect(r!.rows.length).toBeGreaterThan(0);
  });
});
