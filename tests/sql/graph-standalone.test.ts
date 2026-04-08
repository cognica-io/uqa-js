import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";

// =============================================================================
// Helpers
// =============================================================================

async function engineWithGraph(name = "social"): Promise<Engine> {
  const engine = new Engine();
  await engine.sql(`SELECT * FROM create_graph('${name}')`);
  return engine;
}

async function engineWithSocialGraph(): Promise<Engine> {
  const engine = await engineWithGraph("social");
  await engine.sql(
    "SELECT * FROM graph_create_node('social', 'Person', " +
      "'{\"name\":\"Alice\",\"age\":30}')",
  );
  await engine.sql(
    "SELECT * FROM graph_create_node('social', 'Person', " +
      "'{\"name\":\"Bob\",\"age\":25}')",
  );
  await engine.sql(
    "SELECT * FROM graph_create_node('social', 'Person', " +
      "'{\"name\":\"Carol\",\"age\":35}')",
  );
  await engine.sql(
    "SELECT * FROM graph_create_edge('social', 'KNOWS', 1, 2, '{}')",
  );
  await engine.sql(
    "SELECT * FROM graph_create_edge('social', 'KNOWS', 2, 3, '{}')",
  );
  await engine.sql(
    "SELECT * FROM graph_create_edge('social', 'FOLLOWS', 1, 3, '{}')",
  );
  return engine;
}

// =============================================================================
// graph_create / graph_drop aliases
// =============================================================================

describe("GraphCreateDropAliases", () => {
  it("graph_create alias", async () => {
    const e = new Engine();
    const r = await e.sql("SELECT * FROM graph_create('myg')");
    expect(r!.rows[0]).toHaveProperty("create_graph");
  });

  it("graph_drop alias", async () => {
    const e = new Engine();
    await e.sql("SELECT * FROM graph_create('myg')");
    const r = await e.sql("SELECT * FROM graph_drop('myg')");
    expect(r!.rows[0]).toHaveProperty("drop_graph");
  });
});

// =============================================================================
// graph_create_node
// =============================================================================

describe("GraphCreateNode", () => {
  it("basic creation", async () => {
    const e = await engineWithGraph();
    const r = await e.sql(
      "SELECT * FROM graph_create_node('social', 'Person', '{\"name\":\"Alice\"}')",
    );
    expect(r!.rows).toHaveLength(1);
    expect(r!.rows[0]!["id"]).toBe("social:Person:1");
  });

  it("auto increment id", async () => {
    const e = await engineWithGraph();
    const r1 = await e.sql(
      "SELECT * FROM graph_create_node('social', 'A', '{}')",
    );
    const r2 = await e.sql(
      "SELECT * FROM graph_create_node('social', 'B', '{}')",
    );
    expect(r1!.rows[0]!["id"]).toBe("social:A:1");
    expect(r2!.rows[0]!["id"]).toBe("social:B:2");
  });

  it("no properties", async () => {
    const e = await engineWithGraph();
    const r = await e.sql(
      "SELECT * FROM graph_create_node('social', 'Tag')",
    );
    expect(r!.rows[0]!["id"]).toBe("social:Tag:1");
  });

  it("complex properties", async () => {
    const e = await engineWithGraph();
    const props = JSON.stringify({
      name: "Alice",
      scores: [1, 2, 3],
      active: true,
    });
    await e.sql(
      `SELECT * FROM graph_create_node('social', 'User', '${props}')`,
    );
    const rows = (await e.sql("SELECT * FROM graph_nodes('social', 'User')"))!.rows;
    expect(rows).toHaveLength(1);
    const stored = JSON.parse(rows[0]!["properties"] as string);
    expect(stored.scores).toEqual([1, 2, 3]);
    expect(stored.active).toBe(true);
  });

  it("missing label raises", async () => {
    const e = await engineWithGraph();
    await expect(
      e.sql("SELECT * FROM graph_create_node('social')"),
    ).rejects.toThrow();
  });
});

// =============================================================================
// graph_create_edge
// =============================================================================

describe("GraphCreateEdge", () => {
  it("basic creation", async () => {
    const e = await engineWithGraph();
    await e.sql("SELECT * FROM graph_create_node('social', 'P', '{}')");
    await e.sql("SELECT * FROM graph_create_node('social', 'P', '{}')");
    const r = await e.sql(
      "SELECT * FROM graph_create_edge('social', 'KNOWS', 1, 2, '{}')",
    );
    expect(r!.rows).toHaveLength(1);
    expect(r!.rows[0]!["id"]).toBe("social:KNOWS:1");
  });

  it("auto increment edge id", async () => {
    const e = await engineWithGraph();
    await e.sql("SELECT * FROM graph_create_node('social', 'P', '{}')");
    await e.sql("SELECT * FROM graph_create_node('social', 'P', '{}')");
    await e.sql("SELECT * FROM graph_create_node('social', 'P', '{}')");
    const r1 = await e.sql(
      "SELECT * FROM graph_create_edge('social', 'A', 1, 2, '{}')",
    );
    const r2 = await e.sql(
      "SELECT * FROM graph_create_edge('social', 'B', 2, 3, '{}')",
    );
    expect(r1!.rows[0]!["id"]).toBe("social:A:1");
    expect(r2!.rows[0]!["id"]).toBe("social:B:2");
  });

  it("edge with properties", async () => {
    const e = await engineWithGraph();
    await e.sql("SELECT * FROM graph_create_node('social', 'P', '{}')");
    await e.sql("SELECT * FROM graph_create_node('social', 'P', '{}')");
    await e.sql(
      "SELECT * FROM graph_create_edge('social', 'KNOWS', 1, 2, " +
        "'{\"since\":2020,\"weight\":0.9}')",
    );
    const rows = (
      await e.sql(
        "SELECT * FROM graph_neighbors('social', 1, 'KNOWS', 'outgoing', 1)",
      )
    )!.rows;
    expect(rows).toHaveLength(1);
    expect(rows[0]!["id"]).toBe(2);
  });

  it("no properties", async () => {
    const e = await engineWithGraph();
    await e.sql("SELECT * FROM graph_create_node('social', 'P', '{}')");
    await e.sql("SELECT * FROM graph_create_node('social', 'P', '{}')");
    const r = await e.sql(
      "SELECT * FROM graph_create_edge('social', 'LINK', 1, 2)",
    );
    expect(r!.rows[0]!["id"]).toBe("social:LINK:1");
  });

  it("missing args raises", async () => {
    const e = await engineWithGraph();
    await expect(
      e.sql("SELECT * FROM graph_create_edge('social', 'KNOWS', 1)"),
    ).rejects.toThrow();
  });
});

// =============================================================================
// graph_nodes
// =============================================================================

describe("GraphNodes", () => {
  it("all nodes", async () => {
    const e = await engineWithSocialGraph();
    const rows = (await e.sql("SELECT * FROM graph_nodes('social')"))!.rows;
    expect(rows).toHaveLength(3);
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([1, 2, 3]));
  });

  it("filter by label", async () => {
    const e = await engineWithGraph();
    await e.sql(
      "SELECT * FROM graph_create_node('social', 'Person', '{\"name\":\"Alice\"}')",
    );
    await e.sql(
      "SELECT * FROM graph_create_node('social', 'Company', '{\"name\":\"Cognica\"}')",
    );
    const rows = (
      await e.sql("SELECT * FROM graph_nodes('social', 'Person')")
    )!.rows;
    expect(rows).toHaveLength(1);
    expect(rows[0]!["label"]).toBe("Person");
  });

  it("filter by properties", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_nodes('social', 'Person', '{\"name\":\"Bob\"}')",
      )
    )!.rows;
    expect(rows).toHaveLength(1);
    const props = JSON.parse(rows[0]!["properties"] as string);
    expect(props.name).toBe("Bob");
    expect(props.age).toBe(25);
  });

  it("filter no match", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_nodes('social', 'Person', '{\"name\":\"Nobody\"}')",
      )
    )!.rows;
    expect(rows).toHaveLength(0);
  });

  it("properties column is JSON", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql("SELECT * FROM graph_nodes('social', 'Person')")
    )!.rows;
    for (const row of rows) {
      const props = JSON.parse(row["properties"] as string);
      expect(props).toHaveProperty("name");
    }
  });

  it("empty graph", async () => {
    const e = await engineWithGraph();
    const rows = (await e.sql("SELECT * FROM graph_nodes('social')"))!.rows;
    expect(rows).toHaveLength(0);
  });
});

// =============================================================================
// graph_neighbors
// =============================================================================

describe("GraphNeighbors", () => {
  it("one hop outgoing", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_neighbors('social', 1, 'KNOWS', 'outgoing', 1)",
      )
    )!.rows;
    expect(rows).toHaveLength(1);
    expect(rows[0]!["id"]).toBe(2);
    expect(rows[0]!["depth"]).toBe(1);
  });

  it("two hop outgoing", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_neighbors('social', 1, 'KNOWS', 'outgoing', 2)",
      )
    )!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([2, 3]));
  });

  it("incoming direction", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_neighbors('social', 3, 'KNOWS', 'incoming', 1)",
      )
    )!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([2]));
  });

  it("both directions", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_neighbors('social', 2, 'KNOWS', 'both', 1)",
      )
    )!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([1, 3]));
  });

  it("no edge label filter", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_neighbors('social', 1, '', 'outgoing', 1)",
      )
    )!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([2, 3]));
  });

  it("depth path", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_neighbors('social', 1, 'KNOWS', 'outgoing', 2)",
      )
    )!.rows;
    const depthMap = new Map(rows.map((r) => [r["id"], r["depth"]]));
    expect(depthMap.get(2)).toBe(1);
    expect(depthMap.get(3)).toBe(2);

    for (const row of rows) {
      const path = JSON.parse(row["path"] as string);
      expect(path[0]).toBe(1);
      expect(path[path.length - 1]).toBe(row["id"]);
    }
  });

  it("no neighbors", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_neighbors('social', 3, 'KNOWS', 'outgoing', 1)",
      )
    )!.rows;
    expect(rows).toHaveLength(0);
  });

  it("default direction and depth", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql("SELECT * FROM graph_neighbors('social', 1, 'KNOWS')")
    )!.rows;
    expect(rows).toHaveLength(1);
    expect(rows[0]!["id"]).toBe(2);
  });

  it("properties and label returned", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_neighbors('social', 1, 'KNOWS', 'outgoing', 1)",
      )
    )!.rows;
    expect(rows[0]!["label"]).toBe("Person");
    const props = JSON.parse(rows[0]!["properties"] as string);
    expect(props.name).toBe("Bob");
  });
});

// =============================================================================
// graph_delete_node
// =============================================================================

describe("GraphDeleteNode", () => {
  it("delete node", async () => {
    const e = await engineWithSocialGraph();
    await e.sql("SELECT * FROM graph_delete_node('social', 2)");
    const rows = (await e.sql("SELECT * FROM graph_nodes('social')"))!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).not.toContain(2);
    expect(ids).toEqual(new Set([1, 3]));
  });

  it("delete removes incident edges", async () => {
    const e = await engineWithSocialGraph();
    await e.sql("SELECT * FROM graph_delete_node('social', 2)");
    const rows = (
      await e.sql(
        "SELECT * FROM graph_neighbors('social', 1, 'KNOWS', 'outgoing', 1)",
      )
    )!.rows;
    expect(rows).toHaveLength(0);
  });

  it("delete nonexistent node is no-op", async () => {
    const e = await engineWithSocialGraph();
    const r = await e.sql("SELECT * FROM graph_delete_node('social', 999)");
    expect(String(r!.rows[0]!["result"])).toContain("deleted");
    const rows = (await e.sql("SELECT * FROM graph_nodes('social')"))!.rows;
    expect(rows).toHaveLength(3);
  });

  it("delete returns confirmation", async () => {
    const e = await engineWithSocialGraph();
    const r = await e.sql("SELECT * FROM graph_delete_node('social', 1)");
    expect(String(r!.rows[0]!["result"])).toContain("deleted");
  });
});

// =============================================================================
// graph_delete_edge
// =============================================================================

describe("GraphDeleteEdge", () => {
  it("delete edge", async () => {
    const e = await engineWithSocialGraph();
    await e.sql("SELECT * FROM graph_delete_edge('social', 1)");
    const rows = (
      await e.sql(
        "SELECT * FROM graph_neighbors('social', 1, 'KNOWS', 'outgoing', 1)",
      )
    )!.rows;
    expect(rows).toHaveLength(0);
  });

  it("delete edge keeps nodes", async () => {
    const e = await engineWithSocialGraph();
    await e.sql("SELECT * FROM graph_delete_edge('social', 1)");
    const rows = (await e.sql("SELECT * FROM graph_nodes('social')"))!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([1, 2, 3]));
  });

  it("delete nonexistent edge is no-op", async () => {
    const e = await engineWithSocialGraph();
    const r = await e.sql("SELECT * FROM graph_delete_edge('social', 999)");
    expect(String(r!.rows[0]!["result"])).toContain("deleted");
    const rows = (
      await e.sql(
        "SELECT * FROM graph_neighbors('social', 1, 'KNOWS', 'outgoing', 1)",
      )
    )!.rows;
    expect(rows).toHaveLength(1);
  });

  it("delete returns confirmation", async () => {
    const e = await engineWithSocialGraph();
    const r = await e.sql("SELECT * FROM graph_delete_edge('social', 1)");
    expect(String(r!.rows[0]!["result"])).toContain("deleted");
  });
});

// =============================================================================
// graph_edges
// =============================================================================

describe("GraphEdges", () => {
  it("all edges", async () => {
    const e = await engineWithSocialGraph();
    const rows = (await e.sql("SELECT * FROM graph_edges('social')"))!.rows;
    expect(rows).toHaveLength(3);
    const labels = new Set(rows.map((r) => r["label"]));
    expect(labels).toEqual(new Set(["KNOWS", "FOLLOWS"]));
  });

  it("filter by type", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql("SELECT * FROM graph_edges('social', 'KNOWS')")
    )!.rows;
    expect(rows).toHaveLength(2);
    expect(rows.every((r) => r["label"] === "KNOWS")).toBe(true);
  });

  it("filter by properties", async () => {
    const e = await engineWithGraph();
    await e.sql("SELECT * FROM graph_create_node('social', 'P', '{}')");
    await e.sql("SELECT * FROM graph_create_node('social', 'P', '{}')");
    await e.sql(
      "SELECT * FROM graph_create_edge('social', 'KNOWS', 1, 2, " +
        "'{\"since\":2020}')",
    );
    await e.sql(
      "SELECT * FROM graph_create_edge('social', 'KNOWS', 2, 1, " +
        "'{\"since\":2021}')",
    );
    const rows = (
      await e.sql(
        "SELECT * FROM graph_edges('social', 'KNOWS', '{\"since\":2020}')",
      )
    )!.rows;
    expect(rows).toHaveLength(1);
    expect(rows[0]!["source_id"]).toBe(1);
    expect(rows[0]!["target_id"]).toBe(2);
  });

  it("columns", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql("SELECT * FROM graph_edges('social', 'FOLLOWS')")
    )!.rows;
    expect(rows).toHaveLength(1);
    const row = rows[0]!;
    expect(row["source_id"]).toBe(1);
    expect(row["target_id"]).toBe(3);
    expect(row["label"]).toBe("FOLLOWS");
    const props = JSON.parse(row["properties"] as string);
    expect(typeof props).toBe("object");
  });

  it("empty graph", async () => {
    const e = await engineWithGraph();
    const rows = (await e.sql("SELECT * FROM graph_edges('social')"))!.rows;
    expect(rows).toHaveLength(0);
  });

  it("count edges", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql("SELECT COUNT(*) AS cnt FROM graph_edges('social')")
    )!.rows;
    expect(rows[0]!["cnt"]).toBe(3);
  });
});

// =============================================================================
// graph_edges per-vertex mode
// =============================================================================

describe("GraphEdgesPerVertex", () => {
  it("per-vertex outgoing", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql("SELECT * FROM graph_edges('social', 1, NULL, 'outgoing')")
    )!.rows;
    expect(rows).toHaveLength(2);
    const labels = new Set(rows.map((r) => r["label"]));
    expect(labels).toEqual(new Set(["KNOWS", "FOLLOWS"]));
  });

  it("per-vertex incoming", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql("SELECT * FROM graph_edges('social', 3, NULL, 'incoming')")
    )!.rows;
    expect(rows).toHaveLength(2);
  });

  it("per-vertex with type", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_edges('social', 1, 'KNOWS', 'outgoing')",
      )
    )!.rows;
    expect(rows).toHaveLength(1);
    expect(rows[0]!["label"]).toBe("KNOWS");
    expect(rows[0]!["target_id"]).toBe(2);
  });

  it("per-vertex both", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql("SELECT * FROM graph_edges('social', 2, NULL, 'both')")
    )!.rows;
    expect(rows).toHaveLength(2);
  });

  it("per-vertex no edges", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql("SELECT * FROM graph_edges('social', 3, NULL, 'outgoing')")
    )!.rows;
    expect(rows).toHaveLength(0);
  });
});

// =============================================================================
// LATERAL graph_edges
// =============================================================================

describe("LateralGraphEdges", () => {
  it("lateral edge count per vertex", async () => {
    const e = await engineWithSocialGraph();
    await e.sql("CREATE TABLE nodes (node_id INT PRIMARY KEY, name TEXT)");
    await e.sql("INSERT INTO nodes VALUES (1, 'Alice')");
    await e.sql("INSERT INTO nodes VALUES (2, 'Bob')");
    await e.sql("INSERT INTO nodes VALUES (3, 'Carol')");

    const rows = (
      await e.sql(
        "SELECT n.name, sub.cnt " +
          "FROM nodes n, " +
          "LATERAL (SELECT COUNT(*) AS cnt " +
          "FROM graph_edges('social', n.node_id, NULL, 'outgoing')) sub " +
          "ORDER BY n.name",
      )
    )!.rows;
    expect(rows).toHaveLength(3);
    expect(rows[0]!["name"]).toBe("Alice");
    expect(rows[0]!["cnt"]).toBe(2);
    expect(rows[1]!["name"]).toBe("Bob");
    expect(rows[1]!["cnt"]).toBe(1);
    expect(rows[2]!["name"]).toBe("Carol");
    expect(rows[2]!["cnt"]).toBe(0);
  });

  it("lateral edge list per vertex", async () => {
    const e = await engineWithSocialGraph();
    await e.sql("CREATE TABLE nodes (node_id INT PRIMARY KEY, name TEXT)");
    await e.sql("INSERT INTO nodes VALUES (1, 'Alice')");
    await e.sql("INSERT INTO nodes VALUES (2, 'Bob')");

    const rows = (
      await e.sql(
        "SELECT n.name, e.label, e.target_id " +
          "FROM nodes n, " +
          "LATERAL (SELECT label, target_id " +
          "FROM graph_edges('social', n.node_id, NULL, 'outgoing')) e " +
          "ORDER BY n.name, e.label",
      )
    )!.rows;
    expect(rows).toHaveLength(3);
    expect(rows[0]!["name"]).toBe("Alice");
    expect(rows[0]!["label"]).toBe("FOLLOWS");
    expect(rows[0]!["target_id"]).toBe(3);
    expect(rows[1]!["name"]).toBe("Alice");
    expect(rows[1]!["label"]).toBe("KNOWS");
    expect(rows[1]!["target_id"]).toBe(2);
    expect(rows[2]!["name"]).toBe("Bob");
    expect(rows[2]!["label"]).toBe("KNOWS");
    expect(rows[2]!["target_id"]).toBe(3);
  });

  it("lateral graph_traverse per vertex", async () => {
    const e = await engineWithSocialGraph();
    await e.sql("CREATE TABLE start_nodes (sid INT PRIMARY KEY)");
    await e.sql("INSERT INTO start_nodes VALUES (1)");
    await e.sql("INSERT INTO start_nodes VALUES (2)");

    const rows = (
      await e.sql(
        "SELECT s.sid, sub.cnt " +
          "FROM start_nodes s, " +
          "LATERAL (SELECT COUNT(*) AS cnt " +
          "FROM graph_traverse('social', s.sid, '', 'outgoing', 2, 'bfs')) sub " +
          "ORDER BY s.sid",
      )
    )!.rows;
    expect(rows).toHaveLength(2);
    expect(rows[0]!["sid"]).toBe(1);
    expect(rows[0]!["cnt"]).toBe(2);
    expect(rows[1]!["sid"]).toBe(2);
    expect(rows[1]!["cnt"]).toBe(1);
  });
});

// =============================================================================
// graph_traverse
// =============================================================================

describe("GraphTraverse", () => {
  it("bfs single type", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_traverse('social', 1, 'KNOWS', 'outgoing', 2, 'bfs')",
      )
    )!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([2, 3]));
  });

  it("dfs single type", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_traverse('social', 1, 'KNOWS', 'outgoing', 2, 'dfs')",
      )
    )!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([2, 3]));
  });

  it("multiple edge types", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_traverse('social', 1, 'KNOWS,FOLLOWS', " +
          "'outgoing', 1, 'bfs')",
      )
    )!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([2, 3]));
  });

  it("empty types means all", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_traverse('social', 1, '', 'outgoing', 1)",
      )
    )!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([2, 3]));
  });

  it("incoming direction", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_traverse('social', 3, 'KNOWS,FOLLOWS', " +
          "'incoming', 1, 'bfs')",
      )
    )!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([1, 2]));
  });

  it("both directions", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_traverse('social', 2, 'KNOWS', 'both', 1)",
      )
    )!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([1, 3]));
  });

  it("depth and path", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_traverse('social', 1, 'KNOWS', 'outgoing', 2, 'bfs')",
      )
    )!.rows;
    const depthMap = new Map(rows.map((r) => [r["id"], r["depth"]]));
    expect(depthMap.get(2)).toBe(1);
    expect(depthMap.get(3)).toBe(2);
    for (const row of rows) {
      const path = JSON.parse(row["path"] as string);
      expect(path[0]).toBe(1);
      expect(path[path.length - 1]).toBe(row["id"]);
    }
  });

  it("default strategy is bfs", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_traverse('social', 1, 'KNOWS', 'outgoing', 2)",
      )
    )!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([2, 3]));
  });

  it("no results", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_traverse('social', 3, 'KNOWS', 'outgoing', 1)",
      )
    )!.rows;
    expect(rows).toHaveLength(0);
  });

  it("properties returned", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_traverse('social', 1, 'KNOWS', 'outgoing', 1)",
      )
    )!.rows;
    expect(rows).toHaveLength(1);
    expect(rows[0]!["label"]).toBe("Person");
    const props = JSON.parse(rows[0]!["properties"] as string);
    expect(props.name).toBe("Bob");
  });
});

// =============================================================================
// graph_traverse with ARRAY and NULL edge types
// =============================================================================

describe("GraphTraverseArray", () => {
  it("array literal edge types", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_traverse('social', 1, ARRAY['KNOWS','FOLLOWS'], " +
          "'outgoing', 1, 'bfs')",
      )
    )!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([2, 3]));
  });

  it("null edge types", async () => {
    const e = await engineWithSocialGraph();
    const rows = (
      await e.sql(
        "SELECT * FROM graph_traverse('social', 1, NULL, 'outgoing', 1)",
      )
    )!.rows;
    const ids = new Set(rows.map((r) => r["id"]));
    expect(ids).toEqual(new Set([2, 3]));
  });
});

// =============================================================================
// End-to-end workflow
// =============================================================================

describe("EndToEndWorkflow", () => {
  it("full lifecycle", async () => {
    const e = new Engine();

    // Create graph
    await e.sql("SELECT * FROM create_graph('code')");

    // Add nodes
    const r1 = await e.sql(
      "SELECT * FROM graph_create_node('code', 'Function', " +
        "'{\"name\":\"main\",\"lines\":42}')",
    );
    const r2 = await e.sql(
      "SELECT * FROM graph_create_node('code', 'Function', " +
        "'{\"name\":\"helper\",\"lines\":10}')",
    );
    const r3 = await e.sql(
      "SELECT * FROM graph_create_node('code', 'Module', '{\"name\":\"utils\"}')",
    );
    expect(r1!.rows[0]!["id"]).toBe("code:Function:1");
    expect(r2!.rows[0]!["id"]).toBe("code:Function:2");
    expect(r3!.rows[0]!["id"]).toBe("code:Module:3");

    // Add edges
    await e.sql(
      "SELECT * FROM graph_create_edge('code', 'CALLS', 1, 2, '{}')",
    );
    await e.sql(
      "SELECT * FROM graph_create_edge('code', 'BELONGS_TO', 1, 3, '{}')",
    );

    // Query: all Function nodes
    const funcs = (
      await e.sql("SELECT * FROM graph_nodes('code', 'Function')")
    )!.rows;
    expect(funcs).toHaveLength(2);

    // Query: neighbors of main
    let neighbors = (
      await e.sql(
        "SELECT * FROM graph_neighbors('code', 1, 'CALLS', 'outgoing', 1)",
      )
    )!.rows;
    expect(neighbors).toHaveLength(1);
    expect(JSON.parse(neighbors[0]!["properties"] as string).name).toBe(
      "helper",
    );

    // Delete edge
    await e.sql("SELECT * FROM graph_delete_edge('code', 1)");

    // Verify edge gone
    neighbors = (
      await e.sql(
        "SELECT * FROM graph_neighbors('code', 1, 'CALLS', 'outgoing', 1)",
      )
    )!.rows;
    expect(neighbors).toHaveLength(0);

    // Delete node
    await e.sql("SELECT * FROM graph_delete_node('code', 2)");
    const nodes = (await e.sql("SELECT * FROM graph_nodes('code')"))!.rows;
    expect(nodes).toHaveLength(2);

    // Drop graph
    await e.sql("SELECT * FROM drop_graph('code')");
    await expect(e.sql("SELECT * FROM graph_nodes('code')")).rejects.toThrow();
  });

  it("multi-graph isolation", async () => {
    const e = new Engine();
    await e.sql("SELECT * FROM create_graph('g1')");
    await e.sql("SELECT * FROM create_graph('g2')");

    await e.sql(
      "SELECT * FROM graph_create_node('g1', 'X', '{\"val\":1}')",
    );
    await e.sql(
      "SELECT * FROM graph_create_node('g2', 'X', '{\"val\":2}')",
    );

    const g1Nodes = (await e.sql("SELECT * FROM graph_nodes('g1')"))!.rows;
    const g2Nodes = (await e.sql("SELECT * FROM graph_nodes('g2')"))!.rows;
    expect(g1Nodes).toHaveLength(1);
    expect(g2Nodes).toHaveLength(1);
    expect(JSON.parse(g1Nodes[0]!["properties"] as string).val).toBe(1);
    expect(JSON.parse(g2Nodes[0]!["properties"] as string).val).toBe(2);
  });
});
