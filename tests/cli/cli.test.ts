import { describe, expect, it } from "vitest";
import { Engine } from "../../src/engine.js";
import { UQAShell, SQLCompleter } from "../../src/cli/repl.js";
import {
  createForeignServer,
  createForeignTable,
} from "../../src/fdw/foreign-table.js";
import { createColumnDef } from "../../src/sql/table.js";

// =============================================================================
// Helpers
// =============================================================================

function makeShell(): UQAShell {
  const engine = new Engine();
  return new UQAShell(engine);
}

async function makeShellWithTable(): Promise<UQAShell> {
  const shell = makeShell();
  await shell.engine.sql(
    "CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)",
  );
  await shell.engine.sql("INSERT INTO users (id, name, age) VALUES (1, 'Alice', 30)");
  await shell.engine.sql("INSERT INTO users (id, name, age) VALUES (2, 'Bob', 25)");
  // Sync compiler tables to engine._tables for shell commands
  shell.engine.getTable("users");
  return shell;
}

// =============================================================================
// SQL Keywords
// =============================================================================

describe("CLI Keywords", () => {
  it("DDL keywords", () => {
    const shell = makeShell();
    const completions = shell.completer.getCompletions("CRE");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("CREATE");
  });

  it("DML keywords", () => {
    const shell = makeShell();
    const completions = shell.completer.getCompletions("INS");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("INSERT");
  });

  it("join keywords", () => {
    const shell = makeShell();
    const completions = shell.completer.getCompletions("JOI");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("JOIN");
  });

  it("window keywords", () => {
    const shell = makeShell();
    const completions = shell.completer.getCompletions("OVE");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("OVER");
  });

  it("FDW keywords", () => {
    const shell = makeShell();
    const completions = shell.completer.getCompletions("FOR");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("FOREIGN");
  });

  it("cypher keywords", () => {
    const shell = makeShell();
    const completions = shell.completer.getCompletions("cyp");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("cypher");
  });

  it("CTE keywords", () => {
    const shell = makeShell();
    const completions = shell.completer.getCompletions("WIT");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("WITH");
  });

  it("aggregate keywords", () => {
    const shell = makeShell();
    const completions = shell.completer.getCompletions("COU");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("COUNT");
  });

  it("type keywords", () => {
    const shell = makeShell();
    const completions = shell.completer.getCompletions("INT");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("INTEGER");
  });

  it("misc keywords", () => {
    const shell = makeShell();
    const completions = shell.completer.getCompletions("EXP");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("EXPLAIN");
  });
});

// =============================================================================
// Completer
// =============================================================================

describe("CLI Completer", () => {
  it("completes backslash commands", () => {
    const shell = makeShell();
    const completions = shell.completer.getCompletions("\\");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("\\dt");
    expect(texts).toContain("\\q");
  });

  it("completes backslash single", () => {
    const shell = makeShell();
    const completions = shell.completer.getCompletions("\\d");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("\\dt");
    expect(texts).toContain("\\di");
    expect(texts).toContain("\\dF");
    expect(texts).toContain("\\dS");
    expect(texts).toContain("\\dg");
    expect(texts).toContain("\\ds");
    expect(texts).toContain("\\d");
  });

  it("backslash shows descriptions", () => {
    const shell = makeShell();
    const completions = shell.completer.getCompletions("\\d");
    const dtCompletion = completions.find((c) => c.text === "\\dt");
    expect(dtCompletion).toBeDefined();
    expect(dtCompletion!.meta).toBe("List tables");
  });

  it("backslash no SQL keywords", () => {
    const shell = makeShell();
    const completions = shell.completer.getCompletions("\\d");
    const sqlKeywords = completions.filter((c) => c.meta === "keyword");
    expect(sqlKeywords.length).toBe(0);
  });

  it("completes regular table", async () => {
    const shell = await makeShellWithTable();
    const completions = shell.completer.getCompletions("SELECT * FROM use");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("users");
  });

  it("completes foreign table", async () => {
    const shell = makeShell();
    const cols = new Map<string, ReturnType<typeof createColumnDef>>();
    cols.set("col1", createColumnDef("col1", "text"));
    shell.engine._foreignTables.set("ext_data", {
      name: "ext_data",
      serverName: "my_server",
      columns: cols,
      options: {},
    });
    const completions = shell.completer.getCompletions("SELECT * FROM ext");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("ext_data");
  });

  it("completes foreign table columns", async () => {
    const shell = makeShell();
    const cols = new Map<string, ReturnType<typeof createColumnDef>>();
    cols.set("remote_col", createColumnDef("remote_col", "text"));
    shell.engine._foreignTables.set("ext_data", {
      name: "ext_data",
      serverName: "my_server",
      columns: cols,
      options: {},
    });
    const completions = shell.completer.getCompletions("SELECT remote");
    const texts = completions.map((c) => c.text);
    expect(texts).toContain("remote_col");
  });
});

// =============================================================================
// List Tables
// =============================================================================

describe("CLI ListTables", () => {
  it("list tables empty", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\dt");
    expect(output).toMatch(/[Nn]o tables/);
  });

  it("list tables with regular", async () => {
    const shell = await makeShellWithTable();
    const output = await shell.executeCommand("\\dt");
    expect(output).toContain("users");
    expect(output).toContain("table");
  });

  it("list tables includes foreign", async () => {
    const shell = await makeShellWithTable();
    const cols = new Map<string, ReturnType<typeof createColumnDef>>();
    cols.set("col1", createColumnDef("col1", "text"));
    shell.engine._foreignTables.set("ext_data", {
      name: "ext_data",
      serverName: "my_server",
      columns: cols,
      options: {},
    });
    const output = await shell.executeCommand("\\dt");
    expect(output).toContain("users");
    expect(output).toContain("ext_data");
    expect(output).toContain("foreign");
  });
});

// =============================================================================
// Describe Table
// =============================================================================

describe("CLI DescribeTable", () => {
  it("describe regular table", async () => {
    const shell = await makeShellWithTable();
    const output = await shell.executeCommand("\\d users");
    expect(output).toContain("users");
    expect(output).toContain("id");
    expect(output).toContain("name");
    expect(output).toContain("age");
    expect(output).toContain("PK");
  });

  it("describe foreign table", async () => {
    const shell = makeShell();
    const cols = new Map<string, ReturnType<typeof createColumnDef>>();
    cols.set("remote_id", createColumnDef("remote_id", "integer"));
    shell.engine._foreignTables.set("ext_data", {
      name: "ext_data",
      serverName: "my_server",
      columns: cols,
      options: {},
    });
    const output = await shell.executeCommand("\\d ext_data");
    expect(output).toContain("ext_data");
    expect(output).toContain("remote_id");
    expect(output).toContain("my_server");
  });

  it("describe nonexistent", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\d nonexistent");
    expect(output).toMatch(/does not exist/);
  });

  it("describe no arg", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\d");
    expect(output).toMatch(/[Uu]sage/);
  });
});

// =============================================================================
// List Indexes
// =============================================================================

describe("CLI ListIndexes", () => {
  it("no tables", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\di");
    expect(output).toMatch(/[Nn]o tables/);
  });

  it("no indexed fields", async () => {
    const shell = await makeShellWithTable();
    const output = await shell.executeCommand("\\di");
    expect(output).toMatch(/[Nn]o indexed/);
  });

  it("shows indexed text fields", async () => {
    const shell = await makeShellWithTable();
    const table = shell.engine._tables.get("users")!;
    const standardAnalyzer = {
      analyze: (text: string) => text.toLowerCase().split(/\s+/),
    };
    table.invertedIndex.setFieldAnalyzer("name", standardAnalyzer);
    // Re-index documents so the field shows up
    for (const [docId, doc] of table.documentStore.iterAll()) {
      const textFields: Record<string, string> = {};
      for (const [k, v] of Object.entries(doc)) {
        if (typeof v === "string") textFields[k] = v;
      }
      if (Object.keys(textFields).length > 0) {
        table.invertedIndex.addDocument(docId, textFields);
      }
    }
    const output = await shell.executeCommand("\\di");
    expect(output).toContain("users");
    expect(output).toContain("name");
  });
});

// =============================================================================
// List Foreign Tables
// =============================================================================

describe("CLI ListForeignTables", () => {
  it("no foreign tables", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\dF");
    expect(output).toMatch(/[Nn]o foreign tables/);
  });

  it("lists foreign table", async () => {
    const shell = makeShell();
    const cols = new Map<string, ReturnType<typeof createColumnDef>>();
    cols.set("col1", createColumnDef("col1", "text"));
    shell.engine._foreignTables.set("ext_data", {
      name: "ext_data",
      serverName: "my_server",
      columns: cols,
      options: { source: "test.csv" },
    });
    const output = await shell.executeCommand("\\dF");
    expect(output).toContain("ext_data");
    expect(output).toContain("my_server");
  });
});

// =============================================================================
// List Foreign Servers
// =============================================================================

describe("CLI ListForeignServers", () => {
  it("no foreign servers", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\dS");
    expect(output).toMatch(/[Nn]o foreign servers/);
  });

  it("lists foreign server", async () => {
    const shell = makeShell();
    shell.engine._foreignServers.set("my_server", {
      name: "my_server",
      fdwType: "csv_fdw",
      options: { path: "/data" },
    });
    const output = await shell.executeCommand("\\dS");
    expect(output).toContain("my_server");
    expect(output).toContain("csv_fdw");
  });
});

// =============================================================================
// List Graphs
// =============================================================================

describe("CLI ListGraphs", () => {
  it("no graphs", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\dg");
    expect(output).toMatch(/[Nn]o named graphs/);
  });

  it("lists graph", async () => {
    const shell = makeShell();
    shell.engine.graphStore.createGraph("test_graph");
    const output = await shell.executeCommand("\\dg");
    expect(output).toContain("test_graph");
  });
});

// =============================================================================
// Expanded Display
// =============================================================================

describe("CLI ExpandedDisplay", () => {
  it("toggle expanded", async () => {
    const shell = makeShell();
    let output = await shell.executeCommand("\\x");
    expect(output).toContain("on");
    output = await shell.executeCommand("\\x");
    expect(output).toContain("off");
  });

  it("expanded output", async () => {
    const shell = await makeShellWithTable();
    await shell.executeCommand("\\x");
    const output = await shell.executeCommand(
      "SELECT id, name FROM users WHERE id = 1",
    );
    expect(output).toContain("RECORD");
    expect(output).toContain("Alice");
  });

  it("expanded empty", async () => {
    const shell = await makeShellWithTable();
    await shell.executeCommand("\\x");
    const output = await shell.executeCommand(
      "SELECT id, name FROM users WHERE id = 999",
    );
    expect(output).toContain("0 rows");
  });

  it("normal output unchanged", async () => {
    const shell = await makeShellWithTable();
    const output = await shell.executeCommand(
      "SELECT id, name FROM users WHERE id = 1",
    );
    expect(output).not.toContain("RECORD");
    expect(output).toContain("Alice");
  });
});

// =============================================================================
// Output Redirection
// =============================================================================

describe("CLI OutputRedirection", () => {
  it("redirect to file", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\o /tmp/uqa-test-output.txt");
    expect(output).toContain("redirected");
  });

  it("restore stdout", async () => {
    const shell = makeShell();
    await shell.executeCommand("\\o /tmp/uqa-test-output.txt");
    const output = await shell.executeCommand("\\o");
    expect(output).toContain("restored");
  });

  it("output timing to file", async () => {
    const shell = makeShell();
    await shell.executeCommand("\\timing");
    const output = await shell.executeCommand("\\o /tmp/uqa-test-timing.txt");
    expect(output).toContain("redirected");
  });

  it("expanded to file", async () => {
    const shell = makeShell();
    await shell.executeCommand("\\x");
    const output = await shell.executeCommand("\\o /tmp/uqa-test-expanded.txt");
    expect(output).toContain("redirected");
  });
});

// =============================================================================
// Backslash Dispatch
// =============================================================================

describe("CLI BackslashDispatch", () => {
  it("di dispatch", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\di");
    expect(output).toBeDefined();
  });

  it("dF dispatch", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\dF");
    expect(output).toMatch(/[Nn]o foreign tables/);
  });

  it("dS dispatch", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\dS");
    expect(output).toMatch(/[Nn]o foreign servers/);
  });

  it("dg dispatch", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\dg");
    expect(output).toMatch(/[Nn]o named graphs/);
  });

  it("x dispatch", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\x");
    expect(output).toContain("Expanded display");
  });

  it("o dispatch", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\o /tmp/uqa-test-o.txt");
    expect(output).toContain("redirected");
  });

  it("help dispatch", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\?");
    expect(output).toContain("Backslash commands");
    expect(output).toContain("\\dt");
  });

  it("unknown command", async () => {
    const shell = makeShell();
    const output = await shell.executeCommand("\\zzz");
    expect(output).toContain("Unknown command");
  });
});

// =============================================================================
// Toolbar
// =============================================================================

describe("CLI Toolbar", () => {
  it("toolbar default", () => {
    const shell = makeShell();
    const toolbar = shell.toolbar();
    expect(toolbar).toContain("tables: 0");
    expect(toolbar).toContain("timing: off");
    expect(toolbar).toContain("expanded: off");
  });

  it("toolbar with foreign", async () => {
    const shell = makeShell();
    const cols = new Map<string, ReturnType<typeof createColumnDef>>();
    cols.set("col1", createColumnDef("col1", "text"));
    shell.engine._foreignTables.set("ext_data", {
      name: "ext_data",
      serverName: "my_server",
      columns: cols,
      options: {},
    });
    const toolbar = shell.toolbar();
    expect(toolbar).toContain("foreign: 1");
  });

  it("toolbar with output", async () => {
    const shell = makeShell();
    await shell.executeCommand("\\o /tmp/uqa-test-toolbar.txt");
    const toolbar = shell.toolbar();
    expect(toolbar).toContain("output:");
  });

  it("toolbar timing on", async () => {
    const shell = makeShell();
    await shell.executeCommand("\\timing");
    const toolbar = shell.toolbar();
    expect(toolbar).toContain("timing: on");
  });

  it("toolbar expanded on", async () => {
    const shell = makeShell();
    await shell.executeCommand("\\x");
    const toolbar = shell.toolbar();
    expect(toolbar).toContain("expanded: on");
  });
});

// =============================================================================
// Banner
// =============================================================================

describe("CLI Banner", () => {
  it("banner", () => {
    const shell = makeShell();
    const banner = shell.printBanner();
    expect(banner).toContain("usql");
    expect(banner).toContain(":memory:");
  });
});
