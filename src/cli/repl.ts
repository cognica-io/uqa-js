//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- REPL shell
// 1:1 port of uqa/cli.py
//
// Supports SQL execution and backslash meta-commands:
//   \dt             List tables
//   \d  <table>     Describe table schema
//   \di             List inverted-index fields
//   \dF             List foreign tables
//   \dS             List foreign servers
//   \dg             List named graphs
//   \ds <table>     Show column statistics
//   \x              Toggle expanded display
//   \o  [file]      Redirect output to file
//   \timing         Toggle query timing
//   \reset          Reset engine
//   \q              Quit

import type { Engine } from "../engine.js";
import type { SQLResult } from "../sql/compiler.js";
import { appendFileSync } from "fs";

// -- SQL + UQA keyword set for the auto-completer --

const SQL_KEYWORDS: readonly string[] = [
  // DDL
  "CREATE",
  "TABLE",
  "DROP",
  "IF",
  "EXISTS",
  "PRIMARY",
  "KEY",
  "NOT",
  "NULL",
  "DEFAULT",
  "SERIAL",
  "BIGSERIAL",
  "ALTER",
  "ADD",
  "COLUMN",
  "RENAME",
  "TO",
  "SET",
  "TRUNCATE",
  "UNIQUE",
  "CHECK",
  "CONSTRAINT",
  // Types
  "INTEGER",
  "INT",
  "BIGINT",
  "SMALLINT",
  "TEXT",
  "VARCHAR",
  "REAL",
  "FLOAT",
  "DOUBLE",
  "PRECISION",
  "NUMERIC",
  "DECIMAL",
  "BOOLEAN",
  "BOOL",
  "CHAR",
  "CHARACTER",
  "JSON",
  "JSONB",
  // DML
  "INSERT",
  "INTO",
  "VALUES",
  "UPDATE",
  "DELETE",
  "RETURNING",
  "ON",
  "CONFLICT",
  "DO",
  "NOTHING",
  "EXCLUDED",
  // DQL
  "SELECT",
  "FROM",
  "WHERE",
  "AND",
  "OR",
  "IN",
  "BETWEEN",
  "ORDER",
  "BY",
  "ASC",
  "DESC",
  "LIMIT",
  "OFFSET",
  "AS",
  "DISTINCT",
  "GROUP",
  "HAVING",
  "LIKE",
  "ILIKE",
  "IS",
  "CASE",
  "WHEN",
  "THEN",
  "ELSE",
  "END",
  "CAST",
  "COALESCE",
  "NULLIF",
  "UNION",
  "ALL",
  "EXCEPT",
  "INTERSECT",
  // Joins
  "JOIN",
  "INNER",
  "LEFT",
  "RIGHT",
  "FULL",
  "CROSS",
  "OUTER",
  // Subqueries / CTE
  "WITH",
  "RECURSIVE",
  // Aggregates
  "COUNT",
  "SUM",
  "AVG",
  "MIN",
  "MAX",
  "ARRAY_AGG",
  "BOOL_AND",
  "BOOL_OR",
  "FILTER",
  // Window functions
  "OVER",
  "PARTITION",
  "WINDOW",
  "ROWS",
  "RANGE",
  "UNBOUNDED",
  "PRECEDING",
  "FOLLOWING",
  "CURRENT",
  "ROW",
  "ROW_NUMBER",
  "RANK",
  "DENSE_RANK",
  "NTILE",
  "LAG",
  "LEAD",
  "FIRST_VALUE",
  "LAST_VALUE",
  "NTH_VALUE",
  "PERCENT_RANK",
  "CUME_DIST",
  // FDW
  "SERVER",
  "FOREIGN",
  "DATA",
  "WRAPPER",
  "OPTIONS",
  "IMPORT",
  // Utility
  "EXPLAIN",
  "ANALYZE",
  "GENERATE_SERIES",
  // UQA extensions
  "text_match",
  "bayesian_match",
  "knn_match",
  "traverse",
  "rpq",
  "text_search",
  "traverse_match",
  "fuse_log_odds",
  "fuse_prob_and",
  "fuse_prob_or",
  "fuse_prob_not",
  // Cypher integration
  "cypher",
  "create_graph",
  "drop_graph",
];

const BACKSLASH_COMMANDS: readonly [string, string][] = [
  ["\\dt", "List tables"],
  ["\\d", "Describe table"],
  ["\\di", "List indexes"],
  ["\\dF", "List foreign tables"],
  ["\\dS", "List foreign servers"],
  ["\\dg", "List graphs"],
  ["\\ds", "Show statistics"],
  ["\\x", "Expanded display"],
  ["\\o", "Output to file"],
  ["\\timing", "Toggle timing"],
  ["\\reset", "Reset engine"],
  ["\\q", "Quit"],
  ["\\?", "Help"],
];

// -- SQLCompleter -------------------------------------------------------------

export class SQLCompleter {
  private _engine: Engine;
  private _keywordUpper: Set<string>;

  constructor(engine: Engine) {
    this._engine = engine;
    this._keywordUpper = new Set(SQL_KEYWORDS.map((kw) => kw.toUpperCase()));
  }

  getCompletions(textBeforeCursor: string): { text: string; meta: string }[] {
    const text = textBeforeCursor;

    // Backslash command completion
    if (text.trimStart().startsWith("\\")) {
      const prefix = text.trimStart();
      const results: { text: string; meta: string }[] = [];
      for (const [cmd, desc] of BACKSLASH_COMMANDS) {
        if (cmd.startsWith(prefix)) {
          results.push({ text: cmd, meta: desc });
        }
      }
      return results;
    }

    // Extract current word
    const match = text.match(/(\w+)$/);
    if (!match) return [];
    const word = match[1]!;
    const upper = word.toUpperCase();

    // Check if preceding keyword suggests table context
    const beforeWord = text.slice(0, -word.length).toUpperCase();
    const afterTableKw = ["FROM", "INTO", "TABLE", "ANALYZE", "JOIN"].some((kw) =>
      beforeWord.trimEnd().endsWith(kw),
    );

    const candidates: { text: string; kind: string }[] = [];

    // SQL keywords
    for (const kw of SQL_KEYWORDS) {
      if (kw.toUpperCase().startsWith(upper)) {
        candidates.push({ text: kw, kind: "keyword" });
      }
    }

    // Table names
    for (const name of this._engine._tables.keys()) {
      if (name.toUpperCase().startsWith(upper)) {
        candidates.push({ text: name, kind: "table" });
      }
    }
    for (const name of this._engine._foreignTables.keys()) {
      if (name.toUpperCase().startsWith(upper)) {
        candidates.push({ text: name, kind: "foreign table" });
      }
    }

    // Column names (from all known tables)
    if (!afterTableKw) {
      const seen = new Set<string>();
      for (const table of this._engine._tables.values()) {
        for (const colName of table.columns.keys()) {
          if (!seen.has(colName) && colName.toUpperCase().startsWith(upper)) {
            seen.add(colName);
            candidates.push({ text: colName, kind: "column" });
          }
        }
      }
      for (const ftable of this._engine._foreignTables.values()) {
        for (const colName of ftable.columns.keys()) {
          if (!seen.has(colName) && colName.toUpperCase().startsWith(upper)) {
            seen.add(colName);
            candidates.push({ text: colName, kind: "column" });
          }
        }
      }
    }

    // Sort: tables first when after FROM/INTO, otherwise keywords first
    const orderMap = afterTableKw
      ? { table: 0, "foreign table": 1, keyword: 2, column: 3 }
      : { keyword: 0, column: 1, table: 2, "foreign table": 3 };

    candidates.sort((a, b) => {
      const oa = (orderMap as Record<string, number>)[a.kind] ?? 9;
      const ob = (orderMap as Record<string, number>)[b.kind] ?? 9;
      if (oa !== ob) return oa - ob;
      return a.text.toLowerCase().localeCompare(b.text.toLowerCase());
    });

    return candidates.map((c) => ({ text: c.text, meta: c.kind }));
  }
}

// -- UQAShell -----------------------------------------------------------------

export class UQAShell {
  private _dbPath: string | null;
  private _engine: Engine;
  private _showTiming: boolean;
  private _expanded: boolean;
  private _outputFile: string | null;
  private _completer: SQLCompleter;

  constructor(engine: Engine, dbPath?: string | null) {
    this._dbPath = dbPath ?? null;
    this._engine = engine;
    this._showTiming = false;
    this._expanded = false;
    this._outputFile = null;
    this._completer = new SQLCompleter(engine);
  }

  get engine(): Engine {
    return this._engine;
  }

  get completer(): SQLCompleter {
    return this._completer;
  }

  // -- Public API -------------------------------------------------------------

  async runFile(path: string): Promise<void> {
    const fs = await import("fs");
    const text = fs.readFileSync(path, "utf-8");
    await this._executeText(text);
  }

  // -- Statement execution ----------------------------------------------------

  async executeCommand(input: string): Promise<string> {
    const trimmed = input.trim();
    if (trimmed.length === 0) return "";

    // Handle backslash commands
    if (trimmed.startsWith("\\")) {
      return this._handleBackslash(trimmed);
    }

    // Execute SQL
    const t0 = performance.now();
    try {
      const result = await this._engine.sql(trimmed);
      const elapsed = performance.now() - t0;
      let output = "";
      if (result !== null) {
        output = this._formatResult(result);
      } else {
        output = "OK";
      }
      if (this._showTiming) {
        output += `\nTime: ${elapsed.toFixed(3)} ms`;
      }
      return output;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return `ERROR: ${message}`;
    }
  }

  async _executeText(text: string): Promise<void> {
    for (const raw of text.split(";")) {
      const stmt = raw.trim();
      if (!stmt) continue;
      // Skip pure comments
      if (stmt.split("\n").every((ln) => ln.trim().startsWith("--") || !ln.trim())) {
        continue;
      }
      const output = await this.executeCommand(stmt);
      if (output) {
        this._output(output);
      }
    }
  }

  // -- Output -----------------------------------------------------------------

  private _output(text: string): void {
    if (this._outputFile !== null) {
      appendFileSync(this._outputFile, text + "\n");
    } else {
      console.log(text);
    }
  }

  // -- Result formatting ------------------------------------------------------

  private _formatResult(result: SQLResult): string {
    if (result.columns.length === 0 && result.rows.length === 0) {
      return "";
    }
    if (this._expanded) {
      return this._formatExpanded(result);
    }
    return formatSQLResult(result);
  }

  private _formatExpanded(result: SQLResult): string {
    const { rows, columns } = result;
    if (rows.length === 0) return "(0 rows)";

    const colWidth = Math.max(...columns.map((c) => c.length));
    const parts: string[] = [];
    for (let i = 0; i < rows.length; i++) {
      parts.push(
        `-[ RECORD ${String(i + 1)} ]${"-".repeat(Math.max(0, colWidth - 7))}`,
      );
      for (const col of columns) {
        const val = rows[i]![col] ?? "";
        parts.push(`${col.padEnd(colWidth)} | ${String(val as string | number)}`);
      }
    }
    parts.push(`(${String(rows.length)} rows)`);
    return parts.join("\n");
  }

  // -- Backslash commands -----------------------------------------------------

  private _handleBackslash(cmd: string): string {
    const parts = cmd.split(/\s+/);
    const verb = parts[0]!;
    const arg = parts.slice(1).join(" ").trim();

    if (verb === "\\q" || verb === "\\quit") {
      return "Goodbye.";
    }

    switch (verb) {
      case "\\dt":
        return this._cmdListTables();
      case "\\d":
        return this._cmdDescribeTable(arg);
      case "\\di":
        return this._cmdListIndexes();
      case "\\dF":
        return this._cmdListForeignTables();
      case "\\dS":
        return this._cmdListForeignServers();
      case "\\dg":
        return this._cmdListGraphs();
      case "\\ds":
        return this._cmdShowStats(arg);
      case "\\x":
        this._expanded = !this._expanded;
        return `Expanded display is ${this._expanded ? "on" : "off"}.`;
      case "\\o":
        return this._cmdOutput(arg);
      case "\\timing":
        this._showTiming = !this._showTiming;
        return `Timing is ${this._showTiming ? "on" : "off"}.`;
      case "\\reset":
        this._engine.close();
        // Engine must be re-created by caller
        return "Engine reset.";
      case "\\h":
      case "\\help":
      case "\\?":
        return this._cmdHelp();
      default:
        return `Unknown command: ${verb}\n${this._cmdHelp()}`;
    }
  }

  private _cmdListTables(): string {
    const tables = this._engine._tables;
    const ftables = this._engine._foreignTables;
    if (tables.size === 0 && ftables.size === 0) {
      return "No tables.";
    }
    const rows: Record<string, unknown>[] = [];
    for (const [name, table] of [...tables.entries()].sort((a, b) =>
      a[0].localeCompare(b[0]),
    )) {
      rows.push({
        table_name: name,
        type: "table",
        columns: table.columns.size,
        rows: table.rowCount,
      });
    }
    for (const [name, ft] of [...ftables.entries()].sort((a, b) =>
      a[0].localeCompare(b[0]),
    )) {
      rows.push({
        table_name: name,
        type: "foreign",
        columns: ft.columns.size,
        rows: "",
      });
    }
    return formatSQLResult({
      columns: ["table_name", "type", "columns", "rows"],
      rows,
    });
  }

  private _cmdDescribeTable(name: string): string {
    if (!name) return "Usage: \\d <table_name>";

    const table = this._engine._tables.get(name);
    const ftable =
      table === undefined ? this._engine._foreignTables.get(name) : undefined;

    if (table === undefined && ftable === undefined) {
      return `Table '${name}' does not exist.`;
    }

    const rows: Record<string, unknown>[] = [];
    let header: string;

    if (table !== undefined) {
      for (const col of table.columns.values()) {
        const flags: string[] = [];
        if (col.primaryKey) flags.push("PK");
        if (col.notNull) flags.push("NOT NULL");
        if (col.autoIncrement) flags.push("AUTO");
        if (col.defaultValue !== null && col.defaultValue !== undefined) {
          flags.push(`DEFAULT ${String(col.defaultValue as string | number)}`);
        }
        rows.push({
          column: col.name,
          type: col.typeName,
          constraints: flags.length > 0 ? flags.join(" ") : "",
        });
      }
      header = `Table "${name}"`;
    } else {
      for (const col of ftable!.columns.values()) {
        rows.push({
          column: col.name,
          type: col.typeName,
          constraints: "",
        });
      }
      header = `Foreign table "${name}" (server: ${ftable!.serverName})`;
    }
    return `${header}\n${formatSQLResult({ columns: ["column", "type", "constraints"], rows })}`;
  }

  private _cmdShowStats(name: string): string {
    if (!name) return "Usage: \\ds <table_name>";
    const table = this._engine._tables.get(name);
    if (table === undefined) {
      return `Table '${name}' does not exist.`;
    }
    const stats = table.getColumnStats(table.columnNames[0] ?? "");
    if (stats === null) {
      return `No statistics for '${name}'. Run ANALYZE ${name} first.`;
    }
    const rows: Record<string, unknown>[] = [];
    for (const colName of table.columnNames) {
      const cs = table.getColumnStats(colName);
      if (cs === null) continue;
      rows.push({
        column: colName,
        distinct: cs.distinctCount,
        nulls: cs.nullCount,
        min: cs.minValue ?? "",
        max: cs.maxValue ?? "",
        selectivity:
          cs.distinctCount > 0 ? (1.0 / cs.distinctCount).toFixed(6) : "1.000000",
      });
    }
    const header = `Statistics for "${name}" (${String(table.rowCount)} rows)`;
    return `${header}\n${formatSQLResult({ columns: ["column", "distinct", "nulls", "min", "max", "selectivity"], rows })}`;
  }

  private _cmdListIndexes(): string {
    const tables = this._engine._tables;
    if (tables.size === 0) return "No tables.";

    const rows: Record<string, unknown>[] = [];
    for (const [name, table] of [...tables.entries()].sort((a, b) =>
      a[0].localeCompare(b[0]),
    )) {
      const idx = table.invertedIndex;
      const fieldAnalyzers = idx.fieldAnalyzers;
      const fields = Object.keys(fieldAnalyzers).sort();
      if (fields.length > 0) {
        rows.push({
          table_name: name,
          indexed_fields: fields.join(", "),
        });
      }
    }
    if (rows.length === 0) return "No indexed fields.";
    return formatSQLResult({ columns: ["table_name", "indexed_fields"], rows });
  }

  private _cmdListForeignTables(): string {
    const ftables = this._engine._foreignTables;
    if (ftables.size === 0) return "No foreign tables.";

    const rows: Record<string, unknown>[] = [];
    for (const [name, ft] of [...ftables.entries()].sort((a, b) =>
      a[0].localeCompare(b[0]),
    )) {
      const opts: string[] = [];
      if ((ft.options["hive_partitioning"] ?? "").toLowerCase() === "true") {
        opts.push("hive");
      }
      const source = ft.options["source"] ?? "";
      rows.push({
        table_name: name,
        server: ft.serverName,
        columns: ft.columns.size,
        source,
        options: opts.length > 0 ? opts.join(", ") : "",
      });
    }
    return formatSQLResult({
      columns: ["table_name", "server", "columns", "source", "options"],
      rows,
    });
  }

  private _cmdListForeignServers(): string {
    const servers = this._engine._foreignServers;
    if (servers.size === 0) return "No foreign servers.";

    const rows: Record<string, unknown>[] = [];
    for (const [name, srv] of [...servers.entries()].sort((a, b) =>
      a[0].localeCompare(b[0]),
    )) {
      const opts = Object.entries(srv.options)
        .map(([k, v]) => `${k}=${v}`)
        .join(" ");
      rows.push({
        server_name: name,
        fdw_type: srv.fdwType,
        options: opts,
      });
    }
    return formatSQLResult({ columns: ["server_name", "fdw_type", "options"], rows });
  }

  private _cmdListGraphs(): string {
    const gs = this._engine.graphStore;
    const names = gs.graphNames();
    if (names.length === 0) return "No named graphs.";

    const rows: Record<string, unknown>[] = [];
    for (const name of names) {
      rows.push({
        graph_name: name,
        vertices: gs.vertexIdsInGraph(name).size,
        edges: gs.edgesInGraph(name).length,
      });
    }
    return formatSQLResult({ columns: ["graph_name", "vertices", "edges"], rows });
  }

  private _cmdOutput(arg: string): string {
    if (arg) {
      this._outputFile = arg;
      return `Output redirected to: ${arg}`;
    }
    if (this._outputFile !== null) {
      const old = this._outputFile;
      this._outputFile = null;
      return `Output restored to stdout (was: ${old}).`;
    }
    this._outputFile = null;
    return "";
  }

  private _cmdHelp(): string {
    return [
      "Backslash commands:",
      "  \\dt             List tables",
      "  \\d  <table>     Describe table schema",
      "  \\di             List inverted-index fields",
      "  \\dF             List foreign tables",
      "  \\dS             List foreign servers",
      "  \\dg             List named graphs",
      "  \\ds <table>     Show column statistics",
      "  \\x              Toggle expanded display",
      "  \\o  [file]      Redirect output to file",
      "  \\timing         Toggle query timing",
      "  \\reset          Reset engine",
      "  \\q              Quit",
    ].join("\n");
  }

  // -- Toolbar ----------------------------------------------------------------

  toolbar(): string {
    const nt = this._engine._tables.size;
    const nf = this._engine._foreignTables.size;
    const timing = this._showTiming ? "on" : "off";
    const expanded = this._expanded ? "on" : "off";
    const db = this._dbPath ?? ":memory:";
    const parts = [`db: ${db}`, `tables: ${String(nt)}`];
    if (nf > 0) parts.push(`foreign: ${String(nf)}`);
    parts.push(`timing: ${timing}`, `expanded: ${expanded}`);
    if (this._outputFile) parts.push(`output: ${this._outputFile}`);
    parts.push("\\? for help");
    return ` usql | ${parts.join(" | ")} `;
  }

  printBanner(): string {
    const db = this._dbPath ?? ":memory:";
    return [
      `usql -- UQA interactive SQL shell`,
      `Database: ${db}`,
      "Type SQL statements terminated by ';'",
      "Use \\? for help, \\q to quit.",
      "",
    ].join("\n");
  }
}

// -- SQL result formatting ----------------------------------------------------

export function formatSQLResult(result: SQLResult): string {
  const { columns, rows } = result;
  if (columns.length === 0) {
    return "(empty result set)";
  }

  // Compute column widths
  const widths = columns.map((c) => c.length);
  const stringRows: string[][] = [];

  for (const row of rows) {
    const cells: string[] = [];
    for (let i = 0; i < columns.length; i++) {
      const col = columns[i]!;
      const val = row[col];
      const str = formatValue(val);
      cells.push(str);
      if (str.length > widths[i]!) {
        widths[i] = str.length;
      }
    }
    stringRows.push(cells);
  }

  // Build header
  const header = columns.map((c, i) => c.padEnd(widths[i]!)).join(" | ");
  const separator = widths.map((w) => "-".repeat(w)).join("-+-");

  // Build rows
  const bodyLines = stringRows.map((cells) =>
    cells.map((c, i) => c.padEnd(widths[i]!)).join(" | "),
  );

  const lines = [header, separator, ...bodyLines];
  lines.push(`(${String(rows.length)} row${rows.length !== 1 ? "s" : ""})`);

  return lines.join("\n");
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return "NULL";
  if (typeof value === "string") return value;
  if (typeof value === "number") return value.toString(10);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value instanceof Float64Array) {
    return `[${Array.from(value).join(", ")}]`;
  }
  if (Array.isArray(value)) {
    return JSON.stringify(value);
  }
  return JSON.stringify(value);
}
