//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

declare module "sql.js" {
  interface Database {
    run(sql: string, params?: unknown[]): void;
    exec(sql: string, params?: unknown[]): { columns: string[]; values: unknown[][] }[];
    getAsObject(sql: string, params?: unknown[]): Record<string, unknown>;
    prepare(sql: string): Statement;
    export(): Uint8Array;
    close(): void;
    getRowsModified(): number;
  }

  interface Statement {
    bind(params?: unknown[]): boolean;
    step(): boolean;
    getAsObject(params?: Record<string, unknown>): Record<string, unknown>;
    get(params?: unknown[]): unknown[];
    run(params?: unknown[]): void;
    reset(): void;
    free(): boolean;
    getColumnNames(): string[];
  }

  interface SqlJsStatic {
    Database: {
      new (data?: ArrayLike<number>): Database;
    };
  }

  export default function initSqlJs(
    config?: Record<string, unknown>,
  ): Promise<SqlJsStatic>;
}
