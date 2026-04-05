//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- query cancellation support
// 1:1 port of uqa/cancel.py

export class QueryCancelled extends Error {
  constructor(message: string = "canceling statement due to user request") {
    super(message);
    this.name = "QueryCancelled";
  }
}

export class CancellationToken {
  private _cancelled = false;

  cancel(): void {
    this._cancelled = true;
  }

  reset(): void {
    this._cancelled = false;
  }

  get isCancelled(): boolean {
    return this._cancelled;
  }

  check(): void {
    if (this._cancelled) {
      throw new QueryCancelled();
    }
  }
}
