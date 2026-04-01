//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- SQL expression evaluator
// 1:1 port of uqa/sql/expr_evaluator.py
//
// Evaluates pglast/libpg-query AST nodes (Record<string, unknown> from JSON parse).

import { tokenizeFts, FTSParser } from "./fts-query.js";
import type { FTSNode } from "./fts-query.js";
import { extractQueryTerms, highlight } from "../search/highlight.js";

// -- FTS match helper -----------------------------------------------------------

function ftsMatchNode(text: string, node: FTSNode): boolean {
  switch (node.type) {
    case "term":
      return text.includes(node.term.toLowerCase());
    case "phrase":
      return text.includes(node.phrase.toLowerCase());
    case "and":
      return ftsMatchNode(text, node.left) && ftsMatchNode(text, node.right);
    case "or":
      return ftsMatchNode(text, node.left) || ftsMatchNode(text, node.right);
    case "not":
      return !ftsMatchNode(text, node.operand);
    case "vector":
      return false; // vectors cannot match text
    default:
      return false;
  }
}

function ftsMatch(text: string, query: string): boolean {
  try {
    const tokens = tokenizeFts(query);
    const ast = new FTSParser(tokens).parse();
    return ftsMatchNode(text.toLowerCase(), ast);
  } catch {
    // Fallback to simple term matching if parse fails
    const terms = query
      .toLowerCase()
      .split(/\s+/)
      .filter((t) => t.length > 0);
    if (terms.length === 0) return false;
    return terms.every((term) => text.toLowerCase().includes(term));
  }
}

// -- Safe conversion helpers ----------------------------------------------------

/** Convert unknown to string without triggering no-base-to-string. */
function toStr(v: unknown): string {
  if (typeof v === "string") return v;
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return v.toString(10);
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v === "bigint") return v.toString(10);
  return JSON.stringify(v);
}

// -- MD5 implementation (RFC 1321) ----------------------------------------------

/**
 * Pure-JS MD5 hash returning a 32-character lowercase hex string.
 * Implements the standard MD5 algorithm per RFC 1321.
 */
function md5(input: string): string {
  // Pre-computed per-round shift amounts
  const s: number[] = [
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14,
    20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11,
    16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
  ];

  // Pre-computed T table: floor(2^32 * abs(sin(i+1))) for i in 0..63
  const K: number[] = [];
  for (let i = 0; i < 64; i++) {
    K.push(Math.floor(Math.abs(Math.sin(i + 1)) * 0x100000000));
  }

  // Convert string to UTF-8 bytes
  const encoder = new TextEncoder();
  const msgBytes = encoder.encode(input);
  const bitLen = msgBytes.length * 8;

  // Padding: append 0x80, then zeros, then 64-bit little-endian length
  const padLen =
    msgBytes.length % 64 < 56
      ? 56 - (msgBytes.length % 64)
      : 120 - (msgBytes.length % 64);
  const padded = new Uint8Array(msgBytes.length + padLen + 8);
  padded.set(msgBytes);
  padded[msgBytes.length] = 0x80;

  // Append original length in bits as 64-bit LE
  const dv = new DataView(padded.buffer);
  dv.setUint32(padded.length - 8, bitLen >>> 0, true);
  dv.setUint32(padded.length - 4, Math.floor(bitLen / 0x100000000), true);

  // Initialize hash values
  let a0 = 0x67452301;
  let b0 = 0xefcdab89;
  let c0 = 0x98badcfe;
  let d0 = 0x10325476;

  // Process each 64-byte chunk
  for (let offset = 0; offset < padded.length; offset += 64) {
    const M: number[] = [];
    for (let j = 0; j < 16; j++) {
      M.push(dv.getUint32(offset + j * 4, true));
    }

    let A = a0;
    let B = b0;
    let C = c0;
    let D = d0;

    for (let i = 0; i < 64; i++) {
      let F: number;
      let g: number;
      if (i < 16) {
        F = (B & C) | (~B & D);
        g = i;
      } else if (i < 32) {
        F = (D & B) | (~D & C);
        g = (5 * i + 1) % 16;
      } else if (i < 48) {
        F = B ^ C ^ D;
        g = (3 * i + 5) % 16;
      } else {
        F = C ^ (B | ~D);
        g = (7 * i) % 16;
      }
      F = (F + A + K[i]! + M[g]!) >>> 0;
      A = D;
      D = C;
      C = B;
      B = (B + ((F << s[i]!) | (F >>> (32 - s[i]!)))) >>> 0;
    }

    a0 = (a0 + A) >>> 0;
    b0 = (b0 + B) >>> 0;
    c0 = (c0 + C) >>> 0;
    d0 = (d0 + D) >>> 0;
  }

  // Convert to hex string (little-endian byte order)
  function toLEHex(n: number): string {
    let hex = "";
    for (let i = 0; i < 4; i++) {
      const byte = (n >>> (i * 8)) & 0xff;
      hex += (byte < 16 ? "0" : "") + byte.toString(16);
    }
    return hex;
  }

  return toLEHex(a0) + toLEHex(b0) + toLEHex(c0) + toLEHex(d0);
}

/** Format a Date as a local-time ISO string (no UTC conversion). */
/**
 * Parse a date string consistently as local time.
 * ISO strings without timezone suffix (e.g. "2024-06-15T10:30:00")
 * are treated as local time rather than UTC.
 */
function parseLocalDate(s: string): Date {
  // If the string already has a timezone indicator, parse normally
  if (s.endsWith("Z") || /[+-]\d{2}:\d{2}$/.test(s)) {
    return new Date(s);
  }
  // Match ISO-like format: YYYY-MM-DDTHH:MM:SS[.sss]
  const m = /^(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?)?$/.exec(
    s,
  );
  if (m) {
    const year = parseInt(m[1]!, 10);
    const month = parseInt(m[2]!, 10) - 1;
    const day = parseInt(m[3]!, 10);
    const hour = m[4] !== undefined ? parseInt(m[4], 10) : 0;
    const minute = m[5] !== undefined ? parseInt(m[5], 10) : 0;
    const second = m[6] !== undefined ? parseInt(m[6], 10) : 0;
    const ms = m[7] !== undefined ? parseInt(m[7].padEnd(3, "0").slice(0, 3), 10) : 0;
    const d = new Date(year, month, day, hour, minute, second, ms);
    return d;
  }
  return new Date(s);
}

function localISOString(d: Date): string {
  const y = String(d.getFullYear()).padStart(4, "0");
  const mo = String(d.getMonth() + 1).padStart(2, "0");
  const da = String(d.getDate()).padStart(2, "0");
  const h = String(d.getHours()).padStart(2, "0");
  const mi = String(d.getMinutes()).padStart(2, "0");
  const s = String(d.getSeconds()).padStart(2, "0");
  const ms = String(d.getMilliseconds()).padStart(3, "0");
  return `${y}-${mo}-${da}T${h}:${mi}:${s}.${ms}`;
}

// -- Scalar function dispatch ---------------------------------------------------

function callScalarFunction(name: string, args: unknown[]): unknown {
  switch (name) {
    // String functions
    case "upper":
      return args[0] === null ? null : toStr(args[0]).toUpperCase();
    case "lower":
      return args[0] === null ? null : toStr(args[0]).toLowerCase();
    case "length":
    case "char_length":
    case "character_length":
      return args[0] === null ? null : toStr(args[0]).length;
    case "octet_length":
      return args[0] === null ? null : new TextEncoder().encode(toStr(args[0])).length;
    case "trim":
    case "btrim":
      if (args[0] === null) return null;
      if (args.length >= 2 && args[1] !== null && args[1] !== undefined) {
        const chars = toStr(args[1]);
        const s = toStr(args[0]);
        let start = 0;
        let end = s.length;
        while (start < end && chars.includes(s[start]!)) start++;
        while (end > start && chars.includes(s[end - 1]!)) end--;
        return s.slice(start, end);
      }
      return toStr(args[0]).trim();
    case "ltrim":
      if (args[0] === null) return null;
      if (args.length >= 2 && args[1] !== null && args[1] !== undefined) {
        const chars = toStr(args[1]);
        const s = toStr(args[0]);
        let start = 0;
        while (start < s.length && chars.includes(s[start]!)) start++;
        return s.slice(start);
      }
      return toStr(args[0]).replace(/^\s+/, "");
    case "rtrim":
      if (args[0] === null) return null;
      if (args.length >= 2 && args[1] !== null && args[1] !== undefined) {
        const chars = toStr(args[1]);
        const s = toStr(args[0]);
        let end = s.length;
        while (end > 0 && chars.includes(s[end - 1]!)) end--;
        return s.slice(0, end);
      }
      return toStr(args[0]).replace(/\s+$/, "");
    case "replace":
      if (args[0] === null) return null;
      return toStr(args[0]).split(toStr(args[1])).join(toStr(args[2]));
    case "substring":
    case "substr": {
      if (args[0] === null) return null;
      const s = toStr(args[0]);
      // SQL SUBSTRING is 1-based
      const from = Number(args[1]) - 1;
      if (args.length >= 3) {
        const forLen = Number(args[2]);
        return s.slice(from, from + forLen);
      }
      return s.slice(from);
    }
    case "concat":
      return args.map((a) => (a === null ? "" : toStr(a))).join("");
    case "concat_ws": {
      if (args[0] === null || args[0] === undefined) return null;
      const sep = toStr(args[0]);
      return args
        .slice(1)
        .filter((a) => a !== null && a !== undefined)
        .map((a) => toStr(a))
        .join(sep);
    }
    case "left":
      if (args[0] === null) return null;
      return toStr(args[0]).slice(0, Number(args[1]));
    case "right":
      if (args[0] === null) return null;
      return toStr(args[0]).slice(-Number(args[1]));
    case "lpad": {
      if (args[0] === null) return null;
      const s = toStr(args[0]);
      const targetLen = Number(args[1]);
      const pad = args.length >= 3 ? toStr(args[2]) : " ";
      if (s.length >= targetLen) return s.slice(0, targetLen);
      return (
        pad
          .repeat(Math.ceil((targetLen - s.length) / pad.length))
          .slice(0, targetLen - s.length) + s
      );
    }
    case "rpad": {
      if (args[0] === null) return null;
      const s = toStr(args[0]);
      const targetLen = Number(args[1]);
      const pad = args.length >= 3 ? toStr(args[2]) : " ";
      if (s.length >= targetLen) return s.slice(0, targetLen);
      return (
        s +
        pad
          .repeat(Math.ceil((targetLen - s.length) / pad.length))
          .slice(0, targetLen - s.length)
      );
    }
    case "repeat":
      if (args[0] === null) return null;
      return toStr(args[0]).repeat(Number(args[1]));
    case "reverse":
      if (args[0] === null) return null;
      return Array.from(toStr(args[0])).reverse().join("");
    case "position":
    case "strpos": {
      if (args[0] === null || args[1] === null) return null;
      const idx = toStr(args[0]).indexOf(toStr(args[1]));
      return idx === -1 ? 0 : idx + 1; // 1-based, 0 = not found
    }
    case "initcap":
      if (args[0] === null) return null;
      return toStr(args[0]).replace(/\b\w/g, (c) => c.toUpperCase());
    case "md5":
      if (args[0] === null) return null;
      return md5(toStr(args[0]));
    case "encode":
      if (args[0] === null) return null;
      if (toStr(args[1]).toLowerCase() === "base64") {
        if (typeof btoa === "function") {
          return btoa(toStr(args[0]));
        }
      }
      throw new Error(`encode() format not supported: ${toStr(args[1])}`);
    case "decode":
      if (args[0] === null) return null;
      if (toStr(args[1]).toLowerCase() === "base64") {
        if (typeof atob === "function") {
          return atob(toStr(args[0]));
        }
      }
      throw new Error(`decode() format not supported: ${toStr(args[1])}`);

    // Math functions
    case "abs":
      return args[0] === null ? null : Math.abs(Number(args[0]));
    case "round": {
      if (args[0] === null) return null;
      const num = Number(args[0]);
      if (args.length >= 2 && args[1] !== null && args[1] !== undefined) {
        const scale = Number(args[1]);
        const factor = 10 ** scale;
        return Math.round(num * factor) / factor;
      }
      return Math.round(num);
    }
    case "ceil":
    case "ceiling":
      return args[0] === null ? null : Math.ceil(Number(args[0]));
    case "floor":
      return args[0] === null ? null : Math.floor(Number(args[0]));
    case "trunc":
    case "truncate": {
      if (args[0] === null) return null;
      if (args.length >= 2 && args[1] !== null && args[1] !== undefined) {
        const scale = Number(args[1]);
        const factor = 10 ** scale;
        return Math.trunc(Number(args[0]) * factor) / factor;
      }
      return Math.trunc(Number(args[0]));
    }
    case "power":
    case "pow":
      if (args[0] === null || args[1] === null) return null;
      return Math.pow(Number(args[0]), Number(args[1]));
    case "sqrt":
      return args[0] === null ? null : Math.sqrt(Number(args[0]));
    case "cbrt":
      return args[0] === null ? null : Math.cbrt(Number(args[0]));
    case "exp":
      return args[0] === null ? null : Math.exp(Number(args[0]));
    case "ln":
      if (args[0] === null) return null;
      return Math.log(Number(args[0]));
    case "log": {
      if (args[0] === null) return null;
      if (args.length >= 2 && args[1] !== null) {
        // log(base, value)
        return Math.log(Number(args[1])) / Math.log(Number(args[0]));
      }
      // PostgreSQL LOG(x) = log base 10
      return Math.log10(Number(args[0]));
    }
    case "log10":
    case "log2":
      if (args[0] === null) return null;
      return name === "log10"
        ? Math.log10(Number(args[0]))
        : Math.log2(Number(args[0]));
    case "mod":
      if (args[0] === null || args[1] === null) return null;
      return Number(args[0]) % Number(args[1]);
    case "sign":
      return args[0] === null ? null : Math.sign(Number(args[0]));
    case "pi":
      return Math.PI;
    case "random":
      return Math.random();
    case "degrees":
      return args[0] === null ? null : (Number(args[0]) * 180) / Math.PI;
    case "radians":
      return args[0] === null ? null : (Number(args[0]) * Math.PI) / 180;
    case "sin":
      return args[0] === null ? null : Math.sin(Number(args[0]));
    case "cos":
      return args[0] === null ? null : Math.cos(Number(args[0]));
    case "tan":
      return args[0] === null ? null : Math.tan(Number(args[0]));
    case "asin":
      return args[0] === null ? null : Math.asin(Number(args[0]));
    case "acos":
      return args[0] === null ? null : Math.acos(Number(args[0]));
    case "atan":
      return args[0] === null ? null : Math.atan(Number(args[0]));
    case "atan2":
      if (args[0] === null || args[1] === null) return null;
      return Math.atan2(Number(args[0]), Number(args[1]));
    case "div":
      if (args[0] === null || args[1] === null) return null;
      if (Number(args[1]) === 0) return null;
      return Math.trunc(Number(args[0]) / Number(args[1]));
    case "gcd": {
      if (args[0] === null || args[1] === null) return null;
      let ga = Math.abs(Math.trunc(Number(args[0])));
      let gb = Math.abs(Math.trunc(Number(args[1])));
      while (gb) {
        const t = gb;
        gb = ga % gb;
        ga = t;
      }
      return ga;
    }
    case "lcm": {
      if (args[0] === null || args[1] === null) return null;
      const la = Math.abs(Math.trunc(Number(args[0])));
      const lb = Math.abs(Math.trunc(Number(args[1])));
      if (la === 0 || lb === 0) return 0;
      let ga2 = la;
      let gb2 = lb;
      while (gb2) {
        const t = gb2;
        gb2 = ga2 % gb2;
        ga2 = t;
      }
      return Math.abs(la * lb) / ga2;
    }
    case "width_bucket": {
      if (args.length < 4 || args.some((a) => a === null)) return null;
      const val = Number(args[0]);
      const lo = Number(args[1]);
      const hi = Number(args[2]);
      const nb = Math.trunc(Number(args[3]));
      if (hi === lo || nb <= 0) return null;
      if (val < lo) return 0;
      if (val >= hi) return nb + 1;
      return Math.trunc((val - lo) / ((hi - lo) / nb)) + 1;
    }
    case "min_scale": {
      if (args[0] === null) return null;
      const s = String(args[0] as string | number);
      const dotIdx = s.indexOf(".");
      if (dotIdx === -1) return 0;
      let end = s.length;
      while (end > dotIdx + 1 && s[end - 1] === "0") end--;
      return end - dotIdx - 1;
    }
    case "trim_scale": {
      if (args[0] === null) return null;
      const n = Number(args[0]);
      if (Number.isInteger(n)) return n;
      return n;
    }

    // Additional string functions
    case "translate": {
      if (args[0] === null || args[1] === null || args[2] === null) return null;
      const s = toStr(args[0]);
      const from = toStr(args[1]);
      const to = toStr(args[2]);
      let result = "";
      for (const ch of s) {
        const idx = from.indexOf(ch);
        if (idx === -1) result += ch;
        else if (idx < to.length) result += to[idx]!;
        // else: character is deleted (from longer than to)
      }
      return result;
    }
    case "ascii":
      if (args[0] === null) return null;
      return toStr(args[0]).length > 0 ? toStr(args[0]).charCodeAt(0) : 0;
    case "chr":
      if (args[0] === null) return null;
      return String.fromCharCode(Math.trunc(Number(args[0])));
    case "starts_with":
      if (args[0] === null || args[1] === null) return null;
      return toStr(args[0]).startsWith(toStr(args[1]));
    case "split_part": {
      if (args[0] === null || args[1] === null || args[2] === null) return null;
      const parts = toStr(args[0]).split(toStr(args[1]));
      const n = Math.trunc(Number(args[2]));
      return n >= 1 && n <= parts.length ? parts[n - 1] : "";
    }
    case "format": {
      if (args[0] === null) return null;
      let fmt = toStr(args[0]);
      fmt = fmt.replace(/%I/g, "%s").replace(/%L/g, "'%s'");
      let i = 1;
      return fmt.replace(/%s/g, () => (i < args.length ? toStr(args[i++]) : ""));
    }
    case "overlay": {
      if (args[0] === null || args[1] === null || args[2] === null) return null;
      const s = toStr(args[0]);
      const repl = toStr(args[1]);
      const pos = Math.trunc(Number(args[2])) - 1;
      const len =
        args.length > 3 && args[3] != null ? Math.trunc(Number(args[3])) : repl.length;
      return s.slice(0, pos) + repl + s.slice(pos + len);
    }
    case "regexp_match": {
      if (args[0] === null || args[1] === null) return null;
      const m = toStr(args[0]).match(new RegExp(toStr(args[1])));
      if (m === null) return null;
      const groups = m.slice(1);
      return groups.length > 0 ? groups : [m[0]];
    }
    case "regexp_matches": {
      if (args[0] === null || args[1] === null) return null;
      const flagsStr = args.length > 2 && args[2] != null ? toStr(args[2]) : "";
      let flags = "";
      if (flagsStr.includes("i")) flags += "i";
      if (flagsStr.includes("g")) {
        const re = new RegExp(toStr(args[1]), flags + "g");
        const results: string[][] = [];
        let match;
        while ((match = re.exec(toStr(args[0]))) !== null) {
          results.push(match.slice(1).length > 0 ? [...match.slice(1)] : [match[0]]);
        }
        return results;
      }
      const m = toStr(args[0]).match(new RegExp(toStr(args[1]), flags));
      if (m === null) return null;
      return [m.slice(1).length > 0 ? [...m.slice(1)] : [m[0]]];
    }
    case "regexp_replace": {
      if (args[0] === null || args[1] === null) return null;
      const replacement = args.length > 2 && args[2] != null ? toStr(args[2]) : "";
      const flagsStr = args.length > 3 && args[3] != null ? toStr(args[3]) : "";
      let flags = "";
      if (flagsStr.includes("i")) flags += "i";
      if (flagsStr.includes("g")) flags += "g";
      else if (!flags.includes("g")) {
        /* default: replace first only */
      }
      return toStr(args[0]).replace(
        new RegExp(toStr(args[1]), flags || undefined),
        replacement,
      );
    }
    case "regexp_split_to_array": {
      if (args[0] === null || args[1] === null) return null;
      let flags = "";
      if (args.length > 2 && args[2] != null && toStr(args[2]).includes("i"))
        flags = "i";
      return toStr(args[0]).split(new RegExp(toStr(args[1]), flags));
    }
    case "regexp_count": {
      if (args[0] === null || args[1] === null) return 0;
      let flagStr = "";
      if (args.length > 2 && args[2] != null) flagStr = toStr(args[2]);
      const re = new RegExp(toStr(args[1]), "g" + (flagStr.includes("i") ? "i" : ""));
      const matches = toStr(args[0]).match(re);
      return matches !== null ? matches.length : 0;
    }
    case "string_to_array": {
      if (args[0] === null) return null;
      const delimiter = args.length > 1 && args[1] != null ? toStr(args[1]) : ",";
      const nullStr = args.length > 2 && args[2] != null ? toStr(args[2]) : null;
      const parts = toStr(args[0]).split(delimiter);
      if (nullStr !== null) {
        return parts.map((p) => (p === nullStr ? null : p));
      }
      return parts;
    }
    case "array_to_string": {
      if (args[0] === null) return null;
      if (!Array.isArray(args[0])) return null;
      const delimiter = args.length > 1 && args[1] != null ? toStr(args[1]) : ",";
      const nullStr = args.length > 2 && args[2] != null ? toStr(args[2]) : "";
      return (args[0] as unknown[])
        .map((v) => (v === null || v === undefined ? nullStr : toStr(v)))
        .join(delimiter);
    }
    case "array_position": {
      if (args[0] === null || args[1] === null) return null;
      if (!Array.isArray(args[0])) return null;
      const idx = (args[0] as unknown[]).indexOf(args[1]);
      return idx === -1 ? null : idx + 1; // 1-based
    }
    case "array_positions": {
      if (args[0] === null || args[1] === null) return null;
      if (!Array.isArray(args[0])) return null;
      const positions: number[] = [];
      for (let i = 0; i < (args[0] as unknown[]).length; i++) {
        if ((args[0] as unknown[])[i] === args[1]) positions.push(i + 1);
      }
      return positions;
    }
    case "array_replace": {
      if (args[0] === null) return null;
      if (!Array.isArray(args[0])) return null;
      return (args[0] as unknown[]).map((v) => (v === args[1] ? args[2] : v));
    }
    case "array_dims": {
      if (args[0] === null) return null;
      if (!Array.isArray(args[0])) return null;
      return `[1:${String((args[0] as unknown[]).length)}]`;
    }
    case "array_fill": {
      if (args.length < 2 || args[1] === null) return null;
      const fillVal = args[0];
      const dims = Array.isArray(args[1]) ? (args[1] as number[]) : [Number(args[1])];
      const size = dims[0] ?? 0;
      return new Array(size).fill(fillVal);
    }
    case "generate_series": {
      if (args.length < 2 || args[0] === null || args[1] === null) return [];
      const start = Number(args[0]);
      const stop = Number(args[1]);
      const step = args.length > 2 && args[2] != null ? Number(args[2]) : 1;
      if (step === 0) return [];
      const result: number[] = [];
      if (step > 0) {
        for (let i = start; i <= stop; i += step) result.push(i);
      } else {
        for (let i = start; i >= stop; i += step) result.push(i);
      }
      return result;
    }
    case "generate_subscripts": {
      if (args[0] === null) return [];
      if (!Array.isArray(args[0])) return [];
      const arr = args[0] as unknown[];
      return Array.from({ length: arr.length }, (_, i) => i + 1);
    }

    // Hash functions
    case "hashtext": {
      if (args[0] === null) return null;
      // Simple FNV-1a 32-bit hash
      let hash = 0x811c9dc5;
      const str = toStr(args[0]);
      for (let i = 0; i < str.length; i++) {
        hash ^= str.charCodeAt(i);
        hash = Math.imul(hash, 0x01000193);
      }
      return hash >>> 0;
    }

    // String padding/formatting
    case "quote_ident":
      if (args[0] === null) return null;
      return `"${toStr(args[0]).replace(/"/g, '""')}"`;
    case "quote_literal":
      if (args[0] === null) return "NULL";
      return `'${toStr(args[0]).replace(/'/g, "''")}'`;
    case "quote_nullable":
      if (args[0] === null) return "NULL";
      return `'${toStr(args[0]).replace(/'/g, "''")}'`;
    case "string_agg":
      // string_agg is an aggregate; in scalar context just return
      if (args[0] === null) return null;
      return toStr(args[0]);

    // Conversion functions
    case "to_hex": {
      if (args[0] === null) return null;
      const n = Math.trunc(Number(args[0]));
      return n >= 0 ? n.toString(16) : (n >>> 0).toString(16);
    }
    case "to_ascii":
      if (args[0] === null) return null;
      return toStr(args[0]).replace(/[^\x00-\x7F]/g, "?");

    // Date/time functions
    case "date_trunc": {
      if (args[0] === null || args[1] === null) return null;
      const precision = toStr(args[0]).toLowerCase();
      const d = parseLocalDate(toStr(args[1]));
      if (precision === "year") {
        d.setFullYear(d.getFullYear(), 0, 1);
        d.setHours(0, 0, 0, 0);
      } else if (precision === "month") {
        d.setDate(1);
        d.setHours(0, 0, 0, 0);
      } else if (precision === "day") d.setHours(0, 0, 0, 0);
      else if (precision === "hour") d.setMinutes(0, 0, 0);
      else if (precision === "minute") d.setSeconds(0, 0);
      else if (precision === "second") d.setMilliseconds(0);
      return localISOString(d);
    }
    case "make_timestamp": {
      if (args.length < 6 || args.some((a) => a === null)) return null;
      const dt = new Date(
        Number(args[0]),
        Number(args[1]) - 1,
        Number(args[2]),
        Number(args[3]),
        Number(args[4]),
        Math.trunc(Number(args[5])),
      );
      return localISOString(dt);
    }
    case "make_interval": {
      const years = args.length > 0 && args[0] != null ? Number(args[0]) : 0;
      const months = args.length > 1 && args[1] != null ? Number(args[1]) : 0;
      const weeks = args.length > 2 && args[2] != null ? Number(args[2]) : 0;
      const days = args.length > 3 && args[3] != null ? Number(args[3]) : 0;
      const hours = args.length > 4 && args[4] != null ? Number(args[4]) : 0;
      const mins = args.length > 5 && args[5] != null ? Number(args[5]) : 0;
      const secs = args.length > 6 && args[6] != null ? Number(args[6]) : 0;
      const totalDays = years * 365 + months * 30 + weeks * 7 + days;
      const parts: string[] = [];
      if (years > 0) parts.push(`${String(years)} year${years > 1 ? "s" : ""}`);
      if (months > 0) parts.push(`${String(months)} mon${months > 1 ? "s" : ""}`);
      if (totalDays > 0)
        parts.push(`${String(totalDays)} day${totalDays !== 1 ? "s" : ""}`);
      parts.push(
        `${String(hours).padStart(2, "0")}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`,
      );
      return parts.join(" ");
    }
    case "make_date": {
      if (args.length < 3 || args.some((a) => a === null)) return null;
      const yyyy = String(Number(args[0])).padStart(4, "0");
      const mm = String(Number(args[1])).padStart(2, "0");
      const dd = String(Number(args[2])).padStart(2, "0");
      return `${yyyy}-${mm}-${dd}`;
    }
    case "clock_timestamp":
      return new Date().toISOString();
    case "timeofday": {
      const now = new Date();
      return now.toUTCString();
    }
    case "isfinite":
      if (args[0] === null) return null;
      return !["infinity", "-infinity"].includes(toStr(args[0]).trim().toLowerCase());
    case "overlaps": {
      if (args.length < 4 || args.some((a) => a === null)) return null;
      const s1 = new Date(toStr(args[0])).getTime();
      const e1 = new Date(toStr(args[1])).getTime();
      const s2 = new Date(toStr(args[2])).getTime();
      const e2 = new Date(toStr(args[3])).getTime();
      return Math.min(s1, e1) < Math.max(s2, e2) && Math.min(s2, e2) < Math.max(s1, e1);
    }

    // Type checking
    case "typeof":
    case "pg_typeof":
      if (args[0] === null) return "null";
      if (typeof args[0] === "number")
        return Number.isInteger(args[0]) ? "integer" : "real";
      if (typeof args[0] === "string") return "text";
      if (typeof args[0] === "boolean") return "boolean";
      return "unknown";

    // UUID
    case "gen_random_uuid":
      return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
      });

    // Additional array functions
    case "array_remove":
      if (args[0] === null) return null;
      if (Array.isArray(args[0]))
        return (args[0] as unknown[]).filter((x) => x !== args[1]);
      return null;
    case "array_upper":
      if (args[0] === null) return null;
      if (Array.isArray(args[0])) return args[0].length > 0 ? args[0].length : null;
      return null;
    case "array_lower":
      if (args[0] === null) return null;
      if (Array.isArray(args[0])) return args[0].length > 0 ? 1 : null;
      return null;
    case "cardinality":
      if (args[0] === null) return null;
      if (Array.isArray(args[0])) return args[0].length;
      return null;

    // Additional JSON functions
    case "json_extract_path":
    case "jsonb_extract_path":
    case "json_extract_path_text":
    case "jsonb_extract_path_text": {
      if (args[0] === null) return null;
      let obj: unknown = typeof args[0] === "string" ? JSON.parse(args[0]) : args[0];
      for (let i = 1; i < args.length; i++) {
        if (args[i] === null || obj === null || obj === undefined) return null;
        if (typeof obj === "object" && !Array.isArray(obj)) {
          obj = (obj as Record<string, unknown>)[toStr(args[i])];
        } else if (Array.isArray(obj)) {
          obj = obj[Number(args[i])];
        } else return null;
      }
      if (name.endsWith("_text")) {
        if (obj === null || obj === undefined) return null;
        return typeof obj === "object" ? JSON.stringify(obj) : toStr(obj);
      }
      return obj ?? null;
    }
    case "jsonb_set":
    case "jsonb_insert": {
      if (args.length < 3 || args[0] === null) return null;
      let target: unknown = typeof args[0] === "string" ? JSON.parse(args[0]) : args[0];
      target = JSON.parse(JSON.stringify(target)); // deep clone
      const pathStr = toStr(args[1]).replace(/^\{/, "").replace(/\}$/, "");
      const keys = pathStr.split(",").map((k) => k.trim());
      const newValue = args[2];
      let cur: unknown = target;
      for (let i = 0; i < keys.length - 1; i++) {
        if (typeof cur === "object" && cur !== null && !Array.isArray(cur)) {
          cur = (cur as Record<string, unknown>)[keys[i]!];
        } else if (Array.isArray(cur)) {
          cur = cur[Number(keys[i])];
        } else return target;
      }
      const lastKey = keys[keys.length - 1]!;
      if (typeof cur === "object" && cur !== null && !Array.isArray(cur)) {
        (cur as Record<string, unknown>)[lastKey] = newValue;
      } else if (Array.isArray(cur)) {
        cur[Number(lastKey)] = newValue;
      }
      return target;
    }
    case "json_each":
    case "jsonb_each":
    case "json_each_text":
    case "jsonb_each_text": {
      if (args[0] === null) return [];
      const obj: unknown = typeof args[0] === "string" ? JSON.parse(args[0]) : args[0];
      if (typeof obj !== "object" || Array.isArray(obj) || obj === null) return [];
      const asText = name.endsWith("_text");
      return Object.entries(obj as Record<string, unknown>).map(([k, v]) => ({
        key: k,
        value: asText
          ? typeof v === "object" && v !== null
            ? JSON.stringify(v)
            : v != null
              ? toStr(v)
              : null
          : v,
      }));
    }
    case "json_array_elements":
    case "jsonb_array_elements":
    case "json_array_elements_text":
    case "jsonb_array_elements_text": {
      if (args[0] === null) return [];
      const arr: unknown = typeof args[0] === "string" ? JSON.parse(args[0]) : args[0];
      if (!Array.isArray(arr)) return [];
      if (name.endsWith("_text")) {
        return (arr as unknown[]).map((v) =>
          typeof v === "object" && v !== null
            ? JSON.stringify(v)
            : v != null
              ? toStr(v)
              : null,
        );
      }
      return arr;
    }
    case "json_object_keys":
    case "jsonb_object_keys": {
      if (args[0] === null) return null;
      const obj: unknown = typeof args[0] === "string" ? JSON.parse(args[0]) : args[0];
      if (typeof obj !== "object" || Array.isArray(obj) || obj === null) return null;
      return Object.keys(obj as Record<string, unknown>);
    }
    case "jsonb_strip_nulls":
    case "json_strip_nulls": {
      if (args[0] === null) return null;
      const obj: unknown = typeof args[0] === "string" ? JSON.parse(args[0]) : args[0];
      function stripNulls(v: unknown): unknown {
        if (typeof v === "object" && v !== null && !Array.isArray(v)) {
          const result: Record<string, unknown> = {};
          for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
            if (val !== null) result[k] = stripNulls(val);
          }
          return result;
        }
        return v;
      }
      return stripNulls(obj);
    }
    case "json_build_array":
    case "jsonb_build_array":
      return [...args];

    // Spatial functions
    case "point":
      if (args.length !== 2) throw new Error("POINT() requires exactly 2 arguments");
      return [Number(args[0]), Number(args[1])];
    case "st_distance": {
      if (args.length !== 2 || args[0] === null || args[1] === null) return null;
      const p1 = args[0] as number[];
      const p2 = args[1] as number[];
      if (!Array.isArray(p1) || !Array.isArray(p2)) return null;
      // Haversine distance in meters
      const R = 6371000;
      const lat1 = (p1[1]! * Math.PI) / 180;
      const lat2 = (p2[1]! * Math.PI) / 180;
      const dLat = ((p2[1]! - p1[1]!) * Math.PI) / 180;
      const dLon = ((p2[0]! - p1[0]!) * Math.PI) / 180;
      const a =
        Math.sin(dLat / 2) ** 2 +
        Math.cos(lat1) * Math.cos(lat2) * Math.sin(dLon / 2) ** 2;
      return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
    }
    case "st_within":
    case "st_dwithin": {
      if (args.length !== 3 || args[0] === null || args[1] === null) return null;
      const dist = callScalarFunction("st_distance", [args[0], args[1]]) as
        | number
        | null;
      if (dist === null) return null;
      return dist <= Number(args[2]);
    }

    // Conditional / null functions
    case "coalesce":
      for (const a of args) {
        if (a !== null && a !== undefined) return a;
      }
      return null;
    case "nullif":
      if (args[0] === args[1]) return null;
      return args[0];
    case "greatest": {
      let best: unknown = null;
      for (const a of args) {
        if (a === null) continue;
        if (best === null || (a as number) > (best as number)) best = a;
      }
      return best;
    }
    case "least": {
      let best: unknown = null;
      for (const a of args) {
        if (a === null) continue;
        if (best === null || (a as number) < (best as number)) best = a;
      }
      return best;
    }

    // Date/time functions
    case "current_date":
    case "current_timestamp":
    case "now":
      return new Date().toISOString();
    case "date_part":
    case "extract": {
      if (args[0] === null || args[1] === null) return null;
      const part = toStr(args[0]).toLowerCase();
      const d = parseLocalDate(toStr(args[1]));
      switch (part) {
        case "year":
          return d.getFullYear();
        case "month":
          return d.getMonth() + 1;
        case "day":
          return d.getDate();
        case "hour":
          return d.getHours();
        case "minute":
          return d.getMinutes();
        case "second":
          return d.getSeconds();
        case "dow":
        case "dayofweek":
          return d.getDay();
        case "doy":
        case "dayofyear": {
          const start = new Date(d.getFullYear(), 0, 0);
          const diff = d.getTime() - start.getTime();
          return Math.floor(diff / 86400000);
        }
        case "epoch":
          return Math.floor(d.getTime() / 1000);
        case "quarter":
          return Math.ceil((d.getMonth() + 1) / 3);
        case "week": {
          // ISO 8601 week number
          const jan1 = new Date(d.getFullYear(), 0, 1);
          const dayOfYear = Math.floor((d.getTime() - jan1.getTime()) / 86400000) + 1;
          const jan1Dow = jan1.getDay() || 7; // Convert Sunday=0 to 7
          return Math.ceil((dayOfYear + jan1Dow - 1) / 7);
        }
        case "millisecond":
        case "milliseconds":
          return d.getSeconds() * 1000 + d.getMilliseconds();
        case "microsecond":
        case "microseconds":
          return d.getSeconds() * 1000000 + d.getMilliseconds() * 1000;
        default:
          throw new Error(`Unknown date part: "${part}"`);
      }
    }
    case "age": {
      if (args[0] === null) return null;
      const d1 = new Date(toStr(args[0]));
      const d2 =
        args.length >= 2 && args[1] !== null ? new Date(toStr(args[1])) : new Date();
      const diffMs = d2.getTime() - d1.getTime();
      return `${String(Math.floor(diffMs / 86400000))} days`;
    }

    case "date_add":
    case "dateadd": {
      if (args.length < 3 || args.some((a) => a === null)) return null;
      const unit = toStr(args[0]).toLowerCase();
      const amount = Number(args[1]);
      const d = new Date(toStr(args[2]));
      if (unit === "year") d.setFullYear(d.getFullYear() + amount);
      else if (unit === "month") d.setMonth(d.getMonth() + amount);
      else if (unit === "day") d.setDate(d.getDate() + amount);
      else if (unit === "hour") d.setHours(d.getHours() + amount);
      else if (unit === "minute") d.setMinutes(d.getMinutes() + amount);
      else if (unit === "second") d.setSeconds(d.getSeconds() + amount);
      return d.toISOString();
    }
    case "date_subtract":
    case "datesub": {
      if (args.length < 3 || args.some((a) => a === null)) return null;
      const unit = toStr(args[0]).toLowerCase();
      const amount = Number(args[1]);
      const d = new Date(toStr(args[2]));
      if (unit === "year") d.setFullYear(d.getFullYear() - amount);
      else if (unit === "month") d.setMonth(d.getMonth() - amount);
      else if (unit === "day") d.setDate(d.getDate() - amount);
      else if (unit === "hour") d.setHours(d.getHours() - amount);
      else if (unit === "minute") d.setMinutes(d.getMinutes() - amount);
      else if (unit === "second") d.setSeconds(d.getSeconds() - amount);
      return d.toISOString();
    }
    case "date_diff":
    case "datediff": {
      if (args.length < 3 || args.some((a) => a === null)) return null;
      const unit = toStr(args[0]).toLowerCase();
      const d1 = new Date(toStr(args[1]));
      const d2 = new Date(toStr(args[2]));
      const diffMs = d2.getTime() - d1.getTime();
      if (unit === "day") return Math.trunc(diffMs / 86400000);
      if (unit === "hour") return Math.trunc(diffMs / 3600000);
      if (unit === "minute") return Math.trunc(diffMs / 60000);
      if (unit === "second") return Math.trunc(diffMs / 1000);
      if (unit === "year") return d2.getFullYear() - d1.getFullYear();
      if (unit === "month")
        return (
          (d2.getFullYear() - d1.getFullYear()) * 12 + (d2.getMonth() - d1.getMonth())
        );
      return Math.trunc(diffMs / 1000);
    }
    case "justify_days": {
      // Convert excessive days to months (30 days = 1 month)
      if (args[0] === null) return null;
      return args[0]; // Simplified: intervals are represented as strings
    }
    case "justify_hours": {
      // Convert excessive hours to days (24 hours = 1 day)
      if (args[0] === null) return null;
      return args[0];
    }
    case "justify_interval": {
      if (args[0] === null) return null;
      return args[0];
    }
    case "statement_timestamp":
    case "transaction_timestamp":
      return new Date().toISOString();

    // Type cast helpers
    case "to_char": {
      if (args[0] === null) return null;
      // If format string is provided, apply basic formatting
      if (args.length >= 2 && args[1] != null) {
        const val = args[0];
        const fmt = toStr(args[1]);
        // Basic numeric formatting
        if (typeof val === "number") {
          if (fmt.includes("FM")) {
            return String(val);
          }
          if (fmt.includes("9") || fmt.includes("0")) {
            const decimals = (fmt.match(/\.([09]+)/)?.[1] ?? "").length;
            return val.toFixed(decimals);
          }
        }
        // Basic date formatting
        if (typeof val === "string" || val instanceof Date) {
          const d = new Date(toStr(val));
          let result = fmt;
          result = result.replace(/YYYY/g, String(d.getFullYear()));
          result = result.replace(/MM/g, String(d.getMonth() + 1).padStart(2, "0"));
          result = result.replace(/DD/g, String(d.getDate()).padStart(2, "0"));
          result = result.replace(/HH24/g, String(d.getHours()).padStart(2, "0"));
          result = result.replace(
            /HH/g,
            String(d.getHours() % 12 || 12).padStart(2, "0"),
          );
          result = result.replace(/MI/g, String(d.getMinutes()).padStart(2, "0"));
          result = result.replace(/SS/g, String(d.getSeconds()).padStart(2, "0"));
          return result;
        }
      }
      return toStr(args[0]);
    }
    case "to_number": {
      if (args[0] === null) return null;
      // Strip currency symbols, commas, spaces from input
      const cleaned = toStr(args[0]).replace(/[^0-9eE.+-]/g, "");
      return Number(cleaned);
    }
    case "to_date":
    case "to_timestamp":
      if (args[0] === null) return null;
      return new Date(toStr(args[0])).toISOString();

    // Array functions
    case "array_length":
      if (args[0] === null) return null;
      if (Array.isArray(args[0])) return args[0].length;
      return null;
    case "unnest":
      // unnest is a set-returning function; in scalar context return as-is
      return args[0];
    case "array_agg":
      // array_agg is an aggregate; in scalar context just wrap
      return args;
    case "array_cat":
      if (args[0] === null || args[1] === null) return null;
      return [...(args[0] as unknown[]), ...(args[1] as unknown[])];
    case "array_append":
      if (args[0] === null) return null;
      return [...(args[0] as unknown[]), args[1]];
    case "array_prepend":
      if (args[1] === null) return null;
      return [args[0], ...(args[1] as unknown[])];

    // JSON functions
    case "json_build_object":
    case "jsonb_build_object": {
      const obj: Record<string, unknown> = {};
      for (let idx = 0; idx + 1 < args.length; idx += 2) {
        obj[toStr(args[idx])] = args[idx + 1] ?? null;
      }
      return obj;
    }
    case "json_array_length":
    case "jsonb_array_length":
      if (args[0] === null) return null;
      if (Array.isArray(args[0])) return args[0].length;
      if (typeof args[0] === "string") {
        const parsed = JSON.parse(args[0]) as unknown;
        return Array.isArray(parsed) ? parsed.length : null;
      }
      return null;
    case "json_typeof":
    case "jsonb_typeof":
      if (args[0] === null) return "null";
      if (Array.isArray(args[0])) return "array";
      if (typeof args[0] === "object") return "object";
      return typeof args[0];
    case "row_to_json":
      return args[0];
    case "to_json":
    case "to_jsonb":
      return args[0];
    case "jsonb_pretty": {
      if (args[0] === null) return null;
      const obj: unknown =
        typeof args[0] === "string" ? (JSON.parse(args[0]) as unknown) : args[0];
      return JSON.stringify(obj, null, 2);
    }
    case "jsonb_agg":
    case "json_agg":
      // Aggregate; in scalar context return as-is
      return args;
    case "jsonb_object_agg":
    case "json_object_agg":
      // Aggregate; in scalar context return empty object
      if (args.length >= 2) {
        const obj: Record<string, unknown> = {};
        obj[toStr(args[0])] = args[1];
        return obj;
      }
      return {};
    case "jsonb_concat": {
      if (args[0] === null || args[1] === null) return null;
      const a: unknown =
        typeof args[0] === "string" ? (JSON.parse(args[0]) as unknown) : args[0];
      const b: unknown =
        typeof args[1] === "string" ? (JSON.parse(args[1]) as unknown) : args[1];
      if (Array.isArray(a) && Array.isArray(b))
        return [...(a as unknown[]), ...(b as unknown[])];
      if (
        typeof a === "object" &&
        typeof b === "object" &&
        !Array.isArray(a) &&
        !Array.isArray(b)
      ) {
        return { ...(a as Record<string, unknown>), ...(b as Record<string, unknown>) };
      }
      return a;
    }
    case "jsonb_delete_path": {
      if (args[0] === null || args[1] === null) return null;
      const obj =
        typeof args[0] === "string"
          ? (JSON.parse(args[0]) as unknown)
          : (JSON.parse(JSON.stringify(args[0])) as unknown);
      const path = Array.isArray(args[1])
        ? (args[1] as string[])
        : toStr(args[1])
            .replace(/^\{/, "")
            .replace(/\}$/, "")
            .split(",")
            .map((s) => s.trim());
      if (path.length === 0) return obj;
      let cur: unknown = obj;
      for (let i = 0; i < path.length - 1; i++) {
        if (typeof cur === "object" && cur !== null && !Array.isArray(cur)) {
          cur = (cur as Record<string, unknown>)[path[i]!];
        } else if (Array.isArray(cur)) {
          cur = (cur as unknown[])[Number(path[i])];
        } else return obj;
      }
      const lastKey = path[path.length - 1]!;
      if (typeof cur === "object" && cur !== null && !Array.isArray(cur)) {
        // eslint-disable-next-line @typescript-eslint/no-dynamic-delete
        delete (cur as Record<string, unknown>)[lastKey];
      } else if (Array.isArray(cur)) {
        (cur as unknown[]).splice(Number(lastKey), 1);
      }
      return obj;
    }

    // Utility / system functions
    case "current_schema":
    case "current_schemas":
      return "public";
    case "current_database":
      return "uqa";
    case "current_user":
    case "session_user":
      return "uqa_user";
    case "version":
      return "UQA 1.0";
    case "has_table_privilege":
    case "has_schema_privilege":
    case "has_column_privilege":
      return true;
    case "txid_current":
      return 1;

    // UQA text search functions: row-level evaluation for WHERE clause
    case "text_match":
    case "bayesian_match": {
      if (args.length < 2) return false;
      const text = args[0];
      const query = args[1];
      if (text === null || query === null) return false;
      return ftsMatch(
        String(text as string | number),
        String(query as string | number),
      );
    }

    default:
      throw new Error(`Unknown SQL function: ${name}`);
  }
}

// -- Helpers for AST node access ------------------------------------------------

function nodeGet(node: Record<string, unknown>, key: string): unknown {
  return node[key] ?? null;
}

function nodeStr(node: Record<string, unknown>, key: string): string {
  const v = node[key];
  return v === undefined || v === null ? "" : toStr(v);
}

function asObj(value: unknown): Record<string, unknown> {
  return value as Record<string, unknown>;
}

function asList(value: unknown): Record<string, unknown>[] {
  if (Array.isArray(value)) return value as Record<string, unknown>[];
  return [];
}

// -- JSON containment helper ----------------------------------------------------

function jsonContains(container: unknown, contained: unknown): boolean {
  if (contained === null || contained === undefined) return true;
  if (container === null || container === undefined) return false;
  if (typeof contained !== "object") return container === contained;
  if (Array.isArray(contained)) {
    if (!Array.isArray(container)) return false;
    // Every element in contained must be found in container
    for (const item of contained as unknown[]) {
      if (!(container as unknown[]).some((c) => jsonContains(c, item))) return false;
    }
    return true;
  }
  if (typeof container !== "object" || Array.isArray(container)) return false;
  // Both are objects
  for (const [key, val] of Object.entries(contained as Record<string, unknown>)) {
    if (!(key in (container as Record<string, unknown>))) return false;
    if (!jsonContains((container as Record<string, unknown>)[key], val)) return false;
  }
  return true;
}

// -- LIKE matching --------------------------------------------------------------

function likeToRegex(pattern: string, caseInsensitive: boolean): RegExp {
  let regex = "^";
  for (let i = 0; i < pattern.length; i++) {
    const ch = pattern[i]!;
    if (ch === "%") {
      regex += ".*";
    } else if (ch === "_") {
      regex += ".";
    } else if (ch === "\\" && i + 1 < pattern.length) {
      i++;
      regex += pattern[i]!.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    } else {
      regex += ch.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    }
  }
  regex += "$";
  return new RegExp(regex, caseInsensitive ? "is" : "s");
}

// -- SQL type cast helper -------------------------------------------------------

function castValue(value: unknown, typeName: string): unknown {
  if (value === null) return null;
  const lower = typeName.toLowerCase();

  if (
    lower === "integer" ||
    lower === "int" ||
    lower === "int4" ||
    lower === "bigint" ||
    lower === "int8" ||
    lower === "smallint" ||
    lower === "int2"
  ) {
    return Math.trunc(Number(value));
  }
  if (
    lower === "float" ||
    lower === "float4" ||
    lower === "float8" ||
    lower === "double precision" ||
    lower === "real" ||
    lower === "numeric" ||
    lower === "decimal"
  ) {
    return Number(value);
  }
  if (
    lower === "text" ||
    lower === "varchar" ||
    lower === "char" ||
    lower === "character varying"
  ) {
    return toStr(value);
  }
  if (lower === "boolean" || lower === "bool") {
    if (typeof value === "string") {
      const s = value.toLowerCase();
      return s === "true" || s === "t" || s === "1" || s === "yes";
    }
    return Boolean(value);
  }
  if (lower === "json" || lower === "jsonb") {
    if (typeof value === "string") return JSON.parse(value) as unknown;
    return value;
  }
  if (lower === "date" || lower === "timestamp" || lower === "timestamptz") {
    return new Date(toStr(value)).toISOString();
  }

  // Unknown cast -- return as-is
  return value;
}

// -- ExprEvaluator --------------------------------------------------------------

/**
 * Callback type for executing subqueries from within ExprEvaluator.
 * Takes a SelectStmt AST node and an optional outer row (for correlated subqueries)
 * and returns {columns, rows}.
 */
export type SubqueryExecutor = (
  stmt: Record<string, unknown>,
  outerRow?: Record<string, unknown>,
) => { columns: string[]; rows: Record<string, unknown>[] };

export class ExprEvaluator {
  private _params: unknown[];
  private _sequences: Map<string, { current: number; increment: number }> | null;
  private _outerRow: Record<string, unknown> | null;
  private _subqueryExecutor: SubqueryExecutor | null;
  private _analyzer: { analyze(text: string): string[] } | null;

  constructor(opts?: {
    params?: unknown[];
    sequences?: Map<string, { current: number; increment: number }>;
    outerRow?: Record<string, unknown>;
    subqueryExecutor?: SubqueryExecutor;
    analyzer?: { analyze(text: string): string[] } | null;
  }) {
    this._params = opts?.params ?? [];
    this._sequences = opts?.sequences ?? null;
    this._outerRow = opts?.outerRow ?? null;
    this._subqueryExecutor = opts?.subqueryExecutor ?? null;
    this._analyzer = opts?.analyzer ?? null;
  }

  evaluate(node: Record<string, unknown>, row: Record<string, unknown>): unknown {
    // Dispatch on the first key of the node (libpg-query AST convention)
    const keys = Object.keys(node);
    if (keys.length === 0) return null;

    // If the node has a single top-level key that wraps the actual node, unwrap it
    const nodeType = keys[0]!;
    const inner = node[nodeType];

    switch (nodeType) {
      case "ColumnRef":
        return this._evalColumnRef(asObj(inner), row);
      case "A_Const":
        return this._evalConst(asObj(inner));
      case "A_Expr":
        return this._evalAExpr(asObj(inner), row);
      case "BoolExpr":
        return this._evalBoolExpr(asObj(inner), row);
      case "FuncCall":
        return this._evalFuncCall(asObj(inner), row);
      case "NullTest":
        return this._evalNullTest(asObj(inner), row);
      case "CaseExpr":
        return this._evalCaseExpr(asObj(inner), row);
      case "NamedArgExpr": {
        // Named argument -- evaluate the inner arg expression
        const naArg = nodeGet(asObj(inner), "arg");
        if (naArg !== null && naArg !== undefined) {
          return this.evaluate(naArg as Record<string, unknown>, row);
        }
        return null;
      }
      case "TypeCast":
        return this._evalTypeCast(asObj(inner), row);
      case "ParamRef":
        return this._evalParamRef(asObj(inner));
      case "SubLink":
        return this._evalSubLink(asObj(inner), row);
      case "CoalesceExpr":
        return this._evalCoalesceExpr(asObj(inner), row);
      case "MinMaxExpr":
        return this._evalMinMaxExpr(asObj(inner), row);
      case "BooleanTest":
        return this._evalBooleanTest(asObj(inner), row);
      case "SQLValueFunction":
        return this._evalSQLValueFunction(asObj(inner));
      case "A_ArrayExpr":
        return asList(nodeGet(asObj(inner), "elements")).map((item) =>
          this.evaluate(item, row),
        );
      case "List":
        // A List node wraps items in an "items" array
        return asList(nodeGet(asObj(inner), "items")).map((item) =>
          this.evaluate(item, row),
        );
      case "String":
      case "str":
        return nodeStr(asObj(inner), "sval") || nodeStr(asObj(inner), "str");
      case "Integer":
        return nodeGet(asObj(inner), "ival") ?? 0;
      case "Float":
        return Number(nodeStr(asObj(inner), "fval") || nodeStr(asObj(inner), "str"));
      default:
        // If node itself might be a value wrapper
        if ("ival" in node) return node["ival"];
        if ("fval" in node) return Number(node["fval"]);
        if ("sval" in node) return node["sval"];
        if ("boolval" in node) return node["boolval"];
        if ("bsval" in node) return node["bsval"];
        // Return the raw node for unknown types
        return node;
    }
  }

  // -- ColumnRef ----------------------------------------------------------------

  private _evalColumnRef(
    node: Record<string, unknown>,
    row: Record<string, unknown>,
  ): unknown {
    const fields = asList(nodeGet(node, "fields"));
    if (fields.length === 0) return null;

    // Extract column name(s). fields can be [{String: {sval: "col"}}] or
    // [{String: {sval: "table"}}, {String: {sval: "col"}}]
    const names: string[] = [];
    for (const f of fields) {
      const strNode = nodeGet(f, "String") ?? nodeGet(f, "str");
      if (strNode !== null && typeof strNode === "object") {
        const sval = nodeStr(asObj(strNode), "sval") || nodeStr(asObj(strNode), "str");
        if (sval) names.push(sval);
      } else if (typeof strNode === "string") {
        names.push(strNode);
      } else if (typeof f === "object") {
        // Might be a direct sval
        const sval = nodeStr(f, "sval") || nodeStr(f, "str");
        if (sval) names.push(sval);
      }
    }

    // Look up by full qualified name or just column name
    if (names.length === 1) {
      const colName = names[0]!;
      if (colName in row) return row[colName];
      // Check outer row for correlated subquery
      if (this._outerRow !== null && colName in this._outerRow) {
        return this._outerRow[colName];
      }
      // Return null for unknown columns (matches SQL semantics)
      return null;
    }

    if (names.length === 2) {
      // table.column -- try "table.column" first, then just "column"
      const qualified = `${names[0]!}.${names[1]!}`;
      if (qualified in row) return row[qualified];
      // Check outer row for qualified name before falling back to
      // unqualified inner column (correlated subquery resolution)
      if (this._outerRow !== null && qualified in this._outerRow) {
        return this._outerRow[qualified];
      }
      const colName = names[1]!;
      if (colName in row) return row[colName];
      if (this._outerRow !== null && colName in this._outerRow) {
        return this._outerRow[colName];
      }
      return null;
    }

    return null;
  }

  // -- A_Const ------------------------------------------------------------------

  private _evalConst(node: Record<string, unknown>): unknown {
    // libpg-query v17+ uses nested value nodes:
    //   {ival: {ival: N}}, {fval: {fval: "..."}}, {sval: {sval: "..."}},
    //   {boolval: {boolval: T/F}}, {isnull: true}
    // Older or simplified form: {ival: N}, {sval: "..."}, etc.
    if (nodeGet(node, "isnull") === true) return null;

    const ival = nodeGet(node, "ival");
    if (ival !== null && ival !== undefined) {
      if (typeof ival === "object") {
        const inner = nodeGet(asObj(ival), "ival");
        if (inner !== null && inner !== undefined) return inner;
        // Protobuf empty object {} represents integer 0
        if (
          typeof ival === "object" &&
          Object.keys(ival as Record<string, unknown>).length === 0
        )
          return 0;
        return ival;
      }
      return ival;
    }

    const fval = nodeGet(node, "fval");
    if (fval !== null && fval !== undefined) {
      if (typeof fval === "object") return Number(nodeGet(asObj(fval), "fval") ?? fval);
      return Number(fval);
    }

    const sval = nodeGet(node, "sval");
    if (sval !== null && sval !== undefined) {
      if (typeof sval === "object") return nodeGet(asObj(sval), "sval") ?? sval;
      return sval;
    }

    const boolval = nodeGet(node, "boolval");
    if (boolval !== null && boolval !== undefined) {
      if (typeof boolval === "object")
        return nodeGet(asObj(boolval), "boolval") ?? boolval;
      return boolval;
    }

    const bsval = nodeGet(node, "bsval");
    if (bsval !== null && bsval !== undefined) {
      if (typeof bsval === "object") return nodeGet(asObj(bsval), "bsval") ?? bsval;
      return bsval;
    }

    // Nested value node (older format): {Integer: {ival: N}}, {String: {sval: ...}}
    const intNode = nodeGet(node, "Integer");
    if (intNode !== null && intNode !== undefined) {
      return nodeGet(asObj(intNode), "ival") ?? 0;
    }
    const floatNode = nodeGet(node, "Float");
    if (floatNode !== null && floatNode !== undefined) {
      return Number(
        nodeStr(asObj(floatNode), "fval") || nodeStr(asObj(floatNode), "str"),
      );
    }
    const strNode = nodeGet(node, "String");
    if (strNode !== null && strNode !== undefined) {
      return nodeStr(asObj(strNode), "sval") || nodeStr(asObj(strNode), "str");
    }
    const nullNode = nodeGet(node, "Null");
    if (nullNode !== null && nullNode !== undefined) return null;

    return null;
  }

  // -- A_Expr -------------------------------------------------------------------

  private _evalAExpr(
    node: Record<string, unknown>,
    row: Record<string, unknown>,
  ): unknown {
    const kindRaw = nodeGet(node, "kind");
    let kind: number;
    if (typeof kindRaw === "number") {
      kind = kindRaw;
    } else if (typeof kindRaw === "string") {
      // libpg-query may emit string enum values like "AEXPR_NULLIF"
      const kindMap: Record<string, number> = {
        AEXPR_OP: 0,
        AEXPR_OP_ANY: 1,
        AEXPR_OP_ALL: 2,
        AEXPR_DISTINCT: 3,
        AEXPR_NOT_DISTINCT: 4,
        AEXPR_NULLIF: 5,
        AEXPR_IN: 6,
        AEXPR_LIKE: 7,
        AEXPR_ILIKE: 8,
        AEXPR_SIMILAR: 9,
        AEXPR_BETWEEN: 10,
        AEXPR_NOT_BETWEEN: 11,
        AEXPR_BETWEEN_SYM: 12,
        AEXPR_NOT_BETWEEN_SYM: 13,
        AEXPR_PAREN: 14,
      };
      kind = kindMap[kindRaw] ?? 0;
    } else {
      kind = 0;
    }

    // Extract operator name
    const nameList = asList(nodeGet(node, "name"));
    let opName = "";
    for (const n of nameList) {
      const strNode = nodeGet(n, "String") ?? nodeGet(n, "str");
      if (strNode !== null && typeof strNode === "object") {
        opName = nodeStr(asObj(strNode), "sval") || nodeStr(asObj(strNode), "str");
      } else if (typeof strNode === "string") {
        opName = strNode;
      } else {
        const sv = nodeStr(n, "sval") || nodeStr(n, "str");
        if (sv) opName = sv;
      }
    }

    const lexpr = nodeGet(node, "lexpr");
    const rexpr = nodeGet(node, "rexpr");

    // AEXPR_OP = 0, AEXPR_OP_ANY = 1, AEXPR_OP_ALL = 2,
    // AEXPR_DISTINCT = 3, AEXPR_NOT_DISTINCT = 4,
    // AEXPR_NULLIF = 5, AEXPR_IN = 6, AEXPR_LIKE = 7,
    // AEXPR_ILIKE = 8, AEXPR_SIMILAR = 9, AEXPR_BETWEEN = 10,
    // AEXPR_NOT_BETWEEN = 11, AEXPR_BETWEEN_SYM = 12,
    // AEXPR_NOT_BETWEEN_SYM = 13, AEXPR_PAREN = 14

    // BETWEEN: kind = 10
    if (kind === 10 || kind === 11) {
      const val = this.evaluate(asObj(lexpr), row);
      const rexprObj = asObj(rexpr);
      const listNode = nodeGet(rexprObj, "List");
      const itemsRaw =
        listNode !== null
          ? nodeGet(asObj(listNode), "items")
          : nodeGet(rexprObj, "items");
      const rangeItems = asList(itemsRaw ?? rexpr);
      if (rangeItems.length >= 2) {
        const low = this.evaluate(rangeItems[0]!, row);
        const high = this.evaluate(rangeItems[1]!, row);
        if (val === null || low === null || high === null) return null;
        const result =
          (val as number) >= (low as number) && (val as number) <= (high as number);
        return kind === 11 ? !result : result;
      }
      return null;
    }

    // LIKE / ILIKE: kind = 7 / 8
    if (kind === 7 || kind === 8) {
      const left = this.evaluate(asObj(lexpr), row);
      const right = this.evaluate(asObj(rexpr), row);
      if (left === null || right === null) return null;
      const re = likeToRegex(toStr(right), kind === 8);
      const result = re.test(toStr(left));
      // If the operator name contains "!~~" or "NOT", negate
      if (opName === "!~~" || opName === "!~~*") return !result;
      return result;
    }

    // SIMILAR TO: kind = 9
    if (kind === 9) {
      const left = this.evaluate(asObj(lexpr), row);
      const right = this.evaluate(asObj(rexpr), row);
      if (left === null || right === null) return null;
      // SIMILAR TO uses SQL regex syntax: % -> .*, _ -> ., | for alternation
      let pattern = toStr(right);
      pattern = pattern.replace(/%/g, ".*").replace(/_/g, ".");
      const re = new RegExp("^" + pattern + "$");
      return re.test(toStr(left));
    }

    // BETWEEN SYMMETRIC: kind = 12, NOT BETWEEN SYMMETRIC: kind = 13
    if (kind === 12 || kind === 13) {
      const val = this.evaluate(asObj(lexpr), row);
      const rexprObj2 = asObj(rexpr);
      const listNode2 = nodeGet(rexprObj2, "List");
      const itemsRaw2 =
        listNode2 !== null
          ? nodeGet(asObj(listNode2), "items")
          : nodeGet(rexprObj2, "items");
      const rangeItems = asList(itemsRaw2 ?? rexpr);
      if (rangeItems.length >= 2) {
        const a = this.evaluate(rangeItems[0]!, row);
        const b = this.evaluate(rangeItems[1]!, row);
        if (val === null || a === null || b === null) return null;
        const low = (a as number) < (b as number) ? a : b;
        const high = (a as number) >= (b as number) ? a : b;
        const result =
          (val as number) >= (low as number) && (val as number) <= (high as number);
        return kind === 13 ? !result : result;
      }
      return null;
    }

    // AEXPR_OP_ANY: kind = 1 (op ANY (array))
    if (kind === 1) {
      const left = this.evaluate(asObj(lexpr), row);
      const right = this.evaluate(asObj(rexpr), row);
      if (left === null || right === null) return null;
      const arr = Array.isArray(right) ? (right as unknown[]) : [right];
      return arr.some((v) => this._applyOp(opName, left, v) === true);
    }

    // AEXPR_OP_ALL: kind = 2 (op ALL (array))
    if (kind === 2) {
      const left = this.evaluate(asObj(lexpr), row);
      const right = this.evaluate(asObj(rexpr), row);
      if (left === null || right === null) return null;
      const arr = Array.isArray(right) ? (right as unknown[]) : [right];
      return arr.every((v) => this._applyOp(opName, left, v) === true);
    }

    // AEXPR_DISTINCT: kind = 3 (IS DISTINCT FROM)
    if (kind === 3) {
      const left = this.evaluate(asObj(lexpr), row);
      const right = this.evaluate(asObj(rexpr), row);
      if (left === null && right === null) return false;
      if (left === null || right === null) return true;
      return left !== right;
    }

    // AEXPR_NOT_DISTINCT: kind = 4 (IS NOT DISTINCT FROM)
    if (kind === 4) {
      const left = this.evaluate(asObj(lexpr), row);
      const right = this.evaluate(asObj(rexpr), row);
      if (left === null && right === null) return true;
      if (left === null || right === null) return false;
      return left === right;
    }

    // AEXPR_NULLIF: kind = 5
    if (kind === 5) {
      const left = this.evaluate(asObj(lexpr), row);
      const right = this.evaluate(asObj(rexpr), row);
      return left === right ? null : left;
    }

    // IN: kind = 6
    if (kind === 6) {
      const left = this.evaluate(asObj(lexpr), row);
      if (left === null) return null;
      // Unwrap List node if present
      const rexprObj3 = asObj(rexpr);
      const listNodeIn = nodeGet(rexprObj3, "List");
      const itemsIn =
        listNodeIn !== null
          ? asList(nodeGet(asObj(listNodeIn), "items"))
          : asList(rexpr);
      let inList: unknown[];
      if (itemsIn.length > 0) {
        inList = itemsIn.map((item) => this.evaluate(item, row));
      } else {
        const evaluated = this.evaluate(rexprObj3, row);
        inList = Array.isArray(evaluated) ? evaluated : [evaluated];
      }
      // Use loose equality for cross-type comparison (e.g., number vs string)
      return inList.some((v) => v == left);
    }

    // Standard binary operators (kind = 0)
    const left = lexpr !== null ? this.evaluate(asObj(lexpr), row) : null;
    const right = rexpr !== null ? this.evaluate(asObj(rexpr), row) : null;

    return this._applyOp(opName, left, right);
  }

  private _applyOp(op: string, left: unknown, right: unknown): unknown {
    // Arithmetic
    if (op === "+") {
      if (left === null || right === null) return null;
      if (typeof left === "string" || typeof right === "string") {
        return toStr(left) + toStr(right);
      }
      return (left as number) + (right as number);
    }
    if (op === "-") {
      if (left === null || right === null) return null;
      return (left as number) - (right as number);
    }
    if (op === "*") {
      if (left === null || right === null) return null;
      return (left as number) * (right as number);
    }
    if (op === "/") {
      if (left === null || right === null) return null;
      const r = right as number;
      if (r === 0) throw new Error("Division by zero");
      return (left as number) / r;
    }
    if (op === "%") {
      if (left === null || right === null) return null;
      return (left as number) % (right as number);
    }
    if (op === "^") {
      if (left === null || right === null) return null;
      return Math.pow(left as number, right as number);
    }

    // Comparison
    if (op === "=") {
      if (left === null || right === null) return null;
      return left === right;
    }
    if (op === "<>" || op === "!=") {
      if (left === null || right === null) return null;
      return left !== right;
    }
    if (op === "<") {
      if (left === null || right === null) return null;
      return (left as number) < (right as number);
    }
    if (op === ">") {
      if (left === null || right === null) return null;
      return (left as number) > (right as number);
    }
    if (op === "<=") {
      if (left === null || right === null) return null;
      return (left as number) <= (right as number);
    }
    if (op === ">=") {
      if (left === null || right === null) return null;
      return (left as number) >= (right as number);
    }

    // String operators
    if (op === "||") {
      if (left === null || right === null) return null;
      return toStr(left) + toStr(right);
    }
    if (op === "~~") {
      // LIKE
      if (left === null || right === null) return null;
      return likeToRegex(toStr(right), false).test(toStr(left));
    }
    if (op === "!~~") {
      // NOT LIKE
      if (left === null || right === null) return null;
      return !likeToRegex(toStr(right), false).test(toStr(left));
    }
    if (op === "~~*") {
      // ILIKE
      if (left === null || right === null) return null;
      return likeToRegex(toStr(right), true).test(toStr(left));
    }
    if (op === "!~~*") {
      // NOT ILIKE
      if (left === null || right === null) return null;
      return !likeToRegex(toStr(right), true).test(toStr(left));
    }

    // Regex match
    if (op === "~") {
      if (left === null || right === null) return null;
      return new RegExp(toStr(right)).test(toStr(left));
    }
    if (op === "~*") {
      if (left === null || right === null) return null;
      return new RegExp(toStr(right), "i").test(toStr(left));
    }
    if (op === "!~") {
      if (left === null || right === null) return null;
      return !new RegExp(toStr(right)).test(toStr(left));
    }
    if (op === "!~*") {
      if (left === null || right === null) return null;
      return !new RegExp(toStr(right), "i").test(toStr(left));
    }

    // Bitwise
    if (op === "&") {
      if (left === null || right === null) return null;
      return (left as number) & (right as number);
    }
    if (op === "|") {
      if (left === null || right === null) return null;
      return (left as number) | (right as number);
    }
    if (op === "#") {
      if (left === null || right === null) return null;
      return (left as number) ^ (right as number);
    }
    if (op === "<<") {
      if (left === null || right === null) return null;
      return (left as number) << (right as number);
    }
    if (op === ">>") {
      if (left === null || right === null) return null;
      return (left as number) >> (right as number);
    }

    // JSON operators
    if (op === "->") {
      // JSON field access (returns JSON)
      if (left === null || right === null) return null;
      const obj: unknown =
        typeof left === "string" ? (JSON.parse(left) as unknown) : left;
      if (typeof right === "number" && Array.isArray(obj)) {
        return (obj as unknown[])[right] ?? null;
      }
      if (typeof obj === "object" && obj !== null && !Array.isArray(obj)) {
        return (obj as Record<string, unknown>)[toStr(right)] ?? null;
      }
      return null;
    }
    if (op === "->>") {
      // JSON field access (returns text)
      if (left === null || right === null) return null;
      const obj: unknown =
        typeof left === "string" ? (JSON.parse(left) as unknown) : left;
      if (typeof right === "number" && Array.isArray(obj)) {
        const v = (obj as unknown[])[right];
        if (v === null || v === undefined) return null;
        return typeof v === "object" ? JSON.stringify(v) : toStr(v);
      }
      if (typeof obj === "object" && obj !== null && !Array.isArray(obj)) {
        const v = (obj as Record<string, unknown>)[toStr(right)];
        if (v === null || v === undefined) return null;
        return typeof v === "object" ? JSON.stringify(v) : toStr(v);
      }
      return null;
    }
    if (op === "#>") {
      // JSON path access (returns JSON)
      if (left === null || right === null) return null;
      let obj: unknown = typeof left === "string" ? JSON.parse(left) : left;
      const path = Array.isArray(right)
        ? (right as unknown[])
        : toStr(right)
            .replace(/^\{/, "")
            .replace(/\}$/, "")
            .split(",")
            .map((s) => s.trim());
      for (const key of path) {
        if (obj === null || obj === undefined) return null;
        if (Array.isArray(obj)) {
          obj = (obj as unknown[])[Number(key)];
        } else if (typeof obj === "object") {
          obj = (obj as Record<string, unknown>)[toStr(key)];
        } else {
          return null;
        }
      }
      return obj ?? null;
    }
    if (op === "#>>") {
      // JSON path access (returns text)
      if (left === null || right === null) return null;
      const result = this._applyOp("#>", left, right);
      if (result === null || result === undefined) return null;
      return typeof result === "object" ? JSON.stringify(result) : toStr(result);
    }
    if (op === "@>") {
      // JSON containment (left contains right)
      if (left === null || right === null) return null;
      const leftObj: unknown =
        typeof left === "string" ? (JSON.parse(left) as unknown) : left;
      const rightObj: unknown =
        typeof right === "string" ? (JSON.parse(right) as unknown) : right;
      return jsonContains(leftObj, rightObj);
    }
    if (op === "<@") {
      // JSON containment (right contains left)
      if (left === null || right === null) return null;
      const leftObj: unknown =
        typeof left === "string" ? (JSON.parse(left) as unknown) : left;
      const rightObj: unknown =
        typeof right === "string" ? (JSON.parse(right) as unknown) : right;
      return jsonContains(rightObj, leftObj);
    }
    if (op === "?") {
      // JSON key existence
      if (left === null || right === null) return null;
      const obj: unknown =
        typeof left === "string" ? (JSON.parse(left) as unknown) : left;
      if (typeof obj === "object" && obj !== null && !Array.isArray(obj)) {
        return toStr(right) in (obj as Record<string, unknown>);
      }
      if (Array.isArray(obj)) {
        return (obj as unknown[]).includes(toStr(right));
      }
      return false;
    }
    if (op === "?|") {
      // JSON key existence (any)
      if (left === null || right === null) return null;
      const obj: unknown =
        typeof left === "string" ? (JSON.parse(left) as unknown) : left;
      const keys = Array.isArray(right) ? (right as string[]) : [toStr(right)];
      if (typeof obj === "object" && obj !== null && !Array.isArray(obj)) {
        return keys.some((k) => k in (obj as Record<string, unknown>));
      }
      return false;
    }
    if (op === "?&") {
      // JSON key existence (all)
      if (left === null || right === null) return null;
      const obj: unknown =
        typeof left === "string" ? (JSON.parse(left) as unknown) : left;
      const keys = Array.isArray(right) ? (right as string[]) : [toStr(right)];
      if (typeof obj === "object" && obj !== null && !Array.isArray(obj)) {
        return keys.every((k) => k in (obj as Record<string, unknown>));
      }
      return false;
    }

    // Array operators
    if (op === "&&") {
      // Array overlap
      if (left === null || right === null) return null;
      if (Array.isArray(left) && Array.isArray(right)) {
        const rightSet = new Set(right as unknown[]);
        return (left as unknown[]).some((v) => rightSet.has(v));
      }
      return false;
    }

    // Full-text search operator
    if (op === "@@") {
      if (left === null || right === null) return false;
      const text = String(left as string | number).toLowerCase();
      const query = String(right as string | number);
      return ftsMatch(text, query);
    }

    throw new Error(`Unsupported operator: "${op}"`);
  }

  // -- BoolExpr -----------------------------------------------------------------

  private _evalBoolExpr(
    node: Record<string, unknown>,
    row: Record<string, unknown>,
  ): unknown {
    const boolop = nodeGet(node, "boolop");
    const args = asList(nodeGet(node, "args"));

    // BOOL_AND = 0 ("AND_EXPR"), BOOL_OR = 1 ("OR_EXPR"), BOOL_NOT = 2 ("NOT_EXPR")
    if (boolop === 0 || boolop === "BOOL_AND" || boolop === "AND_EXPR") {
      let result: unknown = true;
      for (const arg of args) {
        const v = this.evaluate(arg, row);
        if (v === false) return false;
        if (v === null) result = null;
      }
      return result;
    }

    if (boolop === 1 || boolop === "BOOL_OR" || boolop === "OR_EXPR") {
      let result: unknown = false;
      for (const arg of args) {
        const v = this.evaluate(arg, row);
        if (v === true) return true;
        if (v === null) result = null;
      }
      return result;
    }

    if (boolop === 2 || boolop === "BOOL_NOT" || boolop === "NOT_EXPR") {
      const v = this.evaluate(args[0]!, row);
      if (v === null) return null;
      return !v;
    }

    throw new Error(`Unknown BoolExpr boolop: ${toStr(boolop)}`);
  }

  // -- FuncCall -----------------------------------------------------------------

  private _evalFuncCall(
    node: Record<string, unknown>,
    row: Record<string, unknown>,
  ): unknown {
    // Extract function name
    const funcNameList = asList(nodeGet(node, "funcname"));
    const nameSegments: string[] = [];
    for (const seg of funcNameList) {
      const strNode = nodeGet(seg, "String") ?? nodeGet(seg, "str");
      if (strNode !== null && typeof strNode === "object") {
        const sval = nodeStr(asObj(strNode), "sval") || nodeStr(asObj(strNode), "str");
        if (sval) nameSegments.push(sval);
      } else if (typeof strNode === "string") {
        nameSegments.push(strNode);
      } else {
        const sv = nodeStr(seg, "sval") || nodeStr(seg, "str");
        if (sv) nameSegments.push(sv);
      }
    }
    let funcName = nameSegments.join(".").toLowerCase();
    // Strip pg_catalog schema prefix (PostgreSQL system catalog)
    if (funcName.startsWith("pg_catalog.")) {
      funcName = funcName.slice("pg_catalog.".length);
    }

    // Handle aggregate functions that were pre-computed for HAVING clause
    const AGG_NAMES = new Set([
      "count",
      "sum",
      "avg",
      "min",
      "max",
      "string_agg",
      "array_agg",
      "bool_and",
      "bool_or",
      "stddev",
      "stddev_pop",
      "stddev_samp",
      "variance",
      "var_pop",
      "var_samp",
      "corr",
      "covar_pop",
      "covar_samp",
      "regr_slope",
      "regr_intercept",
      "regr_r2",
      "regr_count",
      "regr_avgx",
      "regr_avgy",
      "regr_sxx",
      "regr_syy",
      "regr_sxy",
    ]);
    if (AGG_NAMES.has(funcName)) {
      // Check if the aggregate was pre-computed and stored in the row
      const isStar = nodeGet(node, "agg_star") === true;
      const argNodes = asList(nodeGet(node, "args"));
      let aggKey: string;
      if (isStar || argNodes.length === 0) {
        aggKey = `${funcName}(*)`;
      } else {
        try {
          const colRef = nodeGet(argNodes[0]!, "ColumnRef");
          if (colRef) {
            const fields = asList(nodeGet(asObj(colRef), "fields"));
            const colParts: string[] = [];
            for (const f of fields) {
              const s = nodeStr(f, "sval") || nodeStr(f, "str");
              if (s) colParts.push(s);
            }
            aggKey = `${funcName}(${colParts.join(".")})`;
          } else {
            aggKey = funcName;
          }
        } catch {
          aggKey = funcName;
        }
      }
      if (aggKey in row) {
        return row[aggKey];
      }
      // Also check for __having_agg_ synthetic key
      const syntheticKey = `__having_agg_${JSON.stringify({ FuncCall: node })}`;
      if (syntheticKey in row) {
        return row[syntheticKey];
      }
    }

    // Handle sequence functions
    if (funcName === "nextval" && this._sequences !== null) {
      const argNodes = asList(nodeGet(node, "args"));
      const seqName =
        argNodes.length > 0 ? toStr(this.evaluate(argNodes[0]!, row)) : "";
      const seq = this._sequences.get(seqName);
      if (seq) {
        seq.current += seq.increment;
        return seq.current;
      }
      throw new Error(`Sequence not found: "${seqName}"`);
    }
    if (funcName === "currval" && this._sequences !== null) {
      const argNodes = asList(nodeGet(node, "args"));
      const seqName =
        argNodes.length > 0 ? toStr(this.evaluate(argNodes[0]!, row)) : "";
      const seq = this._sequences.get(seqName);
      if (seq) {
        return seq.current;
      }
      throw new Error(`Sequence not found: "${seqName}"`);
    }
    if (funcName === "setval" && this._sequences !== null) {
      const argNodes = asList(nodeGet(node, "args"));
      if (argNodes.length < 2)
        throw new Error("setval() requires at least 2 arguments");
      const seqName = toStr(this.evaluate(argNodes[0]!, row));
      const newVal = Number(this.evaluate(argNodes[1]!, row));
      const seq = this._sequences.get(seqName);
      if (seq) {
        // If third argument is true (default), next nextval() returns newVal + increment
        // If false, next nextval() returns newVal
        const isCalled =
          argNodes.length >= 3 ? Boolean(this.evaluate(argNodes[2]!, row)) : true;
        seq.current = isCalled ? newVal : newVal - seq.increment;
        return newVal;
      }
      throw new Error(`Sequence not found: "${seqName}"`);
    }
    if (funcName === "lastval" && this._sequences !== null) {
      // Return the value most recently returned by nextval in the current session
      let latest: number | null = null;
      for (const seq of this._sequences.values()) {
        if (latest === null || seq.current > latest) {
          latest = seq.current;
        }
      }
      if (latest === null)
        throw new Error("lastval is not yet defined in this session");
      return latest;
    }

    // Handle special case: count(*) with agg_star
    if (nodeGet(node, "agg_star") === true) {
      // This is count(*); in row-level evaluation context, return 1
      return 1;
    }

    // Evaluate arguments
    const argNodes = asList(nodeGet(node, "args"));

    // Handle named arguments for make_interval
    if (funcName === "make_interval") {
      const hasNamedArgs = argNodes.some(
        (arg) =>
          nodeGet(arg, "NamedArgExpr") !== null &&
          nodeGet(arg, "NamedArgExpr") !== undefined,
      );
      if (hasNamedArgs) {
        const namedArgs: Record<string, number> = {
          years: 0,
          months: 0,
          weeks: 0,
          days: 0,
          hours: 0,
          mins: 0,
          secs: 0,
        };
        for (const arg of argNodes) {
          const na = nodeGet(arg, "NamedArgExpr");
          if (na !== null && na !== undefined) {
            const naObj = asObj(na);
            const argName = (nodeStr(naObj, "name") || "").toLowerCase();
            const argVal = nodeGet(naObj, "arg");
            const val =
              argVal !== null
                ? Number(this.evaluate(argVal as Record<string, unknown>, row))
                : 0;
            if (argName in namedArgs) {
              namedArgs[argName] = val;
            }
          }
        }
        return callScalarFunction(funcName, [
          namedArgs["years"],
          namedArgs["months"],
          namedArgs["weeks"],
          namedArgs["days"],
          namedArgs["hours"],
          namedArgs["mins"],
          namedArgs["secs"],
        ]);
      }
    }

    // uqa_highlight(field, query [, start_tag, end_tag, max_fragments, fragment_size])
    if (funcName === "uqa_highlight") {
      return this._evalUqaHighlight(argNodes, row);
    }

    const evaluatedArgs = argNodes.map((arg) => this.evaluate(arg, row));

    return callScalarFunction(funcName, evaluatedArgs);
  }

  private _evalUqaHighlight(
    argNodes: Record<string, unknown>[],
    row: Record<string, unknown>,
  ): unknown {
    if (argNodes.length < 2) {
      throw new Error("uqa_highlight() requires at least 2 arguments: field, query");
    }

    const text = this.evaluate(argNodes[0]!, row);
    const queryString = this.evaluate(argNodes[1]!, row);
    const startTag = argNodes.length > 2
      ? String(this.evaluate(argNodes[2]!, row))
      : "<b>";
    const endTag = argNodes.length > 3
      ? String(this.evaluate(argNodes[3]!, row))
      : "</b>";
    const maxFragments = argNodes.length > 4
      ? Number(this.evaluate(argNodes[4]!, row))
      : 0;
    const fragmentSize = argNodes.length > 5
      ? Number(this.evaluate(argNodes[5]!, row))
      : 150;

    if (text === null || text === undefined) {
      return null;
    }

    const textStr = String(text);
    const queryTerms = extractQueryTerms(String(queryString));
    if (queryTerms.length === 0) {
      return textStr;
    }

    return highlight(textStr, queryTerms, {
      startTag,
      endTag,
      maxFragments,
      fragmentSize,
      analyzer: this._analyzer,
    });
  }

  // -- NullTest -----------------------------------------------------------------

  private _evalNullTest(
    node: Record<string, unknown>,
    row: Record<string, unknown>,
  ): unknown {
    const arg = nodeGet(node, "arg");
    const nullTestType = nodeGet(node, "nulltesttype");
    const value = this.evaluate(asObj(arg), row);

    // IS_NULL = 0, IS_NOT_NULL = 1
    if (nullTestType === 0 || nullTestType === "IS_NULL") {
      return value === null || value === undefined;
    }
    return value !== null && value !== undefined;
  }

  // -- SQLValueFunction ---------------------------------------------------------

  private _evalSQLValueFunction(node: Record<string, unknown>): unknown {
    const op = nodeGet(node, "op");
    const opStr = typeof op === "string" ? op : "";
    if (opStr === "SVFOP_CURRENT_DATE") {
      const now = new Date();
      const y = String(now.getFullYear());
      const m = String(now.getMonth() + 1).padStart(2, "0");
      const d = String(now.getDate()).padStart(2, "0");
      return `${y}-${m}-${d}`;
    }
    if (opStr === "SVFOP_CURRENT_TIMESTAMP" || opStr === "SVFOP_CURRENT_TIMESTAMP_N") {
      return new Date().toISOString();
    }
    if (opStr === "SVFOP_CURRENT_TIME" || opStr === "SVFOP_CURRENT_TIME_N") {
      const now = new Date();
      const h = String(now.getHours()).padStart(2, "0");
      const min = String(now.getMinutes()).padStart(2, "0");
      const s = String(now.getSeconds()).padStart(2, "0");
      return `${h}:${min}:${s}`;
    }
    if (opStr === "SVFOP_LOCALTIME" || opStr === "SVFOP_LOCALTIME_N") {
      const now = new Date();
      const h = String(now.getHours()).padStart(2, "0");
      const min = String(now.getMinutes()).padStart(2, "0");
      const s = String(now.getSeconds()).padStart(2, "0");
      return `${h}:${min}:${s}`;
    }
    if (opStr === "SVFOP_LOCALTIMESTAMP" || opStr === "SVFOP_LOCALTIMESTAMP_N") {
      return new Date().toISOString();
    }
    if (opStr === "SVFOP_CURRENT_ROLE") return "current_user";
    if (opStr === "SVFOP_CURRENT_USER") return "current_user";
    if (opStr === "SVFOP_SESSION_USER") return "session_user";
    if (opStr === "SVFOP_USER") return "current_user";
    if (opStr === "SVFOP_CURRENT_CATALOG") return "uqa";
    if (opStr === "SVFOP_CURRENT_SCHEMA") return "public";
    throw new Error(`Unsupported SQLValueFunction op: ${opStr}`);
  }

  // -- CaseExpr -----------------------------------------------------------------

  private _evalCaseExpr(
    node: Record<string, unknown>,
    row: Record<string, unknown>,
  ): unknown {
    const argNode = nodeGet(node, "arg");
    const baseValue =
      argNode !== null && argNode !== undefined
        ? this.evaluate(asObj(argNode), row)
        : null;

    const whenClauses = asList(nodeGet(node, "args"));
    for (const whenClause of whenClauses) {
      const caseWhen = nodeGet(whenClause, "CaseWhen") ?? whenClause;
      const exprNode = nodeGet(asObj(caseWhen), "expr");
      const resultNode = nodeGet(asObj(caseWhen), "result");

      if (baseValue !== null) {
        // Simple CASE: compare base value to each WHEN value
        const whenValue = this.evaluate(asObj(exprNode), row);
        if (baseValue === whenValue) {
          return this.evaluate(asObj(resultNode), row);
        }
      } else {
        // Searched CASE: evaluate WHEN expression as boolean
        const condition = this.evaluate(asObj(exprNode), row);
        if (condition === true) {
          return this.evaluate(asObj(resultNode), row);
        }
      }
    }

    // ELSE clause
    const defresult = nodeGet(node, "defresult");
    if (defresult !== null && defresult !== undefined) {
      return this.evaluate(asObj(defresult), row);
    }

    return null;
  }

  // -- TypeCast -----------------------------------------------------------------

  private _evalTypeCast(
    node: Record<string, unknown>,
    row: Record<string, unknown>,
  ): unknown {
    const arg = nodeGet(node, "arg");
    const typeName = nodeGet(node, "typeName");

    const value = this.evaluate(asObj(arg), row);

    // Extract type name from TypeName node
    const typeNameNode = asObj(typeName);
    const names = asList(nodeGet(typeNameNode, "names"));
    let typeStr = "";
    for (const n of names) {
      const strNode = nodeGet(n, "String") ?? nodeGet(n, "str");
      if (strNode !== null && typeof strNode === "object") {
        const sval = nodeStr(asObj(strNode), "sval") || nodeStr(asObj(strNode), "str");
        if (sval && sval !== "pg_catalog") typeStr = sval;
      } else if (typeof strNode === "string" && strNode !== "pg_catalog") {
        typeStr = strNode;
      } else {
        const sv = nodeStr(n, "sval") || nodeStr(n, "str");
        if (sv && sv !== "pg_catalog") typeStr = sv;
      }
    }

    if (!typeStr) return value;
    return castValue(value, typeStr);
  }

  // -- ParamRef -----------------------------------------------------------------

  private _evalParamRef(node: Record<string, unknown>): unknown {
    const number = nodeGet(node, "number");
    const idx = typeof number === "number" ? number : 0;
    // ParamRef numbers are 1-based
    if (idx > 0 && idx <= this._params.length) {
      return this._params[idx - 1];
    }
    // Numbered param out of range -- throw
    if (idx > 0 && idx > this._params.length) {
      throw new Error(`No value supplied for parameter $${String(idx)}`);
    }
    // $0 or unnumbered -> use next available param (positional)
    return null;
  }

  // -- SubLink ------------------------------------------------------------------

  private _evalSubLink(
    node: Record<string, unknown>,
    row: Record<string, unknown>,
  ): unknown {
    // SubLink is a subquery in expression context (EXISTS, IN, scalar subquery)
    if (this._subqueryExecutor === null) {
      throw new Error(
        "SubLink (subquery) evaluation requires a subqueryExecutor callback",
      );
    }

    const linkType = nodeGet(node, "subLinkType") as number | string;
    const subselect = asObj(nodeGet(node, "subselect"));
    const selectStmt = asObj(nodeGet(subselect, "SelectStmt") ?? subselect);
    const innerResult = this._subqueryExecutor(selectStmt, row);

    // EXISTS_SUBLINK = 0
    if (linkType === 0 || linkType === "EXISTS_SUBLINK") {
      return innerResult.rows.length > 0;
    }

    // ALL_SUBLINK = 1
    if (linkType === 1 || linkType === "ALL_SUBLINK") {
      const testExpr = asObj(nodeGet(node, "testexpr"));
      const lhsValue = this.evaluate(testExpr, row);
      const operName = this._extractSubLinkOperator(node);
      const subCol = innerResult.columns[0]!;
      for (const subRow of innerResult.rows) {
        const rhsValue = subRow[subCol];
        if (!this._compareValues(lhsValue, operName, rhsValue)) return false;
      }
      return true;
    }

    // ANY_SUBLINK = 2 (IN subquery)
    if (linkType === 2 || linkType === "ANY_SUBLINK") {
      const testExpr = asObj(nodeGet(node, "testexpr"));
      const lhsValue = this.evaluate(testExpr, row);
      if (innerResult.columns.length === 0) return false;
      const subCol = innerResult.columns[0]!;
      const operName = this._extractSubLinkOperator(node);
      for (const subRow of innerResult.rows) {
        const rhsValue = subRow[subCol];
        if (this._compareValues(lhsValue, operName, rhsValue)) return true;
      }
      return false;
    }

    // EXPR_SUBLINK = 4 (scalar subquery)
    if (linkType === 4 || linkType === "EXPR_SUBLINK") {
      if (innerResult.rows.length === 0) return null;
      const subCol = innerResult.columns[0]!;
      return innerResult.rows[0]![subCol] ?? null;
    }

    throw new Error(`Unsupported SubLink type: ${String(linkType)}`);
  }

  private _extractSubLinkOperator(node: Record<string, unknown>): string {
    const operNameList = asList(nodeGet(node, "operName"));
    if (operNameList.length > 0) {
      const item = operNameList[0]!;
      const str = nodeGet(item, "str");
      if (str !== null && str !== undefined) return String(str as string | number);
      const sval = nodeGet(asObj(nodeGet(item, "String") ?? item), "sval");
      if (sval !== null && sval !== undefined) return String(sval as string | number);
    }
    return "=";
  }

  private _compareValues(lhs: unknown, op: string, rhs: unknown): boolean {
    switch (op) {
      case "=":
        return lhs == rhs;
      case "<>":
      case "!=":
        return lhs != rhs;
      case "<":
        return (lhs as number) < (rhs as number);
      case "<=":
        return (lhs as number) <= (rhs as number);
      case ">":
        return (lhs as number) > (rhs as number);
      case ">=":
        return (lhs as number) >= (rhs as number);
      default:
        return lhs === rhs;
    }
  }

  // -- CoalesceExpr -------------------------------------------------------------

  private _evalCoalesceExpr(
    node: Record<string, unknown>,
    row: Record<string, unknown>,
  ): unknown {
    const args = asList(nodeGet(node, "args"));
    for (const arg of args) {
      const value = this.evaluate(arg, row);
      if (value !== null && value !== undefined) return value;
    }
    return null;
  }

  // -- MinMaxExpr ---------------------------------------------------------------

  private _evalMinMaxExpr(
    node: Record<string, unknown>,
    row: Record<string, unknown>,
  ): unknown {
    const op = nodeGet(node, "op");
    const args = asList(nodeGet(node, "args"));
    const values = args
      .map((arg) => this.evaluate(arg, row))
      .filter((v) => v !== null && v !== undefined);

    if (values.length === 0) return null;

    // IS_GREATEST = 0, IS_LEAST = 1
    if (op === 0 || op === "IS_GREATEST") {
      return values.reduce((a, b) => ((a as number) > (b as number) ? a : b));
    }
    return values.reduce((a, b) => ((a as number) < (b as number) ? a : b));
  }

  // -- BooleanTest --------------------------------------------------------------

  private _evalBooleanTest(
    node: Record<string, unknown>,
    row: Record<string, unknown>,
  ): unknown {
    const arg = nodeGet(node, "arg");
    const booltesttype = nodeGet(node, "booltesttype");
    const value = this.evaluate(asObj(arg), row);

    // IS_TRUE = 0, IS_NOT_TRUE = 1, IS_FALSE = 2, IS_NOT_FALSE = 3,
    // IS_UNKNOWN = 4, IS_NOT_UNKNOWN = 5
    switch (booltesttype) {
      case 0:
      case "IS_TRUE":
        return value === true;
      case 1:
      case "IS_NOT_TRUE":
        return value !== true;
      case 2:
      case "IS_FALSE":
        return value === false;
      case 3:
      case "IS_NOT_FALSE":
        return value !== false;
      case 4:
      case "IS_UNKNOWN":
        return value === null || value === undefined;
      case 5:
      case "IS_NOT_UNKNOWN":
        return value !== null && value !== undefined;
      default:
        throw new Error(`Unknown BooleanTest type: ${toStr(booltesttype)}`);
    }
  }
}
