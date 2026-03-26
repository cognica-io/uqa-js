//
// Unified Query Algebra
//
// Copyright (c) 2023-2026 Cognica, Inc.
//
// Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0).
// See LICENSE file in the project root for full license text.
//

// Unified Query Algebra -- SpatialIndex
// 1:1 port of uqa/storage/spatial_index.py
// Browser version: in-memory brute force with haversine distance
// (Python version uses SQLite R*Tree; we use Map + bounding box filter)

import type { DocId } from "../core/types.js";
import { createPayload } from "../core/types.js";
import { PostingList } from "../core/posting-list.js";

const EARTH_RADIUS_M = 6_371_000.0; // WGS-84 mean radius
const METERS_PER_DEG_LAT = 111_320.0;

function toRadians(deg: number): number {
  return (deg * Math.PI) / 180;
}

function toDegrees(rad: number): number {
  return (rad * 180) / Math.PI;
}

export function haversineDistance(
  lat1: number,
  lon1: number,
  lat2: number,
  lon2: number,
): number {
  const lat1Rad = toRadians(lat1);
  const lat2Rad = toRadians(lat2);
  const dlat = toRadians(lat2 - lat1);
  const dlon = toRadians(lon2 - lon1);
  const a =
    Math.sin(dlat / 2) ** 2 +
    Math.cos(lat1Rad) * Math.cos(lat2Rad) * Math.sin(dlon / 2) ** 2;
  return 2 * EARTH_RADIUS_M * Math.asin(Math.sqrt(a));
}

interface Point {
  readonly x: number; // longitude
  readonly y: number; // latitude
}

export class SpatialIndex {
  private readonly _tableName: string;
  private readonly _fieldName: string;
  private _points: Map<DocId, Point>;

  constructor(tableName: string, fieldName: string) {
    this._tableName = tableName;
    this._fieldName = fieldName;
    this._points = new Map();
  }

  get tableName(): string {
    return this._tableName;
  }

  get fieldName(): string {
    return this._fieldName;
  }

  add(docId: DocId, x: number, y: number): void {
    this._points.set(docId, { x, y });
  }

  delete(docId: DocId): void {
    this._points.delete(docId);
  }

  clear(): void {
    this._points.clear();
  }

  searchWithin(cx: number, cy: number, distanceM: number): PostingList {
    if (distanceM <= 0) return new PostingList();

    // Bounding box filter (coarse)
    const deltaLat = distanceM / METERS_PER_DEG_LAT;
    const angularDist = distanceM / EARTH_RADIUS_M;
    const cosLat = Math.cos(toRadians(cy));

    let deltaLon: number;
    if (cosLat < 1e-10 || angularDist >= Math.PI) {
      deltaLon = 180.0;
    } else {
      const sinRatio = Math.sin(angularDist) / cosLat;
      if (sinRatio >= 1.0) {
        deltaLon = 180.0;
      } else {
        deltaLon = toDegrees(Math.asin(sinRatio));
      }
    }

    const minX = cx - deltaLon;
    const maxX = cx + deltaLon;
    const minY = cy - deltaLat;
    const maxY = cy + deltaLat;

    // Fine filter with haversine
    const entries: {
      docId: DocId;
      payload: {
        positions: readonly number[];
        score: number;
        fields: Readonly<Record<string, unknown>>;
      };
    }[] = [];

    for (const [docId, pt] of this._points) {
      // Bounding box check
      if (pt.x < minX || pt.x > maxX || pt.y < minY || pt.y > maxY) continue;
      // Haversine check
      const dist = haversineDistance(cy, cx, pt.y, pt.x);
      if (dist <= distanceM) {
        const score = 1.0 - dist / distanceM;
        entries.push({ docId, payload: createPayload({ score }) });
      }
    }

    entries.sort((a, b) => a.docId - b.docId);
    return PostingList.fromSorted(entries);
  }

  count(): number {
    return this._points.size;
  }

  close(): void {
    // Release all point data (no private connection to close in browser).
    this._points.clear();
  }
}
