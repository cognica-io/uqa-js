import { describe, expect, it } from "vitest";
import { haversineDistance, SpatialIndex } from "../../src/storage/spatial-index.js";

// -- Known coordinates -------------------------------------------------------
// NYC:    (40.7128, -74.0060)  -- (lat, lon) -> POINT(-74.0060, 40.7128)
// LA:     (34.0522, -118.2437) -> POINT(-118.2437, 34.0522)
// London: (51.5074, -0.1278)   -> POINT(-0.1278, 51.5074)
// Tokyo:  (35.6762, 139.6503)  -> POINT(139.6503, 35.6762)

const NYC_LON = -74.006;
const NYC_LAT = 40.7128;
const LA_LON = -118.2437;
const LA_LAT = 34.0522;
const LONDON_LON = -0.1278;
const LONDON_LAT = 51.5074;
const TOKYO_LON = 139.6503;
const TOKYO_LAT = 35.6762;

// ==========================================================================
// Unit tests: Haversine distance
// ==========================================================================

describe("Haversine distance", () => {
  it("same point is 0", () => {
    expect(haversineDistance(NYC_LAT, NYC_LON, NYC_LAT, NYC_LON)).toBe(0.0);
  });

  it("NYC to LA (~3940 km)", () => {
    const dist = haversineDistance(NYC_LAT, NYC_LON, LA_LAT, LA_LON);
    expect(dist).toBeGreaterThan(3_900_000);
    expect(dist).toBeLessThan(4_000_000);
  });

  it("NYC to London (~5570 km)", () => {
    const dist = haversineDistance(NYC_LAT, NYC_LON, LONDON_LAT, LONDON_LON);
    expect(dist).toBeGreaterThan(5_500_000);
    expect(dist).toBeLessThan(5_650_000);
  });

  it("is symmetric", () => {
    const d1 = haversineDistance(NYC_LAT, NYC_LON, TOKYO_LAT, TOKYO_LON);
    const d2 = haversineDistance(TOKYO_LAT, TOKYO_LON, NYC_LAT, NYC_LON);
    expect(Math.abs(d1 - d2)).toBeLessThan(0.01);
  });

  it("NYC to Tokyo (~10800 km)", () => {
    const dist = haversineDistance(NYC_LAT, NYC_LON, TOKYO_LAT, TOKYO_LON);
    expect(dist).toBeGreaterThan(10_500_000);
    expect(dist).toBeLessThan(11_000_000);
  });

  it("equator to equator 90 degrees (~10008 km)", () => {
    const dist = haversineDistance(0, 0, 0, 90);
    expect(dist / 1000).toBeCloseTo(10008, -2);
  });
});

// ==========================================================================
// Unit tests: SpatialIndex
// ==========================================================================

describe("SpatialIndex", () => {
  it("add and search within", () => {
    const idx = new SpatialIndex("test", "loc");
    idx.add(1, NYC_LON, NYC_LAT);
    idx.add(2, LA_LON, LA_LAT);
    idx.add(3, LONDON_LON, LONDON_LAT);

    // Search within 100km of NYC -- should only find NYC
    const pl = idx.searchWithin(NYC_LON, NYC_LAT, 100_000);
    const ids = [...pl.docIds];
    expect(ids).toContain(1);
    expect(ids).not.toContain(2);
    expect(ids).not.toContain(3);
    // Proximity score near 1.0 for self
    const entry = pl.getEntry(1);
    expect(entry).not.toBeNull();
    expect(entry!.payload.score).toBeGreaterThan(0.999);
  });

  it("search large radius", () => {
    const idx = new SpatialIndex("test", "loc");
    idx.add(1, NYC_LON, NYC_LAT);
    idx.add(2, LA_LON, LA_LAT);
    idx.add(3, LONDON_LON, LONDON_LAT);
    idx.add(4, TOKYO_LON, TOKYO_LAT);

    // 6000km radius from NYC
    const pl = idx.searchWithin(NYC_LON, NYC_LAT, 6_000_000);
    const ids = [...pl.docIds];
    expect(ids).toContain(1); // NYC
    expect(ids).toContain(2); // LA
    expect(ids).toContain(3); // London
    expect(ids).not.toContain(4); // Tokyo ~10800km
  });

  it("empty index", () => {
    const idx = new SpatialIndex("test", "loc");
    const pl = idx.searchWithin(0, 0, 1000);
    expect(pl.length).toBe(0);
  });

  it("zero distance returns empty", () => {
    const idx = new SpatialIndex("test", "loc");
    idx.add(1, NYC_LON, NYC_LAT);
    const pl = idx.searchWithin(NYC_LON, NYC_LAT, 0);
    expect(pl.length).toBe(0);
  });

  it("delete removes point", () => {
    const idx = new SpatialIndex("test", "loc");
    idx.add(1, NYC_LON, NYC_LAT);
    idx.add(2, NYC_LON + 0.001, NYC_LAT + 0.001);
    expect(idx.count()).toBe(2);

    idx.delete(1);
    expect(idx.count()).toBe(1);

    const pl = idx.searchWithin(NYC_LON, NYC_LAT, 1_000_000);
    const ids = [...pl.docIds];
    expect(ids).not.toContain(1);
    expect(ids).toContain(2);
  });

  it("clear removes all", () => {
    const idx = new SpatialIndex("test", "loc");
    idx.add(1, NYC_LON, NYC_LAT);
    idx.add(2, LA_LON, LA_LAT);
    expect(idx.count()).toBe(2);

    idx.clear();
    expect(idx.count()).toBe(0);
  });

  it("proximity score ordering", () => {
    const idx = new SpatialIndex("test", "loc");
    // Two points: one at center, one slightly away
    idx.add(1, NYC_LON, NYC_LAT);
    idx.add(2, NYC_LON + 0.01, NYC_LAT + 0.01); // ~1.4km away

    const pl = idx.searchWithin(NYC_LON, NYC_LAT, 100_000);
    const scores: Record<number, number> = {};
    for (const entry of pl) {
      scores[entry.docId] = entry.payload.score;
    }
    // doc_id=1 is at center, should have score very close to 1.0
    expect(scores[1]).toBeGreaterThan(0.999);
    // doc_id=2 is farther, should have lower score
    expect(scores[2]!).toBeLessThan(scores[1]!);
    expect(scores[2]).toBeGreaterThan(0.0);
  });

  it("search results sorted by doc id", () => {
    const idx = new SpatialIndex("test", "loc");
    idx.add(5, NYC_LON, NYC_LAT);
    idx.add(3, NYC_LON + 0.001, NYC_LAT + 0.001);
    idx.add(1, NYC_LON + 0.002, NYC_LAT + 0.002);

    const pl = idx.searchWithin(NYC_LON, NYC_LAT, 1_000_000);
    const docIds = pl.entries.map((e) => e.docId);
    // PostingList sorts by docId
    expect(docIds).toEqual([1, 3, 5]);
  });

  it("search with medium radius", () => {
    const idx = new SpatialIndex("test", "loc");
    idx.add(1, NYC_LON, NYC_LAT);
    idx.add(2, LA_LON, LA_LAT);

    // 5000km covers NYC to LA (~3940km)
    const pl = idx.searchWithin(NYC_LON, NYC_LAT, 5_000_000);
    expect(pl.length).toBe(2);
  });

  it("multiple searches on same index", () => {
    const idx = new SpatialIndex("test", "loc");
    idx.add(1, NYC_LON, NYC_LAT);
    idx.add(2, LA_LON, LA_LAT);
    idx.add(3, LONDON_LON, LONDON_LAT);

    // Search from NYC
    const pl1 = idx.searchWithin(NYC_LON, NYC_LAT, 100_000);
    expect([...pl1.docIds]).toContain(1);
    expect(pl1.length).toBe(1);

    // Search from LA
    const pl2 = idx.searchWithin(LA_LON, LA_LAT, 100_000);
    expect([...pl2.docIds]).toContain(2);
    expect(pl2.length).toBe(1);

    // Search from London
    const pl3 = idx.searchWithin(LONDON_LON, LONDON_LAT, 100_000);
    expect([...pl3.docIds]).toContain(3);
    expect(pl3.length).toBe(1);
  });

  it("negative distance returns empty", () => {
    const idx = new SpatialIndex("test", "loc");
    idx.add(1, 0, 0);
    expect(idx.searchWithin(0, 0, -100).length).toBe(0);
  });

  it("large number of points", () => {
    const idx = new SpatialIndex("test", "loc");
    // Add 100 points in a small area
    for (let i = 0; i < 100; i++) {
      idx.add(i, NYC_LON + i * 0.001, NYC_LAT + i * 0.001);
    }
    expect(idx.count()).toBe(100);

    // Search in a 50km radius (should find many but not all)
    const pl = idx.searchWithin(NYC_LON, NYC_LAT, 50_000);
    expect(pl.length).toBeGreaterThan(0);
    expect(pl.length).toBeLessThanOrEqual(100);
  });

  it("delete and re-add same id", () => {
    const idx = new SpatialIndex("test", "loc");
    idx.add(1, NYC_LON, NYC_LAT);
    idx.delete(1);
    idx.add(1, LA_LON, LA_LAT);

    // Should find at LA, not NYC
    const pl1 = idx.searchWithin(NYC_LON, NYC_LAT, 100_000);
    expect([...pl1.docIds]).not.toContain(1);

    const pl2 = idx.searchWithin(LA_LON, LA_LAT, 100_000);
    expect([...pl2.docIds]).toContain(1);
  });

  it("count reflects additions and deletions", () => {
    const idx = new SpatialIndex("test", "loc");
    expect(idx.count()).toBe(0);

    idx.add(1, 0, 0);
    expect(idx.count()).toBe(1);

    idx.add(2, 1, 1);
    expect(idx.count()).toBe(2);

    idx.delete(1);
    expect(idx.count()).toBe(1);

    idx.clear();
    expect(idx.count()).toBe(0);
  });

  it("tableName and fieldName are accessible", () => {
    const idx = new SpatialIndex("places", "location");
    expect(idx.tableName).toBe("places");
    expect(idx.fieldName).toBe("location");
  });
});
