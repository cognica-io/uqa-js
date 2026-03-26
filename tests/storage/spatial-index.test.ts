import { describe, expect, it } from "vitest";
import { haversineDistance, SpatialIndex } from "../../src/storage/spatial-index.js";

describe("haversineDistance", () => {
  it("same point is 0", () => {
    expect(haversineDistance(40.7128, -74.006, 40.7128, -74.006)).toBeCloseTo(0, 5);
  });

  it("known distance: NYC to LA (~3944 km)", () => {
    const dist = haversineDistance(40.7128, -74.006, 34.0522, -118.2437);
    expect(dist / 1000).toBeCloseTo(3944, -2); // within 100 km
  });

  it("equator to north pole (~10008 km)", () => {
    const dist = haversineDistance(0, 0, 90, 0);
    expect(dist / 1000).toBeCloseTo(10008, -2);
  });
});

describe("SpatialIndex", () => {
  it("searchWithin finds nearby points", () => {
    const idx = new SpatialIndex("places", "location");
    // NYC area points
    idx.add(1, -74.006, 40.7128); // NYC
    idx.add(2, -73.9857, 40.7484); // Midtown (about 4 km away)
    idx.add(3, -118.2437, 34.0522); // LA (far away)

    // Search within 10 km of NYC
    const result = idx.searchWithin(-74.006, 40.7128, 10000);
    const ids = [...result.docIds];
    expect(ids).toContain(1);
    expect(ids).toContain(2);
    expect(ids).not.toContain(3);
  });

  it("scores inversely proportional to distance", () => {
    const idx = new SpatialIndex("places", "location");
    idx.add(1, 0.0, 0.0); // origin
    idx.add(2, 0.001, 0.0); // very close

    const result = idx.searchWithin(0.0, 0.0, 1000);
    // doc 1 is at distance 0 -> score = 1.0
    const entry1 = result.getEntry(1);
    expect(entry1).not.toBeNull();
    expect(entry1!.payload.score).toBeCloseTo(1.0, 5);

    // doc 2 is further -> score < 1.0
    const entry2 = result.getEntry(2);
    if (entry2) {
      expect(entry2.payload.score).toBeLessThan(1.0);
      expect(entry2.payload.score).toBeGreaterThan(0);
    }
  });

  it("zero distance returns empty", () => {
    const idx = new SpatialIndex("t", "f");
    idx.add(1, 0, 0);
    expect(idx.searchWithin(0, 0, 0).length).toBe(0);
  });

  it("negative distance returns empty", () => {
    const idx = new SpatialIndex("t", "f");
    idx.add(1, 0, 0);
    expect(idx.searchWithin(0, 0, -100).length).toBe(0);
  });

  it("delete removes point", () => {
    const idx = new SpatialIndex("t", "f");
    idx.add(1, 0, 0);
    idx.delete(1);
    expect(idx.count()).toBe(0);
  });

  it("clear removes all", () => {
    const idx = new SpatialIndex("t", "f");
    idx.add(1, 0, 0);
    idx.add(2, 1, 1);
    idx.clear();
    expect(idx.count()).toBe(0);
  });
});
