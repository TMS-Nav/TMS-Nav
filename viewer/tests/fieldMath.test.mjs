import test from "node:test";
import assert from "node:assert/strict";
import { difference, distance, intensityScale, maxAbs, symmetricTop } from "../src/fieldMath.mjs";

test("difference is vertex by vertex", () => {
  assert.deepEqual([...difference(new Float32Array([3, 1]), new Float32Array([1, 2]))], [2, -1]);
});

test("the symmetric top ignores the sign and one outlier", () => {
  const v = new Float32Array(1001).fill(1);
  v[0] = -50;
  assert.equal(symmetricTop(v), 1);
  assert.equal(maxAbs(v), 50);
  assert.equal(symmetricTop(new Float32Array(0)), 0);
});

test("the field scales linearly with the coil current", () => {
  assert.equal(intensityScale(1), 1);
  assert.equal(intensityScale(80), 80);
  assert.equal(distance([0, 0, 0], [3, 4, 0]), 5);
});
