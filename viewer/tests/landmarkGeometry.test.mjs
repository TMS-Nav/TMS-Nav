import test from "node:test";
import assert from "node:assert/strict";
import { BufferGeometry, Float32BufferAttribute, Vector3 } from "three";
import { closestScalpPoint, surfaceStep } from "../src/landmarkGeometry.mjs";

function plane(indexed) {
  const geometry = new BufferGeometry();
  geometry.setAttribute("position", new Float32BufferAttribute([
    -10, -10, 0, 10, -10, 0, 10, 10, 0,
    -10, -10, 0, 10, 10, 0, -10, 10, 0,
  ], 3));
  if (indexed) geometry.setIndex([0, 1, 2, 3, 4, 5]);
  return geometry;
}
for (const indexed of [false, true]) {
  test("projection onto triangle interiors, indexed=" + indexed, () => {
    const result = closestScalpPoint(plane(indexed), new Vector3(1, 2, 5));
    assert.ok(result.point.distanceTo(new Vector3(1, 2, 0)) < 1e-6);
    assert.ok(result.normal.distanceTo(new Vector3(0, 0, 1)) < 1e-6);
  });
}
test("surface step removes the normal component and stays on mesh", () => {
  const result = surfaceStep(plane(false), new Vector3(0, 0, 2), new Vector3(1, 0, 1), 2);
  assert.ok(result.point.distanceTo(new Vector3(2, 0, 0)) < 1e-6);
});
test("normal-only step stays at the contact", () => {
  const result = surfaceStep(plane(false), new Vector3(0, 0, 2), new Vector3(0, 0, 1), 2);
  assert.ok(result.point.length() < 1e-6);
});
test("projection outside the patch clamps to its boundary", () => {
  const result = closestScalpPoint(plane(false), new Vector3(20, 0, 2));
  assert.ok(result.point.distanceTo(new Vector3(10, 0, 0)) < 1e-6);
});
