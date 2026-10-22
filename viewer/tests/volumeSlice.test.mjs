import test from "node:test";
import assert from "node:assert/strict";
import { invert4, pixelToRas, sampleSlice, sliceGrid, volumeBounds } from "../src/volumeSlice.mjs";

// a 3 x 4 x 5 volume whose every voxel holds its own flat index, x fastest
const dims = [3, 4, 5];
const vol = Uint8Array.from({ length: 60 }, (_, n) => n);
const at = (i, j, k) => i + 3 * (j + 4 * k);
const identity = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]];

test("invert4 undoes a scaled, shifted, permuted affine", () => {
  const a = [[0, 0, 2, -10], [-2, 0, 0, 30], [0, 2, 0, -5], [0, 0, 0, 1]];
  const inv = invert4(a);
  // voxel (1, 2, 3) -> RAS (6-10, -2+30, 4-5) and back
  const ras = [-4, 28, -1];
  const ijk = [0, 1, 2].map((r) => inv[r * 4] * ras[0] + inv[r * 4 + 1] * ras[1] + inv[r * 4 + 2] * ras[2] + inv[r * 4 + 3]);
  assert.deepEqual(ijk.map((v) => Math.round(v * 1e9) / 1e9), [1, 2, 3]);
});

test("bounds cover the voxel centres", () => {
  assert.deepEqual(volumeBounds(dims, identity), [[0, 0, 0], [2, 3, 4]]);
});

test("an axial slice at z = 2 is voxel layer k = 2, right of the head on the left", () => {
  const grid = sliceGrid(volumeBounds(dims, identity), "axial", 2);
  assert.equal(grid.width, 3);
  assert.equal(grid.height, 4);
  const img = sampleSlice(vol, dims, invert4(identity), grid);
  // top left pixel is the most right (x = 2) and most anterior (y = 3) voxel
  assert.equal(img[0], at(2, 3, 2));
  assert.equal(img[grid.width - 1], at(0, 3, 2));
  assert.equal(img[(grid.height - 1) * grid.width], at(2, 0, 2));
});

test("coronal and sagittal pick the right layers", () => {
  const b = volumeBounds(dims, identity);
  const cor = sliceGrid(b, "coronal", 1);
  const imgC = sampleSlice(vol, dims, invert4(identity), cor);
  assert.equal(cor.width, 3);
  assert.equal(cor.height, 5);
  assert.equal(imgC[0], at(2, 1, 4));
  const sag = sliceGrid(b, "sagittal", 0);
  const imgS = sampleSlice(vol, dims, invert4(identity), sag);
  assert.equal(sag.width, 4);
  assert.equal(sag.height, 5);
  // seen from the right, posterior (y = 0) on the left, top row is the highest z
  assert.equal(imgS[0], at(0, 0, 4));
  assert.equal(imgS[sag.width - 1], at(0, 3, 4));
  assert.deepEqual(pixelToRas(sag, 0, 0), [0, 0, 4]);
});

test("outside the grid is zero", () => {
  const grid = sliceGrid(volumeBounds(dims, identity), "axial", 40);
  const img = sampleSlice(vol, dims, invert4(identity), grid);
  assert.ok(img.every((v) => v === 0));
});
