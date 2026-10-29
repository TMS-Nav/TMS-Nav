// one flat slice out of the aim 2 volumes, for the 2d view in the side panel. the
// 3d plane does the same lookup on the gpu, this is the cpu copy so the panel does
// not need a second webgl context. pure, no three.js, so node --test can load it.
//
// a volume is a Uint8Array laid out x fastest (efield/export_web.py to_bytes), dims
// [nx, ny, nz], with a 4x4 voxel to RAS affine as nested rows. slices are taken in
// RAS, not in voxel axes, so axial is always a cut at fixed z (superior) whatever
// way the scan was acquired

// for each axis: which RAS coordinate is fixed, which runs across and which runs up.
// axial and coronal are drawn the radiological way, as if facing the subject, so the
// head's right is on the left of the picture (mirror true). sagittal is seen from
// the subject's right, front of the head on the right. labels are left, right, top
// and bottom edge
export const AXES = {
  axial: { fixed: 2, across: 0, up: 1, mirror: true, labels: ["R", "L", "A", "P"], name: "z" },
  coronal: { fixed: 1, across: 0, up: 2, mirror: true, labels: ["R", "L", "S", "I"], name: "y" },
  sagittal: { fixed: 0, across: 1, up: 2, mirror: false, labels: ["P", "A", "S", "I"], name: "x" },
};

/** row major 4x4 inverse, for the RAS to voxel matrix. */
export function invert4(m) {
  const a = m.flat ? m.flat() : [...m];
  const inv = new Array(16);
  inv[0] = a[5] * a[10] * a[15] - a[5] * a[11] * a[14] - a[9] * a[6] * a[15] + a[9] * a[7] * a[14] + a[13] * a[6] * a[11] - a[13] * a[7] * a[10];
  inv[4] = -a[4] * a[10] * a[15] + a[4] * a[11] * a[14] + a[8] * a[6] * a[15] - a[8] * a[7] * a[14] - a[12] * a[6] * a[11] + a[12] * a[7] * a[10];
  inv[8] = a[4] * a[9] * a[15] - a[4] * a[11] * a[13] - a[8] * a[5] * a[15] + a[8] * a[7] * a[13] + a[12] * a[5] * a[11] - a[12] * a[7] * a[9];
  inv[12] = -a[4] * a[9] * a[14] + a[4] * a[10] * a[13] + a[8] * a[5] * a[14] - a[8] * a[6] * a[13] - a[12] * a[5] * a[10] + a[12] * a[6] * a[9];
  inv[1] = -a[1] * a[10] * a[15] + a[1] * a[11] * a[14] + a[9] * a[2] * a[15] - a[9] * a[3] * a[14] - a[13] * a[2] * a[11] + a[13] * a[3] * a[10];
  inv[5] = a[0] * a[10] * a[15] - a[0] * a[11] * a[14] - a[8] * a[2] * a[15] + a[8] * a[3] * a[14] + a[12] * a[2] * a[11] - a[12] * a[3] * a[10];
  inv[9] = -a[0] * a[9] * a[15] + a[0] * a[11] * a[13] + a[8] * a[1] * a[15] - a[8] * a[3] * a[13] - a[12] * a[1] * a[11] + a[12] * a[3] * a[9];
  inv[13] = a[0] * a[9] * a[14] - a[0] * a[10] * a[13] - a[8] * a[1] * a[14] + a[8] * a[2] * a[13] + a[12] * a[1] * a[10] - a[12] * a[2] * a[9];
  inv[2] = a[1] * a[6] * a[15] - a[1] * a[7] * a[14] - a[5] * a[2] * a[15] + a[5] * a[3] * a[14] + a[13] * a[2] * a[7] - a[13] * a[3] * a[6];
  inv[6] = -a[0] * a[6] * a[15] + a[0] * a[7] * a[14] + a[4] * a[2] * a[15] - a[4] * a[3] * a[14] - a[12] * a[2] * a[7] + a[12] * a[3] * a[6];
  inv[10] = a[0] * a[5] * a[15] - a[0] * a[7] * a[13] - a[4] * a[1] * a[15] + a[4] * a[3] * a[13] + a[12] * a[1] * a[7] - a[12] * a[3] * a[5];
  inv[14] = -a[0] * a[5] * a[14] + a[0] * a[6] * a[13] + a[4] * a[1] * a[14] - a[4] * a[2] * a[13] - a[12] * a[1] * a[6] + a[12] * a[2] * a[5];
  inv[3] = -a[1] * a[6] * a[11] + a[1] * a[7] * a[10] + a[5] * a[2] * a[11] - a[5] * a[3] * a[10] - a[9] * a[2] * a[7] + a[9] * a[3] * a[6];
  inv[7] = a[0] * a[6] * a[11] - a[0] * a[7] * a[10] - a[4] * a[2] * a[11] + a[4] * a[3] * a[10] + a[8] * a[2] * a[7] - a[8] * a[3] * a[6];
  inv[11] = -a[0] * a[5] * a[11] + a[0] * a[7] * a[9] + a[4] * a[1] * a[11] - a[4] * a[3] * a[9] - a[8] * a[1] * a[7] + a[8] * a[3] * a[5];
  inv[15] = a[0] * a[5] * a[10] - a[0] * a[6] * a[9] - a[4] * a[1] * a[10] + a[4] * a[2] * a[9] + a[8] * a[1] * a[6] - a[8] * a[2] * a[5];
  const det = a[0] * inv[0] + a[1] * inv[4] + a[2] * inv[8] + a[3] * inv[12];
  if (Math.abs(det) < 1e-12) throw new Error("affine is not invertible");
  return inv.map((v) => v / det);
}

/** RAS bounding box of the whole grid, [[xmin, ymin, zmin], [xmax, ymax, zmax]]. */
export function volumeBounds(dims, affine) {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  for (const i of [0, dims[0] - 1]) for (const j of [0, dims[1] - 1]) for (const k of [0, dims[2] - 1]) {
    for (let r = 0; r < 3; r++) {
      const v = affine[r][0] * i + affine[r][1] * j + affine[r][2] * k + affine[r][3];
      lo[r] = Math.min(lo[r], v);
      hi[r] = Math.max(hi[r], v);
    }
  }
  return [lo, hi];
}

/**
 * The picture grid of one slice: width x height pixels of mmPerPixel, covering the
 * volume's RAS box across and up, at RAS coordinate `value` on the fixed axis. pixel
 * (0, 0) is the top left, so "up" runs from the last row to the first.
 */
export function sliceGrid(bounds, axis, value, mmPerPixel = 1) {
  const a = AXES[axis];
  if (!a) throw new Error(`unknown slice axis ${axis}`);
  const [lo, hi] = bounds;
  const width = Math.max(1, Math.round((hi[a.across] - lo[a.across]) / mmPerPixel) + 1);
  const height = Math.max(1, Math.round((hi[a.up] - lo[a.up]) / mmPerPixel) + 1);
  return { axis, value, mmPerPixel, width, height, lo, hi };
}

/** RAS point at pixel (px, py) of a slice grid. */
export function pixelToRas(grid, px, py) {
  const a = AXES[grid.axis];
  const p = [0, 0, 0];
  p[a.fixed] = grid.value;
  p[a.across] = a.mirror ? grid.hi[a.across] - px * grid.mmPerPixel : grid.lo[a.across] + px * grid.mmPerPixel;
  p[a.up] = grid.hi[a.up] - py * grid.mmPerPixel;
  return p;
}

/**
 * Nearest voxel lookup of `volume` over a slice grid. rasToVox is the row major
 * inverse of the affine (invert4). returns width x height bytes, row by row from
 * the top, 0 where the slice leaves the grid.
 */
export function sampleSlice(volume, dims, rasToVox, grid) {
  const [nx, ny, nz] = dims;
  const m = rasToVox;
  const out = new Uint8Array(grid.width * grid.height);
  for (let py = 0; py < grid.height; py++) {
    for (let px = 0; px < grid.width; px++) {
      const [x, y, z] = pixelToRas(grid, px, py);
      const i = Math.round(m[0] * x + m[1] * y + m[2] * z + m[3]);
      const j = Math.round(m[4] * x + m[5] * y + m[6] * z + m[7]);
      const k = Math.round(m[8] * x + m[9] * y + m[10] * z + m[11]);
      if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) continue;
      out[py * grid.width + px] = volume[i + nx * (j + ny * k)];
    }
  }
  return out;
}
