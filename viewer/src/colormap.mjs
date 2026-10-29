// turbo, the field colormap for aim 2. it reads like the jet simnibs figures use,
// blue through green to red, without jet's false bands at cyan and yellow. this is
// the polynomial fit google published with turbo (Mikhailov 2019, Apache 2.0), so no
// 256 row table ships here. it tracks the table closely through the middle and
// drifts a little at the two ends, darker at the bottom and a brighter red on top.
// pure, no three.js, so node --test can load it

const R4 = [0.13572138, 4.6153926, -42.66032258, 132.13108234];
const G4 = [0.09140261, 2.19418839, 4.84296658, -14.18503333];
const B4 = [0.1066733, 12.64194608, -60.58204836, 110.36276771];
const R2 = [-152.94239396, 59.28637943];
const G2 = [4.27729857, 2.82956604];
const B2 = [-89.90310912, 27.34824973];

const clamp01 = (x) => Math.min(Math.max(x, 0), 1);

function poly(c4, c2, x, x2, x3, x4, x5) {
  return clamp01(c4[0] + c4[1] * x + c4[2] * x2 + c4[3] * x3 + c2[0] * x4 + c2[1] * x5);
}

/** turbo at t in [0, 1], as [r, g, b] each in [0, 1]. */
export function turbo(t) {
  const x = clamp01(t);
  const x2 = x * x, x3 = x2 * x, x4 = x3 * x, x5 = x4 * x;
  return [poly(R4, R2, x, x2, x3, x4, x5), poly(G4, G2, x, x2, x3, x4, x5), poly(B4, B2, x, x2, x3, x4, x5)];
}

/** n rows of turbo as rgba bytes, for a texture or a canvas gradient. */
export function turboTable(n = 256) {
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = turbo(i / (n - 1));
    out.set([Math.round(r * 255), Math.round(g * 255), Math.round(b * 255), 255], i * 4);
  }
  return out;
}

/** a css linear-gradient running the colormap left to right. */
export function turboGradient(stops = 12) {
  const parts = [];
  for (let i = 0; i <= stops; i++) {
    const [r, g, b] = turbo(i / stops).map((v) => Math.round(v * 255));
    parts.push(`rgb(${r} ${g} ${b}) ${((i / stops) * 100).toFixed(1)}%`);
  }
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}

// the signed maps, the perpendicular field and a difference between two placements,
// go blue below zero and red above it. ColorBrewer's RdBu (Brewer, Apache 2.0) with
// its two darkest ends left off, they sink into the dark panel and the black slice,
// and its two palest steps either side of zero left off too. values that small sit
// under the colour cut anyway, and with them in, everything just past the cut came
// out a washed pastel on the grey cortex
const DIVERGING = [
  [0x21, 0x66, 0xac], [0x43, 0x93, 0xc3], [0x92, 0xc5, 0xde], [0xec, 0xef, 0xf1],
  [0xf4, 0xa5, 0x82], [0xd6, 0x60, 0x4d], [0xb2, 0x18, 0x2b],
].map((c) => c.map((v) => v / 255));

/** the diverging map at s in [-1, 1], blue at -1, near white at 0, red at +1, as [r, g, b]. */
export function diverging(s) {
  const x = clamp01((s + 1) / 2) * (DIVERGING.length - 1);
  const i = Math.min(Math.floor(x), DIVERGING.length - 2);
  const f = x - i;
  const a = DIVERGING[i], b = DIVERGING[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}

/** n rows of the diverging map as rgba bytes, -1 in the first row and +1 in the last. */
export function divergingTable(n = 256) {
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = diverging((2 * i) / (n - 1) - 1);
    out.set([Math.round(r * 255), Math.round(g * 255), Math.round(b * 255), 255], i * 4);
  }
  return out;
}

/** a css linear-gradient running the diverging map left to right. */
export function divergingGradient(stops = 12) {
  const parts = [];
  for (let i = 0; i <= stops; i++) {
    const [r, g, b] = diverging((2 * i) / stops - 1).map((v) => Math.round(v * 255));
    parts.push(`rgb(${r} ${g} ${b}) ${((i / stops) * 100).toFixed(1)}%`);
  }
  return `linear-gradient(90deg, ${parts.join(", ")})`;
}
