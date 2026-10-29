// small numeric helpers for the aim 2 fields, pure so node --test can load them.
// every pose of a head is on the same decimated cortex (efield/export_web.py), so
// two poses' per vertex arrays line up index for index

/** a minus b, vertex by vertex. */
export function difference(a, b) {
  const out = new Float32Array(a.length);
  for (let i = 0; i < a.length; i++) out[i] = a[i] - b[i];
  return out;
}

/**
 * the top of a symmetric colour scale for signed values: a high percentile of |v|,
 * the same rule the export uses for |E|, so one sharp vertex does not wash the rest out
 */
export function symmetricTop(values, pct = 0.999) {
  const abs = Float32Array.from(values, (v) => Math.abs(v)).sort();
  if (!abs.length) return 0;
  return abs[Math.min(abs.length - 1, Math.floor(pct * (abs.length - 1)))];
}

/** the largest |v|. */
export function maxAbs(values) {
  let m = 0;
  for (const v of values) if (Math.abs(v) > m) m = Math.abs(v);
  return m;
}

/** distance between two points, mm. */
export function distance(a, b) {
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

// the simulation runs at 1 A/us, simnibs's reference, and the field is linear in
// the coil current, so the field at another current is the same picture times it
export const REFERENCE_DIDT = 1;

/** how much to multiply a reference field by at a coil current of didt A/us. */
export function intensityScale(didt, reference = REFERENCE_DIDT) {
  return didt / reference;
}
