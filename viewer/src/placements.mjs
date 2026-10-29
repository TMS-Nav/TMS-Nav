// the aim 2 poses grouped the way the study asks its question: per site, an mri
// guided placement and the cap placements, and how much less (or more) field a cap
// puts into the target region. there are two caps, the proposal's two methods: EEG
// is the cap placement brainsight recorded, RECON the one built off the mri
// reconstruction of the cap. the pairing is the one efield/metrics.py uses, each
// cap rep against the mri rep of the same number, or against the only mri
// placement when there is one. pure, so node --test can load it

export const METHODS = [
  { key: "MRI", label: "MRI-guided", title: "Placed by MRI-guided neuronavigation" },
  { key: "EEG", label: "EEG cap", title: "Placed on the EEG cap, as recorded in Brainsight (method 1)" },
  { key: "RECON", label: "Cap from MRI", title: "Placed on the cap sites reconstructed from the MRI (method 2)" },
];

export const CAPS = ["EEG", "RECON"];

/** sites in the order they first appear. */
export function sites(poses) {
  return [...new Set(poses.map((p) => p.site))];
}

/** the poses at one site, each with a button label, mri first. */
export function placementsAt(poses, site) {
  const here = poses.map((p, index) => ({ ...p, index })).filter((p) => p.site === site);
  const out = [];
  for (const m of METHODS) {
    const reps = here.filter((p) => p.method === m.key).sort((a, b) => (a.rep || 0) - (b.rep || 0));
    for (const p of reps) out.push({ ...p, label: reps.length > 1 ? `${m.label} ${p.rep}` : m.label });
  }
  return out;
}

/** the mri placement a cap placement is compared against, or null. */
export function mriPartner(poses, eeg) {
  if (!CAPS.includes(eeg.method)) return null;
  const mri = poses.filter((p) => p.site === eeg.site && p.method === "MRI");
  if (mri.length === 1) return mri[0];
  return mri.find((p) => p.rep === eeg.rep) || null;
}

/**
 * percent difference of the target region mean, a cap (EEG unless told) against mri
 * guided, averaged over the cap reps that have a partner. null when the site lacks
 * either placement.
 */
export function eegVsMri(poses, site, cap = "EEG") {
  const diffs = [];
  for (const eeg of poses.filter((p) => p.site === site && p.method === cap)) {
    const mri = mriPartner(poses, eeg);
    const a = eeg.roi && eeg.roi.mean_E_magn;
    const b = mri && mri.roi && mri.roi.mean_E_magn;
    if (Number.isFinite(a) && Number.isFinite(b) && b !== 0) diffs.push((100 * (a - b)) / b);
  }
  if (!diffs.length) return null;
  return { pct: diffs.reduce((s, v) => s + v, 0) / diffs.length, n: diffs.length };
}
