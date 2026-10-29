import test from "node:test";
import assert from "node:assert/strict";
import { eegVsMri, mriPartner, placementsAt, sites } from "../src/placements.mjs";

const pose = (site, method, rep, mean) => ({ site, method, rep, name: `${site}_${method}_${rep}`, roi: { mean_E_magn: mean } });

test("sites keep their order and placements put mri first", () => {
  const poses = [pose("F4", "EEG", 1, 0.9), pose("SMA", "MRI", 1, 1), pose("F4", "MRI", 1, 1)];
  assert.deepEqual(sites(poses), ["F4", "SMA"]);
  const f4 = placementsAt(poses, "F4");
  assert.deepEqual(f4.map((p) => p.label), ["MRI-guided", "EEG cap"]);
  assert.deepEqual(f4.map((p) => p.index), [2, 0]);
});

test("two reps get numbered labels", () => {
  const poses = [pose("F4", "EEG", 2, 1), pose("F4", "EEG", 1, 1)];
  assert.deepEqual(placementsAt(poses, "F4").map((p) => p.label), ["EEG cap 1", "EEG cap 2"]);
});

test("one mri placement partners every eeg rep", () => {
  const poses = [pose("F4", "MRI", 1, 1.0), pose("F4", "EEG", 1, 0.9), pose("F4", "EEG", 2, 0.8)];
  assert.equal(mriPartner(poses, poses[2]), poses[0]);
  const r = eegVsMri(poses, "F4");
  assert.equal(r.n, 2);
  assert.ok(Math.abs(r.pct - -15) < 1e-9);
});

test("reps pair by number when mri has two", () => {
  const poses = [pose("F4", "MRI", 1, 1.0), pose("F4", "MRI", 2, 2.0), pose("F4", "EEG", 2, 1.0)];
  assert.ok(Math.abs(eegVsMri(poses, "F4").pct - -50) < 1e-9);
});

test("no comparison without both placements", () => {
  assert.equal(eegVsMri([pose("F4", "EEG", 1, 1)], "F4"), null);
  assert.equal(eegVsMri([pose("F4", "MRI", 1, 1)], "F4"), null);
});
