# efield: aim 2, the induced electric field

Aim 1 asks how far the EEG cap puts the coil from where the MRI says it should be.
Aim 2 asks what that does to the field in the brain. Each recorded coil pose becomes a
SimNIBS simulation, and the field in a small region of cortex under the MRI guided
coil is compared between the two localization methods.

## The two cap methods

The proposal gets the EEG cap placement two ways. Both are scored against the same
MRI guided placement and the same ROI, and come out as separate rows in the long table.

| method | `method` | where the pose comes from |
|---|---|---|
| 1, direct recording | `EEG` | Brainsight export, `sub-<ID>_EEGCAP_<SITE>_POSE_REP<n>` |
| 2, MRI reconstruction | `RECON` | `saves/derivatives/subjects/<ID>/markers.json`, the 10-20 site `src/subject_ingest.py` built off the cap markers (`efield/reconstruction.py`) |
| MRI guided | `MRI` | Brainsight export `sub-<ID>_MRI_<SITE>_POSE_REP<n>`; with none, the coil goes on the skin straight over the R01 target (`place = over_target`, finished by the runner on the head model) |

- A RECON pose is a coil flat on the scalp at the site's electrode, 4 mm off it. It
  copies the handle of the recorded MRI pose at that site, so the two differ in where
  the coil is, not in how it was turned. With no recording it uses
  `config.HANDLE_DIRECTIONS` (placeholder: SMA straight back, F4 back and 45° out).
- Only a reconstruction with confirmed anchors (`estimated_from_confirmed_anchors`) is
  used. Charm has to run on the same scan that was ingested, or the poses land in the
  wrong space; the runner's skin distance check catches the gross cases.
- Pairs come out as `EEG-MRI` and `RECON-MRI`. `analysis/efield_stats.py` reports each.

## Two interpreters

SimNIBS ships its own conda python (`<SimNIBS>/bin/simnibs_python`, `.cmd` on Windows,
python 3.11) and cannot be installed into the project's python 3.13. So the project
never imports simnibs. It writes a `job.json`, and the one script that does import
simnibs, `efield/runner_simnibs.py`, is run with `simnibs_python` as a subprocess. The two
sides only ever talk through files:

```
run_efield.py  (project python)                 runner_simnibs.py  (simnibs_python)
  poses -> rois -> job.json          ---->       SESSION / TMSLIST / POSITION
  metrics.py <- results.json         <----       roi numbers per pose
```

Note that `simnibs_python.cmd -c "..."` printed nothing on Windows, so every call
into it uses a script file (see `config.simnibs_version`). `config.find_simnibs_dir`
says where SimNIBS is looked for.

## The dry run rule

Nothing simulates or builds a head model unless someone asks for it by name:

- `run_efield.py` passes `--dry-run` to the runner unless `--run` was typed
- `headmodel.run_charm(..., dry_run=True)` only prints the charm command unless
  `dry_run=False`; the driver needs `--charm`
- the runner in `--dry-run` validates everything (m2m, mesh, coil, rotation matrices,
  coil centre to skin distance, normal pointing into the head, ROI on the cortex),
  writes `dryrun.json` and exits before `run_simnibs`

Simulations take hours per subject and charm about an hour per head, and both write
hundreds of MB. Neither should ever happen as a side effect.

## Running it

Setup, downloads, the Ernie demo and viewing the result: [docs/efield.md](../docs/efield.md).

Outputs go under `saves/efield/<subject>/` (`job.json`, `dryrun.json`, later
`results.json` and `sim/`), head models under `saves/m2m/m2m_<subject>/`, Brainsight
exports are expected at `saves/brainsight/<subject>_coil_pose_samples.txt`. All of it is
git ignored because it derives from patient scans.

## Conventions worth knowing

- `matsimnibs` columns are x', y', z', centre in subject RAS mm. y' is the handle, z' the
  coil normal pointing INTO the head. Aim 1 stores the outward scalp normal, so z' is its
  negative. That sign lives in `poses.py` only.
- Brainsight's coil frame has x and z the other way round (z points away from the head).
  `poses.CoilPose.from_brainsight` flips them, the same flip SimNIBS's own
  `brainsight().read` applies. Exports must be in `NIfTI:Aligned` (or `NIfTI:Q:Aligned`
  from Brainsight 2.5.3 on). A `Brainsight` space export is LPS and is refused unless
  `allow_lps=True`.
- **The ROI is the R01's own fMRI target.** Measuring under the coil only answers "how
  much field landed under the coil", which is close to circular. Measuring at the
  intended target answers what the study is asking. Targets are supplied in
  `saves/fmri_targets.csv`, one row per subject per site, in subject mm or MNI mm (see
  `efield/fmri_targets_example.csv`). MNI rows are mapped in by `mni2subject_coords`.
  The ROI is a 10 mm sphere on the GM central surface around that point, snapped to the
  nearest cortical node, and it is shared by both methods at a site.
  A site with no supplied target falls back to a seed 20 mm down the MRI guided coil
  normal. That fallback is flagged on every run, recorded in the job notes, and is a
  stand in, not a result. The runner also warns when a target has to move more than
  8 mm to reach the cortical surface, which usually means the coordinate is in the
  wrong space.
- `orientation_deg`, `tilt_deg`, `displacement_mm` reuse `src/stats.py`, so a coil
  orientation difference means the same thing in both aims.
- Endpoints (`metrics.py`): percentage difference in area weighted mean |E| in the ROI is
  primary; peak |E|, max |E_normal|, |E| at the ROI centre and the Dice overlap of the
  half maximum hotspots are secondary. Statistics run on subjects, never on reps. Dice
  is left out (NaN) when both hotspots are empty.
- `E_normal` is E · the central surface's outward node normal: positive points out of
  the cortex, negative into it.
- Fields are simulated at 1 A/µs and scale linearly with dI/dt. The viewer's coil
  current slider rescales every number; an absolute V/m cut has to be compared at the
  same current.

## What remains

1. Run charm on the sample scans (`python run_efield.py --charm`, or the printed
   commands one at a time) so `saves/m2m/m2m_T1..T6` exist.
2. Get real Brainsight session exports in NIfTI:Aligned space named per the protocol
   (`sub-<ID>_EEGCAP_<SITE>_POSE_REP<n>`, `sub-<ID>_MRI_<SITE>_POSE_REP<n>`). Until then the
   driver builds stand in poses off the Aim 1 targets plus a fixed nudge.
3. Fill in `saves/fmri_targets.csv` with the R01 targets. Until it exists every ROI is
   the fallback under the coil and the numbers do not mean what the proposal wants.
4. Prespecify the equivalence margin on the primary endpoint (`metrics.PCT_MARGIN`,
   placeholder 10 %), the absolute hotspot cut (`config.SUPRATHRESHOLD_ABS_VM`,
   placeholder 50 V/m) and the handle direction convention per site.
5. `python run_efield.py --run` on a real head model, then `analysis/efield_stats.py` on
   the resulting long table. Install `statsmodels` for the mixed model.
6. Decide the dI/dt each subject is scored at (their stimulator intensity, or 1 A/µs
   with `SUPRATHRESHOLD_ABS_VM` scaled to match).

The Ernie demo (`python -m efield.demo_ernie`) runs method 2 end to end on public data:
RECON poses off Aim 1's 10-20 sites on Ernie, MRI guided poses over Aim 1's targets,
then `export_web.py` writes |E|, E_normal and the slice volumes for the viewer.
