# efield: aim 2, the induced electric field

Aim 1 asks how far the EEG cap puts the coil from where the MRI says it should be.
Aim 2 asks what that does to the field in the brain. Each recorded coil pose becomes a
SimNIBS simulation, and the field in a small region of cortex under the MRI guided
coil is compared between the two localization methods.

## Two interpreters

SimNIBS ships its own conda python (`C:\Users\alizr\SimNIBS-4.6\bin\simnibs_python.cmd`,
python 3.11) and cannot be installed into the project's python 3.13. So the project
never imports simnibs. It writes a `job.json`, and the one script that does import
simnibs, `efield/runner_simnibs.py`, is run with `simnibs_python` as a subprocess. The two
sides only ever talk through files:

```
run_efield.py  (project python)                 runner_simnibs.py  (simnibs_python)
  poses -> rois -> job.json          ---->       SESSION / TMSLIST / POSITION
  metrics.py <- results.json         <----       roi numbers per pose
```

Note that `simnibs_python.cmd -c "..."` prints nothing on this machine, so every call
into it uses a script file (see `config.simnibs_version`).

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

```
python run_efield.py                       every saves/*.nii.gz, dry run
python run_efield.py --subjects T1 T2
python run_efield.py --m2m "..\Sample Datasets\Ernie\m2m_ernie" --subjects T1
                                           end to end dry run on a finished head model
python tests/test_efield.py                or python -m pytest tests/
python analysis/efield_stats.py            the statistics, on a synthetic table until
                                           saves/efield/efield_long.csv exists
```

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
- The ROI is a 10 mm sphere on the GM central surface. Its seed is 20 mm down the MRI
  guided coil normal; the runner snaps that to the nearest cortical node before drawing
  the sphere. One ROI per site, shared by both methods.
- `orientation_deg`, `tilt_deg`, `displacement_mm` reuse `src/stats.py`, so a coil
  orientation difference means the same thing in both aims.
- Endpoints (`metrics.py`): percentage difference in area weighted mean |E| in the ROI is
  primary; peak |E|, max |E_normal|, |E| at the ROI centre and the Dice overlap of the
  half maximum hotspots are secondary. Statistics run on subjects, never on reps.

## What remains

1. Run charm on the six sample scans (`python run_efield.py --charm`, or the printed
   commands one at a time) so `saves/m2m/m2m_T1..T6` exist.
2. Get real Brainsight session exports in NIfTI:Aligned space named per the protocol
   (`sub-<ID>_EEGCAP_<SITE>_POSE_REP<n>`, `sub-<ID>_MRI_<SITE>_POSE_REP<n>`). Until then the
   driver builds stand in poses off the Aim 1 targets plus a fixed nudge.
3. Prespecify the equivalence margin on the primary endpoint (`metrics.PCT_MARGIN`,
   placeholder 10 %) and the handle direction convention per site.
4. `python run_efield.py --run` on a real head model, then `analysis/efield_stats.py` on
   the resulting long table. Install `statsmodels` for the mixed model.
5. `export_web.py` is a stub; the runner should dump the central surface to npz so the
   viewer can paint |E| on the cortex without simnibs.
