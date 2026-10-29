# Codex state

Updated: 2026-10-29

## Current objective
- Aim 2 forward model integrated for both cap methods; viewer shows difference, perpendicular field and coil current.
- PI follow-up: E-field setup runs on other machines; `docs/efield.md` explains downloads, demo and viewing.

## Completed work
- Aim 1 subject reconstruction, MRI review and live EEG (commit 4e24977 and later).
- Aim 2 SimNIBS pipeline, viewer Aim 2 tab (commits e9edbb2, d2b4fcd).
- 2026-10-29:
  - Method 2: `efield/reconstruction.py` builds `RECON` poses from `markers.json`; with no recorded MRI pose the runner places the coil over the R01 target (`place = over_target`). `run_efield.py` takes subjects from Brainsight exports and reconstructions.
  - `metrics.py`: `EEG-MRI` and `RECON-MRI` rows; Dice is NaN (dropped) when both hotspots are empty. `efield_stats.py` reports each method.
  - Export: per-pose `normal_*.f32`, `normal_max`, stale pose files cleared. Ernie demo rerun: 4 poses (RECON from Aim 1 10-20 sites, MRI over Aim 1 targets).
  - Viewer: "Cap from MRI" placement, strength/perpendicular switch, minus-MRI-guided difference (cortex + slice, both coils), coil current slider; shorter sources dialog.
  - Sample heads (Ernie, T1–T5) are registry kind `sample` and show Aim 1 targets, coil and Monte Carlo again (lost in 07fe146). gh-pages redeployed from main_ar.
  - main_ar fast-forwarded to origin/main 78fce6b (PR #3 merge). Portable SimNIBS/sample lookup in `efield/config.py`, `scripts/fetch_sample_data.py`, `sample_data` excluded from scan discovery.

## Verified results
- `py -m pytest tests -q` 87 passed (incl. SimNIBS dry run). `npm test` 27 passed. Scratch production build passed.
- Fetcher tested offline on a fake archive only; the real downloads were not run.
- Real SimNIBS Ernie run: SMA RECON vs MRI −25.5 %, F4 −4.3 % (demo targets). Browser checks at 1280×800 and 375×740.
- `sources.spec.mjs` not run: Playwright browsers not installed.

## Decisions / assumptions
- Forward model = SimNIBS FEM. `src/forward_model.py` is an unused toy.
- RECON copies the recorded MRI handle; else `config.HANDLE_DIRECTIONS` (placeholder: SMA back, F4 back + 45° out).
- E_normal sign: positive out of the cortex (E · outward central-surface normal, checked in SimNIBS source).
- Restricted subject scans must not be searched for or accessed. Native RAS mm stays primary.

## Uncommitted changes
- Committed to main_ar as 34a8f20 and pushed to origin/main_ar on 2026-10-29; PR to main is the user's to open. `analysis/README_stats.md` stays local. Viewer data and saves/ are git ignored.
- README.md still lists the old discovery skip folders (no `sample_data`); README edits are the user's call.

## Remaining blockers / limits
- No study head model, R01 targets or Brainsight NIfTI:Aligned exports on this machine.
- Handle convention and the per-subject dI/dt are unconfirmed with the PI.

## Specific next actions
1. Confirm handle convention and dI/dt with the PI; set `HANDLE_DIRECTIONS` and the absolute cut.
2. Charm one cleared subject; `run_efield.py --run` with both methods.
3. Hold-off items from the user: optimal coil position (ADM) and the Monte Carlo field spread.
4. Aim 2 for MNI152 and T1–T5: not simulated (user's choice); the demo would need generalising beyond Ernie.
