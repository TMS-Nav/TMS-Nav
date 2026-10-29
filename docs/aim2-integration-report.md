# Aim 2 forward model integration report

Updated 2026-10-29.

## What "integrate the forward model" means here

| Term | Meaning in this project |
|---|---|
| Forward model | Coil pose + subject head model → induced E-field. SimNIBS FEM (`efield/runner_simnibs.py`). |
| Our framework | Aim 1: subject MRI with cap markers → reconstructed 10-20 sites → coil placements in native RAS mm. |
| Integration | Every placement the framework produces is simulated, read at the R01 target ROI and fed into the paired statistics and the viewer. |

`src/forward_model.py` is an unused free-space toy (one loop's vector potential). SimNIBS supersedes it.

## Status

| Piece | State | Where |
|---|---|---|
| Method 1: Brainsight poses → SimNIBS → ROI metrics → TOST / Bland–Altman | Done, waiting for real exports and `saves/fmri_targets.csv` | `run_efield.py`, `efield/brainsight.py` |
| **Method 2: reconstructed cap → SimNIBS** | **Done.** `RECON` poses from `markers.json`; MRI-guided pose copied from Brainsight, or placed on the skin over the R01 target | `efield/reconstruction.py`, `runner_simnibs.place_over_target` |
| Pairing and statistics per method | `EEG-MRI` and `RECON-MRI` rows; `efield_stats.py` reports each | `efield/metrics.py`, `analysis/efield_stats.py` |
| Dice with two empty hotspots | Now NaN and left out, was a false 1.0 | `efield/metrics.py` |
| Viewer | Difference map, perpendicular field, coil current control | `viewer/src/aim2.js` |
| Real subject run | Not done: no study head model or R01 targets on this machine | `run_efield.py --charm`, `--run` |

## Method 2 flow

```
saves/derivatives/subjects/<ID>/markers.json  ──► RECON pose: coil flat on the scalp at the 10-20 site
saves/brainsight/<ID>_...txt (MRI pose)       ──► MRI pose, its handle copied onto RECON
   (no MRI pose)  + saves/fmri_targets.csv    ──► MRI pose on the skin over the target (runner)
                                                  ROI at the R01 target, shared by both
```

## Viewer

| Control | Where | What it does |
|---|---|---|
| strength / perpendicular | left panel, field | \|E\| (turbo) or signed E_normal (blue into the cortex, red out of it) |
| minus MRI-guided | left panel, field | cap placement minus its MRI-guided partner on the cortex and slice; both coils drawn |
| coil current | right panel, under the scale | multiplies every V/m number; the % comparison does not change |
| Cap from MRI | coil buttons | the Method 2 placement |

## Verification

| Check | Result |
|---|---|
| `py -m pytest tests -q` | 83 passed, incl. the SimNIBS dry run on Ernie with an over-target pose |
| `npm test` (viewer) | 26 passed |
| Production build (scratch dist) | passed |
| Method 2 driver, scratch reconstruction + targets, SimNIBS dry run | 2 RECON + 2 over-target MRI poses, all 4.0 mm off the skin; MNI SMA target mapped into Ernie |
| Ernie demo, real SimNIBS run (4 poses, D70, 1 A/µs) | passed, export + registry rebuilt |
| In-app browser, 1280×800 and 375×740 | difference, perpendicular, coil current and dialog checked; no overflow |

Ernie demo, method 2 against MRI-guided (demo targets, not study data):

| Site | Coil centres apart | Mean \|E\| in target, RECON / MRI | % difference | Dice, half max |
|---|---|---|---|---|
| F4 | ≈17 mm | 0.86 / 0.90 V/m | −4.3 % | 0.59 |
| SMA | 31.5 mm | 0.62 / 0.83 V/m | −25.5 % | 0.61 |

The 50 V/m absolute hotspot is empty at 1 A/µs for every pose, so its Dice is now left out instead of 1.0.

The previous two-pose demo output is kept at `saves/efield/demo_ernie_2026-10-22/` (git ignored).

## Running it on another machine (PI follow-up)

The PI could not run the simulation: `efield/config.py` hard-coded this machine's Windows
paths for SimNIBS and the sample heads.

| Change | Where |
|---|---|
| SimNIBS found via `TMSNAV_SIMNIBS_DIR`, then `simnibs_python` on the PATH, then the usual install folders; `.cmd` only on Windows | `efield/config.find_simnibs_dir` |
| Sample heads found anywhere under `saves/sample_data/` (or the old sibling `Sample Datasets/`), `TMSNAV_SAMPLE_DATA` to override | `efield/config.sample_m2m_roots` |
| One command to download and unpack Ernie (1.1 GB) and optionally T1–T5 (2.7 GB), copying the T1s into `saves/` | `scripts/fetch_sample_data.py` |
| `saves/sample_data/` excluded from scan discovery, so the datasets' raw T2/dMRI scans never become viewer heads | `src/mri_inputs.RESERVED` |
| Setup, demo, viewing and exporting, one page | `docs/efield.md` |
| Missing pieces print what to run | `efield/demo_ernie.py`, `run_efield.py`, Aim 2 empty state |

Verified here: `config.describe()` finds SimNIBS and the heads without the old paths, a
missing SimNIBS prints the fix, the demo dry run passes, `pytest` 86 passed. The fetcher
was tested offline on a fake archive; the real 1.1 GB download was not run.

## Remaining issues

- Handle directions with no recording are a placeholder (`config.HANDLE_DIRECTIONS`).
- The dI/dt each subject is scored at still needs a decision before an absolute hotspot cut means anything.
- The Ernie demo's MRI-guided targets are Aim 1's anatomical targets, not fMRI targets.
