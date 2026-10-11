# TMS-Nav

Compare EEG cap targeting with MRI-guided TMS neuronavigation. Build scalp meshes,
place approximate study targets and coils, and explore Monte Carlo placement
variability in a Three.js viewer. Recorded Brainsight poses support the Aim 1
geometric comparison; a separate SimNIBS workflow supports Aim 2 electric-field modeling.

The MNI demo and simulated placements are for development and illustration.
Generated targets are anatomical approximations, not individualized fMRI targets
or detected MRI-visible cap fiducials.

## Quick start

Use **Python 3.12** (the verified local version) and **Node.js 22 LTS or newer** with
npm. Run these commands from the repository root after cloning it.

### Windows PowerShell

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install --upgrade pip
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe run_pipeline.py
cd viewer
npm.cmd ci
npm.cmd run dev
```

### macOS and Linux

```bash
python3.12 -m venv venv
./venv/bin/python -m pip install --upgrade pip
./venv/bin/python -m pip install -r requirements.txt
./venv/bin/python run_pipeline.py
cd viewer
npm ci
npm run dev
```

Open [the local viewer](http://localhost:1234). These commands use the venv directly,
so activation and PowerShell execution-policy changes are unnecessary. After
activating a venv yourself, `python` can replace the interpreter paths below.
Linux distributions may require their Python venv package. Headless rendering
requires a working VTK graphics backend (EGL/OSMesa or an X display, depending on
the VTK build); off-screen mode alone does not provide that backend.

## MRI inputs and template fallback

```bash
python run_pipeline.py                         # saves/ scans, otherwise MNI152
python run_pipeline.py --template              # force the MNI152 demo
python run_pipeline.py saves/cohorts/ms         # one cohort, searched recursively
python run_pipeline.py saves/cohorts/ms saves/cohorts/control
python run_pipeline.py "/path/to/subject_T1w.nii.gz"
```

With no arguments, the pipeline searches `saves/` for structural scans, including
nested cohorts. If none are found, or the supplied folders contain no eligible
scans, it downloads and uses the **full-head ICBM152 2009a T1 template**.
The first template run needs internet access; later runs reuse `saves/templates/`.
The fetcher downloads an approximately 63 MB archive with several template images;
only the unmasked T1 is processed. The cache is excluded from patient discovery.
A nonexistent path or unsupported explicit file produces an error.

NiBabel loads NIfTI data and the affine but has no MNI downloader. Downloads use
[`nilearn.datasets.fetch_icbm152_2009`](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.fetch_icbm152_2009.html).
The bundled
[`load_mni152_template`](https://nilearn.github.io/stable/modules/generated/nilearn.datasets.load_mni152_template.html)
is skull-stripped and does not supply the scalp needed here.

For patient scans:

- Supply a **T1-weighted structural NIfTI** (`.nii` or `.nii.gz`) with scalp and
  external head anatomy. Convert DICOM to NIfTI beforehand.
- Prefer intact facial anatomy and ears. Defaced scans are supported and flagged,
  but missing anatomy cannot be recovered and landmarks become approximate.
- Preserve a valid spatial affine in millimeters. Landmark heuristics assume a
  reasonably centered head aligned with anatomical RAS directions.
- Approximately 1 mm resolution is preferred; anisotropic spacing is supported
  and no exact voxel size is enforced.
- A 3D volume is expected; only the first volume of a 4D image is processed.
- Store only intended structural inputs in discovery folders. The pipeline does
  not identify contrast: T2, fMRI, mask and segmentation NIfTIs would also be selected.

### Local data layout

Legacy `saves/*.nii` and `saves/*.nii.gz` inputs remain supported. For multiple
cohorts, use one structural scan per subject, or distinct filenames for visits:

```text
saves/
├── T1.nii.gz                          # optional legacy scan
├── cohorts/
│   ├── ms/
│   │   ├── sub-001/T1w.nii.gz
│   │   └── sub-002/T1w.nii.gz
│   └── control/
│       └── sub-001/T1w.nii.gz
├── templates/                         # automatic download cache
├── previews/                          # generated PNGs
├── brainsight/                        # recorded coil exports
├── aim1/                              # geometric statistics
├── efield/                            # electric-field jobs/results
└── m2m/                               # SimNIBS head models
```

Discovery skips `templates`, `previews`, `brainsight`, `aim1`, `efield`, `m2m`,
`m2m_*` folders and hidden directories. Explicit paths are resolved from the
current working directory. Default inputs and all outputs are anchored to the
repository, so the script can also run from another directory.

### Outputs and viewer compatibility

Each scan produces front/side PNGs and a distance histogram in `saves/previews/`.
The viewer keeps its existing flat per-dataset structure:

```text
viewer/src/data/
├── MNI152/
│   ├── scalp.ply
│   ├── coil.ply
│   ├── markers.json
│   └── montecarlo.json
└── ms__sub-001__T1w/
    └── ...                            # same four files
```

Legacy `T1.nii.gz` still maps to `data/T1/`. Cohort paths map to names such as
`ms__sub-001__T1w` and `control__sub-001__T1w`, keeping shared subject IDs separate.
Characters outside letters, digits, `_`, `-` and `.` become underscores.
Resulting name collisions are rejected before processing. Explicit inputs outside
`saves/` use their filenames as dataset names.

`viewer/src/datasetRegistry.js` is regenerated from **all existing processed
dataset folders**, giving Parcel static imports. Reruns overwrite matching outputs
and keep other datasets. Removing or renaming an input does not delete old viewer
output: remove its generated dataset folder, then rerun to update the dropdown.

`saves/` and `viewer/src/data/` are Git-ignored. Patient scans and derived meshes
stay local. Cohort discovery currently applies to `run_pipeline.py`; the separate
statistics and electric-field drivers retain their documented input layouts.

## Aim 1 statistics

Put Brainsight session exports in `saves/brainsight/` named
`sub-<ID>_coil_pose_samples.txt`, with samples named
`sub-<ID>_EEGCAP_<SITE>_POSE_REP<n>` and `sub-<ID>_MRI_<SITE>_POSE_REP<n>`.

```bash
python analysis/aim1_stats.py
```

This prints per-site statistics and writes `saves/aim1/measurements.csv` (one row
per compared placement), `saves/aim1/aim1_stats.csv` (one number per row), and
`viewer/src/data/aim1_stats.json`. Methods live in `src/stats.py` and use statsmodels.

## Aim 2 electric-field modeling

See [the electric-field workflow](efield/README.md) for SimNIBS setup, head models,
coil poses, individualized target coordinates and dry-run commands. SimNIBS uses
its own interpreter; it is not installed into this venv. `run_pipeline.py` does
not run tissue segmentation or electric-field simulations.

## Tests

Use the same venv, from the repository root:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests/
```

Input tests use temporary files and mocked downloads. They cover fallback,
cohort discovery, exclusions, duplicates, collisions and mesh orientation for
both affine handednesses.

## GitHub Pages

```bash
cd viewer
npm run deploy
```

Deployment checks every processed head against `viewer/public-subjects.json`,
builds `viewer/dist/` and pushes it to `gh-pages`. Patient datasets must never
be added to that allowlist. The six legacy entries reference SimNIBS example data
(Ernie: CC BY-NC 4.0; group dataset: ODC-PDDL). The MNI demo has its own attribution.

Before public deployment, remove patient dataset folders from `viewer/src/data/`
and regenerate the registry with `python run_pipeline.py --template`.
Moving input scans alone does not remove generated meshes. Review whether the
aggregate `aim1_stats.json` should be included too.

In repository settings choose **Pages → Deploy from a branch → gh-pages → / (root)**.
The site appears at `https://<owner>.github.io/TMS-Nav/`; build paths are relative.

### Embedding

```html
<iframe src="https://<owner>.github.io/TMS-Nav/?embed=1"
        width="100%" height="640" style="border:0"
        loading="lazy" title="TMS-Nav viewer"></iframe>
```

`?embed=1` folds both side panels and adds a link to the full page. Frames narrower
than 760 px or shorter than 520 px also fold the panels.

## Template attribution

The MNI demo uses the ICBM152 nonlinear 2009 atlas distributed through Nilearn.
See [McGill's atlas description and license](https://www.mcgill.ca/bic/software/tools-data-analysis/anatomical-mri/atlases/icbm152-non-linear-2009)
for references and full redistribution terms.

Copyright (C) 1993–2004 Louis Collins, McConnell Brain Imaging Centre, Montreal
Neurological Institute, McGill University. Redistribution requires retaining the
copyright notice; the linked license includes the complete warranty statement.
