# TMS-Nav

Compare EEG cap targeting with MRI-guided TMS neuronavigation. Build scalp meshes,
place approximate study targets and coils, and explore Monte Carlo placement
variability in a Three.js viewer. Recorded Brainsight poses support the Aim 1
geometric comparison; a separate SimNIBS workflow supports Aim 2 electric-field modeling.

The MNI demo and simulated placements are for development and illustration.
Template targets are anatomical approximations, not individualized fMRI targets.
Individual subjects use a separate cached reconstruction pipeline with reviewable
MRI fiducial candidates, native scalp meshes, manual landmark confirmation and
subject-specific EEG estimates. See the [complete subject workflow](docs/subject-reconstruction.md)
and [verification report](docs/subject-reconstruction-report.md).

## Quick start

Use **Python 3.12** (the verified local version) and **Node.js 22 LTS or newer** with
npm. Run these commands from the repository root after cloning it.

### Preferred micromamba environment

```bash
micromamba create -y -f environment.yml
eval "$(micromamba shell hook --shell bash)"
micromamba activate tms312
python run_pipeline.py --template
python ingest_subject.py sub-001 /path/to/native_T1w.nii.gz
# Or: python ingest_subject.py sub-001
# reads saves/subjects/sub-001/T1w.nii.gz and optional corrections/transforms
cd viewer
npm ci
npm run dev
```

Python 3.12 and Node 22 were verified in `tms312`. The environment file includes a
Linux library for browser tests; omit `alsa-lib` on other platforms. The alternative
venv instructions below remain available. MNI assets are generated once; subjects
then require only native T1w, with no fMRI or registration prerequisite.

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
  head approximately aligned with anatomical RAS directions; all proposed landmarks require review.
- Approximately 1 mm resolution is preferred; anisotropic spacing is supported
  and no exact voxel size is enforced.
- Native subjects require 3D volumes, explicit spatial units and consistent coded
  qform/sform metadata. Ambiguous headers and 4D inputs are rejected.
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
`derivatives`, `m2m_*` folders and hidden directories. Explicit paths are resolved from the
current working directory. Default inputs and all outputs are anchored to the
repository, so the script can also run from another directory.

### Outputs and viewer compatibility

The MNI demo produces front/side PNGs and a distance histogram in `saves/previews/`.
Subjects produce slice QC, reusable native derivatives and confidence reports in
`saves/derivatives/subjects/<ID>/`. The viewer retains its per-dataset structure:

```text
viewer/src/data/
├── MNI152/
│   ├── scalp.ply
│   ├── coil.ply
│   ├── markers.json
│   └── montecarlo.json
└── ms__sub-001__T1w/
    ├── scalp.ply
    ├── markers.json                   # fiducials, landmarks, EEG, confidence/QC
    └── skull.ply                      # only with supplied skull segmentation
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

## Editing registration landmarks

Select Nasion, Inion, LPA or RPA in the viewer's **edit landmarks** panel, or click
its yellow cube. MRI fiducial diamonds are editable too. Enter X/Y/Z in millimeters and **Apply coordinates**. Entries
are constrained to the displayed scalp by default; uncheck the constraint for an
exact off-surface coordinate. **Snap to scalp** projects the current point.
**Pick on scalp** places it at your next click on the head; Escape cancels.
Rotate first to expose the desired side.

Direction buttons take a step in the local tangent plane and project back onto
the scalp triangles. The step is a tangent displacement, not a geodesic distance.
**Reset selected** restores the pipeline position. Edits are kept per dataset in
browser storage when available; **Export JSON** downloads them with frame metadata.
For reconstructed subjects, save the exported file as
`saves/subjects/<ID>/corrections.json` and rerun `python ingest_subject.py <ID>` to
apply corrections and recompute EEG positions. Original input files stay unchanged.
Template edits do not recompute Monte Carlo registration/statistics. Changed source
landmark exports invalidate saved browser edits.

Regenerate the MNI demo with `python run_pipeline.py --template` for explicit
`MNI152NLin2009aSym` metadata. Primary patient coordinates remain native subject RAS even when secondary MNI coordinates are available.
See the [MNI migration plan](docs/mni-migration.md) for the source audit and rollout.

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

The viewer's scalp projection and movement tests run with Node.js:

```bash
cd viewer
node --test tests/landmarkGeometry.test.mjs
cd ..
```

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

## EEG 10–20 scalp marking

Run `python run_pipeline.py --template` for the demo, or ingest a subject and
confirm its four anatomical anchors as described in the subject workflow, then rebuild
the viewer with `npm run build` in `viewer/`. The **10–20 electrodes** and
**10–20 dashed arcs** layer switches independently control the cyan scalp sites
and tape constructions. Click a site to read its physical coordinates: MNI RAS mm
on the template, native subject RAS mm on an unregistered patient scan.

The pipeline measures connected plane/scalp intersections by arc length:
Nz–Iz at 10/30/50/70/90% gives Fpz/Fz/Cz/Pz/Oz; LPA–Cz–RPA gives
T3/C3/C4/T4 at 10/30/70/90%. The full Fpz–Oz circumference supplies
Fp1/F7/T5/O1 or Fp2/F8/T6/O2 at 5/15/35/45% measured from Fpz
along the respective side. Transverse midpoints on F7–Fz–F8 and
T5–Pz–T6 provide initial F3/F4 and P3/P4 estimates. The final sites are
the intersections of those transverse arcs with the longitudinal Fp1–C3–O1
and Fp2–C4–O2 arcs, so each marker lies on both displayed guides.
**SMA is halfway along Fz–Cz**, at 40% of
Nz–Iz, rather than a fixed centimeter offset. SMA and Fpz/Oz are included
with the 19 conventional scalp electrodes; A1/A2 recording references are
not inferred from LPA/RPA. T3/T4/T5/T6 correspond to modern T7/T8/P7/P8 labels.

The dashed guides also include **Fp1–C3–O1** and **Fp2–C4–O2** longitudinal
scalp arcs. They share the dashed-arcs layer switch. Intersection reconciliation
replaces the independent transverse midpoint rule for F3/F4/P3/P4;
`eeg_qc.transverse_midpoint_shift_mm` reports each resulting displacement.
The workbook calculator retains its separate Fz–F8 midpoint recipe.

`markers.json` stores `electrodes`, `eeg_arcs` and `eeg_qc`. The viewer
reports the sagittal-versus-coronal Cz gap; QC also records temporal-versus-ring
midpoint gaps. Plane cuts approximate snug tape paths on the reconstructed scalp,
and independent constructions can disagree on an irregular head. Landmark
heuristics still need review, particularly on defaced scans. Old exports without
EEG fields continue to load.

The cyan EEG cap is separate from the existing approximate MRI target markers and
Monte Carlo results. These five targets are fixed-direction scalp placeholders,
not anatomically selected Brainsight targets; see the [target-definition audit](docs/target-definitions.md).
Browser point edits hide the cap until saved corrections are applied by subject
ingestion. Unconfirmed subject anchors withhold the cap rather than assign precise
positions. Subject views do not display Monte Carlo dots or template target placeholders.

Marking reference: [Seeck et al., IFCN standardized electrode array (2017)](https://doi.org/10.1016/j.clinph.2017.06.254).
