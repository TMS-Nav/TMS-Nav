# Subject MRI reconstruction and review

## What is available

| Component | Purpose |
|---|---|
| `ingest_subject.py` / `src.subject_ingest` | Native T1w → cached surfaces, marker candidates, coordinates, confidence and QC |
| `run_viewer.py` / `src.viewer_service` | Local MRI access, detector previews, cached EEG recomputation and correction saving |
| Three.js **3D head** tab | Scalp/skull, subject fiducials, anatomical landmarks, EEG electrodes and arcs |
| NiiVue **Anatomical MRI** tab | Linked axial/coronal/sagittal slices, crosshairs, contrast, zoom and point placement |
| MNI demo | Default subject; existing targets/Monte Carlo preserved; pink fiducials hidden |

**Primary coordinates: native RAS millimeters.** Pink markers are MRI candidates;
yellow landmarks define the EEG construction; cyan electrodes are estimates.
The five cap markers and four anatomical anchors are separate point sets.

## Setup and launch

From the repository root:

```bash
micromamba create -y -f environment.yml        # first setup only
# Alternative to activation: prefix commands with micromamba run -n tms312
eval "$(micromamba shell hook --shell bash)"
micromamba activate tms312
npm --prefix viewer ci
python run_pipeline.py --template             # generate/cache MNI once
python ingest_subject.py r-02 ./saves/subjects/r-02/T1w.nii.gz
python run_viewer.py
```

- Open <http://127.0.0.1:1234>; select the subject.
- `run_viewer.py` starts Parcel and a loopback-only API on port **8765**. Ctrl-C stops both.
- `python run_viewer.py --api-only` supports an already-running Parcel viewer.
- `npm --prefix viewer run dev` alone retains static 3D viewing; MRI/live previews/save require the API.
- On launch, saved derivatives refresh the static viewer exports. Nothing is deployed automatically.
- Python 3.12 / Node 22; exact Python versions: `requirements-tms312-lock.txt`.
- `alsa-lib` in `environment.yml` supplies Linux Chromium support; omit it on other platforms.
- SimNIBS remains a separate optional environment; see `efield/README.md`.

Default inputs:

```text
saves/subjects/sub-001/
  T1w.nii.gz                 # or T1w.nii, exactly one
  corrections.json           # reviewed locations; auto-read
  detector.json              # saved detector settings; auto-read
  native_to_mni.json          # optional compatible physical affine
```

Then use `python ingest_subject.py sub-001`. Explicit external anatomical paths
remain supported. Dedicated-folder discovery selects only `T1w.nii[.gz]`, excluding
optional masks. Legacy `run_pipeline.py <scan-or-folder>` uses its existing dataset keys.

## Debug missing cap markers

1. Open **Anatomical MRI**. Locate each expected marker in all three planes; inspect intensity and RAS coordinates.
2. Select a candidate row to center the slices. Enable **Fiducials / landmarks** only when needed; anatomy-only is the default.
3. Inspect the candidate table and `markers.json → detection_qc`: measured volume, elongation, scalp distance/depth and rejection reasons.
4. Adjust filters and click **Preview detection**. **Undo detection preview** restores the previous candidates/review edits.
5. **Include** a rejected candidate, or **Add fiducial at crosshair** for a missed center. Select it and use **Place selected point at crosshair** to refine it.
6. **Confirm** each MRI-supported center; **Exclude** artifacts. Review progress shows confirmed/expected and total active candidates.
7. Click **Save reviewed changes**. Corrections/settings and derivatives are saved locally; existing anatomical input files remain unchanged.

| Setting / diagnostic | Interpretation |
|---|---|
| Brightness | Global cutoff in original image intensity units |
| Local contrast | Difference of 0.6/3 mm Gaussian responses; threshold has different units |
| Blank threshold | Automatically estimate a cutoff for the selected method |
| Min/max volume | Physical segmented component volume; not the nominal package diameter |
| Elongation | Covariance ratio including finite voxel dimensions; helps avoid rejecting faint one-plane components |
| Max depth / outside | Relative to the head mask; cap markers can sit above skin or be included in the mask |
| Separate touching peaks | Optional subdivision of connected components with distinct intensity peaks; review results |
| Expected count | Default **5**, a QC expectation; never creates detections or selects a forced top five |
| Diagnostic limit | Up to 5,000 component rows; omitted rows are counted explicitly |

**LiquiMark 8 mm starting profile:** local contrast, 8–800 mm³, elongation ≤8,
head-mask depth ≤8 mm, outside distance ≤25 mm, expected count 5. This is an
unvalidated starting configuration for review, not an acquisition-specific calibration.
An 8 mm round marker's visible signal need not form an 8 mm sphere.

Offline brightness adjustment is also available:

```bash
python ingest_subject.py sub-001 --marker-threshold 750  # example intensity, not universal
```

Changing detection can renumber candidates. Existing fiducial edits/exclusions
require a matching `detection_signature`; re-review when it changes.
Markers remain at their measured cap centers. **Do not snap cap centers to skin**;
scalp contacts are a different geometric quantity.

## Landmarks and live EEG

- Adjust **Nz (nasion), Iz (inion), LPA, RPA** using the editor, scalp picking, or MRI crosshairs.
- Anatomical placements from MRI are projected to scalp; fiducial placements keep their cap offset.
- After a 300 ms editing pause, the API recomputes electrodes/arcs on the cached full scalp. MRI segmentation and detection do not rerun for previews.
- Pending/invalid constructions hide stale arcs and explain the reason. Late responses from earlier edits/subjects are ignored.
- Usable unconfirmed anchors permit a **PROVISIONAL** preview with limited confidence. Confirm all four for a reviewed-anchor construction.
- Static CLI derivatives withhold EEG until the four anchors are confirmed. Live previews do not silently confirm anchors.
- Fiducial-only edits do not invalidate the landmark-based EEG cap. Five confirmed fiducials are a separate review milestone.
- **Save reviewed changes** persists confirmed points, exclusions and settings; explicit save rebuilds derivatives and reuses an existing registration.
- API-enabled reopening reads saved derivatives immediately. Static exports refresh on the next full `run_viewer.py` launch or CLI ingestion.

Manual/export fallback:

- **Export JSON** retains edits if the local API is unavailable.
- Save it as `saves/subjects/ID/corrections.json`; rerun ingestion and reload.
- Required metadata: subject ID, exact native `source_sha256`, `axes: RAS`, `units: mm`, `coordinate_frame.name: subject_RAS`.
- Detected-fiducial edits additionally carry `detection_signature`.
- `landmarks`: entries with `name`, `label`, `position`; `fiducials`: coordinate entries, optional `anatomical_label`, or `{name, exclude: true}`.
- New manual fiducials use unique names. Retain earlier confirmations you intend to keep.
- `corrections_template.json` is for review, not silently applied. Supplied entries count as human confirmation.

## Coordinates, registration and skull

| Contract | Behavior |
|---|---|
| Native NIfTI | 3D, finite nonconstant T1w, explicit spatial units and consistent coded qform/sform |
| Affines | Reflected/oblique orthogonal grids supported; meter/micron units converted to mm |
| Rejected inputs | Ambiguous headers, sheared grids, 4D images and template-coded native inputs |
| MRI display | Local API normalizes served header units to mm without changing original files or resampling voxels |
| Scalp | Outer soft-tissue surface; review head-mask QC, hair, cropping and defacing |
| Skull | Separate mesh only from a validated binary native-grid mask; no scalp erosion presented as bone |
| Confidence | Detection/construction scores are uncalibrated; human score 1 means confirmed, not a probability |

```bash
python ingest_subject.py sub-001 --skull-mask /path/to/native_binary_skull.nii.gz
python ingest_subject.py sub-001 --transform /path/to/physical_affine.json
python ingest_subject.py sub-001 --register-to /path/to/full_head_MNI_T1w.nii.gz
```

- Skull masks must share shape/affine and lie ≥98% inside the head mask; this checks alignment, not tissue accuracy.
- Affine JSON requires `source_sha256`, `target_sha256`, `target_name`, `units: mm`, `axes: RAS|LPS`, `direction: native_to_MNI|MNI_to_native`, and an invertible 4×4 `matrix`.
- Optional `reference_path` is hash-checked when available. Source hash must match exactly.
- LPS conversion uses `diag(-1,-1,1,1)` on both sides; reverse-direction matrices are inverted.
- Raw FSL/ITK/ANTs matrices and nonlinear warps require explicit external conversion; do not paste ambiguous numeric parameters.
- Optional registration: SimpleITK affine T1w correlation, 4/2/1 levels, 2/1/0 mm smoothing, seed 1729, one thread. Fixed-MNI → moving-native LPS is inverted for native → MNI RAS points.
- Review `qc_registration.png` and `T1w_in_MNI.nii.gz`; arithmetic round trips do not prove anatomical accuracy.

References: [NiBabel coordinates](https://nipy.org/nibabel/coordinate_systems.html),
[SimpleITK registration direction](https://simpleitk.readthedocs.io/en/latest/registrationOverview.html),
[NiiVue MRI controls](https://niivue.com/docs/api/niivue/classes/Niivue/).

## Outputs, caching and scope

| Location | Contents |
|---|---|
| `saves/derivatives/subjects/ID/` | `markers.json`, `coordinates.tsv`, `qc.json`, `manifest.json`, `transforms.json` |
| Same derivative folder | Full `.vtp` and viewer `.ply` scalp; optional skull; native `head_mask.nii.gz` |
| Same derivative folder | `qc_projection.png`, actual `qc_fid-*.png` and `qc_landmark-*.png`, correction template/applied input |
| Registration outputs | Resampled T1w, MNI-to-native LPS transform and overlay QC |
| `viewer/src/data/ID/` | Local static mesh/marker exports; no anatomical volume bundled |

- Content/options/relevant-code/software hashes and output checksums govern reuse; `--force` rebuilds.
- Writer locks and staged derivatives preserve complete outputs on failure. Remove stale locks only after confirming ingestion stopped; recover `.ID-previous` after interrupted replacement.
- Run ingestion serially; viewer exports share a generated registry. Removing input files does not remove old exports.
- Inputs, derivatives and static subject data are Git-ignored; public deployment allowlisting remains active.
- Detection can miss dark/merged/low-contrast markers or accept fat/artifacts. Large tilt, absent anatomy and poor surfaces require manual review.
- Restricted scans stay local. Synthetic tests do not establish accuracy on a real acquisition.

## Reproducible checks

```bash
python scripts/synthetic_subject_demo.py                         # original three-marker fixture
python scripts/synthetic_subject_demo.py --subject synthetic-five --five
python -m pytest tests/ -q
npm --prefix viewer test
cd viewer
npx playwright install chromium
npm run test:browser
npm run build
```

- Demo generation refuses to overwrite inputs. Existing demos can be re-ingested with their `synthetic_anchor_corrections.json` and `synthetic_skull_mask.nii.gz` options.
- Browser tests expect MNI152, `synthetic-demo`, and `synthetic-five`; API/Parcel start automatically.
- Figures and the latest task results: [subject-reconstruction-report.md](subject-reconstruction-report.md).
