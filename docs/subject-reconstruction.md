# Subject-specific MRI reconstruction

`ingest_subject.py` accepts a subject ID and a native anatomical T1w NIfTI. It
reconstructs the outer scalp, detects supported bright marker candidates, proposes
reviewable anatomical landmarks, and exports cached subject derivatives to the
existing viewer. No functional MRI is needed. The reusable Python entry point is
`src.subject_ingest.ingest_subject`; it has no browser dependency.

## Setup and complete workflow

Run from the repository root. Python 3.12 and Node 22 are compatible and were
installed in the user-local micromamba environment `tms312`.

```bash
micromamba create -y -f environment.yml
# For Bash, initialize activation for this shell if needed:
eval "$(micromamba shell hook --shell bash)"
micromamba activate tms312
cd viewer
npm ci
cd ..
# Generate/cache the MNI demo once so it remains the default viewer selection.
python run_pipeline.py --template
# Single subject ingestion command:
python ingest_subject.py sub-001 /absolute/path/to/native_T1w.nii.gz
cd viewer
npm run dev
```

Open <http://localhost:1234>. Select `sub-001`. MNI152 stays the default whenever
its generated assets exist; the first available subject is the fallback if no
MNI assets have been generated. The viewer retains all processed subjects. Adding
subjects requires no source changes. Parcel watches the regenerated static
registry during development; rebuild with `npm run build` for a production build.
Nothing is deployed automatically.

Activation is optional: prefix any command with `micromamba run -n tms312`.
`environment.yml` includes `alsa-lib` for Linux browser tests; omit that package
on other platforms. `requirements-tms312-lock.txt` records the Python package
versions used for delivery. To reproduce those exact versions, create the Python
3.12/Node 22 environment, then `python -m pip install -r requirements-tms312-lock.txt`.
SimNIBS remains an optional external application with its own interpreter, as
specified in the existing electric-field workflow.

Prefer this dedicated project layout:

```text
saves/subjects/sub-001/
  T1w.nii.gz                # user-supplied native structural scan
  corrections.json          # optional; automatically used
  native_to_mni.json        # optional; automatically used
```

Then just run `python ingest_subject.py sub-001`. Exactly one of `T1w.nii.gz` and
`T1w.nii` must exist. Alternate paths remain supported. All defaults are anchored
to the repository even if invoked from another directory. IDs use letters,
numbers, `.`, `_`, `-`, beginning with a letter/number; `MNI152` is reserved.
Automatic discovery within `saves/subjects/` selects only `T1w.nii[.gz]`, so
optional skull masks are not ingested as additional subjects.
Legacy `run_pipeline.py <scan-or-folder>` now routes non-template scans through
this ingestion module, using its existing generated dataset keys. The explicit
subject command above gives stable IDs and picks up the dedicated corrections.

## What is measured, inferred, or unavailable

| Output | Method | Confidence / limits |
| --- | --- | --- |
| Scalp | Existing physical-distance head mask, contour and Taubin smoothing | Outer soft-tissue boundary; threshold/morphology dependent. Review cropping, hair, defacing and mask contour. |
| MRI fiducials | Bright connected components, volume 8–800 mm³, compactness and proximity to scalp, intensity-weighted centers | Candidate detection, no assumed marker count. Scores are uncalibrated morphology/contrast scores, capped at 0.85. False positives and negatives remain possible. |
| Landmarks | Centered scalp profile heuristics for Nz, Iz, LPA and RPA | Automatically flagged `underdetermined`, score 0.2. No claim of precise anatomical localization. Confirm on MRI before use. |
| EEG 10–20 | Existing connected scalp-plane arc lengths from confirmed Nz/Iz/LPA/RPA | Withheld until all four anchors are manually confirmed. Estimated geometry, not MRI detections or functional targets. Cz disagreement above 10 mm flags underdetermination. |
| Skull | Optional externally validated native binary skull mask, outer boundary | T1 alone does not reliably identify bone. No skull is fabricated by eroding the scalp. Skull layer is disabled without an actual mesh. |

Detection supports **bright, compact, superficial** markers. It does not reliably
support dark markers, arbitrary capsule shapes, markers deep inside tissue,
touching components, severe intensity bias or unknown acquisition protocols.
Fat, vessels and artifacts can resemble candidates. No real subject scan was
available for validation in this checkout. Zero detections means no supported
candidates were found, not that the subject has no markers. Review every candidate
in its three actual MRI slices (`qc_fid-*.png`); anchor slices are available as
`qc_landmark-*.png`. Adjust the original-intensity
threshold explicitly if needed:

```bash
python ingest_subject.py sub-001 --marker-threshold 750
```

Positions are not automatically assigned protocol/electrode identities from an
unknown cap arrangement. A unique confirmed landmark or usable electrode within
8 mm yields a **suggested** label requiring confirmation. Ambiguous candidates
stay unlabeled. Manual corrections can set `anatomical_label`, and can add missed
fiducials or exclude false positives. Anatomical heuristics assume the head is
approximately aligned to anatomical RAS directions; large head tilt and absent
facial/ear anatomy require manual correction.

Supply a validated skull segmentation if available:

```bash
python ingest_subject.py sub-001 --skull-mask /path/to/native_binary_skull.nii.gz
```

The mask must be binary, share the T1 shape/affine, and lie at least 98% inside the
head mask. This checks grid alignment/containment, not the correctness of tissue
segmentation. A filled shell is contoured to recover its outer surface; the scalp
and skull remain separate meshes and separate viewer controls.

## Physical coordinates and transforms

Primary point coordinates and every mesh vertex are **native anatomical RAS mm**:
X right, Y anterior, Z superior. No recentering or display-only rotation is applied
to exported meshes. NiBabel's affine maps voxel centers to RAS; see its
[coordinate documentation](https://nipy.org/nibabel/coordinate_systems.html).
Reflected and oblique orthogonal affines are supported. Meter/micron NIfTI units
are converted to mm. Coded qform/sform conflicts, unknown spatial units, singular
matrices, sheared voxel grids, 4D scans and template-coded native inputs are
rejected explicitly. Resolve the header or resample externally rather than
silently guessing. The code cannot determine whether an arbitrary intensity
volume is truly T1-weighted; supply the correct anatomical contrast.

An unregistered subject is fully usable in native space. To reuse a compatible
existing affine, supply `--transform transform.json` or place it in the dedicated
subject folder. The JSON contract is intentionally explicit:

```json
{
  "source_sha256": "SHA256_OF_THE_EXACT_NATIVE_NIFTI_FILE",
  "target_sha256": "SHA256_OF_THE_EXACT_REFERENCE_NIFTI_FILE",
  "target_name": "MNI152NLin2009aSym",
  "reference_path": "/optional/path/to/reference.nii.gz",
  "axes": "RAS",
  "units": "mm",
  "direction": "native_to_MNI",
  "matrix": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
}
```

The matrix above illustrates the format; **it is not a subject registration**.
Calculate hashes with `sha256sum <file>` (or Python `file_hash`). Both `RAS` and
`LPS` conventions and both `native_to_MNI` and `MNI_to_native` directions are
supported. LPS is converted by `diag(-1,-1,1,1)` on both sides; reverse-direction
matrices are inverted. The source hash must match. If `reference_path` exists,
its hash is checked too; otherwise the target identity is recorded as supplied
and cannot be independently verified. FSL voxel/scaled-voxel `.mat`, arbitrary
ITK/ANTs files and nonlinear warps are **not** accepted as ambiguous raw matrices.
Convert them to this physical-coordinate contract with their image geometry and
direction established first. Do not copy their numeric parameters blindly.

Without an existing compatible transform, a reproducible affine registration is
available using an explicit full-head MNI T1 reference:

```bash
python ingest_subject.py sub-001 --register-to \
  saves/templates/icbm152_2009/mni_icbm152_nlin_sym_09a/mni_icbm152_t1_tal_nlin_sym_09a.nii.gz
```

This uses SimpleITK T1w correlation, physical optimizer scales, 4/2/1 resolution
levels, 2/1/0 mm smoothing, 15% sampling with seed 1729 and one thread. The affine
initialization aligns image geometry; this is not nonlinear normalization.
SimpleITK estimates **fixed MNI → moving native LPS**, so the code inverts and
converts it to **native → MNI RAS** for point coordinates, following the official
[registration direction convention](https://simpleitk.readthedocs.io/en/latest/registrationOverview.html).
Round-trip checks establish arithmetic consistency, not anatomical accuracy.
Review `qc_registration.png` and `T1w_in_MNI.nii.gz`; all registration outputs
remain flagged for visual review. A plausible determinant is not proof of a good
registration. Severe cropping, defacing and large rotations can defeat alignment.

`transforms.json` records normalized forward/inverse matrices, source/reference
hashes, imported-transform hash or registration settings, and review status.
Registration also retains `MNI_to_native_LPS.tfm` as the resampling transform.
Copy the normalized `transforms.json` to `saves/subjects/sub-001/native_to_mni.json`
to reuse a computed transform. Primary coordinates stay native; `mni_position`
fields are secondary values, shown on point selection when available.

## Manual correction and confirmation

1. Select a subject. Yellow cubes are anatomical landmarks, pink diamonds are
   MRI fiducials, cyan spheres are scalp electrode estimates. Each category has
   an independent layer checkbox, labels, click selection and coordinate inspection.
2. Choose a landmark/fiducial in the editor or click it. Enter native RAS mm
   coordinates, apply them, or use **Pick on scalp**. For a marker center outside
   skin, uncheck **constrain entries to scalp**; snapping would move it off its
   measured center. Confirm a correct anchor by applying its coordinates unchanged.
3. Edits persist per subject in browser storage and hide the old EEG overlay until
   recomputation. **Reset selected** restores its exported position/status.
4. **Export JSON** downloads source-bound corrections. It includes manually
   confirmed anchors and edited candidates, not unreviewed heuristic anchors.
   Save it as `saves/subjects/sub-001/corrections.json`, then rerun
   `python ingest_subject.py sub-001`. The CLI cannot read browser storage directly.
5. Reload the viewer after ingestion. The saved correction provenance and revised
   electrodes now come from the subject derivatives.

You can instead edit the derivative `corrections_template.json`, review its
coordinates against slice QC, and save the reviewed entries in the dedicated
input folder. Remove any entries you have not confirmed: entries supplied in a
correction file count as human confirmation. Required metadata is subject ID,
`source_sha256`, `axes: RAS`, `units: mm`, and
`coordinate_frame: {"name":"subject_RAS"}`. Editing or excluding detected fiducials
also requires the current `detection_signature` from the viewer export or correction
template. This prevents an old exclusion from removing a different candidate after
threshold/detector changes; review and re-export candidates when the signature changes. Example entries:

```json
{
  "landmarks": [{"name":"nasion","label":"Nz","position":[1,80,-20]}],
  "fiducials": [
    {"name":"fid-001","position":[40,55,65],"anatomical_label":"F4"},
    {"name":"fid-002","exclude":true},
    {"name":"manual-001","label":"manual-001","position":[-40,55,65]}
  ]
}
```

This fragment needs the metadata above and subject-specific measured values.
Landmark names/labels are `nasion/Nz`, `inion/Iz`, `LPA/LPA`, `RPA/RPA`, `Cz/Cz`.
Anchor laterality, spacing, field of view and distance to scalp are checked before
EEG construction. Human confirmation scores of 1 indicate confirmation, not a
calibrated probability. New corrections replace the previous correction input;
keep all previously confirmed entries you intend to retain. Browser exports
include original confirmed anchors automatically.

## Derivatives, cache and QC

```text
saves/derivatives/subjects/sub-001/
  manifest.json              # source/options/code/software hashes and output checksums
  markers.json               # category, native positions, confidence, status; optional MNI positions
  coordinates.tsv            # reusable native RAS mm coordinate table
  scalp.vtp / scalp.ply      # full-resolution / decimated viewer mesh
  skull.vtp / skull.ply      # only when supplied by skull segmentation
  head_mask.nii.gz           # derived native-grid head mask
  transforms.json            # affine provenance or explicit unregistered status
  qc.json                    # segmentation, detection and electrode QC
  qc_projection.png          # three native slices; points explicitly labeled as projections
  qc_fid-*.png               # three actual slices through each marker center
  qc_landmark-*.png          # actual slices through each proposed/corrected anchor
  corrections_template.json  # for review; never silently applied
  corrections_applied.json   # exact supplied correction file, when present
  T1w_in_MNI.nii.gz          # registration only
  MNI_to_native_LPS.tfm      # registration only
  qc_registration.png        # registration only
viewer/src/data/sub-001/
  scalp.ply / markers.json / [skull.ply]
```

Inputs remain unchanged. SHA256 fingerprints include original inputs, correction,
transform/reference/skull inputs, processing options, source code and relevant
software versions. Identical reruns verify derivative checksums and reuse the
cache; `--force` explicitly rebuilds. Each subject has a writer lock, staged
outputs, and replacement only after processing succeeds. A stale lock from a
killed process can be removed after confirming no ingestion is running. A
`.ID-previous` directory after an interrupted replacement needs manual recovery
before retrying. Viewer copies are generated local assets; they can be regenerated
from the cache. Avoid simultaneous ingestion of different subjects while Parcel
builds: the shared static registry is regenerated after each export.

Both input/derivative folders and viewer data are Git-ignored. Existing public
deployment checks reject unallowlisted subjects. Do not add patient datasets to
the public allowlist. Removing an input does not delete its processed subject;
remove the generated subject folders and regenerate the registry if desired.

## Reproducible verification without real subject MRI

```bash
python scripts/synthetic_subject_demo.py
# Creates synthetic-demo: three bright spheres, an ellipsoid scalp,
# explicitly supplied ground-truth anchors and synthetic skull mask.
# Refuses to overwrite an existing demo input; use --subject synthetic-demo-2.
python -m pytest tests/ -q
cd viewer
npm test
npx playwright install chromium
npm run test:browser
npm run build
```

Browser integration expects both `MNI152` (run `run_pipeline.py --template`) and
`synthetic-demo` assets. The test uses software WebGL, switches MNI → subject →
MNI, checks layers, edits a fiducial, inspects coordinates/confidence, validates
correction JSON and browser persistence, and saves a screenshot in
`viewer/test-results/synthetic-subject.png`. Python tests exercise reflected and
oblique affines, units, transform direction/LPS conversion, exact source identity,
known physical translation registration, marker centers, no-marker behavior,
correction recomputation, optional skull alignment and cache corruption.

The phantom is an engineering fixture, not human MRI and not evidence of
clinical marker detection accuracy. Its anchor coordinates are explicitly supplied
from construction truth, not inferred anatomical detections. See the
[delivery report](subject-reconstruction-report.md) for actual verification results.
