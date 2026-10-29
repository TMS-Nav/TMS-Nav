# Codex log

## 2026-10-21 — MRI reconstruction delivery and persistent handoff initialization

Implemented the end-to-end subject pipeline documented in
`docs/subject-reconstruction.md`; verification and limitations are recorded in
`docs/subject-reconstruction-report.md`. Important files include
`ingest_subject.py`, `src/subject_ingest.py`, `src/subject_coordinates.py`,
`src/fiducials.py`, `src/subject_registration.py`, `src/subject_qc.py`,
`src/synthetic_subject.py`, viewer integration and Python/Node/Playwright tests.

Verified during implementation in this thread:

- `micromamba run -n tms312 python -m pytest tests/ -q`: 66 passed.
- `cd viewer && micromamba run -n tms312 npm test`: 8 passed.
- `cd viewer && micromamba run -n tms312 npm run test:browser`: 1 passed.
- `cd viewer && micromamba run -n tms312 npm run build`: passed.
- MNI template rebuilt; synthetic marker detection, native coordinate/surface
  alignment, cache reuse and actual browser-to-Python correction round trip passed.
- Python 3.12.15 / Node 22.23.2 in user-local tms312; pip check passed.

Decisions: native RAS mm remains primary; underdetermined anatomy is explicit;
EEG requires confirmed anchors; no skull is inferred by scalp erosion. Detection
and registration remain reviewable estimates, not validated clinical measurements.
No private subject information is recorded in these checkpoints.

For the requested commit/push, read the newly supplied AGENTS.md. STATE.md and
LOG.md were absent and have been created. Reviewed tracked/untracked changes and
ignored data boundaries, confirmed git diff --check, and fetched origin. Branch
main and origin/main matched at base commit b1ef3ab before the delivery commit.
The latest browser test artifact also reports passed. No implementation changes
were needed for this commit-only task; the full test suite was not repeated.
This entry is included in the implementation delivery commit; its resulting hash
is discoverable with git log. Verify push success and synchronization directly
rather than assuming it from this log.

## 2026-10-22 — MRI review implementation started

- Read STATE/LOG/AGENTS and verified main at the existing delivery; user has a staged AGENTS.md change requiring `[codex]` commit prefixes. Preserve it.
- Implementing candidate rejection reasons, configurable bright/local filters and peak separation, cached loopback MRI/EEG API, NiiVue native slices and crosshair editing, and subject-only fiducials.
- Expected count five remains a QC expectation; no forced detections. Provisional EEG previews retain unconfirmed-anchor status.
- Added requested concise documentation policy to AGENTS.md. No real failing subject scan is present; requested its path while proceeding with synthetic verification.
- Tests/build are in progress; results pending.

## 2026-10-22 — MRI review verification and delivery

- Delivered `run_viewer.py`, `src/viewer_service.py`, `viewer/src/mriViewer.js` and `subjectReview.js`; updated detector, ingestion, editor and viewer integration.
- Found/fixed a faint synthetic sphere rejected as elongated when thresholded voxels occupied one plane: finite voxel-box covariance recovered marker five. Added configurable local contrast/physical filters and optional touching-peak separation.
- Normalized detector numeric types so browser/Python JSON retains stable candidate identity. Guarded late responses and metadata loading against replacing newer point edits; serialized MRI uploads.
- Live previews reuse the full scalp and retain provisional anchor confidence. Save persists reviewed inputs/derivatives and reuses existing registration; API reopening avoids static-bundle/HMR reloads.
- Verification: **75 Python tests, 8 JS tests, 2 Chromium/WebGL workflows**, production build, pip check and diff hygiene passed. Tests cover actual saved browser corrections, reopening, five unequal markers, MRI coordinates and cached EEG changes.
- Consolidated usage into the existing subject guide (177 lines); replaced the existing task report with latest changes/results and refreshed synthetic figures. Added concise documentation policy to AGENTS.md.
- User clarified that the real acquisition is restricted: do not search for or access it. A pending search was rejected after that instruction and was not retried. No private subject details are stored here; all demonstration artifacts are synthetic.
- Main and fetched origin/main matched at base **07fe146** before the `[codex]` delivery commit. Verify final commit/push directly; see Git history for the resulting task hash.

- Final synthetic visual QC isolated a non-preserved WebGL/MSAA reset capture artifact. MRI now retains complete frames without MSAA; a browser pixel check covers both halves of the sagittal phantom after reset.

## 2026-10-29 — Viewer sources dialog and Aim 2 forward model review

- Added a sources, licenses and disclaimer `<dialog>` opened from the left-panel credit: `viewer/src/sources.js`, `index.html`, `style.css`, `tests/browser/sources.spec.mjs`.
- Verified `npm test` (21 passed), a scratch production build, and the dialog by hand in the in-app browser (desktop and 375 px; Escape, backdrop, close and focus return all work). The Playwright spec was not run because the browser binaries are missing.
- Reread the proposal for the PI's "integrate the forward model" goal. The SimNIBS forward model is wired for Brainsight poses (Method 1). Method 2 poses from the MRI reconstruction are not connected. Details in `docs/aim2-integration-report.md`.
- Checked the DOIs for Lepping 2016 and Thielscher 2015 online. The others still need a spot check.

## 2026-10-29 — Method 2 forward model, difference map, perpendicular field, coil current

- Method 2 connected: `efield/reconstruction.py` (RECON poses from the cap reconstruction, over-target MRI pose), `runner_simnibs.place_over_target`, `run_efield.py` subjects from reconstructions too. Pairs `RECON-MRI` alongside `EEG-MRI`; Dice NaN when both hotspots are empty.
- Ernie demo now uses the Aim 1 pipeline's own 10-20 sites (RECON) and Aim 1 targets (MRI over target). Real SimNIBS run of 4 poses passed; SMA −25.5 %, F4 −4.3 % mean |E| in the target. Old demo output kept in `saves/efield/demo_ernie_2026-10-22/`.
- Export writes E_normal per pose; viewer gains strength/perpendicular, minus-MRI-guided difference (diverging RdBu), coil current scaling. Sources dialog cut to one row per source.
- Verified: pytest 83 passed (incl. SimNIBS dry run with over-target pose), npm test 26 passed, scratch build passed, browser checks desktop and 375 px. Playwright not run (browsers missing).

## 2026-10-29 — PI follow-up: run the E-field on another machine

- Fast-forwarded main_ar to origin/main 78fce6b (PR #3 merge, no file changes).
- PI could not run the simulation: `efield/config.py` hard-coded this machine's SimNIBS and sample paths. Now: `TMSNAV_SIMNIBS_DIR` → `simnibs_python` on PATH → usual install folders; `.cmd` only on Windows; sample heads searched under `saves/sample_data/` and the sibling `Sample Datasets/`.
- Added `scripts/fetch_sample_data.py` (SimNIBS example 1.1 GB, group 2.7 GB, from the SimNIBS datasets page) and `docs/efield.md`; `efield/README.md` points there. `sample_data` added to scan discovery exclusions.
- Verified: pytest 86 passed, npm test 26 passed, demo dry run, missing-SimNIBS message. Real downloads not run.

## 2026-10-29 — gh-pages redeploy and sample heads in Aim 1

- Deployed main_ar 34a8f20 to gh-pages (a698659), checked live.
- User found Ernie and T1–T5 showed no Aim 1 targets, coil or Monte Carlo. Cause: since 07fe146 only MNI152 counted as a demo; every other head was treated as a patient scan. `write_registry` now marks heads on `public-subjects.json` (other than MNI152) as kind `sample`; the viewer's `isDemoDataset` gives them the demo layers and skips the cap-review API (removes the public-site 404).
- Aim 2 for MNI152 and T1–T5 left as "no e-field yet" at the user's choice; no simulations run for them.
- Verified: pytest 87 passed, npm test 27 passed, build passed, local viewer Ernie/T3/MNI152 Aim 1 metrics and layers.
