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
