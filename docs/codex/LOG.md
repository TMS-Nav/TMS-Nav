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
