# Codex state

Updated: 2026-10-22

## Current objective
- Deliver native MRI inspection, five-marker review and live EEG updates; commit/push with `[codex]` prefix.

## Completed work
- NiiVue anatomical slice tab: linked planes, sliders, crosshair, intensity, contrast, zoom and overlays.
- Subject candidate review: bright/local detection, rejection diagnostics, physical filters, touching-peak split, manual add/confirm/exclude and undo.
- Loopback Python API and `run_viewer.py`; cached EEG previews, provisional confidence, correction saving/reopening and registration reuse.
- MNI pink fiducials hidden; existing MNI and subject visualization retained.
- Existing usage guide consolidated; existing report describes latest changes with synthetic figures. AGENTS.md documentation policy added; user's staged commit-prefix rule preserved.

## Verified results
- Python: `micromamba run -n tms312 python -m pytest tests/ -q` → **75 passed**.
- Viewer: `npm test` → **8 passed**; `npm run test:browser` → **2 passed**; production build passed.
- Actual browser save/reopen, five unequal markers, cached live cap, native coordinates and registration reuse covered.
- `pip check` and `git diff --check` passed. Main matched fetched origin/main at base **07fe146** before commit.

## Decisions / assumptions
- Restricted subject scans must not be searched for or accessed. User explicitly confirmed the restriction; all validation/figures use synthetic data.
- Five is a QC expectation, not a detection quota. LiquiMark 8 mm profile is an unvalidated starting point.
- Native RAS mm remains primary. Cap-marker centers keep their offset above scalp.
- Usable unconfirmed anchors allow labeled provisional live EEG; saved CLI output still requires confirmed anchors.
- Browser saves avoid Parcel reloads; API reopening reads current derivatives. Full viewer startup refreshes static exports.

## Uncommitted changes
- Task implementation, tests, existing documentation/figures and checkpoint updates are included together in the delivery commit; verify `git status` at session start.
- Inputs, derivatives and browser artifacts remain ignored. No private scan or subject information is included in checkpoints or figures.

## Remaining blockers / limits
- No implementation blocker. Acquisition-specific detection accuracy is unverified by user restriction.
- Existing gh-pages audit findings remain; software-WebGL initial shader setup can be slow.

## Specific next actions
1. Verify the task commit and remote synchronization using Git rather than assuming checkpoint state.
2. Use `docs/subject-reconstruction.md` for local review; `docs/subject-reconstruction-report.md` for latest changes/results.
3. Launch `micromamba run -n tms312 python run_viewer.py`. Review each expected marker locally, confirm four anchors, then save reviewed changes.
4. Do not request or search for the restricted scan in future sessions.
