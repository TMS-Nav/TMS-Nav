# Codex state

Updated: 2026-10-21

## Current objective

Deliver and commit/push the subject-specific MRI reconstruction pipeline and
initialize persistent repository handoff according to AGENTS.md.

## Completed work

- Native T1w subject ingestion CLI/API, default dedicated input folders, cached
  derivatives, confidence/QC, saved manual corrections and explicit affine provenance.
- Bright superficial fiducial candidates, scalp reconstruction, optional external
  skull mask, landmark confirmation and subject-specific EEG scalp constructions.
- Viewer subject overlays, selection/edit/export, MNI default and subject views
  without Monte Carlo samples. Synthetic demo and reproducible registration/tests.
- Environment specification, dependency snapshot, usage guide, delivery report
  and explicitly synthetic QC figures. See docs/subject-reconstruction.md and
  docs/subject-reconstruction-report.md.

## Verified results

- Implementation verification in this thread: 66 Python tests, 8 JavaScript tests,
  1 Chromium/WebGL integration test and production viewer build passed.
- Actual browser-exported correction successfully applied through Python;
  repeated ingestion reused its cache. pip check passed.
- Commit preparation: git diff --check passed; browser last-run artifact reports
  passed. Tests were not rerun for this documentation/commit-only handoff.
- Branch: main. Before commit, HEAD and fetched origin/main matched at b1ef3ab.

## Decisions and assumptions

- Primary coordinates are native anatomical RAS mm; MNI values are secondary.
- Marker scores are uncalibrated candidate scores. No real subject MRI was available.
- Confirm the four anatomical anchors before EEG construction; do not fabricate
  identities or skull geometry. Skull requires a separately validated mask.
- Inputs/derivatives remain Git-ignored. Tracked figures contain synthetic data only.

## Uncommitted changes

The completed implementation, AGENTS.md and these new checkpoint documents are
included together in the delivery commit. Check git status at the next session;
local MRI caches, viewer data and test outputs intentionally remain ignored.

## Remaining blockers / limitations

No blocker to delivery. Real-acquisition validation remains outstanding. Detection
supports bright compact superficial candidates; registration is affine and needs
visual review. Browser corrections require JSON export and CLI recomputation.
Existing gh-pages dependency audit findings are documented in the delivery report.

## Specific next actions

1. Verify the delivery commit and origin/main synchronization with git status -sb
   and git log -1; do not assume remote state from this checkpoint alone.
2. For real data, follow docs/subject-reconstruction.md; review marker/landmark QC
   and confirm anchors before interpreting electrode estimates.
3. Run ingestion serially because viewer exports share a generated registry.
