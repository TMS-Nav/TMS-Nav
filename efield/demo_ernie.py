# a DEMO e field on simnibs's sample head ernie, for the viewer's aim 2 tab.
#
#   python -m efield.demo_ernie            write the job, run simnibs, then export
#   python -m efield.demo_ernie --dry-run  write the job and only dry run it
#
# the two placements come out of this project's own pipeline, the way method 2
# of the study does it. the eeg cap placement is the 10-20 site the aim 1
# pipeline built off ernie's scalp (markers.json electrodes), RECON. the mri
# guided placement goes on the scalp straight over the target, which here is
# aim 1's anatomical target pushed in to the cortex, not an fmri target. the
# runner puts that coil on the skin (place = over_target). handle directions are
# config.HANDLE_DIRECTIONS for both. nothing here is study data and run_efield.py
# never reads it. the export lands in viewer/src/data/Ernie, next to ernie's aim 1
# files, which are on the same t1
import argparse
import subprocess
import sys

import numpy as np

from efield import config
from efield.headmodel import format_command
from efield.job import Job, runner_command
from efield.reconstruction import cap_poses, load_markers, over_target_pose
from efield.roi import RoiSpec

SUBJECT = "demo_ernie"
VIEWER_SUBJECT = "Ernie"
MARKERS = config.ROOT / "viewer" / "src" / "data" / VIEWER_SUBJECT / "markers.json"
SITES = ("F4", "SMA")

# how far below the scalp target the roi seed goes. scalp to the middle of the
# grey matter is roughly this on ernie, so the snap onto the central surface is short
ROI_DEPTH_MM = 18.0


def m2m_ernie():
    for root in config.SAMPLE_M2M_ROOTS:
        p = root / "m2m_ernie"
        if p.is_dir():
            return p
    return None


def build_job(m2m, markers_path=MARKERS):
    markers = load_markers(markers_path)
    targets = {m["label"]: m for m in markers["targets"]}
    rois = []
    for site in SITES:
        t = targets[site]
        seed = np.asarray(t["position"], dtype=float) - np.asarray(t["normal"], dtype=float) * ROI_DEPTH_MM
        rois.append(RoiSpec(kind="fmri_target", centre=[float(x) for x in seed], site=site,
                            reference_method="demo, aim 1 target pushed in",
                            name=f"{site}_demo_{config.ROI_RADIUS_MM:g}mm"))
    # aim 1's sample heads come out of run_pipeline.py with no review status on
    # the cap, so the demo does not ask for one. run_efield.py does
    recon, missing = cap_poses(markers, SUBJECT, SITES, require_status=None)
    if missing:
        raise ValueError(f"ernie's markers.json has no electrode for {', '.join(missing)}")
    mri = [over_target_pose(r, SUBJECT) for r in rois]
    for p in mri + recon:
        p.name = f"demo_{p.site}_{p.method}"
        p.extra["demo"] = True
    return Job(subject=SUBJECT, m2m=str(m2m), poses=mri + recon, rois=rois, map_to_vol=True,
               snap_to_skin=True, poses_source=f"DEMO, {markers_path.name} of the aim 1 pipeline",
               notes=["demo only: the eeg cap placement is the aim 1 pipeline's 10-20 site on ernie,"
                      " the mri guided one sits over aim 1's anatomical target",
                      "roi seeds are aim 1 targets pushed in, not fmri targets",
                      "handle directions are config.HANDLE_DIRECTIONS, a placeholder convention"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="demo e field on ernie for the viewer")
    ap.add_argument("--dry-run", action="store_true", help="dry run the job only")
    ap.add_argument("--no-export", action="store_true", help="skip writing the viewer files")
    args = ap.parse_args(argv)

    m2m = m2m_ernie()
    if m2m is None:
        print("m2m_ernie not found under", ", ".join(str(r) for r in config.SAMPLE_M2M_ROOTS) or "any sample folder")
        print("fetch it with: python scripts/fetch_sample_data.py  (see docs/efield.md)")
        return 1
    if not MARKERS.exists():
        # the cap placements come off aim 1's reconstruction of ernie
        print(f"{MARKERS} is missing. build ernie's aim 1 files first: python run_pipeline.py saves/Ernie.nii.gz")
        return 1
    if not config.simnibs_available():
        print(f"simnibs python not found at {config.SIMNIBS_PYTHON}")
        print("install simnibs, or set TMSNAV_SIMNIBS_DIR to its folder (see docs/efield.md)")
        return 1

    job = build_job(m2m)
    job_path = job.write_json()
    print(job.summary())
    print(f"wrote {job_path}")

    cmd = runner_command(job_path, dry_run=args.dry_run)
    print(f"{'dry run' if args.dry_run else 'RUNNING'}: {format_command(cmd)}")
    out = subprocess.run(cmd, text=True)
    if out.returncode != 0:
        print(f"runner exit {out.returncode}")
        return out.returncode
    if args.dry_run or args.no_export:
        return 0

    from efield.export_web import export_subject
    export_subject(job.job_dir, VIEWER_SUBJECT, demo=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
