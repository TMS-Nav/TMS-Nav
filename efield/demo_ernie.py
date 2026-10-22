# a DEMO e field on simnibs's sample head ernie, for the viewer's aim 2 tab.
#
#   python -m efield.demo_ernie            write the job, run simnibs, then export
#   python -m efield.demo_ernie --dry-run  write the job and only dry run it
#
# nothing here is study data. the coil sits on ernie's own 10-10 cap at F4 and
# at FCz for the SMA (the study puts SMA at the arc midpoint of Fz and Cz, which
# is where 10-10 puts FCz), with handle directions picked for the demo, not the
# protocol's. the roi seed is the electrode pushed straight into the head, not an
# fmri target. run_efield.py never reads this, the study pipeline stays real
# data only. the export lands in viewer/src/data/Ernie, next to ernie's aim 1 files
import argparse
import csv
import subprocess
import sys

import numpy as np

from efield import config
from efield.headmodel import format_command
from efield.job import Job, runner_command
from efield.poses import CoilPose
from efield.roi import RoiSpec

SUBJECT = "demo_ernie"
VIEWER_SUBJECT = "Ernie"
CAP = "EEG10-10_UI_Jurak_2007.csv"

# site, electrode under the coil centre, electrode the handle points at
POSES = [
    ("F4", "F4", "FC2"),
    ("SMA", "FCz", "Cz"),
]

# how far below the electrode the roi seed goes. scalp to the middle of the grey
# matter is roughly this on ernie, so the snap onto the central surface is short
ROI_DEPTH_MM = 18.0


def m2m_ernie():
    for root in config.SAMPLE_M2M_ROOTS:
        p = root / "m2m_ernie"
        if p.is_dir():
            return p
    return None


def read_cap(m2m):
    """Electrode name -> RAS mm, off the cap file charm wrote for this head."""

    out = {}
    with open(m2m / "eeg_positions" / CAP, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 5 and row[0] in ("Electrode", "ReferenceElectrode"):
                out[row[4].strip()] = np.array([float(v) for v in row[1:4]])
    return out


def build_job(m2m):
    cap = read_cap(m2m)
    # the cap's centroid sits inside the head, good enough for "inwards" here.
    # the runner snaps each coil onto the real skin with the real normal anyway
    inside = np.mean(np.stack(list(cap.values())), axis=0)
    poses, rois = [], []
    for site, centre_el, handle_el in POSES:
        c = cap[centre_el]
        inward = (inside - c) / np.linalg.norm(inside - c)
        handle = cap[handle_el] - c
        # placed off the cap, so it is an eeg cap placement in the study's terms
        poses.append(CoilPose(centre=c, ydir=handle, zdir=inward, site=site, method="EEG", rep=1,
                              name=f"demo_{site}", extra={"demo": True, "electrode": centre_el,
                                                          "handle_towards": handle_el}))
        rois.append(RoiSpec(kind="fmri_target", centre=list(c + inward * ROI_DEPTH_MM), site=site,
                            reference_method="demo, electrode pushed in",
                            name=f"{site}_demo_{config.ROI_RADIUS_MM:g}mm"))
    return Job(subject=SUBJECT, m2m=str(m2m), poses=poses, rois=rois, map_to_vol=True,
               snap_to_skin=True, poses_source=f"DEMO, ernie's {CAP}",
               notes=["demo only: poses on ernie's 10-10 cap, handle directions chosen for the demo",
                      "roi seeds are electrodes pushed in, not fmri targets"])


def main(argv=None):
    ap = argparse.ArgumentParser(description="demo e field on ernie for the viewer")
    ap.add_argument("--dry-run", action="store_true", help="dry run the job only")
    ap.add_argument("--no-export", action="store_true", help="skip writing the viewer files")
    args = ap.parse_args(argv)

    m2m = m2m_ernie()
    if m2m is None:
        print("m2m_ernie not found under", ", ".join(str(r) for r in config.SAMPLE_M2M_ROOTS))
        return 1
    if not config.simnibs_available():
        print(f"simnibs python not found at {config.SIMNIBS_PYTHON}")
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
