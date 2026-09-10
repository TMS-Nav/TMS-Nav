# aim 2 driver. for every subject in saves/ find or plan the head model, build
# the coil poses, write the simnibs job, and (dry run by default) ask the
# simnibs python to validate it. results that exist get folded into the long
# table and the site statistics. mirrors run_pipeline.py, which is the aim 1 half.
#
#   python run_efield.py                       dry run, every subject in saves/
#   python run_efield.py --subjects T1 T2      a few of them
#   python run_efield.py --m2m <path>          use a finished m2m (eg the simnibs
#                                              ernie example) for an end to end dry run
#   python run_efield.py --run                 actually simulate, hours, only on purpose
#   python run_efield.py --charm               actually build the head models, same
#
# where the poses come from, in order of preference:
#   1. saves/brainsight/<subject>_coil_pose_samples.txt, the real recorded poses
#   2. the m2m's own eeg_positions csv, when the head model is complete. F4 is
#      the F4 electrode and SMA is between Fz and Cz on the midline. these are in
#      the head model's own space, which is what matters for a dry run on ernie
#   3. viewer/src/data/<subject>/markers.json, the aim 1 targets on the subject's
#      own scalp. cheap to read (no mri reprocessing) and in the scan's RAS, so
#      right once that scan has been through charm. run run_pipeline.py first
# for 2 and 3 the MRI method IS the target and the EEG method is the target
# nudged by a documented offset, so the pairing code has something to pair.
import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from efield import config
from efield.brainsight import make_name, poses_from_export
from efield.headmodel import charm_command, find_m2m, format_command, run_charm
from efield.job import Job, dryrun_path, results_path, runner_command
from efield.metrics import format_site_stats, load_results, long_rows, site_stats, write_csv
from efield.poses import CoilPose, displacement_mm, orientation_deg, tilt_deg
from efield.roi import rois_for_poses
from src.stats import rotation_about

SAVE_DIR = Path("saves")
VIEWER_DATA_DIR = Path("viewer") / "src" / "data"

# the study sites the poses are built for. c3, c4, cz are references and get
# no coil in the protocol
SITES = ("SMA", "F4")

# the stand in eeg cap error, one placement per rep. shift along the coil x
# axis in mm, face tilt about the handle in degrees, handle yaw about the
# normal in degrees. the sizes are about one sd of aim 1's noise model
# (2 mm, 5 deg yaw) so the dry run numbers look like the real thing will
EEG_NUDGE = {1: (3.0, 3.0, 5.0), 2: (-2.0, -2.0, -4.0)}
# the mri guided repeat error, smaller, the navigator's own tre
MRI_NUDGE = {1: (0.0, 0.0, 0.0), 2: (0.8, 0.5, 1.5)}


# --- stand in poses --------------------------------------------------------------
def nudge(pose, shift_mm, tilt, yaw, **kw):
    """A copy of pose moved shift_mm along its x axis, tilted about its handle, handle yawed."""

    x, y, z = pose.xdir, pose.ydir, pose.zdir
    R_tilt = rotation_about(y, tilt)
    z2 = R_tilt @ z
    y2 = rotation_about(z2, yaw) @ (R_tilt @ y)
    return CoilPose(centre=pose.centre + x * shift_mm, ydir=y2, zdir=z2, **kw)


def both_methods(subject, site, reference):
    """MRI and EEG poses, two reps each, off one reference pose at a site."""

    out = []
    for method, table in (("MRI", MRI_NUDGE), ("EEG", EEG_NUDGE)):
        for rep, (s, t, yaw) in table.items():
            out.append(nudge(reference, s, t, yaw, name=make_name(subject, method, site, rep),
                             site=site, method=method, rep=rep))
    return out


def read_eeg_cap(path):
    """{label: xyz} from a simnibs eeg_positions csv."""

    pts = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 5 and row[0] in ("Electrode", "ReferenceElectrode", "Fiducial"):
                pts[row[4]] = np.array([float(v) for v in row[1:4]])
    return pts


def poses_from_eeg_cap(subject, m2m):
    """Stand in poses off the head model's own 10-20 positions."""

    pts = read_eeg_cap(m2m.eeg_cap)
    centre_of_head = np.mean([pts[k] for k in ("Nz", "Iz", "LPA", "RPA", "Cz") if k in pts], axis=0)
    sites = {"F4": pts["F4"], "SMA": 0.5 * (pts["Fz"] + pts["Cz"])}
    poses = []
    for site, p in sites.items():
        # radial direction as the outward normal. the runner snaps the coil onto
        # the skin with the real normal, this only has to be roughly right
        n_out = p - centre_of_head
        n_out /= np.linalg.norm(n_out)
        # handle pointing backwards, the usual for a frontal site. a stand in
        ref = CoilPose.from_scalp_point(p, n_out, [0.0, -1.0, 0.0], site=site)
        poses += both_methods(subject, site, ref)
    return poses


def poses_from_markers(subject, markers_path):
    """Stand in poses off the aim 1 targets the viewer already has."""

    import json
    with open(markers_path) as f:
        markers = json.load(f)
    poses = []
    for t in markers["targets"]:
        if t["label"] not in SITES:
            continue
        # handle backwards here too, projected onto the coil plane by CoilPose
        ref = CoilPose.from_scalp_point(t["position"], t["normal"], [0.0, -1.0, 0.0], site=t["label"])
        poses += both_methods(subject, t["label"], ref)
    return poses


def build_poses(subject, m2m, brainsight_root):
    """The poses for a subject and where they came from, see the header."""

    bs = Path(brainsight_root) / f"{subject}_coil_pose_samples.txt"
    if bs.exists():
        poses, skipped = poses_from_export(bs)
        if skipped:
            print(f"  brainsight: skipped {len(skipped)} rows not named per protocol: {skipped[:4]}")
        return poses, f"brainsight {bs}", False
    if m2m is not None and m2m.complete and m2m.has_eeg_positions:
        return poses_from_eeg_cap(subject, m2m), f"stand in, eeg cap of {m2m.path.name}", True
    markers = VIEWER_DATA_DIR / subject / "markers.json"
    if markers.exists():
        return poses_from_markers(subject, markers), f"stand in, {markers}", True
    return [], "none (no brainsight export, no m2m eeg cap, no markers.json)", False


# --- one subject --------------------------------------------------------------------
def process(subject, args, rows):
    print(f"\n=== {subject} ===")
    t1 = SAVE_DIR / f"{subject}.nii.gz"

    if args.m2m:
        m2m = find_m2m(args.m2m)
    else:
        m2m = find_m2m(subject, roots=[Path(args.m2m_root)] + config.SAMPLE_M2M_ROOTS)

    if m2m is None:
        print(f"no m2m for {subject}")
        if t1.exists():
            run_charm(subject, t1, out_root=args.m2m_root, dry_run=not args.charm)
            if not args.charm:
                print("    (pass --charm to build it, about an hour per head)")
        else:
            print(f"    and no {t1} to build one from")
    else:
        print(m2m.report())

    poses, source, stand_in = build_poses(subject, m2m, args.brainsight_root)
    if not poses:
        print(f"  no poses: {source}")
        return
    rois = rois_for_poses(poses)

    job = Job(subject=subject, m2m=str(m2m.path) if m2m else str(Path(args.m2m_root) / f"m2m_{subject}"),
              poses=poses, rois=rois, snap_to_skin=stand_in, poses_source=source)
    if stand_in:
        job.notes.append("poses are STAND INS built off a target plus a fixed nudge, not recorded coil poses")
    job_path = job.write_json()
    print(job.summary())
    print(f"  wrote {job_path}")

    # the pose to pose numbers aim 1 also reports, here on the actual poses
    by_key = {p.key(): p for p in poses}
    for site in sorted({p.site for p in poses}):
        for rep in sorted({p.rep for p in poses if p.site == site}):
            a, b = by_key.get((site, "EEG", rep)), by_key.get((site, "MRI", rep))
            if a and b:
                print(f"  {site:4s} rep {rep}: eeg vs mri  {displacement_mm(a, b):.1f} mm, "
                      f"tilt {tilt_deg(a, b):.1f} deg, orientation {orientation_deg(a, b):.1f} deg")

    if m2m is None or not m2m.complete:
        print("  head model not complete, runner not called")
    elif args.no_runner:
        print("  --no-runner, job written only")
    elif not config.simnibs_available():
        print(f"  simnibs python not found at {config.SIMNIBS_PYTHON}, runner not called")
    else:
        cmd = runner_command(job_path, dry_run=not args.run)
        print(f"  {'RUNNING' if args.run else 'dry run'}: {format_command(cmd)}")
        out = subprocess.run(cmd, capture_output=True, text=True)
        for ln in out.stdout.splitlines():
            print("    | " + ln)
        if out.returncode != 0:
            print(f"  runner exit {out.returncode}")
            for ln in out.stderr.splitlines()[-15:]:
                print("    ! " + ln)
        elif not args.run:
            print(f"  plan in {dryrun_path(job)}")

    res = results_path(job)
    if res.exists():
        r = load_results(res)
        new = long_rows(r)
        rows += new
        print(f"  results.json found, {len(new)} rows added to the long table")


def main(argv=None):
    ap = argparse.ArgumentParser(description="aim 2 driver, dry run unless told otherwise")
    ap.add_argument("--subjects", nargs="*", help="subject ids, default every saves/*.nii.gz")
    ap.add_argument("--m2m-root", default=str(config.M2M_ROOT), help="where m2m_<subject> folders live")
    ap.add_argument("--m2m", default=None, help="one m2m folder to use for every subject, eg m2m_ernie")
    ap.add_argument("--brainsight-root", default=str(config.BRAINSIGHT_ROOT))
    ap.add_argument("--dry-run", action="store_true", help="the default, accepted for clarity")
    ap.add_argument("--run", action="store_true", help="really run simnibs. hours per subject")
    ap.add_argument("--charm", action="store_true", help="really build missing head models with charm")
    ap.add_argument("--no-runner", action="store_true", help="write the jobs, do not call simnibs at all")
    args = ap.parse_args(argv)
    if args.run and args.dry_run:
        print("--run and --dry-run together, staying dry")
        args.run = False

    print(config.describe())
    if config.simnibs_available():
        print(f"simnibs version  {config.simnibs_version() or 'could not be read'}")

    subjects = args.subjects or sorted(p.name.split(".")[0] for p in SAVE_DIR.glob("*.nii.gz"))
    if not subjects:
        print(f"no subjects, nothing in {SAVE_DIR}")
        return
    print(f"subjects         {', '.join(subjects)}")
    print(f"mode             {'RUN' if args.run else 'dry run'}{', CHARM' if args.charm else ''}")

    rows = []
    for s in subjects:
        process(s, args, rows)

    print()
    if rows:
        out = write_csv(rows, config.SIM_ROOT / "efield_long.csv")
        print(f"wrote {out} ({len(rows)} rows)")
        print(format_site_stats(site_stats(rows)))
    else:
        print("no results.json anywhere yet, so no long table. the endpoints need a real run.")
        print("analysis/efield_stats.py shows the statistics on a synthetic table meanwhile.")


if __name__ == "__main__":
    main()
