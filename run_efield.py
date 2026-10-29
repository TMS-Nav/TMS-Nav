# aim 2 driver. reads the recorded coil poses out of the brainsight exports,
# finds each subject's head model, puts the roi at the r01 target, writes the
# simnibs job and (dry run by default) asks the simnibs python to run it.
# results that exist get folded into the long table and the site statistics.
# mirrors run_pipeline.py, which is the aim 1 half.
#
#   python run_efield.py                       dry run, every subject in saves/brainsight/
#   python run_efield.py --subjects 001 002    a few of them
#   python run_efield.py --run                 actually simulate, only on purpose
#   python run_efield.py --charm               actually build missing head models, same
#
# what each subject needs. when a piece is missing the subject or site is
# skipped and the reason printed, nothing is made up in its place:
#   saves/brainsight/sub-<ID>_coil_pose_samples.txt   the recorded poses, exported in
#       NIfTI:Aligned, samples named sub-<ID>_EEGCAP_<SITE>_POSE_REP<n> and
#       sub-<ID>_MRI_<SITE>_POSE_REP<n>
#   saves/fmri_targets.csv                            the r01 target per site, subject
#                                                     column sub-<ID>
#   saves/m2m/m2m_sub-<ID>                            the head model, charm run on the
#                                                     same t1 brainsight was given
import argparse
import subprocess

from efield import config
from efield.brainsight import poses_from_export
from efield.config import FMRI_TARGETS
from efield.headmodel import find_m2m, format_command, run_charm
from efield.job import Job, dryrun_path, results_path, runner_command
from efield.metrics import format_site_stats, load_results, long_rows, site_stats, write_csv
from efield.poses import displacement_mm, orientation_deg, tilt_deg
from efield.roi import load_target_table, rois_for_subject
from src.aim1 import export_files
from src.mri_inputs import SAVE_DIR, find_mris

# the study sites that get a coil. c3, c4 and cz are cap references only
SITES = ("SMA", "F4")


def subject_id(sub):
    """sub-<ID> whether or not the prefix was typed."""

    return sub if sub.startswith("sub-") else f"sub-{sub}"


# --- the recorded poses -----------------------------------------------------------
def load_poses(folder):
    """{subject: [CoilPose]} and {subject: [export file]} from every export in folder.

    only NIfTI:Aligned exports are used. aim 1 can flip a brainsight LPS export
    because distances and angles survive the flip, but simnibs needs the pose in
    the exact scanner space of the head model, so those files are reported and
    left out. a pose named twice keeps the later one, as in src/aim1.py
    """

    keyed, sources = {}, {}
    for path in export_files(folder):
        try:
            poses, skipped = poses_from_export(path)
        except (ValueError, KeyError, IndexError) as err:
            print(f"  {path.name}: not used, {err}")
            continue
        if skipped:
            print(f"  {path.name}: skipped {len(skipped)} samples not named per the protocol:"
                  f" {', '.join(skipped[:6])}{' ...' if len(skipped) > 6 else ''}")
        for p in poses:
            if p.site not in SITES:
                continue
            subject = subject_id(p.extra["subject"])
            key = (subject, p.method, p.site, int(p.rep))
            if key in keyed:
                print(f"  {subject} {p.method} {p.site} rep {p.rep}: named twice, using the later one ({path.name})")
            keyed[key] = p
            sources.setdefault(subject, [])
            if path.name not in sources[subject]:
                sources[subject].append(path.name)

    by_subject = {}
    for (subject, *_), p in keyed.items():
        by_subject.setdefault(subject, []).append(p)
    return by_subject, sources


# --- the head model ------------------------------------------------------------------
def find_t1(subject):
    """(path, None) for the one structural scan of this subject under saves/, else (None, why).

    a scan belongs to the subject when a folder on its path is named sub-<ID> or the
    file name starts with it. charm has to get the same t1 brainsight was given, so
    zero or several candidates is reported rather than picked from
    """

    hits = [p for p in find_mris(SAVE_DIR)
            if subject in p.relative_to(SAVE_DIR).parts[:-1]
            or p.name.split(".")[0] == subject or p.name.startswith(subject + "_")]
    if len(hits) == 1:
        return hits[0], None
    if not hits:
        return None, f"no t1 for {subject} under {SAVE_DIR}"
    return None, f"{len(hits)} scans could be {subject}'s t1: " + ", ".join(str(h) for h in hits)


def head_model(subject, args):
    """The complete m2m for a subject, or None after saying what is missing."""

    root = args.m2m_root
    m2m = find_m2m(subject, roots=[root]) or find_m2m(subject[4:], roots=[root])
    if m2m is None:
        print(f"  no head model, looked for m2m_{subject} and m2m_{subject[4:]} in {root}")
        t1, why = find_t1(subject)
        if t1 is None:
            print(f"  {why}, so charm cannot be run either")
            return None
        run_charm(subject, t1, out_root=root, dry_run=not args.charm)
        if not args.charm:
            print("    (pass --charm to build it, about an hour per head)")
            return None
        m2m = find_m2m(subject, roots=[root])
        if m2m is None:
            print(f"  charm finished but m2m_{subject} is not in {root}")
            return None
    print("  " + m2m.report())
    if not m2m.complete or not m2m.has_central_surfaces:
        print("  head model is not complete, not simulated")
        return None
    return m2m


# --- one subject --------------------------------------------------------------------
def process(subject, poses, source, args, rows, targets):
    print(f"\n=== {subject} ===")
    print(f"  poses from {', '.join(source)}: {len(poses)}")

    # the pose to pose numbers aim 1 also reports, here on the same poses
    by_key = {p.key(): p for p in poses}
    for site in sorted({p.site for p in poses}):
        for rep in sorted({p.rep for p in poses if p.site == site and p.method == "EEG"}):
            a = by_key.get((site, "EEG", rep))
            b = by_key.get((site, "MRI", rep))
            if b is None:
                mri = [p for p in poses if p.site == site and p.method == "MRI"]
                b = mri[0] if len(mri) == 1 else None
            if a and b:
                print(f"  {site:4s} eeg rep {rep} vs mri rep {b.rep}  {displacement_mm(a, b):.1f} mm, "
                      f"tilt {tilt_deg(a, b):.1f} deg, orientation {orientation_deg(a, b):.1f} deg")

    # the roi is the r01 target. a site without one is not simulated, the field
    # would have nowhere meaningful to be read
    rois, missing = rois_for_subject(subject, poses, table=targets)
    if missing:
        print(f"  no r01 target for {', '.join(missing)} in {args.fmri_targets}, not simulated")
        poses = [p for p in poses if p.site not in missing]
    if not poses:
        print("  nothing left to simulate")
        return
    for r in rois:
        where = "mni" if r.kind == "mni_sphere" else "subject"
        coords = r.mni if r.kind == "mni_sphere" else r.centre
        print(f"  roi {r.site:4s} at the r01 target, {where} [{', '.join(f'{v:.1f}' for v in coords)}]")

    m2m = head_model(subject, args)
    if m2m is None:
        return

    job = Job(subject=subject, m2m=str(m2m.path), poses=poses, rois=rois,
              poses_source=", ".join(source))
    job_path = job.write_json()
    print(job.summary())
    print(f"  wrote {job_path}")

    if args.no_runner:
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
        new = long_rows(load_results(res))
        rows += new
        print(f"  results.json found, {len(new)} rows added to the long table")


def main(argv=None):
    ap = argparse.ArgumentParser(description="aim 2 driver, dry run unless told otherwise")
    ap.add_argument("--subjects", nargs="*", help="subject ids, 001 or sub-001, default every subject in the exports")
    ap.add_argument("--m2m-root", default=str(config.M2M_ROOT), help="where m2m_sub-<ID> folders live")
    ap.add_argument("--brainsight-root", default=str(config.BRAINSIGHT_ROOT))
    ap.add_argument("--fmri-targets", default=str(FMRI_TARGETS),
                    help="csv of the r01 targets, see efield/fmri_targets_example.csv")
    ap.add_argument("--dry-run", action="store_true", help="the default, accepted for clarity")
    ap.add_argument("--run", action="store_true", help="really run simnibs")
    ap.add_argument("--charm", action="store_true", help="really build missing head models with charm")
    ap.add_argument("--no-runner", action="store_true", help="write the jobs, do not call simnibs at all")
    args = ap.parse_args(argv)
    if args.run and args.dry_run:
        print("--run and --dry-run together, staying dry")
        args.run = False

    print(config.describe())
    if config.simnibs_available():
        print(f"simnibs version  {config.simnibs_version() or 'could not be read'}")
    print(f"mode             {'RUN' if args.run else 'dry run'}{', CHARM' if args.charm else ''}")

    # one read of the r01 target table for the whole run
    targets = load_target_table(args.fmri_targets)
    if targets:
        print(f"r01 targets      {len(targets)} rows from {args.fmri_targets}")
    else:
        print(f"r01 targets      NONE at {args.fmri_targets}, no site can be simulated without one")

    print(f"brainsight       {args.brainsight_root}")
    by_subject, sources = load_poses(args.brainsight_root)
    subjects = sorted(by_subject)
    if args.subjects:
        wanted = [subject_id(s) for s in args.subjects]
        for s in wanted:
            if s not in by_subject:
                print(f"  {s}: no poses in any export")
        subjects = [s for s in wanted if s in by_subject]
    if not subjects:
        print(f"no subjects, no protocol named SMA or F4 poses in {args.brainsight_root}")
        return
    print(f"subjects         {', '.join(subjects)}")

    rows = []
    for s in subjects:
        process(s, by_subject[s], sources[s], args, rows, targets)

    print()
    if rows:
        out = write_csv(rows, config.SIM_ROOT / "efield_long.csv")
        print(f"wrote {out} ({len(rows)} rows)")
        print(format_site_stats(site_stats(rows)))
    else:
        print("no results.json anywhere yet, so no long table. run with --run to simulate.")


if __name__ == "__main__":
    main()
