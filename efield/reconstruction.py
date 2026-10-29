# method 2 of the proposal: the cap placements built off the mri reconstruction
# of the cap, and the mri guided placement to compare them with when brainsight
# did not record one.
#
# src/subject_ingest.py reads the cap markers off the scan and writes the 10-20
# sites it builds from them to saves/derivatives/subjects/<subject>/markers.json,
# each a scalp point plus its outward normal in native RAS mm. a coil sat flat
# on the scalp at the site's electrode is the eeg cap placement. it has the
# same frame as the scan, so charm has to be run on that same scan.
#
# the mri guided placement is the brainsight recording when there is one. when
# there is not, the coil goes on the scalp straight over the r01 target, the
# skin point closest to it, which is where neuronavigation puts it. that needs
# the head model's skin, so the pose only carries the target here and the
# simnibs runner finishes it (place = over_target)
import json
from pathlib import Path

import numpy as np

from efield.config import COIL_SKIN_DISTANCE_MM, DERIVATIVE_ROOT, HANDLE_DIRECTIONS
from efield.poses import CoilPose

# what the cap reconstruction has to have reached before its sites are used.
# provisional sites come off anchors nobody confirmed, fine for a live preview,
# not for a result
USABLE_EEG_STATUS = "estimated_from_confirmed_anchors"


def markers_path(subject, root=DERIVATIVE_ROOT):
    """markers.json for sub-<ID>, or for the bare <ID> it was ingested under, else None."""

    root = Path(root)
    bare = subject[4:] if subject.startswith("sub-") else subject
    for name in (subject, bare, f"sub-{bare}"):
        p = root / name / "markers.json"
        if p.exists():
            return p
    return None


def reconstructed_subjects(root=DERIVATIVE_ROOT):
    """sub-<ID> for every ingested subject that has a markers.json."""

    root = Path(root)
    if not root.is_dir():
        return []
    out = []
    for p in sorted(root.glob("*/markers.json")):
        name = p.parent.name
        if name.startswith("."):
            continue
        out.append(name if name.startswith("sub-") else f"sub-{name}")
    return out


def load_markers(path):
    with open(path) as f:
        return json.load(f)


def handle_for(site, handle=None):
    """The handle direction for a site, the recorded one if given, else the convention."""

    if handle is not None:
        return np.asarray(handle, dtype=float)
    if site not in HANDLE_DIRECTIONS:
        raise ValueError(f"no handle direction for site {site!r}, add it to config.HANDLE_DIRECTIONS")
    return np.asarray(HANDLE_DIRECTIONS[site], dtype=float)


def cap_poses(markers, subject, sites, handles=None, distance=COIL_SKIN_DISTANCE_MM,
              require_status=USABLE_EEG_STATUS):
    """([CoilPose], {site: why}) for the reconstructed cap, method RECON, rep 1.

    one pose per site whose electrode the reconstruction built. handles maps a
    site to a handle direction to copy, normally the mri guided recording, so
    the two placements differ in where the coil is and not in how it was turned.
    a site without one gets the convention in config.HANDLE_DIRECTIONS. nothing
    is returned when the reconstruction is not usable, with the reason
    """

    handles = handles or {}
    qc = markers.get("eeg_qc", {})
    status = qc.get("status")
    if require_status and status != require_status:
        why = qc.get("reason") or f"cap reconstruction status is {status!r}, needs {require_status!r}"
        return [], {s: why for s in sites}
    by_label = {m["label"]: m for m in markers.get("electrodes", [])}
    poses, missing = [], {}
    for site in sites:
        m = by_label.get(site)
        if m is None:
            missing[site] = "the reconstruction has no electrode for this site"
            continue
        poses.append(CoilPose.from_scalp_point(
            m["position"], m["normal"], handle_for(site, handles.get(site)), distance,
            name=f"{subject}_RECON_{site}_POSE_REP1", site=site, method="RECON", rep=1,
            extra={"source": "mri reconstruction of the cap", "electrode": m["name"],
                   "eeg_status": status, "handle": "copied" if site in handles else "convention"}))
    return poses, missing


def over_target_pose(roi, subject, handle=None):
    """The mri guided pose for a site with no brainsight recording.

    the coil goes on the skin straight over the roi's target. the runner finds
    that skin point on the head model and builds the matrix there. what is
    stored here is only a placeholder at the target itself so the pose is a
    valid frame until then
    """

    site = roi.site
    ydir = handle_for(site, handle)
    extra = {"place": "over_target", "source": "r01 target, coil over it on the scalp"}
    if roi.kind == "mni_sphere":
        extra["target_mni"] = [float(x) for x in roi.mni]
        centre = [0.0, 0.0, 0.0]
    else:
        extra["target"] = [float(x) for x in roi.centre]
        centre = roi.centre
    # straight down is never parallel to a handle in the convention, which lies flat
    return CoilPose(centre=centre, ydir=ydir, zdir=[0.0, 0.0, -1.0],
                    name=f"{subject}_MRI_{site}_OVERTARGET_REP1", site=site, method="MRI", rep=1,
                    extra=extra)


def method2_poses(subject, recorded, rois, root=DERIVATIVE_ROOT, log=print):
    """The RECON poses for a subject, plus an over target MRI pose where none was recorded.

    recorded is the subject's brainsight poses (may be empty). the mri guided
    pose at a site is copied from there when it exists, and its handle is
    what the reconstructed placement copies. rois are the r01 targets, one per site
    """

    path = markers_path(subject, root)
    if path is None:
        return []
    markers = load_markers(path)
    sites = [r.site for r in rois]
    mri = {}
    for p in recorded:
        if p.method == "MRI" and p.site in sites:
            mri.setdefault(p.site, p)        # the first recorded rep sets the handle
    handles = {site: p.ydir for site, p in mri.items()}
    poses, missing = cap_poses(markers, subject, sites, handles=handles)
    for site, why in missing.items():
        log(f"  method 2 {site}: {why}")
    added = []
    for r in rois:
        if r.site in mri or not any(p.site == r.site for p in poses):
            continue
        added.append(over_target_pose(r, subject))
        log(f"  method 2 {r.site}: no recorded mri guided pose, the coil goes over the r01 target")
    if poses:
        log(f"  method 2 poses from {path}: {', '.join(p.site for p in poses)}")
    return poses + added
