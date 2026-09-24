# aim 1 on the real data. brainsight session exports in, one row per compared pair
# of coil placements out. the statistics on those rows are analysis/aim1_stats.py.
#
# drop the session sample exports (sub-<ID>_coil_pose_samples.txt in the acquisition
# protocol) into saves/brainsight/. every sample named the protocol way,
# sub-<ID>_EEGCAP_<SITE>_POSE_REP<n> and sub-<ID>_MRI_<SITE>_POSE_REP<n>, is one coil
# pose, the centre plus the full 3x3, read by efield/brainsight.py.
#
# what gets compared. at each subject and site, each eeg cap placement against the
# mri guided placement with the same rep number. when the mri side only has one
# placement, every eeg rep is compared with that one. per pair:
#
#   displacement_mm   distance between the two coil centres
#   tilt_deg          angle between the two coil normals, the face tilt
#   orientation_deg   total rotation between the two coil frames, tilt and handle
#   ap_mm             signed eeg minus mri offset, + means the eeg coil sits forward
#   lr_mm             + means the eeg coil sits to the subject's right
#   depth_mm          + means the eeg coil sits further out from the head
#   yaw_deg           signed handle turn, eeg minus mri, about the mri coil normal,
#                     + is anticlockwise seen from outside the head
#
# the three signed offsets are laid out at the mri coil with src/stats.anatomical_axes,
# the same axes the monte carlo uses, so bland altman means the same thing in both.
#
# and within each method, rep 1 against rep 2 at the same site, how well one method
# repeats itself: repeat_mm and repeat_deg.
from pathlib import Path

import numpy as np
import pandas as pd

from efield.brainsight import poses_from_export, read_export
from efield.poses import displacement_mm, orientation_deg, tilt_deg
from src.stats import anatomical_axes

PAIR_METRICS = ("displacement_mm", "tilt_deg", "orientation_deg")
OFFSET_METRICS = ("ap_mm", "lr_mm", "depth_mm", "yaw_deg")
REPEAT_METRICS = ("repeat_mm", "repeat_deg")


def export_files(folder):
    """Every brainsight export in folder, sorted so runs are repeatable."""

    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.glob("*.txt") if p.is_file())


def load_poses(folder):
    """Every protocol named coil pose in every export, and notes on what was skipped.

    a file in brainsight's own LPS space is flipped to RAS. the flip is a proper
    rotation, so distances and angles do not move, and the forward / right axes
    need RAS to mean forward and right
    """

    poses, notes = [], []
    for path in export_files(folder):
        try:
            coord = read_export(path).coord_sys
            found, skipped = poses_from_export(path, which="samples", allow_lps=True)
        except (ValueError, KeyError, IndexError) as err:
            notes.append(f"{path.name}: not read, {err}")
            continue
        flip = ", flipped from brainsight LPS to RAS" if coord.lower() == "brainsight" else ""
        notes.append(f"{path.name}: {len(found)} poses, coordinate system {coord}{flip}")
        if skipped:
            notes.append(f"{path.name}: skipped {len(skipped)} samples not named per the"
                         f" protocol: {', '.join(skipped[:6])}{' ...' if len(skipped) > 6 else ''}")
        poses.extend(found)

    # one pose per subject, method, site and rep. a later sample of the same name is
    # taken to be the corrected one, and the clash is reported, not silently dropped
    keyed = {}
    for p in poses:
        key = (p.extra.get("subject", ""), p.method, p.site, int(p.rep))
        if key in keyed:
            notes.append(f"sub-{key[0]} {key[1]} {key[2]} rep {key[3]}: named twice,"
                         f" using the later one ({p.name})")
        keyed[key] = p
    return list(keyed.values()), notes


def signed_offsets(eeg, mri):
    """eeg minus mri, forward / right / out in mm and the handle turn in degrees."""

    n_out = -mri.zdir     # zdir points into the head, see efield/poses.py
    ap, lr, n = anatomical_axes(n_out)
    d = eeg.centre - mri.centre

    # both handles laid flat in the plane of the mri coil, then the signed angle
    # between them, sign from the right hand rule about the outward normal
    h_mri = mri.ydir - n * float(np.dot(mri.ydir, n))
    h_eeg = eeg.ydir - n * float(np.dot(eeg.ydir, n))
    yaw = np.degrees(np.arctan2(float(np.dot(n, np.cross(h_mri, h_eeg))),
                                float(np.dot(h_mri, h_eeg))))

    return {"ap_mm": float(d @ ap), "lr_mm": float(d @ lr), "depth_mm": float(d @ n),
            "yaw_deg": float(yaw)}


def pair_rows(poses):
    """One row per eeg placement compared with its mri guided placement."""

    by = {}
    for p in poses:
        by.setdefault((p.extra.get("subject", ""), p.site), {}).setdefault(p.method, {})[int(p.rep)] = p

    rows = []
    for (subject, site), methods in sorted(by.items()):
        eeg, mri = methods.get("EEG", {}), methods.get("MRI", {})
        if not eeg or not mri:
            continue
        only = next(iter(mri.values())) if len(mri) == 1 else None
        for rep, e in sorted(eeg.items()):
            m = mri.get(rep, only)
            if m is None:
                continue
            row = {"subject": subject, "site": site, "rep": rep, "mri_rep": int(m.rep),
                   "eeg_name": e.name, "mri_name": m.name,
                   "displacement_mm": displacement_mm(e, m),
                   "tilt_deg": tilt_deg(e, m),
                   "orientation_deg": orientation_deg(e, m)}
            row.update(signed_offsets(e, m))
            rows.append(row)
    return pd.DataFrame(rows, columns=["subject", "site", "rep", "mri_rep", "eeg_name", "mri_name",
                                       *PAIR_METRICS, *OFFSET_METRICS])


def repeat_rows(poses):
    """Rep 1 against rep 2 of the same method at the same site, one row per pair."""

    by = {}
    for p in poses:
        by.setdefault((p.extra.get("subject", ""), p.site, p.method), {})[int(p.rep)] = p

    rows = []
    for (subject, site, method), reps in sorted(by.items()):
        if 1 in reps and 2 in reps:
            a, b = reps[1], reps[2]
            rows.append({"subject": subject, "site": site, "method": method,
                         "repeat_mm": displacement_mm(a, b), "repeat_deg": orientation_deg(a, b)})
    return pd.DataFrame(rows, columns=["subject", "site", "method", *REPEAT_METRICS])


def unmatched(poses, pairs):
    """Subject and site combinations that have poses but no eeg / mri pair, as notes."""

    have = {(p.extra.get("subject", ""), p.site) for p in poses}
    paired = set(zip(pairs["subject"], pairs["site"])) if len(pairs) else set()
    out = []
    for subject, site in sorted(have - paired):
        methods = sorted({p.method for p in poses
                          if p.extra.get("subject", "") == subject and p.site == site})
        out.append(f"sub-{subject} {site}: only {', '.join(methods)} poses, nothing to compare")
    return out


def measurements(folder):
    """(pairs, repeats, notes) for every export in folder."""

    poses, notes = load_poses(folder)
    pairs = pair_rows(poses)
    notes += unmatched(poses, pairs)
    return pairs, repeat_rows(poses), notes


def per_subject(df, metrics, by=("subject", "site")):
    """Reps averaged, one row per subject and site. subjects are the unit of analysis."""

    if df.empty:
        return df
    return df.groupby(list(by), as_index=False)[list(metrics)].mean()
