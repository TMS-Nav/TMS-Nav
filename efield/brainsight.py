# the brainsight session export, read and written without simnibs.
#
# brainsight writes a tab separated text file with a handful of '#' header lines,
# then a '# Target Name ...' column header and the target rows, then a
# '# Sample Name ...' column header and the sample rows. every coil pose the
# operator recorded is a sample. the columns we need are the location and the
# 3x3 orientation, m0n0..m2n2, where m<c>n<r> is row r of column c, so m0 is the
# coil x axis, m1 the y (handle) axis and m2 the z axis. the exact column names
# are copied from simnibs/utils/nnav.py so simnibs's own reader agrees with this
# one on the same file.
#
# coordinate systems. brainsight's internal frame is LPS, x right to left, y
# front to back. an export in NIfTI:Aligned (or NIfTI:Q:Aligned from brainsight
# 2.5.3 on) is already in the scanner RAS the head model uses, and that is what
# the protocol asks for. an export in 'Brainsight' space would need the LPS to
# RAS flip below. simnibs refuses those, so the parser flags them but does not
# hide the problem.
import re
from pathlib import Path

import numpy as np

from efield.poses import CoilPose

# the column names, verbatim from nnav.brainsight.write
COL_NAME_TARGET = "Target Name"
COL_NAME_SAMPLE = "Sample Name"
COL_LOC = ("Loc. X", "Loc. Y", "Loc. Z")
COL_MAT = ("m0n0", "m0n1", "m0n2", "m1n0", "m1n1", "m1n2", "m2n0", "m2n1", "m2n2")
COL_OFFSET = "Offset"
COL_ASSOC = "Assoc. Target"

# the naming rule from the acquisition protocol,
# sub-<ID>_EEGCAP_<SITE>_POSE_REP<n> and sub-<ID>_MRI_<SITE>_POSE_REP<n>.
# the mri guided comparator is accepted under a few spellings
NAME_RE = re.compile(
    r"^sub-(?P<sub>[^_]+)_(?P<method>EEGCAP|EEG|MRIGUIDED|MRI_GUIDED|MRI)_(?P<site>[^_]+)_POSE_REP(?P<rep>\d+)$",
    re.IGNORECASE)
METHOD_ALIAS = {"EEGCAP": "EEG", "EEG": "EEG", "MRIGUIDED": "MRI", "MRI_GUIDED": "MRI", "MRI": "MRI"}


def parse_name(name):
    """(subject, method, site, rep) from a sample name, or None if it is not ours."""

    m = NAME_RE.match(name.strip())
    if not m:
        return None
    return (m.group("sub"), METHOD_ALIAS[m.group("method").upper()], m.group("site").upper(),
            int(m.group("rep")))


def make_name(subject, method, site, rep):
    """The protocol name for a pose. subject without the sub- prefix."""

    tag = "EEGCAP" if method.upper().startswith("EEG") else "MRI"
    return f"sub-{subject}_{tag}_{site}_POSE_REP{int(rep)}"


def lps_to_ras(loc, mat):
    """Flip a location and coil matrix from brainsight LPS into RAS."""

    flip = np.diag([-1.0, -1.0, 1.0])
    return flip @ np.asarray(loc, dtype=float), flip @ np.asarray(mat, dtype=float)


class BrainsightExport:
    """What one export file holds, header and both tables, before any interpretation."""

    def __init__(self):
        self.version = -1
        self.coord_sys = ""
        self.encoding = ""
        self.header = []           # every '#' line before the first table, verbatim
        self.targets = []          # list of dict, column name -> string
        self.samples = []

    @property
    def is_ras(self):
        """True when the coordinates are already scanner RAS, the way simnibs wants them."""

        cs = self.coord_sys.lower()
        return cs == "nifti:aligned" or cs.startswith("nifti:q:") or cs.startswith("nifti:s:")

    @property
    def is_lps(self):
        return self.coord_sys.lower() == "brainsight"


def read_export(path):
    """Parse a brainsight export into tables. nothing is converted here."""

    out = BrainsightExport()
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\r\n") for ln in f]

    cols = None
    table = None
    for ln in lines:
        if ln.startswith("# " + COL_NAME_TARGET) or ln.startswith("# " + COL_NAME_SAMPLE):
            cols = ln[2:].split("\t")
            table = out.targets if cols[0] == COL_NAME_TARGET else out.samples
            continue
        if ln.startswith("#"):
            if table is None:
                out.header.append(ln)
            if ln.startswith("# Version:"):
                out.version = int(ln.split(":", 1)[1].strip())
            elif ln.startswith("# Coordinate system:"):
                out.coord_sys = ln.split(":", 1)[1].strip()
            elif ln.startswith("# Encoding:"):
                out.encoding = ln.split(":", 1)[1].strip()
            else:
                # any other '#' line ends the current table, same as simnibs's reader
                table = None
            continue
        if not ln.strip() or table is None:
            continue
        cells = ln.split("\t")
        table.append({c: cells[i] if i < len(cells) else "" for i, c in enumerate(cols)})

    if out.version == -1:
        raise ValueError(f"no '# Version:' line in {path}")
    return out


def _row_pose(row, name_col, in_lps, **kw):
    loc = np.array([float(row[c]) for c in COL_LOC])
    # m<c>n<r>: fill column c with its three rows
    mat = np.array([[float(row[f"m{c}n{r}"]) for c in range(3)] for r in range(3)])
    if in_lps:
        loc, mat = lps_to_ras(loc, mat)
    offset = float(row.get(COL_OFFSET, "0") or 0.0)
    pose = CoilPose.from_brainsight(loc, mat, offset, name=row[name_col], **kw)
    pose.extra = {k: v for k, v in row.items() if k not in COL_LOC + COL_MAT + (name_col,)}
    return pose


def poses_from_export(path, which="samples", strict_names=True, allow_lps=False):
    """CoilPoses from a brainsight export, grouped by the protocol names.

    which is 'samples' (the recorded poses, default) or 'targets'. with
    strict_names rows whose name does not follow the protocol are dropped with a
    note in the returned skipped list, otherwise they are kept unlabelled. an
    LPS export is refused unless allow_lps, and then flipped to RAS
    """

    ex = read_export(path)
    if ex.is_lps and not allow_lps:
        raise ValueError(f"{path} is in brainsight LPS space, export as NIfTI:Aligned "
                         f"(or pass allow_lps=True to flip it here)")
    if not ex.is_ras and not ex.is_lps:
        raise ValueError(f"{path} coordinate system {ex.coord_sys!r} is not one this reader knows")

    name_col = COL_NAME_SAMPLE if which == "samples" else COL_NAME_TARGET
    rows = ex.samples if which == "samples" else ex.targets
    poses, skipped = [], []
    for row in rows:
        parsed = parse_name(row[name_col])
        if parsed is None:
            if strict_names:
                skipped.append(row[name_col])
                continue
            poses.append(_row_pose(row, name_col, ex.is_lps))
            continue
        sub, method, site, rep = parsed
        pose = _row_pose(row, name_col, ex.is_lps, site=site, method=method, rep=rep)
        pose.extra["subject"] = sub
        poses.append(pose)
    return poses, skipped


def write_fixture(path, poses, coord_sys="NIfTI:Aligned", version=12, targets=None,
                  created_by="TMS-Nav efield.brainsight fixture"):
    """Write a synthetic export in the layout simnibs's brainsight().write produces.

    header lines and column names are copied from nnav.py so that
    simnibs.brainsight().read(path) accepts the file and gives back the same
    matrices. the coil x and z columns are negated on the way out, the inverse of
    what from_brainsight does on the way in. the samples table gets the extra
    columns a real session export carries (associated target, offset, date, time)
    so the parser is exercised on something closer to the real thing
    """

    if version >= 14 and not coord_sys.lower().startswith("nifti:q:"):
        raise ValueError("version 14 exports name the source form, use eg 'NIfTI:Q:Aligned'")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def cells(pose):
        R = pose.rotation.copy()
        R[:, 0] *= -1.0    # brainsight x
        R[:, 2] *= -1.0    # brainsight z, points away from the head
        vals = [f"{v:.4f}" for v in pose.centre]
        for c in range(3):
            vals += [f"{R[r, c]:.4f}" for r in range(3)]
        return vals

    mat_cols = "\t".join(COL_MAT)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# Version: {version}\n")
        f.write(f"# Coordinate system: {coord_sys}\n")
        f.write(f"# Created by: {created_by}\n")
        f.write("# Units: millimetres, degrees, milliseconds, and microvolts\n")
        f.write("# Encoding: UTF-8\n")
        f.write("# Notes: Each column is delimited by a tab. Each value within a column is delimited by a semicolon.\n")
        f.write(f"# {COL_NAME_TARGET}\t" + "\t".join(COL_LOC) + "\t" + mat_cols + "\n")
        for t in (targets or []):
            f.write("\t".join([t.name] + cells(t)) + "\n")
        f.write(f"# {COL_NAME_SAMPLE}\tSession Name\tIndex\t{COL_ASSOC}\t" + "\t".join(COL_LOC)
                + "\t" + mat_cols + f"\tDist. to Target\t{COL_OFFSET}\tDate\tTime\n")
        for i, p in enumerate(poses):
            assoc = p.extra.get("assoc_target", p.site or "(null)")
            f.write("\t".join([p.name, "Session 1", str(i), assoc] + cells(p)
                              + ["0.0", f"{p.offset_mm:.4f}", "2026-01-01", "12:00:00"]) + "\n")
    return path
