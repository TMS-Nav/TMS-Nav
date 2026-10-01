# where on the cortex the e field numbers are read.
#
# the proposal compares the field in "the target region". the region here is a
# sphere on the gm central surface, ROI_RADIUS_MM in radius, centred beneath the
# MRI guided coil. the mri guided pose is the reference method so the roi is the
# same for both methods at a site, and the eeg pose is scored on how well it
# hits the region the mri pose was aimed at. one roi per site per subject.
#
# THE ROI SHOULD BE THE R01'S OWN FMRI TARGET wherever one is known. measuring
# under the coil only answers "how much field landed under the coil", which is
# close to circular. measuring at the intended target answers the question the
# study is actually asking, how much field reached the place we meant to treat.
# so kind 'fmri_target' is the one to use, and 'sphere_under_coil' is the
# fallback for a subject whose target has not been supplied yet.
#
# the spec is only a description. the runner realizes it on the surface with
# simnibs.RegionOfInterest. a target given in mni space is mapped into the
# subject with mni2subject_coords, one given in subject space is used as is.
import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from efield.config import FMRI_TARGETS, ROI_RADIUS_MM

# how far beneath the coil centre the cortex is, mm, a starting guess only.
# the coil centre sits COIL_SKIN_DISTANCE_MM above the scalp and scalp to cortex
# is typically 12 to 20 mm over the convexity, so about 20 mm down the coil
# normal lands in the gm. the runner snaps this guess to the nearest node of the
# central surface before drawing the sphere, so the guess only has to be closer
# to the right gyrus than to the next one
DEFAULT_DEPTH_MM = 20.0

# fmri_target  the r01 target in subject RAS mm, the one to use
# mni_sphere   the same thing given in mni mm, mapped by the runner
# sphere_under_coil  the fallback, a guess down the coil normal
KINDS = ("fmri_target", "mni_sphere", "sphere_under_coil")

# the sites a target can be given for, the protocol marker set
SITES = ("SMA", "F4", "C3", "C4", "Cz")


@dataclass
class RoiSpec:
    kind: str = "sphere_under_coil"
    radius_mm: float = ROI_RADIUS_MM
    centre: list = None                 # subject RAS mm, the seed for the snap
    mni: list = None                    # mni mm, for kind 'mni_sphere'
    site: str = ""
    reference_method: str = "MRI"       # which method's pose seeded the centre
    reference_pose: str = ""            # its name, for the record
    depth_mm: float = DEFAULT_DEPTH_MM
    name: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"roi kind {self.kind!r}, expected one of {KINDS}")
        if self.kind == "mni_sphere" and self.mni is None:
            raise ValueError(f"roi {self.name or self.site!r} is mni_sphere but has no mni coordinate")
        if self.kind in ("fmri_target", "sphere_under_coil") and self.centre is None:
            raise ValueError(f"roi {self.name or self.site!r} is {self.kind} but has no centre")
        if not self.name:
            self.name = f"{self.site or 'roi'}_{self.kind}_{self.radius_mm:g}mm"

    @property
    def is_fallback(self):
        """True when this roi is the guess under the coil, not a real target."""

        return self.kind == "sphere_under_coil"

    def to_dict(self):
        return {
            "kind": self.kind, "radius_mm": float(self.radius_mm),
            "centre": None if self.centre is None else [round(float(x), 4) for x in self.centre],
            "mni": None if self.mni is None else [float(x) for x in self.mni],
            "site": self.site, "reference_method": self.reference_method,
            "reference_pose": self.reference_pose, "depth_mm": float(self.depth_mm),
            "name": self.name, "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(kind=d.get("kind", "sphere_under_coil"), radius_mm=float(d.get("radius_mm", ROI_RADIUS_MM)),
                   centre=d.get("centre"), mni=d.get("mni"), site=d.get("site", ""),
                   reference_method=d.get("reference_method", "MRI"),
                   reference_pose=d.get("reference_pose", ""),
                   depth_mm=float(d.get("depth_mm", DEFAULT_DEPTH_MM)), name=d.get("name", ""),
                   extra=dict(d.get("extra", {})))


def seed_under_coil(pose, depth_mm=DEFAULT_DEPTH_MM):
    """The point depth_mm down the coil normal from the coil centre, RAS mm."""

    return np.asarray(pose.centre, dtype=float) + np.asarray(pose.zdir, dtype=float) * float(depth_mm)


def roi_under_pose(pose, radius_mm=ROI_RADIUS_MM, depth_mm=DEFAULT_DEPTH_MM):
    """The sphere roi seeded beneath a coil pose, normally the mri guided rep 1."""

    return RoiSpec(kind="sphere_under_coil", radius_mm=radius_mm,
                   centre=seed_under_coil(pose, depth_mm).tolist(), site=pose.site,
                   reference_method=pose.method or "MRI", reference_pose=pose.name,
                   depth_mm=depth_mm)


def rois_for_poses(poses, radius_mm=ROI_RADIUS_MM, depth_mm=DEFAULT_DEPTH_MM,
                   reference_method="MRI"):
    """One roi per site, seeded by the lowest rep of the reference method.

    falls back to whichever pose is first at a site when the reference method
    is missing there, and says so in the spec's extra
    """

    by_site = {}
    for p in poses:
        by_site.setdefault(p.site, []).append(p)
    out = []
    for site, ps in by_site.items():
        ref = sorted([p for p in ps if p.method == reference_method], key=lambda p: p.rep)
        if ref:
            roi = roi_under_pose(ref[0], radius_mm, depth_mm)
        else:
            roi = roi_under_pose(ps[0], radius_mm, depth_mm)
            roi.extra["note"] = f"no {reference_method} pose at {site}, seeded from {ps[0].name}"
        out.append(roi)
    return out


# --- the r01 fmri targets ---------------------------------------------------------------
# one csv for the whole study, columns subject, site, x, y, z, space, note.
# space is "subject" for coordinates in the subject's own t1 (the same space the
# brainsight export uses) or "mni" for template coordinates. blank lines and
# lines starting with # are ignored. see efield/fmri_targets_example.csv
TARGET_COLUMNS = ("subject", "site", "x", "y", "z", "space")


def load_target_table(path=None):
    """{(subject, site): {coords, space, note}} from the fmri target csv.

    returns an empty dict when the file is not there, which is the normal state
    until the r01 targets are handed over. the caller decides what to do about it
    """

    path = Path(path or FMRI_TARGETS)
    if not path.exists():
        return {}

    table = {}
    with open(path, newline="", encoding="utf8") as f:
        rows = [r for r in csv.DictReader(_strip_comments(f))]
    for i, r in enumerate(rows, start=2):
        missing = [c for c in TARGET_COLUMNS if not (r.get(c) or "").strip()]
        if missing:
            raise ValueError(f"{path} line {i}: missing {', '.join(missing)}")
        space = r["space"].strip().lower()
        if space not in ("subject", "mni"):
            raise ValueError(f"{path} line {i}: space must be subject or mni, got {r['space']!r}")
        site = r["site"].strip()
        if site not in SITES:
            raise ValueError(f"{path} line {i}: site {site!r}, expected one of {SITES}")
        key = (r["subject"].strip(), site)
        if key in table:
            raise ValueError(f"{path} line {i}: {key[0]} {key[1]} is given twice")
        try:
            coords = [float(r[c]) for c in ("x", "y", "z")]
        except ValueError:
            raise ValueError(f"{path} line {i}: x, y, z must be numbers") from None
        table[key] = {"coords": coords, "space": space, "note": (r.get("note") or "").strip()}
    return table


def _strip_comments(lines):
    for line in lines:
        if line.strip() and not line.lstrip().startswith("#"):
            yield line


def roi_from_target(site, coords, space, radius_mm=ROI_RADIUS_MM, note=""):
    """The roi for one supplied fmri target."""

    if space == "mni":
        return RoiSpec(kind="mni_sphere", radius_mm=radius_mm, mni=list(coords), site=site,
                       reference_method="fMRI", extra={"note": note} if note else {})
    return RoiSpec(kind="fmri_target", radius_mm=radius_mm, centre=list(coords), site=site,
                   reference_method="fMRI", extra={"note": note} if note else {})


def rois_for_subject(subject, poses, table=None, radius_mm=ROI_RADIUS_MM,
                     depth_mm=DEFAULT_DEPTH_MM, reference_method="MRI"):
    """One roi per site: the fmri target when there is one, else the fallback.

    returns the rois and the list of sites that had to fall back, so the caller
    can say out loud which numbers are measured at a guess
    """

    table = load_target_table() if table is None else table
    sites = []
    for p in poses:
        if p.site not in sites:
            sites.append(p.site)

    fallback = {r.site: r for r in rois_for_poses(poses, radius_mm, depth_mm, reference_method)}

    out, guessed = [], []
    for site in sites:
        hit = table.get((subject, site))
        if hit:
            out.append(roi_from_target(site, hit["coords"], hit["space"], radius_mm, hit["note"]))
        else:
            roi = fallback.get(site)
            if roi is None:
                continue
            roi.extra["fallback"] = (
                f"no fmri target for {subject} {site}, this roi is a guess "
                f"{depth_mm:g} mm down the coil normal"
            )
            out.append(roi)
            guessed.append(site)
    return out, guessed
