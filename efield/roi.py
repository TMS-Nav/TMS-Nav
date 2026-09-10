# where on the cortex the e field numbers are read.
#
# the proposal compares the field in "the target region". the region here is a
# sphere on the gm central surface, ROI_RADIUS_MM in radius, centred beneath the
# MRI guided coil. the mri guided pose is the reference method so the roi is the
# same for both methods at a site, and the eeg pose is scored on how well it
# hits the region the mri pose was aimed at. one roi per site per subject.
#
# the spec is only a description. the runner realizes it on the surface with
# simnibs.RegionOfInterest. kind 'mni_sphere' is kept for the day the target is
# given as an mni coordinate, the runner maps it with mni2subject_coords.
from dataclasses import dataclass, field

import numpy as np

from efield.config import ROI_RADIUS_MM

# how far beneath the coil centre the cortex is, mm, a starting guess only.
# the coil centre sits COIL_SKIN_DISTANCE_MM above the scalp and scalp to cortex
# is typically 12 to 20 mm over the convexity, so about 20 mm down the coil
# normal lands in the gm. the runner snaps this guess to the nearest node of the
# central surface before drawing the sphere, so the guess only has to be closer
# to the right gyrus than to the next one
DEFAULT_DEPTH_MM = 20.0

KINDS = ("sphere_under_coil", "mni_sphere")


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
        if not self.name:
            self.name = f"{self.site or 'roi'}_{self.kind}_{self.radius_mm:g}mm"

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
