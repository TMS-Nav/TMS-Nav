# where on the cortex the e field numbers are read.
#
# the proposal compares the field in "the target region", and the target is the
# R01'S OWN FMRI TARGET for that subject and site. the region is a sphere on the
# gm central surface, ROI_RADIUS_MM in radius, centred on that target, and it is
# the same for both methods at a site, so the eeg and mri guided poses are scored
# on how much field reached the place the study meant to treat.
#
# a site with no target supplied gets no roi and is not simulated. measuring
# somewhere under the coil instead would only answer "how much field landed under
# the coil", which says nothing about the target.
#
# the spec is only a description. the runner realizes it on the surface with
# simnibs.RegionOfInterest. a target given in mni space is mapped into the
# subject with mni2subject_coords, one given in subject space is used as is.
import csv
from dataclasses import dataclass, field
from pathlib import Path

from efield.config import FMRI_TARGETS, ROI_RADIUS_MM

# fmri_target  the r01 target in subject RAS mm
# mni_sphere   the same thing given in mni mm, mapped by the runner
KINDS = ("fmri_target", "mni_sphere")

# the sites a target can be given for, the protocol marker set
SITES = ("SMA", "F4", "C3", "C4", "Cz")


@dataclass
class RoiSpec:
    kind: str = "fmri_target"
    radius_mm: float = ROI_RADIUS_MM
    centre: list = None                 # subject RAS mm, the seed for the snap
    mni: list = None                    # mni mm, for kind 'mni_sphere'
    site: str = ""
    reference_method: str = "fMRI"      # where the centre came from
    name: str = ""
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"roi kind {self.kind!r}, expected one of {KINDS}")
        if self.kind == "mni_sphere" and self.mni is None:
            raise ValueError(f"roi {self.name or self.site!r} is mni_sphere but has no mni coordinate")
        if self.kind == "fmri_target" and self.centre is None:
            raise ValueError(f"roi {self.name or self.site!r} is {self.kind} but has no centre")
        if not self.name:
            self.name = f"{self.site or 'roi'}_{self.kind}_{self.radius_mm:g}mm"

    def to_dict(self):
        return {
            "kind": self.kind, "radius_mm": float(self.radius_mm),
            "centre": None if self.centre is None else [round(float(x), 4) for x in self.centre],
            "mni": None if self.mni is None else [float(x) for x in self.mni],
            "site": self.site, "reference_method": self.reference_method,
            "name": self.name, "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(kind=d.get("kind", "fmri_target"), radius_mm=float(d.get("radius_mm", ROI_RADIUS_MM)),
                   centre=d.get("centre"), mni=d.get("mni"), site=d.get("site", ""),
                   reference_method=d.get("reference_method", "fMRI"), name=d.get("name", ""),
                   extra=dict(d.get("extra", {})))


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


def rois_for_subject(subject, poses, table=None, radius_mm=ROI_RADIUS_MM):
    """One roi per site at the r01 target, and the sites that have no target.

    a site with no target gets no roi. the caller leaves its poses out and says
    so, rather than reading the field at a guess
    """

    table = load_target_table() if table is None else table
    sites = []
    for p in poses:
        if p.site not in sites:
            sites.append(p.site)

    rois, missing = [], []
    for site in sites:
        hit = table.get((subject, site))
        if hit:
            rois.append(roi_from_target(site, hit["coords"], hit["space"], radius_mm, hit["note"]))
        else:
            missing.append(site)
    return rois, missing
