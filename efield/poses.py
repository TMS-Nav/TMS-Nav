# one coil pose, and the 4x4 matrix simnibs wants for it.
#
# simnibs calls the matrix matsimnibs. its columns are x', y', z' and the coil
# centre, all in subject RAS mm. y' is the handle direction, z' is the coil
# normal pointing INTO the head, and x' = y' x z' closes the right handed frame.
# aim 1 works with the OUTWARD scalp normal, so z' is the negative of that. one
# sign, easy to get wrong, so it lives in exactly one place, here.
#
# the comparators at the bottom reuse src/stats so "orientation difference"
# means the same thing in aim 1 and aim 2: the single angle of the rotation
# that takes one coil frame onto the other.
from dataclasses import dataclass, field

import numpy as np

from efield.config import COIL_SKIN_DISTANCE_MM
from src.stats import angle_between, rotation_angle

METHODS = ("EEG", "MRI")


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero length direction")
    return v / n


def _orthogonalise(handle, zdir):
    """The handle direction with its component along the normal removed."""

    h = np.asarray(handle, dtype=float)
    h = h - zdir * float(np.dot(h, zdir))
    if np.linalg.norm(h) < 1e-6:
        raise ValueError("handle direction is parallel to the coil normal")
    return _unit(h)


@dataclass
class CoilPose:
    centre: np.ndarray                 # coil centre, RAS mm
    ydir: np.ndarray                   # handle direction, unit
    zdir: np.ndarray                   # coil normal into the head, unit
    name: str = ""
    site: str = ""                     # SMA, F4, ...
    method: str = ""                   # EEG or MRI
    rep: int = 0                       # 1 or 2, the independent placements
    # the coil to scalp offset brainsight recorded, mm, kept for the record. it
    # is already baked into the centre so nothing applies it a second time
    offset_mm: float = 0.0
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.centre = np.asarray(self.centre, dtype=float).reshape(3)
        self.zdir = _unit(self.zdir)
        self.ydir = _orthogonalise(self.ydir, self.zdir)

    # --- the matrix -------------------------------------------------------------
    @property
    def xdir(self):
        return np.cross(self.ydir, self.zdir)

    @property
    def rotation(self):
        """3x3 with columns x', y', z'."""

        return np.column_stack([self.xdir, self.ydir, self.zdir])

    def matsimnibs(self):
        m = np.eye(4)
        m[:3, :3] = self.rotation
        m[:3, 3] = self.centre
        return m

    @classmethod
    def from_matsimnibs(cls, m, **kw):
        m = np.asarray(m, dtype=float).reshape(4, 4)
        return cls(centre=m[:3, 3], ydir=m[:3, 1], zdir=m[:3, 2], **kw)

    # --- builders ----------------------------------------------------------------
    @classmethod
    def from_scalp_point(cls, contact, outward_normal, handle_dir,
                         distance=COIL_SKIN_DISTANCE_MM, **kw):
        """A coil sat flat on the scalp at contact, its housing distance mm off it.

        outward_normal is the scalp normal pointing away from the head, the way
        aim 1 stores it. the coil normal is the opposite
        """

        n_out = _unit(outward_normal)
        centre = np.asarray(contact, dtype=float) + n_out * distance
        return cls(centre=centre, ydir=handle_dir, zdir=-n_out, **kw)

    @classmethod
    def from_aim1_target(cls, target, handle_dir=None, distance=COIL_SKIN_DISTANCE_MM, **kw):
        """A pose from an aim 1 Target (src/targets.py), contact plus normal.

        aim 1's coil model does not define a handle, its spin about the normal
        is arbitrary, so with no handle_dir the second tangent of coil_frame is
        used. that is a stand in. real handle directions come from brainsight
        """

        if handle_dir is None:
            # imported here so this module stays light. coil_model pulls pyvista
            from src.coil_model import coil_frame
            _, e2, _ = coil_frame(target.normal)
            handle_dir = e2
        kw.setdefault("site", target.label)
        kw.setdefault("name", target.name)
        return cls.from_scalp_point(target.contact, target.normal, handle_dir, distance, **kw)

    @classmethod
    def from_brainsight(cls, loc, mat3x3, offset=0.0, **kw):
        """A pose from one brainsight sample row, already in RAS.

        brainsight's coil frame has x and z the other way round from simnibs, z
        points away from the head. simnibs's own reader flips those two columns
        (nnav.brainsight._simnibs2brainsight), and so does this. the columns of
        mat3x3 are m0, m1, m2 from the export, ie m<column><row>
        """

        m = np.asarray(mat3x3, dtype=float).reshape(3, 3)
        return cls(centre=loc, ydir=m[:, 1], zdir=-m[:, 2], offset_mm=float(offset), **kw)

    # --- json ---------------------------------------------------------------------
    def to_dict(self):
        return {
            "name": self.name, "site": self.site, "method": self.method, "rep": int(self.rep),
            "centre": [round(float(x), 4) for x in self.centre],
            "ydir": [round(float(x), 6) for x in self.ydir],
            "zdir": [round(float(x), 6) for x in self.zdir],
            "offset_mm": float(self.offset_mm),
            "matsimnibs": [[round(float(x), 6) for x in row] for row in self.matsimnibs()],
            "extra": dict(self.extra),
        }

    @classmethod
    def from_dict(cls, d):
        return cls(centre=d["centre"], ydir=d["ydir"], zdir=d["zdir"], name=d.get("name", ""),
                   site=d.get("site", ""), method=d.get("method", ""), rep=int(d.get("rep", 0)),
                   offset_mm=float(d.get("offset_mm", 0.0)), extra=dict(d.get("extra", {})))

    def key(self):
        return (self.site, self.method, int(self.rep))

    def __str__(self):
        c, y, z = np.round(self.centre, 1), np.round(self.ydir, 3), np.round(self.zdir, 3)
        return f"{self.name or '?'}  centre {c}  handle {y}  normal(in) {z}"


# --- comparators, shared definitions with aim 1 ------------------------------------
def displacement_mm(a, b):
    """Distance between two coil centres, mm."""

    return float(np.linalg.norm(a.centre - b.centre))


def tilt_deg(a, b):
    """Angle between the two coil normals, degrees. the face tilt only."""

    return float(angle_between(a.zdir, b.zdir))


def orientation_deg(a, b):
    """Total rotation between the two coil frames, degrees.

    the angle of Ra^T Rb, which folds tilt and handle spin into one number, the
    same number aim 1 reports as the coil orientation difference
    """

    return float(rotation_angle(a.rotation.T @ b.rotation))


def is_rotation(R, tol=1e-6):
    """True when R is orthonormal with determinant +1."""

    R = np.asarray(R, dtype=float)
    return bool(np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) < tol)
