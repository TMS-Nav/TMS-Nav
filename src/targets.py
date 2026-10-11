# the study targets and the eeg registration landmarks, all placed by anatomical
# direction so the same code works on any head. approximate, not true 10-20, swap
# this module out later if we get real cap markers off the mri
from dataclasses import dataclass

import numpy as np

from src.head_surface import scalp_target


@dataclass
class Target:
    name: str
    label: str
    color: str
    aim: tuple = None
    contact: np.ndarray = None
    normal: np.ndarray = None
    optional: bool = False   # placed only when a fifth fiducial is available


# the marker set from the acquisition protocol, in its priority order. sma and f4
# are the therapeutic targets, c3 and c4 the motor references, cz the midline
# reference and the one to drop when only four fiducials are available. f3 is
# out, the stimulation protocol uses the right dlpfc. aim directions in RAS, x is
# R+, y is A+, z is S+
TARGET_SPECS = [
    Target("SMA", "SMA", "#d62728", (0.0, 0.15, 1.0)),                    # ahead of the vertex
    Target("R-DLPFC", "F4", "#ff7f0e", (0.55, 1.0, 0.35)),                # forehead right
    Target("L-M1", "C3", "#9467bd", (-0.9, -0.1, 0.5)),                   # motor strip left
    Target("R-M1", "C4", "#8c564b", (0.9, -0.1, 0.5)),                    # motor strip right
    Target("Cz", "Cz", "#2ca02c", (0.0, 0.0, 1.0), optional=True),        # vertex
]

# one accent color for all the eeg landmarks
LANDMARK_COLOR = "#f2c14e"


def standard_targets(surf):
    """The protocol markers as points on the scalp, protocol priority order."""

    out = []
    for spec in TARGET_SPECS:
        contact, normal, _ = scalp_target(surf, spec.aim)
        out.append(Target(spec.name, spec.label, spec.color, spec.aim, contact, normal,
                          optional=spec.optional))
    return out


def standard_landmarks(surf):
    """EEG registration landmarks inferred from scalp profiles in RAS mm."""

    p = surf.points
    x, z = p[:, 0], p[:, 2]
    center = p.mean(axis=0)
    nasion = _nasion(p)

    picks = [
        ("Cz", "Cz", _pick(p, np.abs(x) < 20, z, "max")),        # vertex
        ("nasion", "Nz", nasion),                               # dip between brow and nose
        ("inion", "Iz", _inion(p)),                             # posterior midline prominence
        ("LPA", "LPA", _lpa(p, nasion=nasion)),                  # in front of the left pinna
        ("RPA", "RPA", _rpa(p, nasion=nasion)),                  # in front of the right pinna
    ]

    out = []
    for name, label, pos in picks:
        n = pos - center
        n = n / np.linalg.norm(n)
        out.append(Target(name, label, LANDMARK_COLOR, None, pos, n))
    return out


def _nasion(points, half_width=4.0, tip_below_top=(55.0, 170.0), window=60.0,
            min_dip=5.0, min_rise=2.0):
    """The nasion, read off the midline profile of the face.

    walk down the front of the head along the midline and the profile bulges out
    at the brow, dips in at the bridge of the nose, and bulges out again at the
    nose tip. the nasion is that dip. so: find the nose tip as the most anterior
    point in the band of heights a nose can sit at, measured down from the vertex
    rather than from the middle of the mesh, because the mesh may or may not have
    a neck on it. then look up from the tip for the deepest point, and accept it
    only if the nose really sticks out past it and the brow comes forward again
    above it. a defaced head has no nose, its profile just slopes back from the
    cut, so nothing passes and the pick falls back to the most anterior point,
    which is the top edge of the cut at brow height, the best that scan can do.
    """

    p = points
    z_top = p[:, 2].max()

    # anterior most vertex for every mm of height along the midline strip
    mid = np.flatnonzero(np.abs(p[:, 0]) < half_width)
    best = {}
    for i, zz in zip(mid, np.round(p[mid, 2]).astype(int)):
        if zz not in best or p[i, 1] > p[best[zz], 1]:
            best[zz] = i
    zs = np.array(sorted(best))
    idx = np.array([best[k] for k in zs])
    ys = p[idx, 1]

    band = (zs >= z_top - tip_below_top[1]) & (zs <= z_top - tip_below_top[0])
    if not band.any():
        return p[idx[np.argmax(ys)]]
    k_tip = np.flatnonzero(band)[np.argmax(ys[band])]

    win = np.flatnonzero((zs > zs[k_tip]) & (zs <= zs[k_tip] + window))
    if len(win):
        k_min = win[np.argmin(ys[win])]
        above = win[win > k_min]
        dip = ys[k_tip] - ys[k_min]
        rise = ys[above].max() - ys[k_min] if len(above) else 0.0
        if dip >= min_dip and rise >= min_rise:
            return p[idx[k_min]]

    return p[idx[k_tip]]


def _inion(points, half_width=4.0, below_top=(70.0, 145.0)):
    """Posterior-most scalp point in a narrow occipital midline strip.

    The search is measured down from the vertex rather than up from the mesh
    midpoint, so neck coverage does not push the pick toward the crown. The broad
    occipital band includes the lower posterior prominence of the MNI head.
    This is a surface heuristic, not detection of a bony feature in the MRI.
    """
    p = _landmark_points(points)
    midline = 0.5 * (p[:, 0].min() + p[:, 0].max())
    lateral = np.abs(p[:, 0] - midline)
    mid = lateral < half_width
    if not mid.any():
        mid = lateral <= lateral.min() + 1e-6
    depth = p[:, 2].max() - p[:, 2]
    band = mid & (depth >= below_top[0]) & (depth <= below_top[1])
    # A cropped/short surface may miss the band: stay on the midline, never jump
    # to the back of an ear simply because it is the global posterior extreme.
    return _pick(p, band if band.any() else mid, p[:, 1], "min")


def _lpa(points, nasion=None, **kwargs):
    """Left preauricular scalp prominence, anterior to the outer ear."""
    return _preauricular(points, side=-1, nasion=nasion, **kwargs)


def _rpa(points, nasion=None, **kwargs):
    """Right preauricular scalp prominence, anterior to the outer ear."""
    return _preauricular(points, side=1, nasion=nasion, **kwargs)


def _preauricular(points, side, nasion=None, half_height=2.0,
                 pinna_clearance=15.0, anterior_reach=60.0):
    """Find each ear separately near nasion height, without selecting the pinna.

    In a thin horizontal slice, the outer ear is the largest lateral prominence.
    Look anterior to it for the smaller preauricular prominence. Clearance and
    slice thickness scale with head width (180 mm nominal); no template XYZ
    coordinates are used. The two sides are not forced to be mirror images.
    Missing/cropped slices use the closest available heights on the same side.
    """
    p = _landmark_points(points)
    if side not in (-1, 1):
        raise ValueError("preauricular side must be -1 (left) or +1 (right)")
    width = float(np.ptp(p[:, 0]))
    if width <= 0:
        raise ValueError("need a scalp with nonzero left/right extent")
    midline = 0.5 * (p[:, 0].min() + p[:, 0].max())
    scale = width / 180.0
    reference = _nasion(p) if nasion is None else np.asarray(nasion, dtype=float)
    height = float(reference[2])
    lateral = side * (p[:, 0] - midline)
    side_mask = lateral > 0.65 * width / 2
    if not side_mask.any():
        raise ValueError("no scalp points on the requested preauricular side")
    height_error = np.abs(p[:, 2] - height)
    band = side_mask & (height_error <= half_height * scale)
    if not band.any():
        nearest_height = height_error[side_mask].min()
        band = side_mask & (height_error <= nearest_height + half_height * scale)
    q = p[band]
    pinna = q[np.argmax(side * (q[:, 0] - midline))]
    anterior = ((q[:, 1] > pinna[1] + pinna_clearance * scale)
                & (q[:, 1] <= pinna[1] + anterior_reach * scale))
    if anterior.any():
        candidates = q[anterior]
        return candidates[np.argmax(side * (candidates[:, 0] - midline))]
    # If the anterior patch is cropped, return the nearest available point to
    # the expected anterior edge at nasion height, retaining correct laterality.
    score = ((q[:, 1] - pinna[1] - pinna_clearance * scale) ** 2
             + (q[:, 2] - height) ** 2)
    return q[np.argmin(score)]


def _landmark_points(points):
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 3 or not len(p) or not np.isfinite(p).all():
        raise ValueError("landmarks need a nonempty finite Nx3 scalp point array")
    return p


def _pick(points, mask, key, mode):
    # index of the point where key is largest/smallest within the masked region
    idx = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.arange(len(points))
    k = key[idx]
    j = idx[np.argmax(k) if mode == "max" else np.argmin(k)]
    return points[j]
