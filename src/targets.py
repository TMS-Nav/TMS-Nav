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
    """The eeg registration landmarks, picked as extreme points on the scalp."""

    p = surf.points
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    center = p.mean(axis=0)
    zc = 0.5 * (z.min() + z.max())

    # keep to eye/brow height so the front pick lands near the nose bridge, not
    # the chin, and the ears land by the ear canal, not the jaw
    face_band = np.abs(x) < 15
    face_band &= (z > zc) & (z < zc + 45)
    back_band = (np.abs(x) < 15) & (z > zc - 5)
    ear_band = (y > -25) & (y < 35) & (z > zc - 3) & (z < zc + 35)

    picks = [
        ("Cz", "Cz", _pick(p, np.abs(x) < 20, z, "max")),        # vertex
        ("nasion", "Nz", _nasion(p)),                            # dip between brow and nose
        ("inion", "Iz", _pick(p, back_band, y, "min")),          # bump at the back
        ("LPA", "LPA", _pick(p, ear_band, x, "min")),            # left ear
        ("RPA", "RPA", _pick(p, ear_band, x, "max")),            # right ear
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


def _pick(points, mask, key, mode):
    # index of the point where key is largest/smallest within the masked region
    idx = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.arange(len(points))
    k = key[idx]
    j = idx[np.argmax(k) if mode == "max" else np.argmin(k)]
    return points[j]
