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


# aim directions in RAS, x is R+, y is A+, z is S+
TARGET_SPECS = [
    Target("SMA", "SMA", "#d62728", (0.0, 0.15, 1.0)),         # about the vertex, cz
    Target("L-DLPFC", "F3", "#2ca02c", (-0.55, 1.0, 0.35)),    # forehead left, f3
    Target("R-DLPFC", "F4", "#ff7f0e", (0.55, 1.0, 0.35)),     # forehead right, f4
    Target("L-M1", "C3", "#9467bd", (-0.9, -0.1, 0.5)),        # motor strip left, c3
    Target("R-M1", "C4", "#8c564b", (0.9, -0.1, 0.5)),         # motor strip right, c4
]

# one accent color for all the eeg landmarks
LANDMARK_COLOR = "#f2c14e"


def standard_targets(surf):
    """The 5 stimulation targets as points on the scalp."""

    out = []
    for spec in TARGET_SPECS:
        contact, normal, _ = scalp_target(surf, spec.aim)
        out.append(Target(spec.name, spec.label, spec.color, spec.aim, contact, normal))
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
        ("nasion", "Nz", _pick(p, face_band, y, "max")),         # front, over the nose
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


def _pick(points, mask, key, mode):
    # index of the point where key is largest/smallest within the masked region
    idx = np.where(mask)[0]
    if len(idx) == 0:
        idx = np.arange(len(points))
    k = key[idx]
    j = idx[np.argmax(k) if mode == "max" else np.argmin(k)]
    return points[j]
