# the 3 study targets, placed by anatomical direction so the same code works on
# any head. approximate, not true 10-20, swap this module out later if we need
# real electrode positions off the eeg cap markers
from dataclasses import dataclass

import numpy as np

from src.head_surface import scalp_target


@dataclass
class Target:
    name: str
    label: str
    color: str
    aim: tuple
    contact: np.ndarray = None
    normal: np.ndarray = None


# aim directions in RAS, x is R+, y is A+, z is S+
TARGET_SPECS = [
    Target("SMA", "SMA", "#d62728", (0.0, 0.15, 1.0)),      # about the vertex, cz
    Target("L-DLPFC", "L-DLPFC", "#2ca02c", (-0.55, 1.0, 0.35)),  # forehead left, f3
    Target("R-DLPFC", "R-DLPFC", "#ff7f0e", (0.55, 1.0, 0.35)),   # forehead right, f4
]


def standard_targets(surf):
    """The 3 study targets as marker points on the scalp."""

    out = []
    for spec in TARGET_SPECS:
        contact, normal, _ = scalp_target(surf, spec.aim)
        out.append(Target(spec.name, spec.label, spec.color, spec.aim, contact, normal))
    return out
