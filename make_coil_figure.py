# scalp + figure-8 coil held 2 mm off, saves the png for the abstract
from pathlib import Path

import numpy as np
import pyvista as pv

from src.coil_model import clearance_to, coil_frame, figure8_coil, seat_coil, to_world
from src.head_surface import scalp_target, scalp_surface
from src.skull_boundary import load_mri_volume
from src.viz import render_coil

SAVE_DIR = Path("saves")
T1_PATH = SAVE_DIR / "T1.nii.gz"

# hair plus coil casing
GAP_MM = 2.0

# left and up, lands somewhere around c3
AIM = (-1.0, 0.0, 1.2)


def main():
    pv.OFF_SCREEN = True

    volume, affine = load_mri_volume(T1_PATH)

    # sigma 2 not 1, kills the leftover ripples on the scalp
    scalp = scalp_surface(volume, affine, smooth_sigma=2.0)

    contact, normal, flatness = scalp_target(scalp, AIM)
    e1, e2, n = coil_frame(normal)

    coil = figure8_coil()
    coil = to_world(coil, contact, e1, e2, n)
    coil = seat_coil(coil, scalp, n, gap=GAP_MM)

    gap = clearance_to(coil, scalp)

    lo = np.array(scalp.bounds[0::2])
    hi = np.array(scalp.bounds[1::2])
    print(f"scalp verts   {scalp.n_points}")
    print(f"scalp bbox mm {np.round(lo, 1)} to {np.round(hi, 1)}")
    print(f"scalp span mm {np.round(hi - lo, 1)}")
    print(f"contact RAS   {np.round(contact, 1)}")
    print(f"normal        {np.round(n, 3)}")
    print(f"patch flatness mm {flatness:.2f}")
    print(f"measured gap mm   {gap:.3f}")

    out = render_coil(scalp, coil, contact, n, gap, SAVE_DIR / "head_coil_2mm.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
