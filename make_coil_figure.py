# scalp + figure-8 coil held 2 mm off, saves the png for the abstract
from pathlib import Path

import numpy as np
import pyvista as pv

from src.coil_model import clearance_to, coil_frame, figure8_coil, seat_coil, to_world
from src.head_surface import scalp_target, scalp_surface
from src.skull_boundary import load_mri_volume

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

    # build in the local frame, drop it at the contact point, then back it off
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

    plot(scalp, coil, contact, n, gap)


def plot(scalp, coil, contact, n, gap):
    p = pv.Plotter(off_screen=True, window_size=(1600, 1200))

    # otherwise the coil edges come out jagged
    p.enable_anti_aliasing("ssaa")

    p.add_mesh(scalp, color="lightgray", smooth_shading=True, specular=0.15)
    p.add_mesh(coil, color="#1f77b4", smooth_shading=True, specular=0.5, specular_power=20)

    # ticks in mm so the scale is checkable
    p.show_grid(
        xtitle="x  R+ [mm]",
        ytitle="y  A+ [mm]",
        ztitle="z  S+ [mm]",
        n_xlabels=3,
        n_ylabels=3,
        n_zlabels=3,
        fmt="%.0f",
        font_size=16,
        location="outer",
        padding=0.02,
    )

    # corner triad, tells us which way the head is facing
    p.add_axes(
        xlabel="R",
        ylabel="A",
        zlabel="S",
        line_width=5,
        labels_off=False,
        viewport=(0.0, 0.0, 0.22, 0.22),
    )

    p.add_text(
        f"figure-8 coil, {gap:.2f} mm off scalp\n"
        f"contact RAS ({contact[0]:.0f}, {contact[1]:.0f}, {contact[2]:.0f}) mm",
        font_size=14,
    )

    # off the coil normal toward the front, shows both wings and the face
    view = n + np.array([0.0, 0.9, 0.0])
    view /= np.linalg.norm(view)

    focus = np.array(scalp.center)
    p.camera_position = [tuple(focus + view * 500.0), tuple(focus), (0.0, 0.0, 1.0)]

    # keeps the direction, pulls back until the head fits
    p.reset_camera()
    p.camera.zoom(0.95)

    out = SAVE_DIR / "head_coil_2mm.png"
    p.screenshot(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
