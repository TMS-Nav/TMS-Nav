# feed an mri or a folder of mris, build the head and drop the study targets on
# it, save a preview png for each. this is the sanity check view, the real
# showcase will be three.js later off the same scene
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from src.head_surface import scalp_surface
from src.scene import Scene
from src.skull_boundary import load_mri_volume
from src.targets import standard_targets
from src.viz import render_targets

SAVE_DIR = Path("saves")


def find_mris(path):
    """One mri file, or every nifti in a folder."""

    path = Path(path)
    if path.is_dir():
        return sorted(path.glob("*.nii")) + sorted(path.glob("*.nii.gz"))
    return [path]


def build_scene(mri_path):
    """Mri in, finished head scene out."""

    volume, affine = load_mri_volume(mri_path)

    # sigma 2 keeps the scalp smooth for the figure
    scalp = scalp_surface(volume, affine, smooth_sigma=2.0)
    targets = standard_targets(scalp)

    return Scene(name=Path(mri_path).name.split(".")[0], scalp=scalp, targets=targets)


def main():
    pv.OFF_SCREEN = True

    # default to the saves folder, or take a path off the command line
    src = sys.argv[1] if len(sys.argv) > 1 else SAVE_DIR
    mris = find_mris(src)
    if not mris:
        print(f"no mri found at {src}")
        return

    for mri in mris:
        print(f"\n=== {mri} ===")
        scene = build_scene(mri)

        scalp = scene.scalp
        lo = np.array(scalp.bounds[0::2])
        hi = np.array(scalp.bounds[1::2])
        print(f"scalp verts   {scalp.n_points}")
        print(f"scalp span mm {np.round(hi - lo, 1)}")
        for t in scene.targets:
            print(f"{t.label:9s} contact RAS {np.round(t.contact, 1)}")

        out = render_targets(
            scalp,
            scene.targets,
            SAVE_DIR / f"{scene.name}_targets.png",
            title=f"{scene.name}  study targets",
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
