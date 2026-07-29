# feed an mri or a folder of mris, build the head, drop the study targets and a
# coil on it, save front and side previews for each into saves/previews. this is
# the sanity check view, the real showcase will be three.js later off the scene
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from src.coil_model import place_coil
from src.export_web import export_web
from src.head_surface import scalp_surface
from src.scene import Scene
from src.skull_boundary import load_mri_volume
from src.targets import standard_targets
from src.viz import render_view

SAVE_DIR = Path("saves")
PREVIEW_DIR = SAVE_DIR / "previews"

# three.js viewer reads its data from here, one folder per subject
VIEWER_DATA_DIR = Path("viewer") / "src" / "data"

GAP_MM = 2.0

# where the coil sits, out on the left side around c3. kept low and lateral so
# it clears the 3 markers instead of covering the vertex one
COIL_AIM = (-1.0, -0.15, 0.6)

VIEWS = ["front", "side"]


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
    coil = place_coil(scalp, COIL_AIM, gap=GAP_MM)

    return Scene(
        name=Path(mri_path).name.split(".")[0],
        scalp=scalp,
        targets=targets,
        coil=coil,
    )


def main():
    pv.OFF_SCREEN = True

    # default to the saves folder, or take a path off the command line
    src = sys.argv[1] if len(sys.argv) > 1 else SAVE_DIR
    mris = find_mris(src)
    if not mris:
        print(f"no mri found at {src}")
        return

    PREVIEW_DIR.mkdir(exist_ok=True)

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

        for view in VIEWS:
            out = render_view(
                scalp,
                PREVIEW_DIR / f"{scene.name}_{view}.png",
                coil=scene.coil,
                targets=scene.targets,
                view=view,
                title=f"{scene.name}  {view}",
            )
            print(f"wrote {out}")

        # dump the same scene out for the three.js viewer
        web = export_web(scene, VIEWER_DATA_DIR / scene.name)
        print(f"wrote {web}")


if __name__ == "__main__":
    main()
