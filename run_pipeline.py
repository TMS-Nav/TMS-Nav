# feed an mri or a folder of mris, build the head, drop the study targets and a
# coil on it, save front and side previews for each into saves/previews. this is
# the sanity check view, the real showcase will be three.js later off the scene
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from src.coil_model import place_coil
from src.export_mc import export_monte_carlo
from src.export_web import export_web, write_registry
from src.head_surface import brain_surface, scalp_surface
from src.scene import Scene
from src.skull_boundary import head_mask, load_mri_volume
from src.stats import render_distance_hist
from src.targets import standard_landmarks, standard_targets
from src.viz import render_view

SAVE_DIR = Path("saves")
PREVIEW_DIR = SAVE_DIR / "previews"

# three.js viewer reads its data from here, one folder per subject
VIEWER_DATA_DIR = Path("viewer") / "src" / "data"

# regenerated every run so the viewer dropdown matches whatever is in saves/
REGISTRY_PATH = Path("viewer") / "src" / "datasetRegistry.js"

GAP_MM = 2.0

# where the demo coil sits. the protocol records coil poses at sma and f4, and f4
# is the one off the midline, so the coil is centred over f4 the way the operator
# would place it
COIL_AIM = (0.55, 1.0, 0.35)

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

    # one mask for both surfaces, and the qc that came with it
    mask, info = head_mask(volume, affine)

    # sigma 2 keeps the scalp smooth for the figure
    scalp = scalp_surface(volume, affine, smooth_sigma=2.0, mask=mask, wipe=info.wipe)
    brain = brain_surface(volume, affine, mask=mask)
    targets = standard_targets(scalp)
    landmarks = standard_landmarks(scalp)
    coil = place_coil(scalp, COIL_AIM, gap=GAP_MM)

    return Scene(
        name=Path(mri_path).name.split(".")[0],
        scalp=scalp,
        targets=targets,
        coil=coil,
        brain=brain,
        landmarks=landmarks,
        qc=info.as_dict(),
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
        qc = scene.qc
        print(f"voxel mm      {qc['voxel_mm']}   air/tissue cut {qc['threshold']:.0f}")
        print(f"head          {100 * qc['head_frac']:.0f}% of volume,"
              f" {100 * qc['nodata_frac']:.0f}% of volume is zero padding or wipe")
        if qc["defaced"]:
            print(f"DEFACED       {qc['cut_area_cm2']:.0f} cm2 of head surface is the"
                  f" wipe, the face is not in this file and cannot be rebuilt")
        print(f"scalp verts   {scalp.n_points}")
        print(f"brain verts   {scene.brain.n_points}")
        print(f"scalp span mm {np.round(hi - lo, 1)}")
        for t in scene.targets:
            print(f"target {t.label:4s} RAS {np.round(t.contact, 1)}")
        for m in scene.landmarks:
            print(f"lmark  {m.label:4s} RAS {np.round(m.contact, 1)}")

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

        # and the monte carlo cloud that sits on top of the targets
        mc_path, mc = export_monte_carlo(scene, VIEWER_DATA_DIR / scene.name)
        print(f"wrote {mc_path}")
        print(f"  registered onto the subject, scale {mc['fit_scale']:.3f},"
              f" landmark residual {mc['fit_residual_mm']:.1f} mm")
        for site in mc["sites"]:
            sig = ", ".join(f"{v:.2f}" for v in site["sigmas"])
            m, q = site["miss_stats"], site["pair_stats"]
            opt = " (optional)" if site.get("optional") else ""
            print(f"  {site['label']:4s} to target mean {m['mean_mm']:.2f} sd {m['sd_mm']:.2f}"
                  f" p95 {m['p95_mm']:.2f} mm, centroid off {site['bias_mm']:.2f} mm"
                  f" | between placements mean {q['mean_mm']:.2f}"
                  f" p95 {q['p95_mm']:.2f} mm | sigmas {sig}{opt}")

        # the distributions themselves, as a figure next to the previews
        fig = render_distance_hist(mc, PREVIEW_DIR / f"{scene.name}_distances.png", subject=scene.name)
        print(f"wrote {fig}")

    # last, point the viewer at every subject that came out of this run
    reg, subjects = write_registry(VIEWER_DATA_DIR, REGISTRY_PATH)
    print()
    print(f"wrote {reg}")
    print(f"viewer dropdown: {', '.join(subjects)}")


if __name__ == "__main__":
    main()
