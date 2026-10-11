# Build scalp scenes, previews and Three.js assets from structural MRIs or MNI152.
import argparse
import zlib

import numpy as np
import pyvista as pv

from src.coil_model import place_coil
from src.export_mc import export_monte_carlo
from src.export_web import export_web, write_registry
from src.head_surface import scalp_surface
from src.mri_inputs import ROOT, SAVE_DIR, dataset_name, find_mris, resolve_inputs
from src.scene import Scene
from src.skull_boundary import head_mask, load_mri_volume
from src.stats import render_distance_hist
from src.targets import standard_landmarks, standard_targets
from src.viz import render_view

PREVIEW_DIR = SAVE_DIR / "previews"

# three.js viewer reads its data from here, one folder per subject
VIEWER_DATA_DIR = ROOT / "viewer" / "src" / "data"

# Regenerated from existing processed dataset folders on every run.
REGISTRY_PATH = ROOT / "viewer" / "src" / "datasetRegistry.js"

GAP_MM = 2.0

# where the demo coil sits. the protocol records coil poses at sma and f4, and f4
# is the one off the midline, so the coil is centred over f4 the way the operator
# would place it
COIL_AIM = (0.55, 1.0, 0.35)

VIEWS = ["front", "side"]


def build_scene(mri_path, name=None):
    """Mri in, finished head scene out."""

    volume, affine = load_mri_volume(mri_path)

    # one mask for both surfaces, and the qc that came with it
    mask, info = head_mask(volume, affine)

    # sigma 2 keeps the scalp smooth for the figure
    scalp = scalp_surface(volume, affine, smooth_sigma=2.0, mask=mask, wipe=info.wipe)
    targets = standard_targets(scalp)
    landmarks = standard_landmarks(scalp)
    coil = place_coil(scalp, COIL_AIM, gap=GAP_MM)

    return Scene(
        name=name or dataset_name(mri_path),
        scalp=scalp,
        targets=targets,
        coil=coil,
        landmarks=landmarks,
        qc=info.as_dict(),
    )


def main(argv=None):
    pv.OFF_SCREEN = True

    parser = argparse.ArgumentParser(description="Build scalp scenes and Monte Carlo viewer data from T1w NIfTI scans.")
    parser.add_argument("paths", nargs="*", help="MRI files or folders (default: saves/; empty folders use MNI152)")
    parser.add_argument("--template", action="store_true", help="Use only the cached/downloaded full-head MNI152 demo")
    args = parser.parse_args(argv)
    if args.template and args.paths:
        parser.error("--template cannot be combined with MRI paths")
    try:
        inputs = resolve_inputs(args.paths, template=args.template)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    for mri, name in inputs:
        print(f"\n=== {mri} ===")
        scene = build_scene(mri, name=name)

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

        # and the monte carlo cloud that sits on top of the targets. the seed comes
        # off the subject name, so each subject is its own independent run of the
        # cap procedure and the run is still reproducible file by file
        seed = zlib.crc32(scene.name.encode())
        mc_path, mc = export_monte_carlo(scene, VIEWER_DATA_DIR / scene.name, seed=seed)
        print(f"wrote {mc_path}")
        print(f"  registered onto the subject, scale {mc['fit_scale']:.3f},"
              f" landmark residual {mc['fit_residual_mm']:.1f} mm")
        for site in mc["sites"]:
            m, q = site["miss_stats"], site["pair_stats"]
            a, o = site["tilt_stats"], site["orient_stats"]
            opt = " (optional)" if site.get("optional") else ""
            print(f"  {site['label']:4s} to target mean {m['mean']:.2f} sd {m['sd']:.2f}"
                  f" p95 {m['p95']:.2f} mm, centroid off {site['bias_mm']:.2f} mm"
                  f" | between placements mean {q['mean']:.2f} p95 {q['p95']:.2f} mm"
                  f" | tilt mean {a['mean']:.2f} p95 {a['p95']:.2f} deg"
                  f" | orientation mean {o['mean']:.2f} p95 {o['p95']:.2f} deg{opt}")
            ba = site["bland_altman"]
            print("       bland altman  " + "  ".join(
                f"{k} {v['bias']:+.2f} [{v['loa_lo']:+.2f}, {v['loa_hi']:+.2f}]"
                for k, v in ba.items()) + " mm")

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
