# the aim 2 layer for the three.js viewer: |E| on the cortex and in an mri slice.
#
#   python -m efield.export_web saves/efield/<subject> <viewer subject> [--demo]
#
# mirrors src/export_web.py, which drops ply meshes plus a small json per subject
# into viewer/src/data/<subject>/. this writes viewer/src/data/<subject>/efield/:
#
#   cortex.ply              the gm central surface, decimated, RAS mm
#   efield_<pose>.f32       |E| per cortex vertex, float32, one file per pose
#   normal_<pose>.f32       E_normal per cortex vertex, float32, signed. simnibs
#                           takes it as E . the central surface's node normal,
#                           which points out of the cortex, so positive is a
#                           field pointing out of the cortex, negative into it
#   t1.u8                   the t1, cropped to the head and resampled, uint8
#   field_<pose>.u8         |E| on the same grid, uint8, grey and white matter only
#   manifest.json           what all of the above is, grid, scale, roi numbers
#
# everything is read from what the runner left behind: results.json, the
# web/<pose>.npz surface dumps and the subject_volumes niftis (map_to_vol). so
# this runs in the project python, no simnibs needed.
import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from efield.config import ROOT

VIEWER_DATA_DIR = ROOT / "viewer" / "src" / "data"

# faces the browser gets for both hemispheres together. the full central surface
# is ~600k, this keeps the gyri readable at ~1.5 MB
WEB_CORTEX_FACES = 80_000

# voxel size of the slice volumes, mm. ernie's head and neck at 1.5 mm is 3 MB per
# volume, 2 mm brings it to about 1.3 MB and the slice still reads as an mri
STEP_MM = 2.0

# final_tissues labels the field is kept in for the slice view, white and grey
# matter, so the slice shows the field in the brain rather than the scalp
BRAIN_LABELS = (1, 2)

# the top of the colour scale, a high percentile of the cortex field rather than
# the max so one sharp node at a sulcal edge does not wash out the rest
COLOR_PCT = 99.9


# --- cortex ---------------------------------------------------------------------
def nearest_values(src_points, src_values, dst_points):
    """src_values carried onto dst_points by nearest source point."""

    _, idx = cKDTree(np.asarray(src_points, dtype=float)).query(np.asarray(dst_points, dtype=float))
    return np.asarray(src_values, dtype=float)[idx]


def export_cortex(npz_paths, names, out_dir, faces=WEB_CORTEX_FACES):
    """cortex.ply plus one efield_<name>.f32 per pose, all off the same surface.

    every pose of a subject is on the same central surface, so the mesh is
    decimated once off the first npz and every pose's |E| is carried onto the
    kept vertices by nearest original node
    """

    # pyvista only here, it is slow to import and the volume half does not need it
    import pyvista as pv
    from src.export_web import web_mesh

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    first = np.load(npz_paths[0])
    nodes, tri = first["nodes"], first["triangles"]
    cells = np.hstack([np.full((len(tri), 1), 3), tri]).astype(np.int64).ravel()
    web = web_mesh(pv.PolyData(np.asarray(nodes, dtype=float), cells), faces=faces)
    web.save(str(out_dir / "cortex.ply"))

    out = {"file": "cortex.ply", "n_vertices": int(web.n_points), "n_faces": int(web.n_cells),
           "poses": {}}
    for path, name in zip(npz_paths, names):
        d = np.load(path)
        if d["nodes"].shape != nodes.shape:
            raise ValueError(f"{path} is not on the same surface as {npz_paths[0]}")
        e = nearest_values(d["nodes"], d["E_magn"], web.points).astype("<f4")
        fname = f"efield_{name}.f32"
        e.tofile(out_dir / fname)
        en = nearest_values(d["nodes"], d["E_normal"], web.points).astype("<f4")
        nname = f"normal_{name}.f32"
        en.tofile(out_dir / nname)
        out["poses"][name] = {"file": fname, "peak": float(np.max(d["E_magn"])),
                              "p_color": float(np.percentile(d["E_magn"], COLOR_PCT)),
                              "normal_file": nname,
                              "normal_p_color": float(np.percentile(np.abs(d["E_normal"]), COLOR_PCT))}
    return out


# what a previous export of a subject leaves behind, one file per pose. cleared
# before every export so a pose that was renamed or dropped does not linger
POSE_FILES = ("efield_*.f32", "normal_*.f32", "field_*.u8")


def clear_pose_files(out_dir):
    for pattern in POSE_FILES:
        for f in Path(out_dir).glob(pattern):
            f.unlink()


# --- volumes -------------------------------------------------------------------
def crop_grid(ref_affine, mask, step_mm, pad_mm=4.0):
    """Shape and voxel to RAS affine of a step_mm grid around the mask.

    the grid keeps the reference scan's axes, only the spacing and the origin
    change, so a slice in the viewer is a slice of the scan, not an oblique cut
    """

    ijk = np.argwhere(mask)
    lo, hi = ijk.min(axis=0), ijk.max(axis=0)
    vox = np.sqrt((ref_affine[:3, :3] ** 2).sum(axis=0))
    pad = np.ceil(pad_mm / vox).astype(int)
    lo = np.maximum(lo - pad, 0)
    hi = np.minimum(hi + pad, np.array(mask.shape) - 1)
    stride = step_mm / vox
    shape = (np.floor((hi - lo) / stride).astype(int) + 1)
    sub = np.diag(np.append(stride, 1.0))
    sub[:3, 3] = lo
    return tuple(int(n) for n in shape), ref_affine @ sub


def resample(img, shape, affine, order=1):
    """img sampled on the grid (shape, affine), RAS mm in both."""

    data = np.asarray(img.dataobj, dtype=np.float32)
    if data.ndim == 4:
        data = data[..., 0]
    to_src = np.linalg.inv(img.affine) @ affine
    grid = np.indices(shape, dtype=np.float32).reshape(3, -1)
    src = to_src[:3, :3] @ grid + to_src[:3, 3:4]
    return map_coordinates(data, src, order=order, mode="constant", cval=0.0).reshape(shape)


def to_bytes(vol):
    """x fastest, then y, then z, the order a three.js Data3DTexture reads."""

    return np.ascontiguousarray(np.asarray(vol, dtype=np.uint8).transpose(2, 1, 0)).tobytes()


def quantise(values, top):
    """values in [0, top] onto 0..255, and the size of one step."""

    step = float(top) / 255.0 if top > 0 else 1.0
    return np.clip(np.rint(np.asarray(values) / step), 0, 255).astype(np.uint8), step


def export_volumes(t1_path, tissues_path, field_paths, names, out_dir, color_max, step_mm=STEP_MM):
    """t1.u8 and one field_<name>.u8 per pose on one cropped grid."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t1 = nib.load(str(t1_path))
    tissues = nib.load(str(tissues_path))
    labels = np.asarray(tissues.dataobj)
    if labels.ndim == 4:
        labels = labels[..., 0]
    shape, affine = crop_grid(tissues.affine, labels > 0, step_mm)

    lab = resample(tissues, shape, affine, order=0)
    head = lab > 0
    brain = np.isin(np.rint(lab).astype(int), BRAIN_LABELS)

    t = resample(t1, shape, affine)
    lo, hi = np.percentile(t[head], [1, 99])
    t_u8 = np.clip((t - lo) / max(hi - lo, 1e-6) * 255.0, 0, 255)
    t_u8[~head] = 0
    (out_dir / "t1.u8").write_bytes(to_bytes(t_u8))

    fields = {}
    for path, name in zip(field_paths, names):
        if path is None:
            continue
        e = resample(nib.load(str(path)), shape, affine)
        e[~brain] = 0.0
        q, step = quantise(e, color_max)
        fname = f"field_{name}.u8"
        (out_dir / fname).write_bytes(to_bytes(q))
        fields[name] = {"file": fname}

    return {"dims": list(shape), "affine": np.round(affine, 6).tolist(), "step_mm": float(step_mm),
            "order": "x fastest", "t1": "t1.u8", "field_step": float(color_max) / 255.0,
            "field_mask": "grey and white matter", "fields": fields}


# --- one subject ----------------------------------------------------------------
def _pose_frame(entry, dry):
    """matsimnibs of a pose, from results.json, else the snapped one in dryrun.json."""

    m = entry.get("matsimnibs")
    if m is None and dry:
        m = next((p["matsimnibs"] for p in dry.get("poses", []) if p["name"] == entry["name"]), None)
    if m is None:
        return None
    m = np.asarray(m, dtype=float)
    return {"centre": np.round(m[:3, 3], 3).tolist(), "handle": np.round(m[:3, 1], 5).tolist(),
            "normal_in": np.round(m[:3, 2], 5).tolist()}


def _pose_roi(entry, job, dry):
    """The roi numbers for the pose's own site."""

    site_of = {r["name"]: r.get("site") for r in job.get("rois", [])}
    for name, m in entry.get("rois", {}).items():
        if site_of.get(name) != entry.get("site"):
            continue
        centre = m.get("centre")
        if centre is None and dry:
            centre = next((r.get("centre_snapped") for r in dry.get("rois", []) if r["name"] == name), None)
        radius = next((r["radius_mm"] for r in job.get("rois", []) if r["name"] == name), None)
        return {"name": name, "centre": centre, "radius_mm": radius,
                "mean_E_magn": m["mean_E_magn"], "peak_E_magn": m["peak_E_magn"],
                "max_abs_E_normal": m.get("max_abs_E_normal")}
    return None


def same_frame(a_path, b_path, tol=1e-3):
    """True when two niftis share one voxel to RAS mapping and grid."""

    if not (Path(a_path).exists() and Path(b_path).exists()):
        return False
    a, b = nib.load(str(a_path)), nib.load(str(b_path))
    return a.shape[:3] == b.shape[:3] and np.allclose(a.affine, b.affine, atol=tol)


def export_subject(job_dir, viewer_subject, demo=False, out_root=VIEWER_DATA_DIR, aim1_t1=None):
    """viewer/src/data/<viewer_subject>/efield/ from one job folder."""

    job_dir = Path(job_dir)
    with open(job_dir / "results.json") as f:
        res = json.load(f)
    with open(job_dir / "job.json") as f:
        job = json.load(f)
    dry = None
    if (job_dir / "dryrun.json").exists():
        with open(job_dir / "dryrun.json") as f:
            dry = json.load(f)

    poses = [p for p in res["poses"] if p.get("surface_npz")]
    if not poses:
        raise ValueError(f"{job_dir / 'results.json'} has no surface dumps, rerun the runner")
    names = [p["name"] for p in poses]
    out_dir = Path(out_root) / viewer_subject / "efield"
    clear_pose_files(out_dir)

    cortex = export_cortex([p["surface_npz"] for p in poses], names, out_dir)
    # one colour scale for every pose of the subject, so two poses compare by eye
    color_max = max(c["p_color"] for c in cortex["poses"].values())

    m2m = Path(res["m2m"])
    volume = None
    if any(p.get("volume") for p in poses):
        volume = export_volumes(m2m / "T1.nii.gz", m2m / "final_tissues.nii.gz",
                                [p.get("volume") for p in poses], names, out_dir, color_max)

    aim1_t1 = Path(aim1_t1) if aim1_t1 else ROOT / "saves" / f"{viewer_subject}.nii.gz"
    manifest = {
        "subject": viewer_subject,
        "job": job["subject"],
        "demo": bool(demo),
        "notes": job.get("notes", []),
        "didt": float(res.get("didt", 1e6)),
        "units": "V/m",
        "color_max": float(color_max),
        # the signed perpendicular field gets its own symmetric scale
        "normal_max": float(max(c["normal_p_color"] for c in cortex["poses"].values())),
        "normal_sign": "positive points out of the cortex, negative into it",
        # the aim 1 scalp can be drawn around this cortex only if both came off one scan
        "shares_aim1_frame": same_frame(m2m / "T1.nii.gz", aim1_t1),
        "cortex": {k: cortex[k] for k in ("file", "n_vertices", "n_faces")},
        "volume": volume,
        "poses": [],
    }
    for p in poses:
        vol = (volume or {}).get("fields", {}).get(p["name"])
        manifest["poses"].append({
            "name": p["name"], "site": p.get("site"), "method": p.get("method"), "rep": p.get("rep"),
            "surface": cortex["poses"][p["name"]]["file"],
            "surface_peak": cortex["poses"][p["name"]]["peak"],
            "surface_normal": cortex["poses"][p["name"]]["normal_file"],
            "volume": vol["file"] if vol else None,
            "coil": _pose_frame(p, dry),
            "roi": _pose_roi(p, job, dry),
        })
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    sizes = {q.name: q.stat().st_size for q in sorted(out_dir.iterdir())}
    print(f"wrote {out_dir}")
    for k, v in sizes.items():
        print(f"    {k:28s} {v / 1e6:6.2f} MB")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description="aim 2 viewer files from one simnibs job folder")
    ap.add_argument("job_dir", nargs="?", default=str(ROOT / "saves" / "efield" / "demo_ernie"))
    ap.add_argument("subject", nargs="?", default="Ernie", help="the viewer subject folder it belongs to")
    ap.add_argument("--demo", action="store_true", help="mark the export as a demo")
    args = ap.parse_args(argv)
    demo = args.demo or Path(args.job_dir).name.startswith("demo")
    export_subject(args.job_dir, args.subject, demo=demo)


if __name__ == "__main__":
    main()
