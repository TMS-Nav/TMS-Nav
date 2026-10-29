# dump a scene out for the three.js viewer, ply meshes plus a small json. the
# coil is already placed in RAS so the viewer just loads it as is
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import pyvista as pv

# the scalp the browser gets. the full mesh is ~430k triangles and 8 to 11 MB, which
# is slow to pull into an embedded page. 60k triangles sits within 0.07 mm of the
# full surface (checked with pyvista compute_implicit_distance on Ernie) at about 1.5 MB.
# the targets, the monte carlo and the previews all still use the full mesh
WEB_SCALP_FACES = 60_000


def web_mesh(surf, faces=WEB_SCALP_FACES):
    """A lighter copy of a triangle mesh for the browser, point data dropped."""

    # polygons only, then triangles. a defaced scan can leave stray line cells and
    # the odd polygon where the wipe was capped, and decimation takes triangles only
    mesh = pv.PolyData(surf.points, surf.faces).triangulate().clean()
    if mesh.n_cells > faces:
        mesh = mesh.decimate(1.0 - faces / mesh.n_cells)
    mesh = mesh.copy()
    # the viewer computes its own normals, shipping them only doubles the file
    mesh.clear_data()
    return mesh


def export_web(scene, out_dir):
    """Write the meshes and markers.json for one subject."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # all meshes already in RAS mm
    web_mesh(scene.scalp).save(str(out_dir / "scalp.ply"))
    scene.coil.save(str(out_dir / "coil.ply"))

    payload = {
        "subject": scene.name,
        "space": "RAS_mm",
        "coordinate_frame": scene.coordinate_frame,
        "targets": [_marker_json(t) for t in scene.targets],
        "landmarks": [_marker_json(m) for m in scene.landmarks],
        "electrodes": [_marker_json(m) for m in scene.electrodes],
        "eeg_arcs": scene.eeg_arcs,
        "eeg_qc": scene.eeg_qc,
        # threshold, defaced flag and so on, so the viewer can say what it is showing
        "qc": scene.qc,
    }

    with open(out_dir / "markers.json", "w") as f:
        json.dump(payload, f, indent=2)

    return out_dir


def _marker_json(m):
    return {
        "name": m.name,
        "label": m.label,
        "color": m.color,
        "optional": bool(getattr(m, "optional", False)),
        "position": _vec(m.contact),
        "normal": _vec(m.normal),
    }


def _vec(v):
    return [round(float(x), 3) for x in np.asarray(v)]


def efield_files(manifest_path):
    """Every file an aim 2 manifest (efield/export_web.py) points at, in a fixed order."""

    with open(manifest_path) as f:
        m = json.load(f)
    files = [m["cortex"]["file"]]
    if m.get("volume"):
        files.append(m["volume"]["t1"])
    for p in m["poses"]:
        files += [fn for fn in (p.get("surface"), p.get("surface_normal"), p.get("volume")) if fn]
    return files


def write_registry(data_dir, registry_path):
    """Regenerate datasetRegistry.js from whatever subject folders actually exist.

    parcel needs static imports, so the viewer cannot read the folder at runtime.
    the pipeline writes this file instead, which means the dropdown always matches
    the mris that have actually been processed. rename a file in saves, rerun the
    pipeline, and the dropdown follows.

    the monte carlo json is half a MB per subject, so it goes in as a url: asset
    and the viewer fetches only the subject on screen, rather than every subject
    riding in the script bundle. the aim 1 statistics (analysis/aim1_stats.py)
    come in when that file exists, and as null until then. the same goes for a
    subject's aim 2 e field (efield/export_web.py), its manifest plus a url per file
    """

    data_dir = Path(data_dir)
    registry_path = Path(registry_path)

    subjects = sorted(
        (d.name for d in data_dir.iterdir()
        if d.is_dir() and (d / "markers.json").exists()),
        key=lambda name: (name != "MNI152", name))
    if not subjects:
        raise ValueError(f"no processed subjects found under {data_dir}")

    imports = []
    if (data_dir / "aim1_stats.json").exists():
        imports.append('import AIM1_DATA from "./data/aim1_stats.json";')
    entries = []
    for i, name in enumerate(subjects):
        # the folder name is whatever the mri was called, so it can be anything.
        # keep it verbatim as the label and the key, and use a safe made up
        # identifier for the import binding
        var = f"S{i}"
        imports.append(f'import {var}_MARKERS from "./data/{name}/markers.json";')
        with open(data_dir / name / "markers.json") as f:
            payload = json.load(f)
        qc = payload.get("qc", {})
        label = f"{name} (defaced)" if qc.get("defaced") else name
        template = name == "MNI152"
        if template and (data_dir / name / "montecarlo.json").exists():
            imports.append(f'import {var}_MC from "url:./data/{name}/montecarlo.json";')
            mc = f"{var}_MC"
        else:
            mc = "null"
        def asset(filename):
            if (data_dir / name / filename).exists():
                return f'new URL("./data/{name}/{filename}", import.meta.url)'
            return "null"

        efield = "null"
        manifest = data_dir / name / "efield" / "manifest.json"
        if manifest.exists():
            imports.append(f'import {var}_EFIELD from "./data/{name}/efield/manifest.json";')
            urls = "".join(
                f'        {json.dumps(fn)}: new URL("./data/{name}/efield/{fn}", import.meta.url),\n'
                for fn in efield_files(manifest)
            )
            efield = f"{{\n      manifest: {var}_EFIELD,\n      files: {{\n{urls}      }},\n    }}"

        entries.append(
            f'  {json.dumps(name)}: {{\n'
            f'    label: {json.dumps(label)},\n'
            f'    kind: {json.dumps("template" if template else "subject")},\n'
            f'    scalpURL: {asset("scalp.ply")},\n'
            f'    skullURL: {asset("skull.ply")},\n'
            f'    coilURL: {asset("coil.ply") if template else "null"},\n'
            f'    markers: {var}_MARKERS,\n'
            f'    monteCarloURL: {mc},\n'
            f'    efield: {efield},\n'
            f'  }},'
        )

    aim1 = "AIM1_DATA" if (data_dir / "aim1_stats.json").exists() else "null"
    text = (
        "// GENERATED by src/export_web.py, do not edit by hand.\n"
        "// rerun run_pipeline.py to pick up new or renamed mris in saves/\n"
        + "\n".join(imports)
        + "\n\nexport const DATASETS = {\n"
        + "\n".join(entries)
        + "\n};\n\n"
        # the real aim 1 numbers, written by analysis/aim1_stats.py off brainsight
        + f"export const AIM1 = {aim1};\n"
    )
    if not registry_path.exists() or registry_path.read_text(encoding="utf-8") != text:
        # Parcel should observe a complete registry, including on cache re-export.
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".js",
                                         dir=registry_path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(text)
        try:
            os.replace(temporary, registry_path)
        finally:
            temporary.unlink(missing_ok=True)
    return registry_path, subjects

