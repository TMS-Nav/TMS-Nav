# dump a scene out for the three.js viewer, ply meshes plus a small json. the
# coil is already placed in RAS so the viewer just loads it as is
import json
from pathlib import Path

import numpy as np


def export_web(scene, out_dir):
    """Write the meshes and markers.json for one subject."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # all meshes already in RAS mm
    scene.scalp.save(str(out_dir / "scalp.ply"))
    scene.coil.save(str(out_dir / "coil.ply"))
    if scene.brain is not None:
        scene.brain.save(str(out_dir / "brain.ply"))

    payload = {
        "subject": scene.name,
        "space": "RAS_mm",
        "targets": [_marker_json(t) for t in scene.targets],
        "landmarks": [_marker_json(m) for m in scene.landmarks],
    }

    with open(out_dir / "markers.json", "w") as f:
        json.dump(payload, f, indent=2)

    return out_dir


def _marker_json(m):
    return {
        "name": m.name,
        "label": m.label,
        "color": m.color,
        "position": _vec(m.contact),
        "normal": _vec(m.normal),
    }


def _vec(v):
    return [round(float(x), 3) for x in np.asarray(v)]
