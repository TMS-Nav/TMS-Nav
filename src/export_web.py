# dump a scene out for the three.js viewer, ply meshes plus a small json. the
# coil is already placed in RAS so the viewer just loads it as is
import json
from pathlib import Path

import numpy as np


def export_web(scene, out_dir):
    """Write scalp.ply, coil.ply and targets.json for one subject."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # head shell and the demo coil, both already in RAS mm
    scene.scalp.save(str(out_dir / "scalp.ply"))
    scene.coil.save(str(out_dir / "coil.ply"))

    payload = {
        "subject": scene.name,
        "space": "RAS_mm",
        "targets": [_target_json(t) for t in scene.targets],
    }

    with open(out_dir / "targets.json", "w") as f:
        json.dump(payload, f, indent=2)

    return out_dir


def _target_json(target):
    return {
        "name": target.name,
        "label": target.label,
        "color": target.color,
        "position": _vec(target.contact),
        "normal": _vec(target.normal),
    }


def _vec(v):
    return [round(float(x), 3) for x in np.asarray(v)]
