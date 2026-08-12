# the finished head scene, everything a viewer needs in one place. this is also
# the handoff point for three.js later, a gltf writer would read straight off
# this instead of the pyvista plotter
from dataclasses import dataclass, field


@dataclass
class Scene:
    name: str
    scalp: object          # pyvista surface mesh in RAS mm
    targets: list = field(default_factory=list)
    coil: object = None    # seated figure-8 coil mesh in RAS mm
    brain: object = None   # rough intracranial surface in RAS mm
    landmarks: list = field(default_factory=list)  # eeg registration points
