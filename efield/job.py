# the job file. everything the simnibs side needs to know, as json, so the two
# interpreters only ever talk through files. one job per subject.
import json
from dataclasses import dataclass, field
from pathlib import Path

from efield.config import (
    COIL_FILE,
    DIDT,
    SIMNIBS_PYTHON,
    SIM_ROOT,
    SUPRATHRESHOLD_ABS_VM,
    SUPRATHRESHOLD_FRAC,
)
from efield.poses import CoilPose
from efield.roi import RoiSpec

RUNNER = Path(__file__).resolve().parent / "runner_simnibs.py"

# what simnibs computes. e is the field vector, E its magnitude. j/J would add
# current density, not needed for the endpoints
DEFAULT_FIELDS = "eE"


@dataclass
class Job:
    subject: str
    m2m: str                                   # path to the m2m folder
    poses: list                                # [CoilPose]
    rois: list                                 # [RoiSpec]
    coil_file: str = COIL_FILE
    didt: float = DIDT
    fields: str = DEFAULT_FIELDS
    map_to_surf: bool = True
    # the two hotspot cuts, carried in the job so a result can always say what
    # threshold produced it. see roi_metrics in runner_simnibs.py for why there
    # are two of them
    supra_frac: float = SUPRATHRESHOLD_FRAC
    supra_abs_vm: float = SUPRATHRESHOLD_ABS_VM
    out_dir: str = ""                          # where simnibs writes, default SIM_ROOT/<subject>/sim
    # when true the runner rebuilds each matsimnibs with mesh.calc_matsimnibs,
    # which projects the centre onto the skin and takes the skin normal. wanted
    # for stand in poses built off aim 1's approximate scalp, not for real
    # brainsight poses which are already where the coil was
    snap_to_skin: bool = False
    poses_source: str = ""                     # brainsight file, markers.json, eeg cap ...
    notes: list = field(default_factory=list)

    def __post_init__(self):
        if not self.out_dir:
            self.out_dir = str(SIM_ROOT / self.subject / "sim")

    @property
    def job_dir(self):
        return SIM_ROOT / self.subject

    def to_dict(self):
        return {
            "subject": self.subject, "m2m": str(self.m2m), "coil_file": self.coil_file,
            "didt": float(self.didt), "fields": self.fields, "map_to_surf": bool(self.map_to_surf),
            "supra_frac": float(self.supra_frac), "supra_abs_vm": float(self.supra_abs_vm),
            "out_dir": str(self.out_dir), "snap_to_skin": bool(self.snap_to_skin),
            "poses_source": self.poses_source, "notes": list(self.notes),
            "poses": [p.to_dict() for p in self.poses],
            "rois": [r.to_dict() for r in self.rois],
        }

    @classmethod
    def from_dict(cls, d):
        return cls(subject=d["subject"], m2m=d["m2m"],
                   poses=[CoilPose.from_dict(p) for p in d.get("poses", [])],
                   rois=[RoiSpec.from_dict(r) for r in d.get("rois", [])],
                   coil_file=d.get("coil_file", COIL_FILE), didt=float(d.get("didt", DIDT)),
                   fields=d.get("fields", DEFAULT_FIELDS), map_to_surf=bool(d.get("map_to_surf", True)),
                   supra_frac=float(d.get("supra_frac", SUPRATHRESHOLD_FRAC)),
                   supra_abs_vm=float(d.get("supra_abs_vm", SUPRATHRESHOLD_ABS_VM)),
                   out_dir=d.get("out_dir", ""), snap_to_skin=bool(d.get("snap_to_skin", False)),
                   poses_source=d.get("poses_source", ""), notes=list(d.get("notes", [])))

    def write_json(self, path=None):
        path = Path(path) if path else self.job_dir / "job.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return path

    @classmethod
    def read_json(cls, path):
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def summary(self):
        lines = [f"job {self.subject}: {len(self.poses)} poses, {len(self.rois)} rois, coil {self.coil_file}",
                 f"    m2m {self.m2m}", f"    out {self.out_dir}", f"    poses from {self.poses_source}"]
        for p in self.poses:
            lines.append(f"    {p}")
        for r in self.rois:
            seed = None if r.centre is None else [round(x, 1) for x in r.centre]
            lines.append(f"    roi {r.name} seed {seed}")
        return "\n".join(lines)


def runner_command(job_path, dry_run=True):
    """The argv that runs the simnibs side on a job file."""

    cmd = [str(SIMNIBS_PYTHON), str(RUNNER), str(job_path)]
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def results_path(job):
    return Path(job.job_dir) / "results.json"


def dryrun_path(job):
    return Path(job.job_dir) / "dryrun.json"
