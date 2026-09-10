# finding, checking and (only when asked) building the simnibs head model.
#
# charm takes the t1 (and t2 if there is one) and writes m2m_<sub>/ with the
# tetrahedral mesh <sub>.msh, the tissue labels, the gm central surfaces and the
# electrode positions. it takes an hour or so per head. nothing here runs it
# unless dry_run=False is passed on purpose, the default only prints the command.
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from efield.config import CHARM, M2M_ROOT, SAMPLE_M2M_ROOTS

# the electrode file the stand in poses read. 10-20 with the study labels in it
EEG_CAP_FILE = "EEG10-20_Okamoto_2004.csv"


@dataclass
class M2M:
    path: Path
    sub_id: str
    mesh: Path
    t1: Path
    complete: bool                      # <sub>.msh is there
    has_eeg_positions: bool
    has_central_surfaces: bool
    missing: list = field(default_factory=list)

    @property
    def eeg_cap(self):
        return self.path / "eeg_positions" / EEG_CAP_FILE

    def report(self):
        state = "complete" if self.complete else "INCOMPLETE"
        lines = [f"m2m {self.sub_id:8s} {state}  {self.path}"]
        if self.missing:
            lines.append("    missing: " + ", ".join(self.missing))
        return "\n".join(lines)


def inspect_m2m(path):
    """What is in an m2m folder, complete or not."""

    path = Path(path)
    name = path.name
    sub_id = name[4:] if name.startswith("m2m_") else name
    mesh = path / f"{sub_id}.msh"
    t1 = path / "T1.nii.gz"
    surfaces = path / "surfaces"
    central = all((surfaces / f"{h}.central.gii").exists() for h in ("lh", "rh"))
    eeg = (path / "eeg_positions" / EEG_CAP_FILE).exists()

    missing = []
    if not mesh.exists():
        missing.append(mesh.name)
    if not t1.exists():
        missing.append(t1.name)
    if not central:
        missing.append("surfaces/lh|rh.central.gii")
    if not eeg:
        missing.append(f"eeg_positions/{EEG_CAP_FILE}")
    return M2M(path=path, sub_id=sub_id, mesh=mesh, t1=t1, complete=mesh.exists(),
               has_eeg_positions=eeg, has_central_surfaces=central, missing=missing)


def find_m2m(subject, roots=None):
    """The m2m folder for a subject, searching the project root then the samples.

    subject can be a plain id (T1, ernie, sub01) or a path to an m2m folder.
    returns an M2M or None when nothing is found
    """

    p = Path(subject)
    if p.is_dir() and p.name.startswith("m2m_"):
        return inspect_m2m(p)
    roots = [Path(r) for r in (roots if roots is not None else [M2M_ROOT] + SAMPLE_M2M_ROOTS)]
    for root in roots:
        cand = root / f"m2m_{subject}"
        if cand.is_dir():
            return inspect_m2m(cand)
    return None


def charm_command(subject, t1, t2=None, out_root=M2M_ROOT):
    """The argv for charm, to run FROM out_root so m2m_<subject> lands there."""

    cmd = [str(CHARM), str(subject), str(Path(t1).resolve())]
    if t2 is not None:
        cmd.append(str(Path(t2).resolve()))
    return cmd


def format_command(cmd):
    """Argv as one shell line, quoting what has spaces in it."""

    return " ".join(f'"{c}"' if " " in c else c for c in cmd)


def run_charm(subject, t1, t2=None, out_root=M2M_ROOT, dry_run=True):
    """Print the charm command, and only run it when dry_run is False.

    building a head model is an hour of cpu and writes a few hundred mb, so it
    never happens as a side effect. the returned value is the command in a dry
    run and the CompletedProcess otherwise
    """

    out_root = Path(out_root)
    cmd = charm_command(subject, t1, t2, out_root)
    print(f"charm (cwd {out_root}):  {format_command(cmd)}")
    if dry_run:
        print("    dry run, not executed")
        return cmd
    out_root.mkdir(parents=True, exist_ok=True)
    return subprocess.run(cmd, cwd=str(out_root))


def status(subjects, t1_of=None, roots=None):
    """One line per subject, found or the charm command that would build it."""

    lines = []
    for s in subjects:
        m = find_m2m(s, roots)
        if m is not None:
            lines.append(m.report())
        elif t1_of is not None and t1_of(s) is not None:
            lines.append(f"m2m {s:8s} none, would build with: {format_command(charm_command(s, t1_of(s)))}")
        else:
            lines.append(f"m2m {s:8s} none, and no t1 to build it from")
    return "\n".join(lines)
