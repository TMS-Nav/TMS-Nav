# where simnibs lives on this machine, the coil and stimulation defaults, and
# the folders the aim 2 files go to. every path can be overridden with an
# environment variable so the same code runs on another machine untouched.
import os
import subprocess
import tempfile
from pathlib import Path

# the project root, one level up from this file
ROOT = Path(__file__).resolve().parent.parent


def _env_path(name, default):
    """A path from the environment if set, else the default."""

    return Path(os.environ.get(name, default))


# --- simnibs install ------------------------------------------------------------
SIMNIBS_DIR = _env_path("TMSNAV_SIMNIBS_DIR", r"C:\Users\alizr\SimNIBS-4.6")
# the conda python that has simnibs on it. it is a .cmd wrapper on windows, and
# passing it `-c "..."` printed nothing in testing, so it is always given a script
# FILE to run, never an inline string. see simnibs_version below
SIMNIBS_PYTHON = _env_path("TMSNAV_SIMNIBS_PYTHON", SIMNIBS_DIR / "bin" / "simnibs_python.cmd")
# head model builder, `charm <subID> <T1> [T2]`
CHARM = _env_path("TMSNAV_CHARM", SIMNIBS_DIR / "bin" / "charm.cmd")

# --- coil and stimulation -----------------------------------------------------
# relative names resolve inside simnibs resources/coil_models. the magstim 70 mm
# figure of eight is the study coil. the other two on this install are
# Drakaki_BrainStim_2022/MagVenture_Cool-B65.ccd and .../MagVenture_C-B70.ccd
COIL_FILE = os.environ.get("TMSNAV_COIL_FILE", os.path.join("legacy_and_other", "Magstim_70mm_Fig8.ccd"))
# rate of change of coil current, A/s. 1e6 is 1 A/us, the simnibs default, and
# the e field scales linearly with it so any intensity can be read off later
DIDT = float(os.environ.get("TMSNAV_DIDT", 1e6))
# coil housing to scalp, mm. simnibs default, used when a pose is built from a
# scalp point rather than a recorded matrix
COIL_SKIN_DISTANCE_MM = float(os.environ.get("TMSNAV_COIL_SKIN_DISTANCE_MM", 4.0))

# --- roi and overlap -----------------------------------------------------------
# radius of the sphere on the gm central surface that the roi numbers come from
ROI_RADIUS_MM = float(os.environ.get("TMSNAV_ROI_RADIUS_MM", 10.0))
# a node is suprathreshold when its |E| is at least this fraction of the peak |E|
# over the whole surface. half the peak is the usual convention for the hotspot
SUPRATHRESHOLD_FRAC = float(os.environ.get("TMSNAV_SUPRATHRESHOLD_FRAC", 0.5))

# and an ABSOLUTE threshold in V/m, used for the same hotspot at a cut that does
# not move between the two poses being compared. the fraction above is relative to
# each pose's own peak, so if the eeg placement is weaker overall its threshold
# drops with it and the overlap partly measures the threshold rather than the
# location. an absolute cut is the honest one for a paired comparison. 50 V/m is a
# PLACEHOLDER, it wants a value from the pilot simulations and his sign off
SUPRATHRESHOLD_ABS_VM = float(os.environ.get("TMSNAV_SUPRATHRESHOLD_ABS_VM", 50.0))

# --- folders -------------------------------------------------------------------
# all under saves/, which is git ignored, because everything here is derived from
# patient scans. m2m_<subject> head models, one folder per subject
M2M_ROOT = _env_path("TMSNAV_M2M_ROOT", ROOT / "saves" / "m2m")
# jobs, dry run plans and results, one folder per subject
SIM_ROOT = _env_path("TMSNAV_SIM_ROOT", ROOT / "saves" / "efield")
# brainsight session exports, <subject>_coil_pose_samples.txt
BRAINSIGHT_ROOT = _env_path("TMSNAV_BRAINSIGHT_ROOT", ROOT / "saves" / "brainsight")

# finished example head models that ship with simnibs, used for dry runs until
# the study scans have been through charm
SAMPLE_M2M_ROOTS = [
    Path(r"C:\Users\alizr\Documents\.TMSResearch\Sample Datasets\Ernie"),
    Path(r"C:\Users\alizr\Documents\.TMSResearch\Sample Datasets\GroupDataset"),
]


# --- is simnibs there ---------------------------------------------------------
def simnibs_available():
    """True when the simnibs python wrapper exists on disk."""

    return Path(SIMNIBS_PYTHON).exists()


def simnibs_version(timeout=120):
    """The simnibs version string, or None if it cannot be run.

    runs a two line script through simnibs_python. importing simnibs takes a few
    seconds, so this is not something to call in a loop
    """

    if not simnibs_available():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "version.py"
        script.write_text("import simnibs\nprint(simnibs.__version__)\n")
        try:
            out = subprocess.run([str(SIMNIBS_PYTHON), str(script)], capture_output=True,
                                 text=True, timeout=timeout)
        except (OSError, subprocess.TimeoutExpired):
            return None
    if out.returncode != 0:
        return None
    lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def describe():
    """One line per setting, for the driver to print."""

    return "\n".join([
        f"simnibs dir      {SIMNIBS_DIR}  ({'found' if simnibs_available() else 'MISSING'})",
        f"simnibs python   {SIMNIBS_PYTHON}",
        f"charm            {CHARM}",
        f"coil             {COIL_FILE}",
        f"didt             {DIDT:g} A/s",
        f"coil-skin        {COIL_SKIN_DISTANCE_MM} mm",
        f"roi radius       {ROI_RADIUS_MM} mm",
        f"suprathreshold   {SUPRATHRESHOLD_FRAC:g} x peak, or {SUPRATHRESHOLD_ABS_VM:g} V/m absolute",
        f"m2m root         {M2M_ROOT}",
        f"sim root         {SIM_ROOT}",
        f"brainsight root  {BRAINSIGHT_ROOT}",
    ])
