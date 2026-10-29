"""Structural scan discovery and the cached full-head MNI demo input."""
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = ROOT / "saves"
TEMPLATE_DIR = SAVE_DIR / "templates"
# These contain outputs or auxiliary inputs, never patient structural scans.
# sample_data is where scripts/fetch_sample_data.py unpacks the simnibs datasets,
# which carry raw T2 and diffusion scans of their own; the T1s it wants processed
# it copies up into saves/ itself
RESERVED = {"templates", "previews", "brainsight", "aim1", "efield", "m2m", "derivatives", "sample_data"}


def is_nifti(path):
    return path.name.lower().endswith((".nii", ".nii.gz"))


def find_mris(path):
    """Flat legacy saves scans and nested cohorts; skip generated directories."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"MRI input does not exist: {path}")
    if path.is_file():
        if not is_nifti(path):
            raise ValueError(f"Expected a .nii or .nii.gz structural MRI: {path}")
        return [path]
    scans = []
    for scan in path.rglob("*"):
        relative = scan.relative_to(path)
        # Dedicated subject folders can also hold optional masks/registration QC.
        # Only the declared T1w input belongs in automatic scan discovery there.
        if (scan.is_relative_to(SAVE_DIR / "subjects")
                and scan.name.lower() not in {"t1w.nii", "t1w.nii.gz"}):
            continue
        if scan.is_file() and is_nifti(scan) and not any(
            part.lower() in RESERVED or part.lower().startswith("m2m_")
            or part.startswith(".") for part in relative.parts[:-1]
        ):
            scans.append(scan)
    return sorted(scans)


def dataset_name(path, base=SAVE_DIR):
    """Keep legacy names; flatten cohort/subject/scan into a safe viewer key."""
    path = Path(path).resolve()
    try:
        relative = path.relative_to(Path(base).resolve())
    except ValueError:
        relative = Path(path.name)
    parts = list(relative.parts)
    if parts[0] == "cohorts":
        parts = parts[1:]
    parts[-1] = parts[-1][:-7] if parts[-1].lower().endswith(".nii.gz") else parts[-1][:-4]
    return "__".join(re.sub(r"[^A-Za-z0-9_.-]", "_", part) for part in parts)


def mni_template():
    """Fetch unmasked ICBM152 2009a T1; NiBabel loads it downstream.

    load_mni152_template is skull-stripped and cannot supply a scalp.
    Nilearn's fetcher reuses the downloaded files on subsequent runs.
    """
    from nilearn.datasets import fetch_icbm152_2009

    TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        dataset = fetch_icbm152_2009(data_dir=str(TEMPLATE_DIR))
    except Exception as exc:
        raise RuntimeError(
            "Could not fetch the full-head MNI template. The first run needs "
            "internet access; alternatively supply a local T1w NIfTI file. "
            f"Template cache: {TEMPLATE_DIR}"
        ) from exc
    return Path(dataset.t1)


def resolve_inputs(paths=(), template=False):
    """Return (path, viewer key) pairs, rejecting collisions before writing."""
    scans = []
    if not template:
        for path in paths or ([SAVE_DIR] if SAVE_DIR.exists() else []):
            scans.extend(find_mris(path))
        scans = list(dict.fromkeys(scans))
    if not scans:
        print("Using the full-head MNI152 template (demo, not patient data).")
        return [(mni_template(), "MNI152")]
    inputs = [(scan, dataset_name(scan)) for scan in scans]
    names = {}
    for scan, name in inputs:
        key = name.casefold()
        if key in names:
            raise ValueError(f"Viewer dataset name collision: {names[key]} and {scan} -> {name}")
        names[key] = scan
    return inputs
