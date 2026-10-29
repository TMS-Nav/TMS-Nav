#!/usr/bin/env python3
"""Download the SimNIBS sample heads and put them where the project looks.

    python scripts/fetch_sample_data.py                 ernie, about 1.1 GB
    python scripts/fetch_sample_data.py --group         plus T1 to T5, about 2.7 GB more
    python scripts/fetch_sample_data.py --from-zip simnibs4_examples.zip
                                                        an archive already downloaded by hand

The archives come from the SimNIBS datasets page
(https://simnibs.github.io/simnibs/build/html/dataset.html) and unpack into
saves/sample_data/, where efield/config.py finds the m2m head models. Each
head's T1 is also copied to saves/<name>.nii.gz (Ernie, T1 ... T5), the scans
run_pipeline.py turns into the viewer's sample heads. A step that is already
done is skipped, so the script can be rerun after a broken download.

Ernie is CC BY-NC 4.0, the group dataset ODC-PDDL. Neither is study data.
"""
import argparse
import shutil
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAVES = ROOT / "saves"
SAMPLE_DATA = SAVES / "sample_data"

# name, url, approximate size, folder under sample_data, {viewer name: m2m folder}
DATASETS = {
    "examples": ("simnibs4_examples.zip",
                 "https://github.com/simnibs/example-dataset/releases/latest/download/simnibs4_examples.zip",
                 "1.1 GB", {"Ernie": "m2m_ernie"}),
    "group": ("simnibs_group_dataset.zip", "https://osf.io/5ysve/download", "2.7 GB",
              # the order public-subjects.json gives them, sub01 is T1 and so on
              {"T1": "m2m_sub01", "T2": "m2m_sub09", "T3": "m2m_sub10", "T4": "m2m_sub12", "T5": "m2m_sub15"}),
}


def download(url, dest, size):
    """url to dest, through a .part file so an interrupted download is not mistaken for a finished one."""

    if dest.exists():
        print(f"  have {dest.name}, not downloading again")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    print(f"  downloading {dest.name} ({size}) from {url}")
    with urllib.request.urlopen(url) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, shown = 0, -1
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            pct = int(100 * done / total) if total else -1
            if pct != shown and pct % 5 == 0:
                print(f"    {pct:3d} %  {done / 1e9:.2f} GB", flush=True)
                shown = pct
    part.rename(dest)
    return dest


def unpack(archive, out):
    """Unpack a zip or tar archive into out. both modules refuse paths that leave out."""

    out.mkdir(parents=True, exist_ok=True)
    print(f"  unpacking {archive.name} into {out}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as z:
            z.extractall(out)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as t:
            t.extractall(out, filter="data")
    else:
        raise SystemExit(f"{archive} is neither a zip nor a tar archive, download it again")


def find_m2m(base, name):
    """The m2m folder called name anywhere under base, or None."""

    hits = [p for p in base.rglob(name) if p.is_dir()]
    return min(hits, key=lambda p: len(p.parts)) if hits else None


def fetch(key, from_zip=None):
    fname, url, size, heads = DATASETS[key]
    out = SAMPLE_DATA / key
    print(f"{key}:")
    if out.is_dir() and all(find_m2m(out, m) for m in heads.values()):
        print(f"  already unpacked in {out}")
    else:
        archive = Path(from_zip) if from_zip else download(url, SAMPLE_DATA / "downloads" / fname, size)
        unpack(archive, out)
    for viewer_name, m2m_name in heads.items():
        m2m = find_m2m(out, m2m_name)
        if m2m is None:
            print(f"  {m2m_name} is not in the archive, {viewer_name} skipped")
            continue
        t1 = SAVES / f"{viewer_name}.nii.gz"
        if t1.exists():
            print(f"  have saves/{t1.name}")
        else:
            shutil.copy2(m2m / "T1.nii.gz", t1)
            print(f"  saves/{t1.name} <- {m2m / 'T1.nii.gz'}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", action="store_true", help="also fetch the group dataset (T1 to T5)")
    ap.add_argument("--from-zip", help="use this already downloaded examples archive instead of downloading")
    ap.add_argument("--group-from-zip", help="use this already downloaded group archive instead of downloading")
    args = ap.parse_args(argv)
    fetch("examples", args.from_zip)
    if args.group or args.group_from_zip:
        fetch("group", args.group_from_zip)
    print("done. next: python run_pipeline.py, then python -m efield.demo_ernie (docs/efield.md)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
