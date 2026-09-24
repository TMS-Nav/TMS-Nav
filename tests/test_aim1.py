# the aim 1 real data path. brainsight exports are written with known offsets
# (efield.brainsight.write_fixture, the same layout simnibs reads), dropped in a
# folder, and the numbers that come out are checked against what went in.
#
#   python -m pytest tests/          if pytest is installed
#   python tests/test_aim1.py        otherwise, the main at the bottom runs them all
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import aim1_stats  # noqa: E402
from efield.brainsight import make_name, write_fixture  # noqa: E402
from efield.poses import CoilPose  # noqa: E402
from src.aim1 import measurements, per_subject, pair_rows  # noqa: E402
from src.stats import rotation_about  # noqa: E402

UP = np.array([0.0, 0.0, 1.0])
BACK = np.array([0.0, -1.0, 0.0])


def _pose(subject, method, site, rep, contact, normal=UP, handle=BACK):
    return CoilPose.from_scalp_point(np.asarray(contact, float), normal, handle,
                                     name=make_name(subject, method, site, rep),
                                     site=site, method=method, rep=rep)


def _flip_to_lps(p):
    f = np.diag([-1.0, -1.0, 1.0])
    return CoilPose(centre=f @ p.centre, ydir=f @ p.ydir, zdir=f @ p.zdir, name=p.name,
                    site=p.site, method=p.method, rep=p.rep)


def _known_poses(subject="K01"):
    """At the vertex the eeg coil sits 3 mm forward, 4 mm right, handle turned +5 deg.
    at F4 it sits on the same spot with the face tilted 7 deg. rep 2 of the eeg sits
    a further 1 mm forward of rep 1."""

    poses = []
    for rep, extra in ((1, 0.0), (2, 1.0)):
        poses.append(_pose(subject, "MRI", "SMA", rep, (0, 0, 100)))
        poses.append(_pose(subject, "EEG", "SMA", rep, (4, 3 + extra, 100),
                           handle=rotation_about(UP, 5.0) @ BACK))
    n = np.array([0.55, 1.0, 0.35])
    n /= np.linalg.norm(n)
    tilt_axis = np.cross(n, UP)
    tilted = rotation_about(tilt_axis, 7.0) @ n
    c = np.array([45.0, 60.0, 55.0])
    poses.append(_pose(subject, "MRI", "F4", 1, c, normal=n))
    poses.append(_pose(subject, "EEG", "F4", 1, c, normal=tilted))
    return poses


def test_known_offsets_come_back():
    with tempfile.TemporaryDirectory() as d:
        write_fixture(Path(d) / "sub-K01_coil_pose_samples.txt", _known_poses())
        pairs, repeats, notes = measurements(d)

    sma = pairs[pairs["site"] == "SMA"].set_index("rep")
    r1 = sma.loc[1]
    assert abs(r1["displacement_mm"] - 5.0) < 1e-3
    assert abs(r1["ap_mm"] - 3.0) < 1e-3 and abs(r1["lr_mm"] - 4.0) < 1e-3
    assert abs(r1["depth_mm"]) < 1e-3
    assert abs(r1["yaw_deg"] - 5.0) < 1e-2
    assert abs(r1["tilt_deg"]) < 1e-2 and abs(r1["orientation_deg"] - 5.0) < 1e-2
    assert abs(sma.loc[2]["ap_mm"] - 4.0) < 1e-3

    f4 = pairs[pairs["site"] == "F4"].iloc[0]
    assert abs(f4["tilt_deg"] - 7.0) < 1e-2
    assert f4["orientation_deg"] >= f4["tilt_deg"] - 1e-2

    rep = repeats[(repeats["site"] == "SMA") & (repeats["method"] == "EEG")].iloc[0]
    assert abs(rep["repeat_mm"] - 1.0) < 1e-3 and abs(rep["repeat_deg"]) < 1e-2
    assert any("NIfTI:Aligned" in n for n in notes)


def test_lps_export_gives_the_same_numbers():
    with tempfile.TemporaryDirectory() as d:
        write_fixture(Path(d) / "a.txt", _known_poses())
        ras, _, _ = measurements(d)
    with tempfile.TemporaryDirectory() as d:
        write_fixture(Path(d) / "a.txt", [_flip_to_lps(p) for p in _known_poses()],
                      coord_sys="Brainsight")
        lps, _, notes = measurements(d)
    cols = ["displacement_mm", "tilt_deg", "orientation_deg", "ap_mm", "lr_mm", "depth_mm", "yaw_deg"]
    assert np.allclose(ras[cols].to_numpy(), lps[cols].to_numpy(), atol=2e-3)
    assert any("flipped from brainsight LPS" in n for n in notes)


def test_pairing_rules():
    poses = [
        # one mri placement, two eeg reps, both compared with it
        _pose("P1", "MRI", "SMA", 1, (0, 0, 100)),
        _pose("P1", "EEG", "SMA", 1, (1, 0, 100)),
        _pose("P1", "EEG", "SMA", 2, (2, 0, 100)),
        # eeg only, nothing to compare, and it is reported
        _pose("P1", "EEG", "C3", 1, (-60, 0, 70)),
    ]
    pairs = pair_rows(poses)
    assert list(pairs["rep"]) == [1, 2] and list(pairs["mri_rep"]) == [1, 1]
    assert np.allclose(pairs["lr_mm"], [1.0, 2.0])

    with tempfile.TemporaryDirectory() as d:
        write_fixture(Path(d) / "a.txt", poses)
        # a stray sample that does not follow the naming is skipped and said so
        with open(Path(d) / "a.txt", "a", encoding="utf-8") as f:
            f.write("my test sample\tSession 1\t9\t(null)\t0\t0\t0\t1\t0\t0\t0\t1\t0\t0\t0\t1\t0\t0\t2026-01-01\t12:00:00\n")
        _, _, notes = measurements(d)
    assert any("only EEG poses" in n and "C3" in n for n in notes)
    assert any("skipped 1" in n for n in notes)

    # subjects are the unit, reps are averaged first
    subj = per_subject(pairs, ["lr_mm"])
    assert len(subj) == 1 and abs(subj["lr_mm"].iloc[0] - 1.5) < 1e-9


def _study(folder, n_subjects=10, seed=2):
    rng = np.random.default_rng(seed)
    for s in range(n_subjects):
        sub = f"S{s + 1:02d}"
        poses = []
        for site, c, shift in (("SMA", (0, 10, 95), 0.0), ("F4", (45, 60, 55), 3.0)):
            gap = rng.normal(0, 3, 3) + np.array([0, shift, 0])
            for rep in (1, 2):
                poses.append(_pose(sub, "MRI", site, rep, np.array(c) + rng.normal(0, .5, 3)))
                tilt = rotation_about([np.cos(a := rng.uniform(0, 2 * np.pi)), np.sin(a), 0.0],
                                      rng.normal(0, 3))
                poses.append(_pose(sub, "EEG", site, rep, np.array(c) + gap + rng.normal(0, 1, 3),
                                   normal=tilt @ UP,
                                   handle=rotation_about(UP, rng.normal(0, 4)) @ BACK))
        write_fixture(Path(folder) / f"sub-{sub}_coil_pose_samples.txt", poses)


def test_drop_the_files_in_and_run():
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as out:
        _study(d)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = aim1_stats.main([d], out_dir=out, viewer_dir=Path(out) / "no_viewer")
        text = buf.getvalue()
        assert code == 0, text
        for name in ("measurements.csv", "repeats.csv", "aim1_stats.csv"):
            assert (Path(out) / name).exists(), name
        assert "=== F4   n = 10 subjects" in text and "=== SMA   n = 10 subjects" in text

        # the table agrees with a by hand average of the measurements, subjects first
        meas = pd.read_csv(Path(out) / "measurements.csv")
        long = pd.read_csv(Path(out) / "aim1_stats.csv")
        by_hand = meas.groupby(["subject", "site"])["displacement_mm"].mean().groupby("site").mean()
        for site, v in by_hand.items():
            got = long[(long["site"] == site) & (long["quantity"] == "displacement_mm")
                       & (long["statistic"] == "describe.mean")]["value"].iloc[0]
            assert abs(got - v) < 1e-9
        assert (long["statistic"] == "bias.q").sum() == 8        # 2 sites x 4 axes, fdr'd together
        assert (long["statistic"] == "mixed.p").sum() == 3       # one per distance


def test_viewer_gets_the_numbers_and_the_registry_points_at_them():
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as out:
        _study(d, n_subjects=4)
        viewer = Path(out) / "data"
        (viewer / "T1").mkdir(parents=True)
        (viewer / "T1" / "markers.json").write_text(json.dumps({"qc": {}}))
        reg = Path(out) / "datasetRegistry.js"
        with contextlib.redirect_stdout(io.StringIO()):
            aim1_stats.main([d], out_dir=out, viewer_dir=viewer, registry_path=reg)
        payload = json.loads((viewer / "aim1_stats.json").read_text())
        assert payload["source"] == "brainsight" and payload["n_subjects"] == 4
        # aggregates only, no subject ids go to the website
        assert "S01" not in json.dumps(payload)
        text = reg.read_text()
        assert 'import AIM1_DATA from "./data/aim1_stats.json";' in text
        assert "export const AIM1 = AIM1_DATA;" in text


def test_empty_folder_exits_cleanly():
    with tempfile.TemporaryDirectory() as d, tempfile.TemporaryDirectory() as out:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = aim1_stats.main([d], out_dir=Path(out) / "aim1", viewer_dir=Path(out) / "v")
        assert code == 0
        assert "no brainsight exports" in buf.getvalue()
        assert not (Path(out) / "aim1").exists()


TESTS = [
    test_known_offsets_come_back,
    test_lps_export_gives_the_same_numbers,
    test_pairing_rules,
    test_drop_the_files_in_and_run,
    test_viewer_gets_the_numbers_and_the_registry_points_at_them,
    test_empty_folder_exits_cleanly,
]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        print(f"{t.__name__} ...")
        try:
            t()
            print("  ok")
        except Exception as e:      # keep going so every failure is seen at once
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")
    print(f"\n{len(TESTS) - failed} of {len(TESTS)} passed")
    sys.exit(1 if failed else 0)
