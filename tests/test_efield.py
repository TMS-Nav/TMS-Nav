# tests for the aim 2 framework. pure python except the last one, which runs
# the simnibs runner as a subprocess in dry run on the ernie example head.
#
#   python -m pytest tests/          if pytest is installed
#   python tests/test_efield.py      otherwise, the main at the bottom runs them all
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from efield import config  # noqa: E402
from efield.brainsight import make_name, parse_name, poses_from_export, read_export, write_fixture  # noqa: E402
from efield.headmodel import charm_command, find_m2m  # noqa: E402
from efield.job import Job, runner_command  # noqa: E402
from efield.metrics import dice, long_rows, paired_metrics  # noqa: E402
from efield.poses import CoilPose, displacement_mm, is_rotation, orientation_deg, tilt_deg  # noqa: E402
from efield.roi import RoiSpec, roi_under_pose, rois_for_poses  # noqa: E402
from efield.runner_simnibs import roi_metrics, snap_to_nodes, sphere_mask  # noqa: E402
from src.stats import rotation_about  # noqa: E402
from src.targets import Target  # noqa: E402

ERNIE = Path(r"C:\Users\alizr\Documents\.TMSResearch\Sample Datasets\Ernie\m2m_ernie")


def _pose(**kw):
    return CoilPose(centre=[40.0, 80.0, 60.0], ydir=[0.0, -1.0, 0.0], zdir=[-0.5, -0.5, -0.7], **kw)


def test_matsimnibs_round_trip():
    p = _pose(name="a")
    m = p.matsimnibs()
    assert m.shape == (4, 4) and is_rotation(m[:3, :3])
    # x' = y' x z', right handed
    assert np.allclose(np.cross(m[:3, 1], m[:3, 2]), m[:3, 0])
    q = CoilPose.from_matsimnibs(m, name="b")
    assert np.allclose(q.matsimnibs(), m)
    assert displacement_mm(p, q) < 1e-9 and orientation_deg(p, q) < 1e-6
    # json round trip too
    r = CoilPose.from_dict(json.loads(json.dumps(p.to_dict())))
    assert np.allclose(r.matsimnibs(), m, atol=1e-5)


def test_orientation_of_known_rotation():
    p = _pose()
    # tilt the face 7 deg about the handle, then yaw the handle 5 deg about the new normal
    z2 = rotation_about(p.ydir, 7.0) @ p.zdir
    y2 = rotation_about(z2, 5.0) @ p.ydir
    q = CoilPose(centre=p.centre, ydir=y2, zdir=z2)
    assert abs(tilt_deg(p, q) - 7.0) < 1e-6
    total = orientation_deg(p, q)
    # perpendicular axes, the total is more than either and less than the sum
    assert 7.0 < total < 12.0, total
    # matches aim 1's own composition of the same two rotations
    R = rotation_about(z2, 5.0) @ rotation_about(p.ydir, 7.0)
    from src.stats import rotation_angle
    assert abs(total - rotation_angle(R)) < 1e-6


def test_from_aim1_target_points_into_head():
    n_out = np.array([0.52, 0.528, 0.672])
    n_out /= np.linalg.norm(n_out)
    t = Target("R-DLPFC", "F4", "#ff7f0e", (0.55, 1.0, 0.35), np.array([40.4, 84.1, 66.5]), n_out)
    p = CoilPose.from_aim1_target(t, method="MRI", rep=1)
    assert np.allclose(p.zdir, -n_out)
    assert abs(np.dot(p.ydir, p.zdir)) < 1e-9
    # the centre is the coil skin distance above the contact, along the outward normal
    assert abs(np.linalg.norm(p.centre - t.contact) - config.COIL_SKIN_DISTANCE_MM) < 1e-9
    assert np.dot(p.centre - t.contact, n_out) > 0
    assert p.site == "F4"


def test_brainsight_fixture_round_trip():
    poses = []
    for site in ("SMA", "F4"):
        for method in ("MRI", "EEG"):
            for rep in (1, 2):
                c = np.array([10.0 * rep, 80.0 if site == "F4" else 20.0, 90.0])
                z = np.array([-0.3 if site == "F4" else 0.0, -0.4, -0.85])
                poses.append(CoilPose(centre=c, ydir=[0.1, -1.0, 0.0], zdir=z, offset_mm=1.5,
                                      name=make_name("07", method, site, rep), site=site,
                                      method=method, rep=rep))
    with tempfile.TemporaryDirectory() as tmp:
        fn = write_fixture(Path(tmp) / "sub-07_coil_pose_samples.txt", poses)
        ex = read_export(fn)
        assert ex.version == 12 and ex.is_ras and not ex.is_lps
        assert len(ex.samples) == 8 and len(ex.targets) == 0
        back, skipped = poses_from_export(fn)
    assert not skipped and len(back) == 8
    for a, b in zip(poses, back):
        assert a.name == b.name and a.key() == b.key(), (a.name, b.name)
        assert np.allclose(a.matsimnibs(), b.matsimnibs(), atol=2e-4)
        assert abs(b.offset_mm - 1.5) < 1e-9
        assert b.extra["subject"] == "07"
    assert parse_name("sub-07_EEGCAP_F4_POSE_REP2") == ("07", "EEG", "F4", 2)
    assert parse_name("sub-07_MRI_SMA_POSE_REP1") == ("07", "MRI", "SMA", 1)
    assert parse_name("C3_reference") is None


def test_job_json_round_trip():
    poses = [_pose(name="p1", site="F4", method="MRI", rep=1), _pose(name="p2", site="F4", method="EEG", rep=1)]
    rois = rois_for_poses(poses)
    assert len(rois) == 1 and rois[0].reference_method == "MRI" and rois[0].reference_pose == "p1"
    with tempfile.TemporaryDirectory() as tmp:
        job = Job(subject="T9", m2m=str(Path(tmp) / "m2m_T9"), poses=poses, rois=rois, out_dir=tmp)
        path = job.write_json(Path(tmp) / "job.json")
        back = Job.read_json(path)
    assert back.subject == "T9" and len(back.poses) == 2 and len(back.rois) == 1
    assert back.coil_file == config.COIL_FILE and back.didt == config.DIDT
    assert np.allclose(back.poses[0].matsimnibs(), poses[0].matsimnibs(), atol=1e-5)
    assert back.rois[0].to_dict() == rois[0].to_dict()
    cmd = runner_command(path)
    assert cmd[0] == str(config.SIMNIBS_PYTHON) and cmd[-1] == "--dry-run" and cmd[1].endswith("runner_simnibs.py")


def test_roi_seed_is_below_the_coil():
    p = _pose(name="ref", site="F4", method="MRI", rep=1)
    roi = roi_under_pose(p)
    seed = np.array(roi.centre)
    assert abs(np.linalg.norm(seed - p.centre) - roi.depth_mm) < 1e-9
    assert np.dot(seed - p.centre, p.zdir) > 0
    assert RoiSpec.from_dict(roi.to_dict()).name == roi.name


def test_post_processing_on_fake_arrays():
    rng = np.random.default_rng(1)
    nodes = rng.uniform(-50, 50, size=(2000, 3))
    centre = np.array([0.0, 0.0, 0.0])
    k = snap_to_nodes(nodes, centre)
    mask = sphere_mask(nodes, nodes[k], 10.0)
    assert mask[k] and 0 < mask.sum() < 2000
    # a field that falls off from the centre, so the peak and the hotspot are known
    e = 100.0 * np.exp(-np.linalg.norm(nodes, axis=1) / 40.0)
    en = 0.5 * e
    areas = np.ones(len(nodes))
    out = roi_metrics(e, en, areas, mask, k, supra_frac=0.5)
    assert out["n_roi_nodes"] == mask.sum()
    assert out["peak_E_magn"] == e[mask].max() and abs(out["mean_E_magn"] - e[mask].mean()) < 1e-9
    assert out["max_abs_E_normal"] == np.abs(en[mask]).max()
    assert out["E_magn_at_centre"] == e[k]
    assert all(e[i] >= 0.5 * e.max() for i in out["supra_nodes"])
    # and the metrics pair up
    a = dict(out)
    b = dict(out, mean_E_magn=out["mean_E_magn"] * 0.9, peak_E_magn=out["peak_E_magn"] - 5.0)
    pm = paired_metrics(b, a)
    assert abs(pm["pct_diff_mean_roi"] + 10.0) < 1e-9 and abs(pm["diff_peak"] + 5.0) < 1e-9
    assert pm["dice_supra"] == 1.0 and dice([1, 2], [3]) == 0.0
    assert "dice_supra_abs" not in pm, "no absolute cut was asked for, so none should appear"
    rows = long_rows({"subject": "S", "poses": [
        {"name": "m", "site": "F4", "method": "MRI", "rep": 1, "rois": {"F4_x": a}},
        {"name": "e", "site": "F4", "method": "EEG", "rep": 1, "rois": {"F4_x": b}}]})
    assert any(r["metric"] == "pct_diff_mean_roi" and r["method"] == "EEG-MRI" for r in rows)

    # the absolute cut. it does not move with the pose, so a weaker pose really
    # does lose nodes off its hotspot, which is the whole point of having it
    strong = roi_metrics(e, en, areas, mask, k, supra_frac=0.5, supra_abs=30.0)
    weak = roi_metrics(0.9 * e, en, areas, mask, k, supra_frac=0.5, supra_abs=30.0)
    assert all(e[i] >= 30.0 for i in strong["supra_abs_nodes"])
    assert weak["n_supra_abs_nodes"] < strong["n_supra_abs_nodes"], \
        "a weaker field must lose nodes at a fixed cut"
    # at each pose's own peak the two hotspots are identical, a pure scaling
    # cannot move them, so the relative dice is blind to the drop and the
    # absolute one is not
    assert weak["n_supra_nodes"] == strong["n_supra_nodes"]
    pm2 = paired_metrics(weak, strong)
    assert pm2["dice_supra"] == 1.0
    assert pm2["dice_supra_abs"] < 1.0


def test_headmodel_finds_ernie():
    if not ERNIE.is_dir():
        print("  skip: ernie example not on this machine")
        return
    m = find_m2m("ernie")
    assert m is not None and m.complete and m.sub_id == "ernie"
    assert m.mesh.exists() and m.has_eeg_positions and m.has_central_surfaces and not m.missing
    assert find_m2m(str(ERNIE)).path == ERNIE
    assert find_m2m("nobody_here") is None
    cmd = charm_command("T1", "saves/T1.nii.gz")
    assert cmd[0] == str(config.CHARM) and cmd[1] == "T1" and cmd[2].endswith("T1.nii.gz")


def test_runner_dry_run_on_ernie():
    if not ERNIE.is_dir():
        print("  skip: ernie example not on this machine")
        return
    if not config.simnibs_available():
        print(f"  skip: simnibs python not found at {config.SIMNIBS_PYTHON}")
        return
    # F4 off ernie's own electrode file, so the coil is on ernie's head
    f4 = None
    with open(ERNIE / "eeg_positions" / "EEG10-20_Okamoto_2004.csv") as f:
        for ln in f:
            cells = ln.strip().split(",")
            if cells[-1] == "F4":
                f4 = np.array([float(v) for v in cells[1:4]])
    assert f4 is not None
    n_out = f4 / np.linalg.norm(f4)   # radial, good enough since the runner snaps to skin
    mri = CoilPose.from_scalp_point(f4, n_out, [0, -1, 0], name="sub-ernie_MRI_F4_POSE_REP1",
                                    site="F4", method="MRI", rep=1)
    eeg = CoilPose.from_scalp_point(f4 + np.array([3.0, 0, 0]), n_out, [0.1, -1, 0],
                                    name="sub-ernie_EEGCAP_F4_POSE_REP1", site="F4", method="EEG", rep=1)
    rois = rois_for_poses([mri, eeg])
    assert len(rois) == 1

    tmp = Path(tempfile.mkdtemp(prefix="efield_test_"))
    job = Job(subject="ernie", m2m=str(ERNIE), poses=[mri, eeg], rois=rois, out_dir=str(tmp / "sim"),
              snap_to_skin=True, poses_source="test")
    path = job.write_json(tmp / "job.json")
    cmd = runner_command(path, dry_run=True)
    assert "--dry-run" in cmd
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    print(out.stdout)
    if out.returncode != 0:
        print(out.stderr[-3000:])
    assert out.returncode == 0, "runner dry run failed"
    plan = json.loads((tmp / "dryrun.json").read_text())
    assert plan["dry_run"] is True and plan["session_ok"] and not plan["problems"]
    assert len(plan["poses"]) == 2 and len(plan["rois"]) == 1
    assert plan["rois"][0]["n_nodes"] > 0
    for p in plan["poses"]:
        assert p["normal_into_head"] and p["skin_distance_mm"] < 10.0
    # and nothing was simulated
    assert not (tmp / "sim").exists() or not list((tmp / "sim").rglob("*.msh"))
    assert not list(ERNIE.rglob("subject_overlays"))


TESTS = [
    test_matsimnibs_round_trip,
    test_orientation_of_known_rotation,
    test_from_aim1_target_points_into_head,
    test_brainsight_fixture_round_trip,
    test_job_json_round_trip,
    test_roi_seed_is_below_the_coil,
    test_post_processing_on_fake_arrays,
    test_headmodel_finds_ernie,
    test_runner_dry_run_on_ernie,
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
