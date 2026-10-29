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
from efield.roi import RoiSpec, load_target_table, rois_for_subject  # noqa: E402
from efield.runner_simnibs import roi_metrics, snap_to_nodes, sphere_mask  # noqa: E402
from src.stats import rotation_about  # noqa: E402
from src.targets import Target  # noqa: E402
import run_efield  # noqa: E402

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
    rois = [RoiSpec(kind="fmri_target", centre=[36.0, 35.0, 29.4], site="F4")]
    with tempfile.TemporaryDirectory() as tmp:
        job = Job(subject="T9", m2m=str(Path(tmp) / "m2m_T9"), poses=poses, rois=rois, out_dir=tmp)
        path = job.write_json(Path(tmp) / "job.json")
        back = Job.read_json(path)
    assert back.subject == "T9" and len(back.poses) == 2 and len(back.rois) == 1
    assert back.coil_file == config.COIL_FILE and back.didt == config.DIDT
    assert back.map_to_vol == config.MAP_TO_VOL
    with tempfile.TemporaryDirectory() as tmp:
        vol = Job(subject="T9", m2m=tmp, poses=poses, rois=rois, out_dir=tmp, map_to_vol=True)
        assert Job.read_json(vol.write_json(Path(tmp) / "job.json")).map_to_vol is True
    assert np.allclose(back.poses[0].matsimnibs(), poses[0].matsimnibs(), atol=1e-5)
    assert back.rois[0].to_dict() == rois[0].to_dict()
    cmd = runner_command(path)
    assert cmd[0] == str(config.SIMNIBS_PYTHON) and cmd[-1] == "--dry-run" and cmd[1].endswith("runner_simnibs.py")


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
    n_out = f4 / np.linalg.norm(f4)   # radial, close enough to the skin normal at f4
    mri = CoilPose.from_scalp_point(f4, n_out, [0, -1, 0], name="sub-ernie_MRI_F4_POSE_REP1",
                                    site="F4", method="MRI", rep=1)
    eeg = CoilPose.from_scalp_point(f4 + np.array([3.0, 0, 0]), n_out, [0.1, -1, 0],
                                    name="sub-ernie_EEGCAP_F4_POSE_REP1", site="F4", method="EEG", rep=1)
    # a target 15 mm in from the electrode, about where the cortex is under f4
    rois = [RoiSpec(kind="fmri_target", centre=(f4 - 15.0 * n_out).tolist(), site="F4")]

    tmp = Path(tempfile.mkdtemp(prefix="efield_test_"))
    job = Job(subject="ernie", m2m=str(ERNIE), poses=[mri, eeg], rois=rois, out_dir=str(tmp / "sim"),
              poses_source="test")
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


def test_fmri_target_table():
    """The r01 target csv parses, and every bad row is caught."""

    good = """subject,site,x,y,z,space,note
# a comment, and a blank line follow

S01,SMA,-4.2,-2.1,62.0,subject,left of midline
S01,F4,38.5,34.0,32.4,mni,
"""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "t.csv"
        f.write_text(good, encoding="utf8")
        table = load_target_table(f)
        assert set(table) == {("S01", "SMA"), ("S01", "F4")}
        assert table[("S01", "SMA")]["coords"] == [-4.2, -2.1, 62.0]
        assert table[("S01", "SMA")]["space"] == "subject"
        assert table[("S01", "F4")]["space"] == "mni"
        assert table[("S01", "SMA")]["note"] == "left of midline"

        # a missing file is normal until the targets are handed over
        assert load_target_table(Path(d) / "nope.csv") == {}

        bad = [
            ("subject,site,x,y,z,space\nS01,SMA,1,2,3,scanner\n", "space"),
            ("subject,site,x,y,z,space\nS01,NOSE,1,2,3,subject\n", "site"),
            ("subject,site,x,y,z,space\nS01,SMA,1,2,three,subject\n", "numbers"),
            ("subject,site,x,y,z,space\nS01,SMA,1,2,3,subject\nS01,SMA,4,5,6,subject\n", "twice"),
            ("subject,site,x,y,z,space\nS01,SMA,1,,3,subject\n", "missing"),
        ]
        for text, why in bad:
            f.write_text(text, encoding="utf8")
            try:
                load_target_table(f)
            except ValueError:
                continue
            raise AssertionError(f"a row with a bad {why} was accepted")


def test_roi_is_the_r01_target():
    """With a target the roi sits on it. Without one the site gets no roi at all."""

    t = Target("R-DLPFC", "F4", "#ff7f0e", None,
               np.array([40.0, 84.0, 66.0]), np.array([0.52, 0.53, 0.67]))
    poses = [CoilPose.from_aim1_target(t, site="F4", method=m, rep=1) for m in ("MRI", "EEG")]

    # supplied: the roi is the target itself, not anything derived from the coil
    table = {("S01", "F4"): {"coords": [36.0, 35.0, 29.4], "space": "subject", "note": ""}}
    rois, missing = rois_for_subject("S01", poses, table=table)
    assert missing == [] and len(rois) == 1
    roi = rois[0]
    assert roi.kind == "fmri_target"
    assert roi.centre == [36.0, 35.0, 29.4]
    assert roi.reference_method == "fMRI"
    assert RoiSpec.from_dict(roi.to_dict()).to_dict() == roi.to_dict()

    # an mni target keeps its coordinate for the runner to map
    table = {("S01", "F4"): {"coords": [38.0, 34.0, 32.0], "space": "mni", "note": ""}}
    roi = rois_for_subject("S01", poses, table=table)[0][0]
    assert roi.kind == "mni_sphere" and roi.mni == [38.0, 34.0, 32.0]

    # nothing supplied: no roi, and the site is reported missing
    rois, missing = rois_for_subject("S01", poses, table={})
    assert rois == [] and missing == ["F4"]

    # the old guess under the coil is not a kind any more
    try:
        RoiSpec(kind="sphere_under_coil", centre=[0.0, 0.0, 0.0], site="F4")
    except ValueError:
        pass
    else:
        raise AssertionError("sphere_under_coil was accepted")

    # a spec that claims a kind it has no coordinate for is refused
    for kind, kw in (("fmri_target", {}), ("mni_sphere", {})):
        try:
            RoiSpec(kind=kind, site="F4", **kw)
        except ValueError:
            continue
        raise AssertionError(f"{kind} with no coordinate was accepted")


def test_poses_come_from_the_exports():
    """Every subject in the exports, sma and f4 only, an LPS file left out."""

    def poses_for(sub, sites):
        out = []
        for site in sites:
            for method in ("MRI", "EEG"):
                c = np.array([40.0, 80.0 if site == "F4" else 20.0, 90.0])
                out.append(CoilPose(centre=c, ydir=[0.0, -1.0, 0.0], zdir=[-0.3, -0.4, -0.85],
                                    name=make_name(sub, method, site, 1), site=site,
                                    method=method, rep=1))
        return out

    with tempfile.TemporaryDirectory() as tmp:
        write_fixture(Path(tmp) / "sub-001_coil_pose_samples.txt", poses_for("001", ("SMA", "F4", "C3")))
        write_fixture(Path(tmp) / "sub-002_coil_pose_samples.txt", poses_for("002", ("F4",)))
        write_fixture(Path(tmp) / "sub-003_coil_pose_samples.txt", poses_for("003", ("F4",)),
                      coord_sys="Brainsight")
        by_subject, sources = run_efield.load_poses(tmp)

    assert sorted(by_subject) == ["sub-001", "sub-002"], "the LPS export must not be used"
    assert sorted(p.key() for p in by_subject["sub-001"]) == [
        ("F4", "EEG", 1), ("F4", "MRI", 1), ("SMA", "EEG", 1), ("SMA", "MRI", 1)]
    assert len(by_subject["sub-002"]) == 2
    assert sources["sub-001"] == ["sub-001_coil_pose_samples.txt"]
    assert run_efield.subject_id("001") == run_efield.subject_id("sub-001") == "sub-001"

    # with a target for sma only, f4 is reported and gets no roi
    table = {("sub-001", "SMA"): {"coords": [0.0, -2.0, 60.0], "space": "subject", "note": ""}}
    rois, missing = rois_for_subject("sub-001", by_subject["sub-001"], table=table)
    assert [r.site for r in rois] == ["SMA"] and missing == ["F4"]


def test_one_mri_rep_pairs_every_eeg_rep():
    """The aim 1 pairing rule: a single mri placement is compared with each eeg rep."""

    def block(mean):
        return {"mean_E_magn": mean, "peak_E_magn": mean * 1.5, "max_abs_E_normal": mean * 0.7,
                "E_magn_at_centre": mean, "supra_nodes": [1, 2, 3]}

    results = {"subject": "sub-001", "poses": [
        {"name": "m1", "site": "F4", "method": "MRI", "rep": 1, "rois": {"F4_t": block(100.0)}},
        {"name": "e1", "site": "F4", "method": "EEG", "rep": 1, "rois": {"F4_t": block(90.0)}},
        {"name": "e2", "site": "F4", "method": "EEG", "rep": 2, "rois": {"F4_t": block(80.0)}},
        # sma has two of each, so rep 2 goes with rep 2
        {"name": "m1", "site": "SMA", "method": "MRI", "rep": 1, "rois": {"SMA_t": block(100.0)}},
        {"name": "m2", "site": "SMA", "method": "MRI", "rep": 2, "rois": {"SMA_t": block(50.0)}},
        {"name": "e1", "site": "SMA", "method": "EEG", "rep": 1, "rois": {"SMA_t": block(100.0)}},
        {"name": "e2", "site": "SMA", "method": "EEG", "rep": 2, "rois": {"SMA_t": block(100.0)}},
    ]}
    pct = {(r["site"], r["rep"]): r["value"] for r in long_rows(results) if r["metric"] == "pct_diff_mean_roi"}
    assert set(pct) == {("F4", 1), ("F4", 2), ("SMA", 1), ("SMA", 2)}
    assert abs(pct[("F4", 1)] + 10.0) < 1e-9 and abs(pct[("F4", 2)] + 20.0) < 1e-9
    assert abs(pct[("SMA", 1)]) < 1e-9 and abs(pct[("SMA", 2)] - 100.0) < 1e-9


def test_nearest_values_keep_the_field():
    from efield.export_web import nearest_values

    rng = np.random.default_rng(3)
    src = rng.normal(size=(500, 3)) * 40.0
    e = rng.random(500)
    # a subset of the original nodes, nudged less than half the closest spacing
    keep = np.arange(0, 500, 7)
    dst = src[keep] + 1e-4
    out = nearest_values(src, e, dst)
    assert np.array_equal(out, e[keep])
    assert out.max() == e[keep].max()


def _surface_npz(path, scale=1.0):
    # an octahedron is enough of a closed surface for the decimation to pass through
    nodes = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float) * 50.0
    tri = np.array([[0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4], [2, 0, 5], [1, 2, 5], [3, 1, 5], [0, 3, 5]])
    e = np.arange(6, dtype=float) * scale
    np.savez(path, nodes=nodes, triangles=tri, E_magn=e, E_normal=-e)
    return e


def test_export_cortex_writes_one_field_per_pose():
    from efield.export_web import export_cortex

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        e1 = _surface_npz(tmp / "a.npz")
        e2 = _surface_npz(tmp / "b.npz", scale=2.0)
        out = export_cortex([tmp / "a.npz", tmp / "b.npz"], ["a", "b"], tmp / "web", faces=8)
        assert (tmp / "web" / "cortex.ply").exists()
        n = out["n_vertices"]
        f1 = np.fromfile(tmp / "web" / "efield_a.f32", dtype="<f4")
        f2 = np.fromfile(tmp / "web" / "efield_b.f32", dtype="<f4")
    assert n == 6 and len(f1) == n and len(f2) == n
    assert sorted(f1.tolist()) == sorted(e1.tolist()) and np.allclose(f2, 2.0 * f1)
    assert out["poses"]["b"]["peak"] == e2.max()


def test_export_volumes_grid_and_scale():
    import nibabel as nib
    from efield.export_web import export_volumes

    shape = (40, 44, 36)
    aff = np.array([[1.0, 0, 0, -20], [0, 1.0, 0, -22], [0, 0, 1.0, -18], [0, 0, 0, 1]])
    lab = np.zeros(shape, np.uint8)
    lab[5:35, 6:38, 4:32] = 5                    # scalp
    lab[10:30, 12:32, 9:27] = 2                  # grey matter inside it
    t1 = (lab * 20).astype(np.float32)
    field = np.where(lab == 2, 0.8, 1.5).astype(np.float32)     # higher in the scalp, masked out
    field[lab == 0] = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name, data in (("t1", t1), ("lab", lab), ("e", field)):
            nib.save(nib.Nifti1Image(data, aff), str(tmp / f"{name}.nii.gz"))
        vol = export_volumes(tmp / "t1.nii.gz", tmp / "lab.nii.gz", [tmp / "e.nii.gz"], ["p"],
                             tmp / "web", color_max=1.0, step_mm=2.0)
        nx, ny, nz = vol["dims"]
        raw_t1 = (tmp / "web" / "t1.u8").read_bytes()
        raw_e = np.frombuffer((tmp / "web" / "field_p.u8").read_bytes(), np.uint8)
    assert len(raw_t1) == nx * ny * nz and len(raw_e) == nx * ny * nz
    A = np.asarray(vol["affine"])
    assert np.allclose(np.sqrt((A[:3, :3] ** 2).sum(axis=0)), 2.0)
    # x fastest: the voxel at grid (i, j, k) is byte i + nx * (j + ny * k)
    e = raw_e.reshape(nz, ny, nx).transpose(2, 1, 0)
    # the grey matter centre, mapped back through the affine, holds 0.8 V/m within one step
    centre = np.linalg.inv(A) @ np.array([0.0, 0.0, 0.0, 1.0])
    i, j, k = np.rint(centre[:3]).astype(int)
    assert abs(e[i, j, k] * vol["field_step"] - 0.8) <= vol["field_step"]
    # nothing outside the brain, though the scalp had more field
    assert e.max() * vol["field_step"] <= 0.8 + vol["field_step"]


def test_registry_has_efield_only_with_a_manifest():
    from src.export_web import write_registry

    with tempfile.TemporaryDirectory() as tmp:
        data = Path(tmp) / "data"
        for s in ("A", "B"):
            (data / s).mkdir(parents=True)
            (data / s / "markers.json").write_text("{}")
        (data / "A" / "efield").mkdir()
        manifest = {"cortex": {"file": "cortex.ply"}, "volume": {"t1": "t1.u8"},
                    "poses": [{"surface": "efield_p.f32", "volume": "field_p.u8"}]}
        (data / "A" / "efield" / "manifest.json").write_text(json.dumps(manifest))
        path, subjects = write_registry(data, Path(tmp) / "reg.js")
        text = path.read_text()
    assert subjects == ["A", "B"]
    assert 'import S0_EFIELD from "./data/A/efield/manifest.json";' in text
    for fn in ("cortex.ply", "t1.u8", "efield_p.f32", "field_p.u8"):
        assert f'new URL("./data/A/efield/{fn}", import.meta.url)' in text
    assert "S1_EFIELD" not in text and "efield: null" in text


TESTS = [
    test_matsimnibs_round_trip,
    test_orientation_of_known_rotation,
    test_from_aim1_target_points_into_head,
    test_brainsight_fixture_round_trip,
    test_job_json_round_trip,
    test_fmri_target_table,
    test_roi_is_the_r01_target,
    test_poses_come_from_the_exports,
    test_one_mri_rep_pairs_every_eeg_rep,
    test_post_processing_on_fake_arrays,
    test_nearest_values_keep_the_field,
    test_export_cortex_writes_one_field_per_pose,
    test_export_volumes_grid_and_scale,
    test_registry_has_efield_only_with_a_manifest,
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
