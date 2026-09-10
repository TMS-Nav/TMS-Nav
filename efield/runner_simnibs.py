# the simnibs side. THE ONLY FILE IN THE PROJECT THAT IMPORTS SIMNIBS.
#
# usage, from the simnibs python, never the project one:
#
#     simnibs_python efield/runner_simnibs.py saves/efield/<subject>/job.json --dry-run
#
# reads a job.json written by efield/job.py, builds the simnibs SESSION for it
# and either prints the plan (--dry-run, writes dryrun.json, exits 0 without
# simulating) or runs run_simnibs and writes results.json with the roi numbers
# for every pose. the driver always passes --dry-run unless someone typed --run.
#
# the post processing functions take plain numpy arrays so they can be imported
# and tested from the project python, which has no simnibs. that is why the
# simnibs import is guarded, the module loads without it, only main() needs it.
import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import simnibs
    from simnibs import RegionOfInterest, mesh_io, run_simnibs, sim_struct
    from simnibs.utils.file_finder import SubjectFiles
    HAVE_SIMNIBS = True
except ImportError:            # project python, the pure functions below still work
    simnibs = None
    HAVE_SIMNIBS = False

# a coil centre further than this from the skin is almost certainly in the wrong
# space (wrong scan, lps instead of ras) rather than a badly placed coil
MAX_SKIN_DISTANCE_MM = 30.0

# node field names simnibs writes on the central surface overlay for field 'e'
FIELD_MAGN = "E_magn"
FIELD_NORMAL = "E_normal"


# --- pure numpy post processing, importable anywhere ---------------------------------
def sphere_mask(node_coords, centre, radius_mm):
    """Boolean node mask of the nodes within radius_mm of centre."""

    d = np.linalg.norm(np.asarray(node_coords, dtype=float) - np.asarray(centre, dtype=float), axis=1)
    return d <= float(radius_mm)


def snap_to_nodes(node_coords, point):
    """Index of the node nearest to point. the roi seed is snapped with this."""

    d = np.linalg.norm(np.asarray(node_coords, dtype=float) - np.asarray(point, dtype=float), axis=1)
    return int(np.argmin(d))


def roi_metrics(e_magn, e_normal, node_areas, roi_mask, centre_node, supra_frac,
                supra_abs=None):
    """The per pose numbers the endpoints are built from.

    e_magn and e_normal are per node on the whole central surface, node_areas
    the area each node owns, roi_mask a boolean over nodes, centre_node the
    index the roi was drawn around. the mean is area weighted so a densely
    meshed patch does not count for more than a coarse one. the suprathreshold
    mask is taken over the WHOLE surface, it is the hotspot, and its overlap
    between methods is a secondary endpoint.

    two hotspots come back, at two different cuts, because the choice matters.
    supra_frac is a fraction of THIS pose's own peak, so a weaker placement gets
    a lower bar and the overlap then mixes up "the hotspot moved" with "the
    hotspot got weaker". supra_abs is a fixed V/m cut, the same number for both
    poses of a pair, so its overlap answers the question actually being asked.
    metrics.py reports dice on both and prefers the absolute one
    """

    e_magn = np.asarray(e_magn, dtype=float)
    e_normal = np.asarray(e_normal, dtype=float)
    node_areas = np.asarray(node_areas, dtype=float)
    roi_mask = np.asarray(roi_mask, dtype=bool)
    if roi_mask.sum() == 0:
        raise ValueError("empty roi")

    peak_surface = float(np.max(e_magn))
    thresh = supra_frac * peak_surface
    supra = np.flatnonzero(e_magn >= thresh)

    # the same hotspot at a cut that does not move with the pose
    abs_block = {}
    if supra_abs is not None:
        supra_a = np.flatnonzero(e_magn >= float(supra_abs))
        abs_block = {
            "supra_abs_vm": float(supra_abs),
            "n_supra_abs_nodes": int(len(supra_a)),
            "supra_abs_area_mm2": float(node_areas[supra_a].sum()),
            "supra_abs_nodes": [int(i) for i in supra_a],
        }

    return {
        "n_roi_nodes": int(roi_mask.sum()),
        "roi_area_mm2": float(node_areas[roi_mask].sum()),
        "peak_E_magn": float(np.max(e_magn[roi_mask])),
        "mean_E_magn": float(np.average(e_magn[roi_mask], weights=node_areas[roi_mask])),
        "max_abs_E_normal": float(np.max(np.abs(e_normal[roi_mask]))),
        "E_magn_at_centre": float(e_magn[centre_node]),
        "surface_peak_E_magn": peak_surface,
        "supra_threshold": float(thresh),
        "supra_frac": float(supra_frac),
        "n_supra_nodes": int(len(supra)),
        "supra_area_mm2": float(node_areas[supra].sum()),
        "supra_nodes": [int(i) for i in supra],
        **abs_block,
    }


def check_rotation(R, tol=1e-4):
    R = np.asarray(R, dtype=float)
    return bool(np.allclose(R.T @ R, np.eye(3), atol=tol) and abs(np.linalg.det(R) - 1.0) < tol)


# --- helpers that need simnibs ------------------------------------------------------
def resolve_coil(coil_file):
    """Absolute path of the coil file, relative names looked up in simnibs's resources."""

    if os.path.isabs(coil_file) and os.path.exists(coil_file):
        return coil_file
    root = Path(simnibs.__file__).resolve().parent / "resources" / "coil_models"
    cand = root / coil_file
    return str(cand) if cand.exists() else None


def load_skin(m2m):
    """The scalp surface of the head mesh, for projecting coil centres onto."""

    files = SubjectFiles(subpath=str(m2m))
    mesh = mesh_io.read_msh(files.fnamehead)
    return mesh


def skin_distance(mesh, centre):
    """Signed-ish distance from a coil centre to the nearest skin point, mm."""

    proj = mesh.project_points_on_surface(np.asarray(centre, dtype=float).reshape(1, 3),
                                          surface_tags=int(simnibs.ElementTags.SCALP_TH_SURFACE))
    return float(np.linalg.norm(proj[0] - centre)), proj[0]


def realize_roi(roi, m2m):
    """The roi on the gm central surface: snapped centre, node mask, node coords.

    the spec's centre is a guess (see efield/roi.py). it is snapped to the
    nearest node of the joined lh+rh central surface, and the sphere is drawn
    around that node, so the roi is guaranteed to sit on cortex. the node order
    (lh then rh) is the same order simnibs writes the overlay in, so the mask
    indexes the overlay's node fields directly
    """

    R = RegionOfInterest()
    R.load_surfaces("central", subpath=str(m2m))
    nodes = R.get_nodes()          # mask is all ones straight after loading
    if roi["kind"] == "mni_sphere":
        seed = np.asarray(simnibs.mni2subject_coords(roi["mni"], str(m2m)), dtype=float)
    else:
        seed = np.asarray(roi["centre"], dtype=float)
    k = snap_to_nodes(nodes, seed)
    centre = nodes[k]
    # node_type must be said out loud. it defaults to "elm_center" even on a
    # surface roi, and then the kd tree is over triangles while the mask is over
    # nodes, which raises an index error (simnibs 4.6.0)
    R.apply_sphere_mask(node_type="node", roi_sphere_center=[float(x) for x in centre],
                        roi_sphere_radius=float(roi["radius_mm"]),
                        roi_sphere_center_space="subject")
    inside = R.get_nodes()
    mask = sphere_mask(nodes, centre, roi["radius_mm"])
    if mask.sum() != len(inside):
        print(f"    note: own sphere mask has {mask.sum()} nodes, simnibs's has {len(inside)}")
    return {"centre_node": k, "centre": centre, "seed": seed, "snap_mm": float(np.linalg.norm(centre - seed)),
            "mask": mask, "n_nodes": int(mask.sum()), "nodes": nodes}


def build_session(job, coil_path):
    S = sim_struct.SESSION()
    S.subpath = str(job["m2m"])
    S.pathfem = str(job["out_dir"])
    S.fields = job.get("fields", "eE")
    S.map_to_surf = bool(job.get("map_to_surf", True))
    S.open_in_gmsh = False
    tms = S.add_tmslist()
    tms.fnamecoil = coil_path
    for p in job["poses"]:
        pos = tms.add_position()
        pos.matsimnibs = np.asarray(p["matsimnibs"], dtype=float)
        pos.didt = float(job.get("didt", 1e6))
        pos.name = p["name"]
    return S, tms


def find_overlay(out_dir, index):
    """The central surface overlay for pose index (1 based) of the first tmslist.

    simnibs names it <sub>_TMS_1-<index:04d>_<coil>_scalar_central.msh under
    subject_overlays. glob on the index so the coil name does not have to be
    reproduced here
    """

    pat = os.path.join(str(out_dir), "subject_overlays", f"*_TMS_1-{index:04d}_*_central.msh")
    hits = sorted(glob.glob(pat))
    return hits[0] if hits else None


def read_overlay(path):
    m = mesh_io.read_msh(path)
    return {
        "nodes": m.nodes.node_coord,
        "areas": np.asarray(m.nodes_areas().value, dtype=float).squeeze(),
        "E_magn": np.asarray(m.field[FIELD_MAGN].value, dtype=float).squeeze(),
        "E_normal": np.asarray(m.field[FIELD_NORMAL].value, dtype=float).squeeze(),
    }


# --- main ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="simnibs side of aim 2, see efield/README.md")
    ap.add_argument("job", help="job.json written by efield/job.py")
    ap.add_argument("--dry-run", action="store_true",
                    help="validate and print the plan, write dryrun.json, never simulate")
    ap.add_argument("--no-mesh", action="store_true",
                    help="skip loading the head mesh in the dry run (no skin distance check)")
    ap.add_argument("--supra-frac", type=float, default=None,
                    help="suprathreshold fraction of the surface peak, default from the job or 0.5")
    args = ap.parse_args(argv)

    if not HAVE_SIMNIBS:
        print("this script needs simnibs, run it with simnibs_python")
        return 2

    job_path = Path(args.job).resolve()
    with open(job_path) as f:
        job = json.load(f)
    job_dir = job_path.parent
    supra_frac = args.supra_frac if args.supra_frac is not None else float(job.get("supra_frac", 0.5))
    supra_abs = job.get("supra_abs_vm")

    print(f"simnibs {simnibs.__version__}   job {job_path}")
    print(f"subject {job['subject']}   m2m {job['m2m']}")
    problems = []

    # --- the head model
    m2m = Path(job["m2m"])
    files = None
    if not m2m.is_dir():
        problems.append(f"m2m folder missing: {m2m}")
    else:
        files = SubjectFiles(subpath=str(m2m))
        if not os.path.exists(files.fnamehead):
            problems.append(f"head mesh missing: {files.fnamehead}")
        else:
            print(f"mesh    {files.fnamehead}")

    # --- the coil
    coil_path = resolve_coil(job["coil_file"])
    if coil_path is None:
        problems.append(f"coil file not found: {job['coil_file']}")
    else:
        print(f"coil    {coil_path}")
    print(f"didt    {job.get('didt', 1e6):g} A/s   fields {job.get('fields', 'eE')}   map_to_surf {job.get('map_to_surf', True)}")

    # --- the poses
    mesh = None
    if files is not None and os.path.exists(files.fnamehead) and (not args.no_mesh or job.get("snap_to_skin")):
        t0 = time.time()
        mesh = load_skin(m2m)
        print(f"loaded head mesh in {time.time() - t0:.1f} s, {mesh.nodes.nr} nodes")

    plan_poses = []
    for i, p in enumerate(job["poses"], start=1):
        m = np.asarray(p["matsimnibs"], dtype=float)
        entry = {"index": i, "name": p["name"], "site": p.get("site"), "method": p.get("method"),
                 "rep": p.get("rep"), "matsimnibs": m.tolist()}
        if m.shape != (4, 4) or not check_rotation(m[:3, :3]):
            problems.append(f"pose {p['name']}: matsimnibs is not a proper rotation")
        centre, ydir, zdir = m[:3, 3], m[:3, 1], m[:3, 2]
        line = (f"  pose {i} {p['name']}: centre {np.round(centre, 1)} handle {np.round(ydir, 3)} "
                f"normal(in) {np.round(zdir, 3)}")
        if mesh is not None:
            if job.get("snap_to_skin"):
                # rebuild from the centre and handle with simnibs's own projection,
                # so the coil sits on this head's skin with this head's normal
                ydir_point = centre + ydir * 50.0
                m_new = mesh.calc_matsimnibs(centre, ydir_point, float(job.get("coil_skin_distance_mm", 4.0)))
                moved = float(np.linalg.norm(m_new[:3, 3] - centre))
                p["matsimnibs"] = m_new.tolist()
                entry["matsimnibs"] = m_new.tolist()
                entry["snapped_mm"] = moved
                centre, ydir, zdir = m_new[:3, 3], m_new[:3, 1], m_new[:3, 2]
                line += f"\n         snapped to skin, moved {moved:.1f} mm -> centre {np.round(centre, 1)} normal(in) {np.round(zdir, 3)}"
            d, proj = skin_distance(mesh, centre)
            entry["skin_distance_mm"] = d
            # is the normal pointing into the head: the skin point should be along +z from the centre
            into = float(np.dot(proj - centre, zdir))
            entry["normal_into_head"] = bool(into >= -1e-6)
            line += f"\n         {d:.1f} mm from skin, normal points {'into' if into >= -1e-6 else 'AWAY FROM'} the head"
            if d > MAX_SKIN_DISTANCE_MM:
                problems.append(f"pose {p['name']}: centre is {d:.0f} mm from the skin, wrong space?")
            if into < -1e-6:
                problems.append(f"pose {p['name']}: coil normal points away from the head")
        else:
            line += "\n         (skin distance not checked, no mesh)"
        print(line)
        plan_poses.append(entry)

    # --- the rois
    plan_rois = []
    for r in job.get("rois", []):
        entry = dict(r)
        if files is not None and m2m.is_dir():
            try:
                real = realize_roi(r, m2m)
                entry.update({"n_nodes": real["n_nodes"], "centre_node": real["centre_node"],
                              "centre_snapped": [float(x) for x in real["centre"]],
                              "snap_mm": real["snap_mm"]})
                print(f"  roi {r['name']}: seed {np.round(real['seed'], 1)} snapped {real['snap_mm']:.1f} mm "
                      f"onto the central surface at {np.round(real['centre'], 1)}, {real['n_nodes']} nodes")
                if real["snap_mm"] > 15.0:
                    problems.append(f"roi {r['name']}: seed is {real['snap_mm']:.0f} mm from the cortex, depth guess off?")
            except Exception as e:      # the central surfaces may be missing on a partial m2m
                entry["error"] = str(e)
                print(f"  roi {r['name']}: could not be realized ({e})")
        else:
            print(f"  roi {r['name']}: seed {r.get('centre')} radius {r['radius_mm']} mm (not realized, no m2m)")
        plan_rois.append(entry)

    # --- build the session, validate what simnibs itself checks cheaply
    session_ok = False
    if not problems and coil_path is not None:
        try:
            S, tms = build_session(job, coil_path)
            session_ok = True
            print(f"session: {len(tms.pos)} positions, pathfem {S.pathfem}")
        except Exception as e:
            problems.append(f"could not build SESSION: {e}")

    for pr in problems:
        print("PROBLEM:", pr)

    if args.dry_run:
        plan = {"dry_run": True, "simnibs_version": simnibs.__version__, "subject": job["subject"],
                "m2m": str(m2m), "mesh": files.fnamehead if files else None, "coil": coil_path,
                "didt": job.get("didt", 1e6), "poses": plan_poses, "rois": plan_rois,
                "session_ok": session_ok, "problems": problems,
                "would_run": "run_simnibs(S) then read subject_overlays/*_central.msh"}
        out = job_dir / "dryrun.json"
        with open(out, "w") as f:
            json.dump(plan, f, indent=2)
        print(f"dry run, nothing simulated. plan written to {out}")
        return 0 if not problems else 1

    if problems:
        print("not running, fix the problems above")
        return 1

    # --- the real thing
    print("running simnibs ...")
    run_simnibs(S)

    results = {"subject": job["subject"], "m2m": str(m2m), "coil": coil_path, "didt": job.get("didt", 1e6),
               "out_dir": str(job["out_dir"]), "supra_frac": supra_frac,
               "supra_abs_vm": supra_abs, "ran": True, "poses": []}
    rois = {r["name"]: realize_roi(r, m2m) for r in job.get("rois", [])}
    for i, p in enumerate(job["poses"], start=1):
        ov = find_overlay(job["out_dir"], i)
        if ov is None:
            print(f"  pose {p['name']}: no overlay found for index {i}")
            continue
        data = read_overlay(ov)
        entry = {"index": i, "name": p["name"], "site": p.get("site"), "method": p.get("method"),
                 "rep": p.get("rep"), "overlay": ov, "rois": {}}
        for name, real in rois.items():
            if real["mask"].shape[0] != data["E_magn"].shape[0]:
                raise RuntimeError(f"roi {name} has {real['mask'].shape[0]} nodes, overlay {ov} has "
                                   f"{data['E_magn'].shape[0]}; the surfaces do not match")
            entry["rois"][name] = roi_metrics(data["E_magn"], data["E_normal"], data["areas"],
                                              real["mask"], real["centre_node"], supra_frac,
                                              supra_abs=supra_abs)
        results["poses"].append(entry)
        print(f"  pose {p['name']}: " + ", ".join(
            f"{k} mean {v['mean_E_magn']:.1f} peak {v['peak_E_magn']:.1f} V/m" for k, v in entry["rois"].items()))

    out = job_dir / "results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"results written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
