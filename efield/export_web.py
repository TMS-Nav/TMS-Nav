# a future layer for the three.js viewer: the e field painted on the cortex.
#
# not wired up yet. the plan is to mirror src/export_web.py, which drops ply
# meshes plus a small json per subject into viewer/src/data/<subject>/, so the
# viewer can load one more mesh and one more json without learning anything new.
from pathlib import Path


def export_efield_surface(overlay_msh, out_dir, pose_name, roi_nodes=None, supra_nodes=None):
    """Write E_magn on the gm central surface as ply + json for the viewer.

    overlay_msh is the subject_overlays/*_central.msh simnibs wrote for one pose
    (node fields E_magn, E_normal). the plan:

      out_dir/cortex.ply            the central surface, RAS mm, written once per
                                    subject, with E_magn of each pose as a scalar
                                    array named after the pose
      out_dir/efield_<pose>.json    peak, mean and the roi and suprathreshold
                                    node lists, so the viewer can colour the
                                    hotspot and outline the roi

    reading the msh needs simnibs (mesh_io.read_msh) so this has to run under
    simnibs_python, or the runner has to dump the node coordinates, triangles
    and fields to npz first and this function reads the npz with numpy. the
    second is the better fit for the two interpreter rule and is what will be
    done. until then this raises so nobody mistakes it for a working export
    """

    raise NotImplementedError(
        "efield.export_web.export_efield_surface is a documented stub. the runner does not "
        "yet dump the central surface to npz, so there is nothing for the viewer to load. "
        f"asked for {Path(overlay_msh).name} -> {out_dir} ({pose_name})")
