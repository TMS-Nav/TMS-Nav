import numpy as np
import pyvista as pv
import pytest

from src.eeg_cap import arc_intersection, eeg_cap, sample_arc, scalp_arc
from src.targets import Target


def sphere_cap(offset=(0, 0, 0)):
    offset = np.asarray(offset)
    surf = pv.Sphere(radius=100, theta_resolution=180, phi_resolution=181).translate(offset)
    positions = {"Nz": [0, 100, 0], "Iz": [0, -100, 0],
                 "LPA": [-100, 0, 0], "RPA": [100, 0, 0]}
    landmarks = [Target(label, label, "yellow", contact=np.asarray(p) + offset)
                 for label, p in positions.items()]
    return surf, landmarks


def test_spherical_tape_fractions_and_surface():
    surf, landmarks = sphere_cap()
    markers, arcs, qc = eeg_cap(surf, landmarks)
    sites = {m.label: m.contact for m in markers}
    assert len(sites) == 22
    assert set(sites) >= {"F3", "F4", "C3", "C4", "P3", "P4", "SMA"}
    assert qc["cz_coronal_gap_mm"] < .1
    assert np.linalg.norm(sites["Cz"] - [0, 0, 100]) < .1
    # Arc position 40% is not the Euclidean midpoint of the two contacts.
    assert np.linalg.norm(sites["SMA"] - [0, 100*np.cos(.4*np.pi), 100*np.sin(.4*np.pi)]) < .1
    assert np.linalg.norm(sites["SMA"] - .5*(sites["Fz"]+sites["Cz"])) > 4
    assert np.allclose(sites["C3"]*[-1, 1, 1], sites["C4"], atol=.1)
    assert abs(arcs[0]["length_mm"] - 100*np.pi) < .1
    for name, labels in [("Fp1–C3–O1", ("Fp1", "C3", "O1")),
                         ("Fp2–C4–O2", ("Fp2", "C4", "O2"))]:
        arc = np.asarray(next(a["points"] for a in arcs if a["name"] == name))
        assert np.linalg.norm(arc[0] - sites[labels[0]]) < 1
        assert np.linalg.norm(arc[-1] - sites[labels[2]]) < 1
        assert np.min(np.linalg.norm(arc - sites[labels[1]], axis=1)) < 1
        assert np.max(np.linalg.norm(np.diff(arc, axis=0), axis=1)) < 3
        assert np.max(np.abs(np.linalg.norm(arc, axis=1) - 100)) < .1
    for marker in markers:
        assert abs(np.linalg.norm(marker.contact)-100) < .1


def test_cap_is_translation_equivariant():
    base, _, _ = eeg_cap(*sphere_cap())
    offset = np.array([35, -20, 11])
    shifted, _, _ = eeg_cap(*sphere_cap(offset))
    for a, b in zip(base, shifted):
        np.testing.assert_allclose(b.contact-a.contact, offset, atol=.1)


def test_arc_uses_length_not_vertex_index():
    np.testing.assert_allclose(sample_arc([[0,0,0],[1,0,0],[10,0,0]], [.5]), [[5,0,0]])
    with pytest.raises(ValueError, match="collinear"):
        scalp_arc(sphere_cap()[0], [0,0,0], [1,0,0], [2,0,0])


def distance_to_polyline(point, points):
    a, b = points[:-1], points[1:]
    direction = b-a
    length_squared = np.sum(direction**2, axis=1)
    t = np.clip(np.divide(np.sum((point-a)*direction, axis=1), length_squared,
                          out=np.zeros(len(a)), where=length_squared > 0), 0, 1)
    return np.linalg.norm(a+t[:, None]*direction-point, axis=1).min()


@pytest.mark.parametrize("axes", [(100, 100, 100), (85, 105, 115)])
def test_lateral_sites_lie_on_both_construction_arcs(axes):
    surf, landmarks = sphere_cap()
    scale = np.asarray(axes)/100
    surf.points *= scale
    for landmark in landmarks:
        landmark.contact = landmark.contact*scale
    markers, arcs, _ = eeg_cap(surf, landmarks)
    sites = {m.label: m.contact for m in markers}
    paths = {a["name"]: np.asarray(a["points"]) for a in arcs}
    for longitudinal, frontal, posterior in [("Fp1–C3–O1", "F3", "P3"),
                                              ("Fp2–C4–O2", "F4", "P4")]:
        for label, transverse in [(frontal, "F7–Fz–F8"), (posterior, "T5–Pz–T6")]:
            assert distance_to_polyline(sites[label], paths[longitudinal]) < .003
            assert distance_to_polyline(sites[label], paths[transverse]) < .003


def test_arc_intersection_interpolates_and_selects_the_near_crossing():
    arc = [[-1, 0, 0], [1, 2, 0], [-1, 4, 0]]
    plane = [[0, 0, 0], [0, 1, 0], [0, 0, 1]]
    np.testing.assert_allclose(arc_intersection(arc, plane, [0, 3, 0]), [0, 3, 0])
    with pytest.raises(ValueError, match="do not intersect"):
        arc_intersection([[1, 0, 0], [2, 1, 0]], plane, [0, 0, 0])
