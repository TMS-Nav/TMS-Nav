"""10-20 tape constructions on the MRI scalp, in its physical RAS mm frame.

Tape paths are connected plane/mesh intersections, not straight XYZ distances.
On an irregular head, independent tape constructions need not agree; retain QC.
Transverse midpoints provide initial estimates; F3/F4/P3/P4 are reconciled at
the intersections of their transverse and longitudinal scalp arcs.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from src.targets import Target


def sample_arc(points, fractions):
    points = np.asarray(points, dtype=float)
    length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if length[-1] <= 0:
        raise ValueError("zero length scalp arc")
    return np.column_stack([np.interp(np.asarray(fractions) * length[-1], length, points[:, i])
                            for i in range(3)])


def scalp_arc(surf, start, via, end):
    """Connected surface cut from start to end through via (nearest cut vertices).

    The via point selects the intended branch of the closed cut, e.g. the crown
    rather than the face/neck. Vertex snapping is limited by mesh resolution.
    """
    start, via, end = np.asarray([start, via, end], dtype=float)
    normal = np.cross(via - start, end - start)
    if np.linalg.norm(normal) < 1e-8:
        raise ValueError("scalp arc anchors are collinear")
    cut = surf.slice(normal=normal / np.linalg.norm(normal), origin=start).clean(tolerance=1e-5)
    points = np.asarray(cut.points)
    edges, cursor = [], 0
    while cursor < len(cut.lines):
        n = int(cut.lines[cursor]); ids = cut.lines[cursor+1:cursor+1+n]
        edges.extend(zip(ids[:-1], ids[1:])); cursor += n + 1
    if not edges:
        raise ValueError("scalp arc plane does not intersect the mesh")
    edges = np.asarray(edges, dtype=int)
    weights = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    graph = coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(len(points), len(points))).tocsr()
    ids = [int(np.argmin(np.linalg.norm(points - p, axis=1))) for p in (start, via, end)]
    paths = []
    for a, b in zip(ids[:-1], ids[1:]):
        distances, prev = dijkstra(graph, directed=False, indices=a, return_predecessors=True)
        if not np.isfinite(distances[b]):
            raise ValueError("disconnected scalp arc: inspect the mesh and landmarks")
        path = [b]
        while path[-1] != a:
            path.append(int(prev[path[-1]]))
        paths.append(path[::-1])
    return points[paths[0] + paths[1][1:]]


def arc_intersection(points, plane_anchors, near):
    """Intersect a scalp polyline with a second cut plane, near a site estimate.

    Linear interpolation stays on the original triangle/cut segment, placing
    the site on the scalp and on both planes instead of snapping to a vertex.
    The estimate selects the intended crossing when there is more than one.
    """
    a, via, b = np.asarray(plane_anchors, dtype=float)
    normal = np.cross(via - a, b - a)
    magnitude = np.linalg.norm(normal)
    if magnitude < 1e-8:
        raise ValueError("scalp arc anchors are collinear")
    normal /= magnitude
    points = np.asarray(points, dtype=float)
    signed = (points - a) @ normal
    candidates = list(points[np.abs(signed) < 1e-7])
    for i in np.flatnonzero(signed[:-1] * signed[1:] < 0):
        fraction = signed[i] / (signed[i] - signed[i+1])
        candidates.append(points[i] + fraction * (points[i+1] - points[i]))
    if not candidates:
        raise ValueError("EEG construction arcs do not intersect: recheck landmarks")
    candidates = np.asarray(candidates)
    return candidates[np.argmin(np.linalg.norm(candidates - near, axis=1))]


def eeg_cap(surf, landmarks):
    """Return scalp electrodes, construction arcs and consistency measurements.

    Includes 19 conventional scalp electrodes, Fpz/Oz construction sites and SMA.
    Ear/mastoid recording references A1/A2 are not these preauricular landmarks.
    """
    anchors = {m.label: np.asarray(m.contact) for m in landmarks}
    nz, iz, lpa, rpa = [anchors[k] for k in ("Nz", "Iz", "LPA", "RPA")]
    center = np.mean([nz, iz, lpa, rpa], axis=0)
    crown = center.copy(); crown[2] = surf.bounds[5]
    sites, arcs = {}, []

    def construct(name, a, v, b, labels, fractions):
        points = scalp_arc(surf, a, v, b)
        for label, point in zip(labels, sample_arc(points, fractions)):
            sites[label] = point
        length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
        # Keep the cut vertices: dashed lines follow the exact measured path.
        arcs.append({"name": name, "length_mm": round(length, 3),
                     "points": np.round(points, 3).tolist()})
        return points

    construct("Nasion–inion", nz, crown, iz,
                         ["Fpz", "Fz", "Cz", "Pz", "Oz", "SMA"], [.1, .3, .5, .7, .9, .4])
    coronal = construct("LPA–Cz–RPA", lpa, sites["Cz"], rpa,
                        ["T3", "C3", "C4", "T4"], [.1, .3, .7, .9])
    cz_gap = float(np.linalg.norm(sample_arc(coronal, [.5])[0] - sites["Cz"]))
    ring_gaps, rings = [], []
    for temporal, labels in [("T3", ["Fp1", "F7", "T5", "O1"]),
                             ("T4", ["Fp2", "F8", "T6", "O2"])]:
        ring = construct("Circumference " + temporal, sites["Fpz"], sites[temporal], sites["Oz"],
                         labels, [.1, .3, .7, .9])
        rings.append((ring, temporal, labels))
    full_ring = sum(np.linalg.norm(np.diff(ring, axis=0), axis=1).sum() for ring, _, _ in rings)
    for ring, temporal, labels in rings:
        length = np.linalg.norm(np.diff(ring, axis=0), axis=1).sum()
        fractions = np.array([.05, .15, .35, .45]) * full_ring / length
        if fractions[-1] >= 1:
            raise ValueError("circumference halves too asymmetric: recheck landmarks")
        sites.update(zip(labels, sample_arc(ring, fractions)))
        ring_gaps.append(float(np.linalg.norm(sample_arc(ring, [.25 * full_ring / length])[0] - sites[temporal])))
    # Upper transverse arcs supply initial midpoint estimates and crossing planes.
    transverse_planes = {}
    for left, mid, right, labels in [("F7", "Fz", "F8", ("F3", "F4")),
                                      ("T5", "Pz", "T6", ("P3", "P4"))]:
        arc = construct(left + "–" + mid + "–" + right,
                        sites[left], sites[mid], sites[right], [], [])
        split = int(np.argmin(np.linalg.norm(arc - sites[mid], axis=1)))
        sites[labels[0]] = sample_arc(arc[:split+1], [.5])[0]
        sites[labels[1]] = sample_arc(arc[split:], [.5])[0]
        for label in labels:
            transverse_planes[label] = [sites[left], sites[mid], sites[right]]
    # Reconcile the independently measured chains at their actual crossings.
    # This leaves all tape guides unchanged and puts each site on both arcs.
    crossing_shifts = {}
    for fp, c, o, labels in [("Fp1", "C3", "O1", ("F3", "P3")),
                             ("Fp2", "C4", "O2", ("F4", "P4"))]:
        longitudinal = construct(fp + "–" + c + "–" + o,
                                 sites[fp], sites[c], sites[o], [], [])
        for label in labels:
            old = sites[label]
            sites[label] = arc_intersection(longitudinal, transverse_planes[label], old)
            crossing_shifts[label] = round(float(np.linalg.norm(sites[label] - old)), 3)
    markers = []
    for label, point in sites.items():
        normal = point - center; normal /= np.linalg.norm(normal)
        markers.append(Target("EEG-" + label, label, "#43d9df", contact=point, normal=normal))
    return markers, arcs, {"cz_coronal_gap_mm": round(cz_gap, 3),
                           "temporal_ring_gap_mm": [round(g, 3) for g in ring_gaps],
                           "transverse_midpoint_shift_mm": crossing_shifts,
                           "method": "plane/scalp tape arcs; arc-length fractions",
                           "sma": "midpoint of Fz–Cz sagittal arc (40% Nz–Iz)",
                           "references": "A1/A2 not inferred"}
