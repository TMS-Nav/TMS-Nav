# international 10-20 eeg positions from tape measure head dimensions. the head is a
# half ellipsoid sitting on the plane through nasion, inion and the two preauricular
# points, the three tape arcs pin down its three semi axes, and every 10-20 chain is
# then an arc cut out of that ellipsoid by a plane. no mri needed.
#
# head frame is the usual RAS mm, x is R+, y is A+, z is S+, origin in the middle of
# the ellipsoid so nz/iz/lpa/rpa all sit at z = 0 and the vertex at z = c.
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ellipe


@dataclass
class HeadDimensions:
    nasion_inion: float    # arc nz over the vertex to iz, mm
    lpa_rpa: float         # arc lpa over the vertex to rpa, mm
    circumference: float   # around through nz and iz, above the ears, mm


# where each site sits along its chain, as a fraction of the whole arc
SAGITTAL_SITES = [("Fpz", 0.10), ("Fz", 0.30), ("Cz", 0.50), ("Pz", 0.70), ("Oz", 0.90)]
CORONAL_SITES = [("T3", 0.10), ("C3", 0.30), ("C4", 0.70), ("T4", 0.90)]

# the ring runs fpz -> ear -> oz, so half the circumference. jasper puts fp1 5% of the
# full loop off the midline and then steps by 10%, which on the half arc is 10/30/70/90.
# the 50% mark is t3/t4 and those come off the coronal chain instead
RING_LEFT = [("Fp1", 0.10), ("F7", 0.30), ("T5", 0.70), ("O1", 0.90)]
RING_RIGHT = [("Fp2", 0.10), ("F8", 0.30), ("T6", 0.70), ("O2", 0.90)]

# fp1 -> c3 -> o1 and its mirror, f3/p3 land midway between their neighbours
PARA_CHAINS = [("Fp1", "C3", "O1", "F3", "P3"), ("Fp2", "C4", "O2", "F4", "P4")]

# sma is 15% of the nasion inion arc in front of cz, so 35% along the sagittal chain.
# TODO read this out of the mantovani 2010 methods, the abstract does not state the rule
SMA_FRACTION = 0.35

# dense samples per arc before interpolating by length. 2001 puts the interpolation
# error a few orders below the mm we care about
_N_SAMPLES = 2001


def ellipse_perimeter(p, q):
    """Exact perimeter of an ellipse with semi axes p and q."""

    p, q = max(p, q), min(p, q)
    if q <= 0:
        raise ValueError(f"ellipse semi axes must be positive, got {p} and {q}")

    # no elementary closed form, 4pE(m) is exact, m is the squared eccentricity
    return 4.0 * p * float(ellipe(1.0 - (q / p) ** 2))


def fit_ellipsoid(dims):
    """Semi axes (a, b, c) in mm that reproduce the three tape measurements."""

    def residuals(v):
        a, b, c = v
        return [
            ellipse_perimeter(b, c) / 2.0 - dims.nasion_inion,   # sagittal, x = 0
            ellipse_perimeter(a, c) / 2.0 - dims.lpa_rpa,        # coronal, y = 0
            ellipse_perimeter(a, b) - dims.circumference,        # axial, z = 0
        ]

    # a sphere of the right girth gets a and b close, then treat the sagittal ellipse
    # as near circular to guess the height
    r0 = dims.circumference / (2.0 * np.pi)
    c0 = max(2.0 * dims.nasion_inion / np.pi - r0, 1.0)

    out = least_squares(residuals, [r0, r0, c0], bounds=(1.0, np.inf))
    if not out.success or np.max(np.abs(out.fun)) > 1e-6:
        raise ValueError(
            f"no ellipsoid matches those measurements, residuals {np.round(out.fun, 4)} mm"
        )
    return tuple(float(v) for v in out.x)


def measure(axes):
    """The three tape arcs implied by semi axes, the inverse of fit_ellipsoid."""

    a, b, c = axes
    return HeadDimensions(
        nasion_inion=ellipse_perimeter(b, c) / 2.0,
        lpa_rpa=ellipse_perimeter(a, c) / 2.0,
        circumference=ellipse_perimeter(a, b),
    )


def project_to_ellipsoid(p, axes):
    """Push a point out onto the ellipsoid, radially in the scaled frame."""

    axes = np.asarray(axes, dtype=float)
    s = np.asarray(p, dtype=float) / axes
    return (s / np.linalg.norm(s, axis=-1, keepdims=True)) * axes


def _arc_samples(axes, start, via, end):
    """Dense points along start -> via -> end, their cumulative length, and via.

    the plane through the three points cuts the ellipsoid in an exact ellipse, so
    scale everything to the unit sphere where that cut is a plain circle, walk the
    circle, then scale back. one primitive covers every 10-20 chain.
    """

    axes = np.asarray(axes, dtype=float)
    s0, sv, s1 = (np.asarray(p, dtype=float) / axes for p in (start, via, end))

    nrm = np.cross(sv - s0, s1 - s0)
    n_len = np.linalg.norm(nrm)
    if n_len < 1e-9:
        raise ValueError("start, via and end are collinear, no plane through them")
    nrm = nrm / n_len

    # a plane cuts the unit sphere in a circle, offset d from the origin
    d = float(np.dot(nrm, s0))
    center = d * nrm
    radius = np.sqrt(max(1.0 - d * d, 0.0))
    if radius < 1e-9:
        raise ValueError("the three points collapse to a point on the sphere")

    u = s0 - center
    u = u / np.linalg.norm(u)
    v = np.cross(nrm, u)

    def angle(s):
        w = s - center
        return np.arctan2(float(np.dot(w, v)), float(np.dot(w, u))) % (2.0 * np.pi)

    th_v, th_1 = angle(sv), angle(s1)

    # go round the way that passes through via, not the short way that skips it
    if th_v > th_1:
        th_v -= 2.0 * np.pi
        th_1 -= 2.0 * np.pi

    th = np.linspace(0.0, th_1, _N_SAMPLES)
    ring = center + radius * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)
    pts = ring * axes

    # arc length has to be measured on the ellipsoid, not on the sphere we walked
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])

    # th may run negative, abs(th) is monotone either way so interpolate against that
    via_frac = float(np.interp(abs(th_v), np.abs(th), cum)) / cum[-1]

    return pts, cum, via_frac


def arc_on_ellipsoid(axes, start, via, end, fractions):
    """Points at those fractions of start -> via -> end, where via sits, total mm."""

    pts, cum, via_frac = _arc_samples(axes, start, via, end)
    total = cum[-1]

    want = np.atleast_1d(np.asarray(fractions, dtype=float)) * total
    out = np.stack([np.interp(want, cum, pts[:, k]) for k in range(3)], axis=1)
    if len(out):
        out = project_to_ellipsoid(out, axes)

    return out, via_frac, float(total)


def arc_length(axes, start, via, end):
    """Length in mm of the arc start -> via -> end, what a tape over the head reads."""

    return float(_arc_samples(axes, start, via, end)[1][-1])


def loop_on_ellipsoid(axes, points):
    """Perimeter of the closed curve the least squares plane through points cuts.

    a tape run round the head touches four landmarks that in general are not
    coplanar, so it settles on the plane that misses all four by as little as
    possible. that is the least squares plane.
    """

    axes = np.asarray(axes, dtype=float)
    s = np.asarray(points, dtype=float) / axes
    mid = s.mean(axis=0)

    # smallest singular direction of the centred points is the plane normal
    nrm = np.linalg.svd(s - mid)[2][2]

    d = float(np.dot(nrm, mid))
    if abs(d) >= 1.0:
        raise ValueError(f"the tape plane misses the head, offset {d:.3f}")
    center = d * nrm
    radius = np.sqrt(1.0 - d * d)

    u = s[0] - center
    u = u / np.linalg.norm(u)
    v = np.cross(nrm, u)

    th = np.linspace(0.0, 2.0 * np.pi, _N_SAMPLES)
    pts = (center + radius * (np.cos(th)[:, None] * u + np.sin(th)[:, None] * v)) * axes
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def electrode_positions(dims):
    """The 21 standard 10-20 sites, the 4 anchors and the 2 study targets, head frame mm."""

    a, b, c = fit_ellipsoid(dims)
    axes = np.array([a, b, c])

    nz = np.array([0.0, b, 0.0])
    iz = np.array([0.0, -b, 0.0])
    lpa = np.array([-a, 0.0, 0.0])
    rpa = np.array([a, 0.0, 0.0])
    vertex = np.array([0.0, 0.0, c])

    sites = {"Nz": nz, "Iz": iz, "LPA": lpa, "RPA": rpa}

    # chain 1, nasion over the vertex to inion. sma rides along on the same arc
    fr = [f for _, f in SAGITTAL_SITES] + [SMA_FRACTION]
    pts, _, _ = arc_on_ellipsoid(axes, nz, vertex, iz, fr)
    for (name, _), p in zip(SAGITTAL_SITES, pts):
        sites[name] = p
    sites["SMA"] = pts[-1]

    # chain 2, ear to ear through cz. its own 50% point is the vertex again by
    # symmetry so cz is left alone here, consistency() reports how well that holds
    pts, _, _ = arc_on_ellipsoid(axes, lpa, sites["Cz"], rpa, [f for _, f in CORONAL_SITES])
    for (name, _), p in zip(CORONAL_SITES, pts):
        sites[name] = p

    # chain 3, the circumference ring. fpz and t3 come out at different heights so no
    # single plane passes through both, and the least squares plane through fpz, oz,
    # t3 and t4 is horizontal at their mean height, which is the path a tape round
    # the head actually follows
    z_ring = 0.5 * (sites["Fpz"][2] + sites["T3"][2])
    k = np.sqrt(max(1.0 - (z_ring / c) ** 2, 0.0))
    front = np.array([0.0, b * k, z_ring])
    back = np.array([0.0, -b * k, z_ring])

    for side, spec in ((-1.0, RING_LEFT), (1.0, RING_RIGHT)):
        ear = np.array([side * a * k, 0.0, z_ring])
        pts, _, _ = arc_on_ellipsoid(axes, front, ear, back, [f for _, f in spec])
        for (name, _), p in zip(spec, pts):
            sites[name] = p

    # chain 4, fp1 over c3 to o1. the standard spaces f3 and p3 evenly between their
    # neighbours, and c3 came off the coronal chain so it need not land exactly half
    # way here, hence two calls, one to find c3 then one to place f3 and p3
    for fp, ck, o, f_name, p_name in PARA_CHAINS:
        _, f_c, _ = arc_on_ellipsoid(axes, sites[fp], sites[ck], sites[o], [])
        pts, _, _ = arc_on_ellipsoid(
            axes, sites[fp], sites[ck], sites[o], [f_c / 2.0, (1.0 + f_c) / 2.0]
        )
        sites[f_name], sites[p_name] = pts[0], pts[1]

    return sites


def consistency(dims):
    """Where the chains disagree with each other, in mm and in fraction.

    the 10-20 recipe over determines the head. on a real skull the tape reconciles
    the chains as you go, on an ellipsoid they miss each other slightly. these are
    the misses, they are the honest error bar on the geometry itself.
    """

    a, b, c = fit_ellipsoid(dims)
    axes = np.array([a, b, c])
    sites = electrode_positions(dims)

    # the coronal chain midpoint against the sagittal cz
    pts, _, _ = arc_on_ellipsoid(axes, [-a, 0.0, 0.0], sites["Cz"], [a, 0.0, 0.0], [0.5])
    cz_gap = float(np.linalg.norm(pts[0] - sites["Cz"]))

    # the ring 50% mark against the coronal t3
    z_ring = 0.5 * (sites["Fpz"][2] + sites["T3"][2])
    k = np.sqrt(max(1.0 - (z_ring / c) ** 2, 0.0))
    pts, _, _ = arc_on_ellipsoid(
        axes,
        [0.0, b * k, z_ring],
        [-a * k, 0.0, z_ring],
        [0.0, -b * k, z_ring],
        [0.5],
    )
    t3_gap = float(np.linalg.norm(pts[0] - sites["T3"]))

    # how far along fp1 -> o1 the coronal c3 lands, the recipe assumes 0.50
    _, c3_frac, _ = arc_on_ellipsoid(axes, sites["Fp1"], sites["C3"], sites["O1"], [])

    return {"cz_gap_mm": cz_gap, "t3_gap_mm": t3_gap, "c3_fraction": float(c3_frac)}


if __name__ == "__main__":
    # a typical adult head, all mm
    dims = HeadDimensions(nasion_inion=360.0, lpa_rpa=350.0, circumference=570.0)

    axes = fit_ellipsoid(dims)
    print("semi axes a, b, c:", tuple(round(v, 2) for v in axes), "mm")

    # the fit has to reproduce the tape it was built from
    back = measure(axes)
    err = [
        back.nasion_inion - dims.nasion_inion,
        back.lpa_rpa - dims.lpa_rpa,
        back.circumference - dims.circumference,
    ]
    assert max(abs(e) for e in err) < 1e-2, f"fit does not round trip, off by {err} mm"
    print("round trip error:", [f"{e:.1e}" for e in err], "mm")

    # the numerical arc walk has to agree with the closed form elliptic integral,
    # otherwise the noise model would be measuring with a different ruler
    a, b, c = axes
    nz, iz = [0.0, b, 0.0], [0.0, -b, 0.0]
    lpa, rpa, vertex = [-a, 0.0, 0.0], [a, 0.0, 0.0], [0.0, 0.0, c]
    walk = [
        arc_length(axes, nz, vertex, iz) - dims.nasion_inion,
        arc_length(axes, lpa, vertex, rpa) - dims.lpa_rpa,
        loop_on_ellipsoid(axes, [nz, rpa, iz, lpa]) - dims.circumference,
    ]
    assert max(abs(e) for e in walk) < 1e-2, f"arc walk disagrees with the integral, {walk}"
    print("arc walk vs integral:", [f"{e:.1e}" for e in walk], "mm")

    sites = electrode_positions(dims)
    print()
    print(len(sites), "sites")
    for name in sorted(sites):
        x, y, z = sites[name]
        print(f"  {name:5s} {x:8.1f} {y:8.1f} {z:8.1f}")

    # spacings that ought to come out near 10% and 20% of the nasion inion arc
    print()
    print("nz  -> fpz:", round(float(np.linalg.norm(sites["Fpz"] - sites["Nz"])), 1), "mm")
    print("fpz -> fz :", round(float(np.linalg.norm(sites["Fz"] - sites["Fpz"])), 1), "mm")
    print("fz  -> cz :", round(float(np.linalg.norm(sites["Cz"] - sites["Fz"])), 1), "mm")
    print("(straight line so a bit under 10% and 20% of the", dims.nasion_inion, "mm arc)")

    print()
    print("chain disagreement:", {k: round(v, 4) for k, v in consistency(dims).items()})
