# the probabilistic half of the head model. where you decide the nasion is, and what
# the tape reads, are both random, so a head is a distribution over 10-20 caps rather
# than one cap. each realization here is one honest repeat of the whole cap drawing
# procedure on the same skull.
#
# the key modelling choice is that landmark error lives in the TANGENT PLANE at the
# landmark. palpating for the nasion you can slide along the skin, but you cannot push
# into the skull or float off it, so the error is genuinely two dimensional. treating
# it as an isotropic 3d blob would invent a radial component that physically cannot
# happen and would overstate the spread by about a factor of sqrt(3/2).
from dataclasses import dataclass

import numpy as np

from src.coil_model import coil_frame
from src.ten_twenty import (
    HeadDimensions,
    arc_length,
    fit_ellipsoid,
    loop_on_ellipsoid,
    project_to_ellipsoid,
)


@dataclass
class NoiseModel:
    # gaussian sd, or box half width, per in plane axis, mm. the meeting said 0.2 mm
    # which is optimistic, reported palpation error for these fiducials is nearer
    # 1-3 mm, so 1.0 is the default and 0.2 is reachable as the best case
    landmark: float = 1.0

    # sd on each of the three tape readings, mm. tape slip, tension, reading to the
    # nearest mm over an arc of 350 mm or so
    tape: float = 1.0

    # "gaussian" or "box". box is the literal reading of the meeting note, gaussian is
    # the default because a box has no tails and makes the 95% region ill defined
    shape: str = "gaussian"


# the four landmarks the tape is anchored on. cz is not one of them, it is derived
LANDMARKS = ("Nz", "Iz", "LPA", "RPA")


def anchors(axes):
    """The four tape landmarks and the vertex, on the ellipsoid, head frame mm."""

    a, b, c = axes
    return {
        "Nz": np.array([0.0, b, 0.0]),
        "Iz": np.array([0.0, -b, 0.0]),
        "LPA": np.array([-a, 0.0, 0.0]),
        "RPA": np.array([a, 0.0, 0.0]),
        "vertex": np.array([0.0, 0.0, c]),
    }


def ellipsoid_normal(axes, p):
    """Outward unit normal of the ellipsoid at p, the gradient of the implicit form."""

    axes = np.asarray(axes, dtype=float)
    n = np.asarray(p, dtype=float) / axes**2
    return n / np.linalg.norm(n)


def tangent_basis(axes, p):
    """Orthonormal (e1, e2, n) at p, e1 and e2 spanning the tangent plane."""

    # coil_model already builds a frame from a normal and handles the degenerate
    # up_hint, no reason to write that twice
    return coil_frame(ellipsoid_normal(axes, p))


def perturb_landmark(axes, p, noise, rng):
    """Slide p around inside its own tangent plane, then put it back on the head."""

    e1, e2, _ = tangent_basis(axes, p)

    if noise.shape == "gaussian":
        u, v = rng.normal(0.0, noise.landmark, size=2)
    elif noise.shape == "box":
        u, v = rng.uniform(-noise.landmark, noise.landmark, size=2)
    else:
        raise ValueError(f"noise shape must be gaussian or box, got {noise.shape}")

    # the slide leaves the surface by a second order amount, project it back so the
    # landmark stays somewhere a finger could actually reach
    return project_to_ellipsoid(np.asarray(p, dtype=float) + e1 * u + e2 * v, axes)


def draw_dimensions(true_axes, noise, rng):
    """One realization of the three tape readings on a head whose axes we know.

    two independent error layers. first the four landmarks get misplaced in their
    tangent planes, then the tape is run between the misplaced points and misread.
    """

    a = anchors(true_axes)

    # layer A, where the clinician decides each landmark is
    put = {k: perturb_landmark(true_axes, a[k], noise, rng) for k in LANDMARKS}

    # the tape still goes over the physical top of the head, misplacing the nasion
    # does not move the vertex, so the true vertex stays the via point
    ni = arc_length(true_axes, put["Nz"], a["vertex"], put["Iz"])
    lr = arc_length(true_axes, put["LPA"], a["vertex"], put["RPA"])
    circ = loop_on_ellipsoid(true_axes, [put["Nz"], put["RPA"], put["Iz"], put["LPA"]])

    # layer B, what the tape actually reads
    if noise.tape:
        ni, lr, circ = np.array([ni, lr, circ]) + rng.normal(0.0, noise.tape, size=3)

    return HeadDimensions(nasion_inion=float(ni), lpa_rpa=float(lr), circumference=float(circ))


if __name__ == "__main__":
    from src.ten_twenty import measure

    rng = np.random.default_rng(0)
    axes = fit_ellipsoid(HeadDimensions(360.0, 350.0, 570.0))

    # the tangent basis has to be orthonormal and actually tangent
    for name, p in anchors(axes).items():
        e1, e2, n = tangent_basis(axes, p)
        gram = np.array([[float(np.dot(x, y)) for y in (e1, e2, n)] for x in (e1, e2, n)])
        assert np.allclose(gram, np.eye(3), atol=1e-9), f"{name} frame is not orthonormal"
    print("tangent frames orthonormal at all five anchors")

    # zero noise has to be the identity, otherwise every displacement below is a bug
    quiet = NoiseModel(landmark=0.0, tape=0.0)
    same = draw_dimensions(axes, quiet, rng)
    exact = measure(axes)
    gap = [
        same.nasion_inion - exact.nasion_inion,
        same.lpa_rpa - exact.lpa_rpa,
        same.circumference - exact.circumference,
    ]
    # not bit exact, the arcs are re-measured by walking a polyline, so this carries
    # the discretization error of that walk. a few times 1e-5 mm is the floor
    assert max(abs(g) for g in gap) < 1e-2, f"zero noise moved the tape by {gap} mm"
    print("zero noise reproduces the tape to the arc walk floor:",
          [f"{g:.1e}" for g in gap], "mm")

    # a perturbed landmark must stay on the ellipsoid and move by about the sd asked for
    noise = NoiseModel(landmark=1.0, tape=0.0)
    nz = anchors(axes)["Nz"]
    draws = np.array([perturb_landmark(axes, nz, noise, rng) for _ in range(4000)])
    on_surface = np.abs(((draws / axes) ** 2).sum(axis=1) - 1.0).max()
    assert on_surface < 1e-9, f"perturbed landmarks left the ellipsoid by {on_surface}"
    print("perturbed landmarks stay on the surface, max residual", f"{on_surface:.1e}")
    print("nasion slide rms:", round(float(np.linalg.norm(draws - nz, axis=1).mean()), 3),
          "mm (expect about 1.25 for sd 1.0, that is the mean of a 2d rayleigh)")
