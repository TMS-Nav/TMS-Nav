# the probabilistic half of the head model. where you decide the nasion is, and what
# the tape reads, are both random, so a head is a distribution over 10-20 caps rather
# than one cap. each realization here is one honest repeat of the whole cap drawing
# procedure on the same skull.
#
# the error at a landmark is an ANISOTROPIC 3d gaussian, not a flat 2d one. sliding a
# finger along the scalp is easy so the two tangent directions get the big sd. pressing
# in or lifting off is resisted by bone and by the tape tension, so the normal gets a
# smaller sd, but not zero. an earlier version used zero there, which quietly claimed
# you can always find the exact depth of a bony landmark through skin, and that is not
# true.
#
# the two directions also enter the model at different places, which is the part worth
# understanding:
#   tangent  moves the landmark ALONG the surface, so the tape spans a different arc.
#            first order effect on every reading.
#   normal   moves it perpendicular to the tape run, which to first order does not
#            change an arc length at all. it shows up instead as the tape having to
#            reach a mark that sits proud of, or pressed into, the skin. press 1 mm in
#            at both ears and the ear to ear reading comes back 2 mm short.
from dataclasses import dataclass

import numpy as np

from src.coil_model import coil_frame
from src.ten_twenty import (
    HeadDimensions,
    arc_length,
    electrode_positions,
    fit_ellipsoid,
    loop_on_ellipsoid,
    project_to_ellipsoid,
)


@dataclass
class NoiseModel:
    # --- finding the tape landmarks, mm -------------------------------------------
    # sliding along the skin, the big one. reported palpation error for these
    # fiducials is around 1-3 mm, the meeting note said 0.2 mm which is optimistic
    landmark_tangent: float = 1.0
    # pressing in or lifting off. about a third of the slide, that is roughly how
    # much soft tissue there is to compress over the nasion and the preauriculars
    # before you are on bone
    landmark_normal: float = 0.35

    # --- drawing the actual mark on the scalp, mm ---------------------------------
    # the same two errors again, now at the target rather than at the landmark. the
    # pen slides, and the skin squashes under it
    mark_tangent: float = 1.0
    mark_normal: float = 0.35

    # sd on each of the three tape readings, mm. slip, tension, reading to the mm
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


def _draw(scale, shape, rng, size):
    # one noise draw of the requested shape, scale is an sd or a box half width
    if not scale:
        return np.zeros(size)
    if shape == "gaussian":
        return rng.normal(0.0, scale, size=size)
    if shape == "box":
        return rng.uniform(-scale, scale, size=size)
    raise ValueError(f"noise shape must be gaussian or box, got {shape}")


def perturb_landmark(axes, p, noise, rng):
    """Misplace a tape landmark. Returns the slid point and the normal offset.

    the two come back separately on purpose. the slide is a position on the head, the
    normal offset is a length the tape has to make up, and they feed the arc
    measurement in different ways.
    """

    e1, e2, _ = tangent_basis(axes, p)

    u, v = _draw(noise.landmark_tangent, noise.shape, rng, 2)
    w = float(_draw(noise.landmark_normal, noise.shape, rng, 1)[0])

    # the slide leaves the surface by a second order amount, project it back so the
    # landmark stays somewhere a finger could actually reach
    slid = project_to_ellipsoid(np.asarray(p, dtype=float) + e1 * u + e2 * v, axes)
    return slid, w


def perturb_mark(axes, p, noise, rng):
    """Where the X actually lands when you draw the target on the scalp.

    no projection back onto the surface here. the mark really does sit off the ideal
    head, that is skin squash and hair, and it is what gives the scatter its third
    dimension.
    """

    e1, e2, n = tangent_basis(axes, p)

    u, v = _draw(noise.mark_tangent, noise.shape, rng, 2)
    w = _draw(noise.mark_normal, noise.shape, rng, 1)[0]

    return np.asarray(p, dtype=float) + e1 * u + e2 * v + n * w


def draw_dimensions(true_axes, noise, rng):
    """One realization of the three tape readings on a head whose axes we know.

    three error layers. the landmarks slide along the skin, they also sit proud of or
    pressed into it, and then the tape itself is misread.
    """

    a = anchors(true_axes)

    # layer A, where the clinician decides each landmark is
    put, off = {}, {}
    for k in LANDMARKS:
        put[k], off[k] = perturb_landmark(true_axes, a[k], noise, rng)

    # the tape still goes over the physical top of the head, misplacing the nasion
    # does not move the vertex, so the true vertex stays the via point
    ni = arc_length(true_axes, put["Nz"], a["vertex"], put["Iz"])
    lr = arc_length(true_axes, put["LPA"], a["vertex"], put["RPA"])
    circ = loop_on_ellipsoid(true_axes, [put["Nz"], put["RPA"], put["Iz"], put["LPA"]])

    # layer B, the normal offsets. a mark sitting w proud of the skin costs the tape
    # an extra w to reach, pressed in it saves w. each arc picks up its two ends, the
    # circumference passes through all four
    ni += off["Nz"] + off["Iz"]
    lr += off["LPA"] + off["RPA"]
    circ += sum(off.values())

    # layer C, what the tape actually reads
    if noise.tape:
        ni, lr, circ = np.array([ni, lr, circ]) + rng.normal(0.0, noise.tape, size=3)

    return HeadDimensions(nasion_inion=float(ni), lpa_rpa=float(lr), circumference=float(circ))


def simulate_caps(true_dims, noise, n_draws, rng):
    """Rebuild the whole cap n_draws times. Returns the truth and one array per site.

    two stages per draw. first the head gets measured badly and the cap is rebuilt
    from those readings, then the target actually gets drawn on the scalp and the pen
    misses too. the second stage is what gives the cloud its third dimension, without
    it a reconstructed nasion can only ever move along one axis.
    """

    truth = electrode_positions(true_dims)
    true_axes = fit_ellipsoid(true_dims)

    draws = []
    for _ in range(n_draws):
        dims = draw_dimensions(true_axes, noise, rng)
        sites = electrode_positions(dims)

        # the frame comes off the TRUE head, that is the surface being drawn on
        draws.append({k: perturb_mark(true_axes, v, noise, rng) for k, v in sites.items()})

    samples = {name: np.array([d[name] for d in draws]) for name in truth}
    return truth, samples


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

    # zero noise has to be the identity, otherwise every displacement is a bug
    quiet = NoiseModel(landmark_tangent=0.0, landmark_normal=0.0, tape=0.0)
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

    # the slid landmark must stay on the ellipsoid, and move by about the sd asked for
    noise = NoiseModel(landmark_tangent=1.0, landmark_normal=0.35, tape=0.0)
    nz = anchors(axes)["Nz"]
    slid = np.array([perturb_landmark(axes, nz, noise, rng)[0] for _ in range(4000)])
    on_surface = np.abs(((slid / axes) ** 2).sum(axis=1) - 1.0).max()
    assert on_surface < 1e-9, f"slid landmarks left the ellipsoid by {on_surface}"
    print("slid landmarks stay on the surface, max residual", f"{on_surface:.1e}")
    print("nasion slide rms:", round(float(np.linalg.norm(slid - nz, axis=1).mean()), 3),
          "mm (expect about 1.25 for sd 1.0, the mean of a 2d rayleigh)")

    # the marked point must be anisotropic, wide in the tangent plane and thin along
    # the normal, in the ratio the noise model was given
    marks = np.array([perturb_mark(axes, nz, noise, rng) for _ in range(20000)])
    e1, e2, n = tangent_basis(axes, nz)
    d = marks - nz
    sd_t1 = float((d @ e1).std(ddof=1))
    sd_t2 = float((d @ e2).std(ddof=1))
    sd_n = float((d @ n).std(ddof=1))
    print(f"marked point sd: tangent {sd_t1:.3f} and {sd_t2:.3f} mm, normal {sd_n:.3f} mm")
    assert abs(sd_t1 - noise.mark_tangent) < 0.05, "tangent sd is not what was asked for"
    assert abs(sd_n - noise.mark_normal) < 0.05, "normal sd is not what was asked for"
    print(f"ratio normal/tangent {sd_n / sd_t1:.2f}, asked for"
          f" {noise.mark_normal / noise.mark_tangent:.2f}")
