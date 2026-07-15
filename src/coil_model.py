# figure-8 coil geometry, same R and d as forward_model, just in mm instead of m
import numpy as np
import pyvista as pv

# forward_model works in meters, R=0.035 and d=0.035, mri is in mm so scale up
R_MM = 35.0
D_MM = 35.0


def figure8_coil(R=R_MM, d=D_MM, tube_radius=3.0, n_theta=240):
    """Two counter-wound loops as tubes, local frame, windings in z=0, +z is up."""

    theta = np.linspace(0, 2 * np.pi, n_theta)

    wings = []
    for sign in (-1.0, 1.0):
        # loop centers sit at +/-d on the local x axis
        x = sign * d + R * np.cos(theta)
        y = R * np.sin(theta)
        z = np.zeros_like(theta)

        ring = pv.MultipleLines(np.column_stack([x, y, z]))
        wings.append(ring.tube(radius=tube_radius, n_sides=16))

    return wings[0].merge(wings[1])


def coil_frame(normal, up_hint=(0.0, 0.0, 1.0)):
    """Local axes for the coil, e1/e2 span the winding plane, n is the normal."""

    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)

    # spin about n is arbitrary, any tangent works
    e1 = np.cross(n, np.asarray(up_hint, dtype=float))
    if np.linalg.norm(e1) < 1e-6:
        e1 = np.cross(n, [1.0, 0.0, 0.0])
    e1 = e1 / np.linalg.norm(e1)

    e2 = np.cross(n, e1)

    return e1, e2, n


def to_world(coil, origin, e1, e2, n):
    """Drop the local coil into RAS at origin with the given axes."""

    m = np.eye(4)
    m[:3, 0] = e1
    m[:3, 1] = e2
    m[:3, 2] = n
    m[:3, 3] = origin

    out = coil.copy()
    out.transform(m, inplace=True)
    return out


def clearance_to(coil, scalp):
    """Smallest distance from coil surface to scalp, negative means overlapping."""

    # implicit distance is point-to-surface, not point-to-nearest-vertex
    probe = coil.compute_implicit_distance(scalp)
    return float(probe.point_data["implicit_distance"].min())


def seat_coil(coil, scalp, n, gap=2.0, iters=12, tol=1e-3):
    """Slide the coil along n until its closest point is gap mm off the scalp."""

    out = coil.copy()

    # measure off the tube surface not the centerline, otherwise the 3 mm tube
    # radius eats the whole 2 mm and the windings end up in the head
    err = np.inf
    for _ in range(iters):
        err = gap - clearance_to(out, scalp)
        if abs(err) < tol:
            return out

        # not exactly 1:1 since the scalp curves under the coil, but it lands
        out.translate(n * err, inplace=True)

    raise RuntimeError(f"coil never seated, last error {err:.3f} mm")
