# monte carlo cloud for the three.js viewer.
#
# the probabilistic model lives in its own ellipsoid head frame, the viewer lives in
# the subject scanner RAS, so the cloud has to be landed on the subject. it is landed
# with the same eeg landmarks that register everything else, via the umeyama fit in
# src/align.py. one uniform scale, so a bigger subject stretches the cloud with the
# head rather than leaving it the wrong size.
#
# on top of the positions this now carries the aim 1 metrics per draw. the coil face
# tilt and the total orientation difference, computed in the model frame where the
# surface normal is known exactly, and the signed offsets in the local scalp frame
# that bland altman runs on. angles do not care about the rotation onto the subject
import json
from pathlib import Path

import numpy as np

from src.align import similarity_transform
from src.landmark_noise import NoiseModel, ellipsoid_normal, simulate_caps, tangent_basis
from src.stats import (
    NAV_TRE_MM,
    bland_altman,
    coil_orientation_error,
    miss_distances,
    pair_distances,
    summary,
)
from src.ten_twenty import HeadDimensions, fit_ellipsoid

# the model names its anchors Nz/Iz, targets.standard_landmarks says nasion/inion
LANDMARK_ALIAS = {"nasion": "Nz", "inion": "Iz", "Cz": "Cz", "LPA": "LPA", "RPA": "RPA"}

# nominal adult head, mm. the umeyama scale below soaks up most of the difference
# between this and the actual subject, and the fitted scale gets reported so it is
# visible when it does not
NOMINAL_DIMS = HeadDimensions(nasion_inion=360.0, lpa_rpa=350.0, circumference=570.0)

# how many draws show at once. enough to read as a cloud, few enough that the
# individual dots stay distinguishable
N_SHOW = 50

# how many draws to ship in the json. the viewer picks n_show of these at random
# and can pick again, so the cloud can be redrawn without rerunning python
N_KEEP = 400

# mahalanobis radii to draw shells at. 1/2/3 sigma
SHELLS = (1.0, 2.0, 3.0)

# the three signed directions bland altman is reported in, local scalp frame
COMPONENTS = ("along_1", "along_2", "normal")


def monte_carlo_cloud(landmarks, targets, dims=NOMINAL_DIMS, noise=None,
                      n_draws=1200, n_show=N_SHOW, n_keep=N_KEEP, seed=0):
    """Per target cloud, mean and covariance ellipsoid, all in subject RAS mm."""

    rng = np.random.default_rng(seed)
    noise = noise or NoiseModel()

    truth, samples, yaws = simulate_caps(dims, noise, n_draws, rng, with_yaw=True)
    axes = fit_ellipsoid(dims)

    # register the model head onto this subject off the shared landmarks
    src, dst = [], []
    for m in landmarks:
        key = LANDMARK_ALIAS.get(m.name)
        if key and key in truth:
            src.append(truth[key])
            dst.append(np.asarray(m.contact, dtype=float))
    if len(src) < 4:
        raise ValueError(f"need at least 4 shared landmarks to register, got {len(src)}")

    xform = similarity_transform(src, dst)
    rot, shift = xform[:3, :3], xform[:3, 3]

    # the uniform scale is the cube root of the determinant of the linear part
    scale = float(abs(np.linalg.det(rot)) ** (1.0 / 3.0))

    # how well the landmarks line up after the fit, in mm. this is reported but NOT
    # relied on for placement, see the note in the site loop. it is large here because
    # standard_landmarks is approximate by its own admission
    fitted = np.asarray(src) @ rot.T + shift
    resid = float(np.linalg.norm(fitted - np.asarray(dst), axis=1).mean())

    sites = []
    for t in targets:
        key = t.label  # viewer labels are SMA, F4, C3, C4, Cz, same as the model names
        if key not in samples:
            continue

        # the cloud is CENTRED on the subject own target, not on the transformed model
        # target. the registration is only trusted for orientation and scale, because
        # standard_landmarks picks its fiducials off extreme scalp vertices and lands
        # them tens of mm from where the ellipsoid wants them. the model is the right
        # tool for the SHAPE of the uncertainty, the mri is the right tool for WHERE
        # the target is, so each is used for the thing it is good at
        model_offsets = samples[key] - truth[key]
        offsets = model_offsets @ rot.T
        pts = np.asarray(t.contact, dtype=float) + offsets
        mean = pts.mean(axis=0)
        cov = np.cov(pts.T)

        # principal axes of the scatter, largest first
        eigval, eigvec = np.linalg.eigh(cov)
        order = np.argsort(eigval)[::-1]
        eigval, eigvec = np.clip(eigval[order], 0.0, None), eigvec[:, order]
        sigmas = np.sqrt(eigval)

        pick = rng.choice(len(pts), size=min(n_keep, len(pts)), replace=False)

        ideal = np.asarray(t.contact, dtype=float)

        # the two distance distributions aim 1 is about. how far a placement lands
        # from the mri target, and how far two placements land from each other
        miss = miss_distances(pts, ideal)
        pair = pair_distances(pts, rng)

        # and the centroid offset, how far the middle of the cloud sits from the
        # mri target. the systematic part of the miss, the rest is spread
        bias = float(np.linalg.norm(mean - ideal))

        # the two angle distributions. the coil sits tangent to the model head at
        # the mark, so its normal is the ellipsoid normal there, and the handle is
        # turned by the drawn yaw. all in the model frame, angles survive the fit
        normals = np.array([ellipsoid_normal(axes, p) for p in samples[key]])
        tilt, orient = coil_orientation_error(ellipsoid_normal(axes, truth[key]), normals, yaws[key])

        # signed offsets in the local frame at the true site, the bland altman
        # inputs. two along the scalp and one in and out of it
        e1, e2, nvec = tangent_basis(axes, truth[key])
        local = np.stack([model_offsets @ e1, model_offsets @ e2, model_offsets @ nvec], axis=1)

        sites.append({
            "name": t.name,
            "label": t.label,
            "color": t.color,
            "optional": bool(getattr(t, "optional", False)),
            "bias_mm": round(bias, 3),
            "truth": _vec(ideal),
            "mean": _vec(mean),
            "sigmas": _vec(sigmas),
            # rows are the three principal directions, paired with sigmas
            "axes": [_vec(eigvec[:, i]) for i in range(3)],
            "samples": [_vec(p) for p in pts[pick]],
            "rms_mm": round(float(np.sqrt((miss**2).mean())), 3),
            "p95_mm": round(float(np.percentile(miss, 95)), 3),
            # every draw, not just the shipped subset, so the histograms are the
            # real thing. a few hundred floats per site
            "miss_mm": _list(miss),
            "miss_stats": summary(miss),
            "pair_mm": _list(pair),
            "pair_stats": summary(pair),
            "tilt_deg": _list(tilt),
            "tilt_stats": summary(tilt),
            "orient_deg": _list(orient),
            "orient_stats": summary(orient),
            "bland_altman": {
                name: bland_altman(local[:, i]) for i, name in enumerate(COMPONENTS)
            },
        })

    return {
        "space": "RAS_mm",
        "n_draws": n_draws,
        "n_show": int(min(n_show, n_keep, n_draws)),
        "n_keep": int(min(n_keep, n_draws)),
        "shells": list(SHELLS),
        "fit_scale": round(scale, 4),
        "fit_residual_mm": round(resid, 2),
        "nav_tre_mm": NAV_TRE_MM,
        "noise": {
            "landmark_tangent": noise.landmark_tangent,
            "landmark_normal": noise.landmark_normal,
            "mark_tangent": noise.mark_tangent,
            "mark_normal": noise.mark_normal,
            "tape": noise.tape,
            "coil_yaw": noise.coil_yaw,
            "shape": noise.shape,
        },
        "sites": sites,
    }


def export_monte_carlo(scene, out_dir, **kw):
    """Write montecarlo.json next to the meshes for one subject."""

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = monte_carlo_cloud(scene.landmarks, scene.targets, **kw)
    with open(out_dir / "montecarlo.json", "w") as f:
        json.dump(payload, f, indent=2)

    return out_dir / "montecarlo.json", payload


def _vec(v):
    return [round(float(x), 3) for x in np.asarray(v)]


def _list(v):
    return [round(float(x), 3) for x in np.asarray(v)]
