# monte carlo cloud for the three.js viewer.
#
# the probabilistic model lives in its own ellipsoid head frame, the viewer lives in
# the subject scanner RAS, so the cloud has to be landed on the subject. it is landed
# with the same eeg landmarks that register everything else, via the umeyama fit in
# src/align.py. one uniform scale, so a bigger subject stretches the cloud with the
# head rather than leaving it the wrong size.
import json
from pathlib import Path

import numpy as np

from src.align import similarity_transform
from src.landmark_noise import NoiseModel, simulate_caps
from src.ten_twenty import HeadDimensions

# the model names its anchors Nz/Iz, targets.standard_landmarks says nasion/inion
LANDMARK_ALIAS = {"nasion": "Nz", "inion": "Iz", "Cz": "Cz", "LPA": "LPA", "RPA": "RPA"}

# nominal adult head, mm. the umeyama scale below soaks up most of the difference
# between this and the actual subject, and the fitted scale gets reported so it is
# visible when it does not
NOMINAL_DIMS = HeadDimensions(nasion_inion=360.0, lpa_rpa=350.0, circumference=570.0)

# how many of the draws to actually ship. enough to read as a cloud, few enough that
# the json stays small and the individual dots stay distinguishable
N_SHOW = 50

# mahalanobis radii to draw shells at. 1/2/3 sigma
SHELLS = (1.0, 2.0, 3.0)


def monte_carlo_cloud(landmarks, targets, dims=NOMINAL_DIMS, noise=None,
                      n_draws=1200, n_show=N_SHOW, seed=0):
    """Per target cloud, mean and covariance ellipsoid, all in subject RAS mm."""

    rng = np.random.default_rng(seed)
    noise = noise or NoiseModel()

    truth, samples = simulate_caps(dims, noise, n_draws, rng)

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
        key = t.label  # viewer labels are SMA, F3, F4, C3, C4, same as the model names
        if key not in samples:
            continue

        # the cloud is CENTRED on the subject own target, not on the transformed model
        # target. the registration is only trusted for orientation and scale, because
        # standard_landmarks picks its fiducials off extreme scalp vertices and lands
        # them tens of mm from where the ellipsoid wants them. the model is the right
        # tool for the SHAPE of the uncertainty, the mri is the right tool for WHERE
        # the target is, so each is used for the thing it is good at
        offsets = (samples[key] - truth[key]) @ rot.T
        pts = np.asarray(t.contact, dtype=float) + offsets
        mean = pts.mean(axis=0)
        cov = np.cov(pts.T)

        # principal axes of the scatter, largest first
        eigval, eigvec = np.linalg.eigh(cov)
        order = np.argsort(eigval)[::-1]
        eigval, eigvec = np.clip(eigval[order], 0.0, None), eigvec[:, order]
        sigmas = np.sqrt(eigval)

        pick = rng.choice(len(pts), size=min(n_show, len(pts)), replace=False)

        ideal = np.asarray(t.contact, dtype=float)
        miss = np.linalg.norm(pts - ideal, axis=1)

        sites.append({
            "name": t.name,
            "label": t.label,
            "color": t.color,
            "truth": _vec(ideal),
            "mean": _vec(mean),
            "sigmas": _vec(sigmas),
            # rows are the three principal directions, paired with sigmas
            "axes": [_vec(eigvec[:, i]) for i in range(3)],
            "samples": [_vec(p) for p in pts[pick]],
            "rms_mm": round(float(np.sqrt((miss**2).mean())), 3),
            "p95_mm": round(float(np.percentile(miss, 95)), 3),
        })

    return {
        "space": "RAS_mm",
        "n_draws": n_draws,
        "n_show": int(min(n_show, n_draws)),
        "shells": list(SHELLS),
        "fit_scale": round(scale, 4),
        "fit_residual_mm": round(resid, 2),
        "noise": {
            "landmark_tangent": noise.landmark_tangent,
            "landmark_normal": noise.landmark_normal,
            "mark_tangent": noise.mark_tangent,
            "mark_normal": noise.mark_normal,
            "tape": noise.tape,
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
