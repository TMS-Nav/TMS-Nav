# fit the mni template brain into a subject head off the eeg landmarks. rough
# similarity transform (rotation, one scale, translation), the template is a
# stand in and not the subject's own brain
import numpy as np

# rough mni scalp fiducials in mm, RAS. approximate but they bracket the head so
# the fit lands the template in about the right place
MNI_FIDUCIALS = {
    "Cz": (0.0, -18.0, 88.0),
    "nasion": (0.0, 85.0, -40.0),
    "inion": (0.0, -115.0, -25.0),
    "LPA": (-82.0, -18.0, -45.0),
    "RPA": (82.0, -18.0, -45.0),
}

# mni brain centroid, used to sanity check the fit against the subject brain blob
MNI_BRAIN_CENTER = (0.0, -18.0, 10.0)


def similarity_transform(src, dst):
    """Umeyama fit, 4x4 mapping src onto dst with rotation, one scale, translation."""

    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)

    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    S = src - mu_s
    D = dst - mu_d

    cov = (D.T @ S) / len(src)
    U, sig, Vt = np.linalg.svd(cov)

    # guard against a reflection sneaking in
    d = np.sign(np.linalg.det(U @ Vt))
    diag = np.diag([1.0, 1.0, d])
    R = U @ diag @ Vt

    var_s = (S**2).sum() / len(src)
    scale = (sig * np.array([1.0, 1.0, d])).sum() / var_s

    t = mu_d - scale * (R @ mu_s)

    m = np.eye(4)
    m[:3, :3] = scale * R
    m[:3, 3] = t
    return m


def fit_mni_to_head(landmarks):
    """4x4 that places the mni template in the subject head using the landmarks."""

    names = [m.name for m in landmarks if m.name in MNI_FIDUCIALS]
    src = [MNI_FIDUCIALS[n] for n in names]
    dst = [next(m.contact for m in landmarks if m.name == n) for n in names]
    return similarity_transform(src, dst)
