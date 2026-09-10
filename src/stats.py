# the numbers behind the monte carlo cloud, and the tests that aim 1 reports.
#
# every draw is one cap laid on the same head, so the cloud answers four questions
# about a target. how far is a placement from where the mri says the target is, how
# far apart do two placements of the same target land (session to session), how much
# does the coil face tilt because the mark is off, and how far off is the whole coil
# orientation once the handle is turned too. all four are distributions, and the
# figure here shows them as such.
#
# the subject level tests live here as well. they run on SUBJECTS, one number per
# head, never on draws. see subject_test for why, it matters.
import numpy as np
from scipy import stats as sps
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --- the thresholds aim 1 is judged against, one copy for the whole project -------
# smallest average offset worth detecting, the null for the one sample t test on a
# displacement magnitude, which cannot be negative so testing against 0 is meaningless
BIAS_MM = 1.5
# how close counts as equivalent, the tost margin on displacement
EQ_MARGIN_MM = 2.0
# and on coil orientation, degrees. total rotation between the two coil frames
EQ_MARGIN_DEG = 10.0
# brainsight's own registration error, target registration error, taken as fixed
# and added in quadrature to the cap error. double check against the manual
NAV_TRE_MM = 0.75


# --- distances --------------------------------------------------------------------
def miss_distances(pts, ideal):
    """Distance from each draw to the mri target, mm."""

    return np.linalg.norm(np.asarray(pts, dtype=float) - np.asarray(ideal, dtype=float), axis=1)


def pair_distances(pts, rng):
    """Distance between two independent placements of the same target, mm.

    the draws are shuffled and taken two at a time, so every pair is two honest
    repeats of the whole cap procedure and no draw is used twice
    """

    pts = np.asarray(pts, dtype=float)
    idx = rng.permutation(len(pts))
    half = len(pts) // 2
    return np.linalg.norm(pts[idx[:half]] - pts[idx[half:2 * half]], axis=1)


# --- angles -----------------------------------------------------------------------
def angle_between(a, b):
    """Angle between two vectors or two stacks of vectors, degrees."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a, axis=-1, keepdims=True)
    b = b / np.linalg.norm(b, axis=-1, keepdims=True)
    cos = np.clip((a * b).sum(axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def rotation_between(a, b):
    """The smallest rotation taking unit vector a onto unit vector b, 3x3."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # antiparallel, any axis perpendicular to a will do
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return rotation_about(axis, 180.0)
    # rodrigues
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s * s))


def rotation_about(axis, deg):
    """Rotation of deg degrees about a unit axis, 3x3."""

    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    th = np.radians(deg)
    kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(th) * kx + (1.0 - np.cos(th)) * kx @ kx


def rotation_angle(R):
    """The single angle of a rotation matrix, degrees. arccos((tr R - 1) / 2)."""

    R = np.asarray(R, dtype=float)
    tr = np.trace(R, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def coil_orientation_error(true_normal, normals, yaw_deg):
    """Tilt and total orientation error of a coil sat tangent to a mislaid mark.

    the coil face follows the scalp, so a mark that is off tilts the face by the
    angle between the surface normal at the mark and at the true site. then the
    handle is turned by yaw about the new normal. the total orientation difference
    is the angle of the composed rotation, which is what a 3x3 out of brainsight
    would give when compared to the reference pose.

    returns tilt and total, both in degrees, one per draw
    """

    n0 = np.asarray(true_normal, dtype=float)
    n0 = n0 / np.linalg.norm(n0)
    normals = np.asarray(normals, dtype=float)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    yaw_deg = np.asarray(yaw_deg, dtype=float)

    tilt = angle_between(normals, n0)
    total = np.empty(len(normals))
    for i, (n1, yaw) in enumerate(zip(normals, yaw_deg)):
        R = rotation_about(n1, yaw) @ rotation_between(n0, n1)
        total[i] = rotation_angle(R)
    return tilt, total


# --- descriptive ------------------------------------------------------------------
def summary(d):
    """The handful of numbers that describe a distance or angle distribution."""

    d = np.asarray(d, dtype=float)
    return {
        "n": int(len(d)),
        "mean": round(float(d.mean()), 3),
        "sd": round(float(d.std(ddof=1)), 3) if len(d) > 1 else 0.0,
        "median": round(float(np.median(d)), 3),
        "rms": round(float(np.sqrt((d**2).mean())), 3),
        "p95": round(float(np.percentile(d, 95)), 3),
        "max": round(float(d.max()), 3),
    }


def bland_altman(diff, alpha=0.05):
    """Bias and limits of agreement of a set of signed differences.

    bias is the mean, the limits are bias +/- 1.96 sd, so 95% of single differences
    are expected inside them. on monte carlo draws this is a statement about where
    ONE placement lands, a tolerance, which is the thing a patient experiences. the
    ci on the bias uses t with n-1 df
    """

    d = np.asarray(diff, dtype=float)
    n = len(d)
    bias = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    z = float(sps.norm.ppf(1.0 - alpha / 2.0))
    half = float(sps.t.ppf(1.0 - alpha / 2.0, n - 1)) * sd / np.sqrt(n) if n > 1 else 0.0
    return {
        "n": int(n),
        "bias": round(bias, 3),
        "sd": round(sd, 3),
        "loa_lo": round(bias - z * sd, 3),
        "loa_hi": round(bias + z * sd, 3),
        "bias_ci": [round(bias - half, 3), round(bias + half, 3)],
    }


# --- the subject level tests -----------------------------------------------------
def subject_test(values, mu0=BIAS_MM, alpha=0.05):
    """One sample t test of subject level means against mu0. n is SUBJECTS.

    do not run this on monte carlo draws. the standard error is s/sqrt(n) and n
    there is the number of times you chose to run the loop, so the p value is
    something you set rather than something you measure. simulate ten times longer
    and p drops by orders of magnitude without a single new head being measured.
    the simulation is there to estimate the spread, sigma. the spread then feeds
    the sample size formula, and the t test runs once, on real subjects.

    mu0 is whatever null the question needs. for a signed difference between two
    localization methods that is 0. for a displacement magnitude, which cannot be
    negative, testing against 0 is meaningless, so use the bias threshold instead.
    """

    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        raise ValueError(f"need at least 2 subjects for a t test, got {n}")

    df = n - 1
    mean = float(v.mean())
    se = float(v.std(ddof=1)) / np.sqrt(n)   # ddof=1, the sample sd, not the population one

    t = (mean - mu0) / se if se > 0 else float("inf")
    p = 2.0 * float(sps.t.sf(abs(t), df))
    half = float(sps.t.ppf(1.0 - alpha / 2.0, df)) * se

    return {
        "n": n, "df": df, "mean": mean, "sd": float(v.std(ddof=1)), "se": se,
        "t": float(t), "p": p, "mu0": float(mu0),
        "ci": (mean - half, mean + half),
    }


def subject_tost(values, margin=EQ_MARGIN_MM, alpha=0.05):
    """Two one sided tests. Rejecting BOTH is what buys you 'equivalent'.

    a plain t test that fails to reject says nothing, absence of evidence. tost
    turns the question round, the null is that the difference is at least as big
    as the margin, and rejecting that is a positive claim of equivalence. the
    equivalent statement is that the 90% ci (1 - 2 alpha) sits inside the margin,
    so the ci is returned too, it is the thing the proposal reports.
    """

    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        raise ValueError(f"need at least 2 subjects for tost, got {n}")
    df = n - 1
    mean = float(v.mean())
    se = float(v.std(ddof=1)) / np.sqrt(n)

    if se > 0:
        p_lower = float(sps.t.sf((mean + margin) / se, df))   # H0: mean <= -margin
        p_upper = float(sps.t.cdf((mean - margin) / se, df))  # H0: mean >= +margin
    else:
        p_lower = p_upper = 0.0 if abs(mean) < margin else 1.0
    worst = max(p_lower, p_upper)
    half = float(sps.t.ppf(1.0 - alpha, df)) * se

    return {
        "n": n, "mean": mean, "margin": float(margin),
        "p_lower": p_lower, "p_upper": p_upper, "p": worst,
        "ci90": (mean - half, mean + half),
        "equivalent": bool(worst < alpha),
    }


# --- the figure --------------------------------------------------------------------
# the panels of the figure, json key, stats key, title and unit
METRICS = [
    ("miss_mm", "miss_stats", "distance to the MRI target", "mm"),
    ("pair_mm", "pair_stats", "distance between two placements", "mm"),
    ("tilt_deg", "tilt_stats", "coil face tilt", "deg"),
    ("orient_deg", "orient_stats", "coil orientation difference", "deg"),
]


def render_distance_hist(mc, out_path, subject="", bins=30):
    """One row per target, one column per metric, all on shared axes per column.

    thin bars with a gap of surface between them, a solid line at the mean and a
    dotted one at the 95th percentile, and the numbers in the panel title rather
    than on every bar. same colour per target as the viewer uses
    """

    sites = mc["sites"]
    metrics = [m for m in METRICS if m[0] in sites[0]]
    fig, axes = plt.subplots(len(sites), len(metrics),
                             figsize=(4.4 * len(metrics), 1.9 * len(sites) + 0.8),
                             sharex="col", squeeze=False)
    fig.patch.set_facecolor("white")

    for col, (key, stats_key, title, unit) in enumerate(metrics):
        xmax = max(s[stats_key]["max"] for s in sites)
        step = 0.5 if unit == "mm" else 1.0
        edges = np.linspace(0.0, np.ceil(xmax / step) * step, bins + 1)

        for row, site in enumerate(sites):
            ax = axes[row, col]
            d = np.asarray(site[key], dtype=float)
            st = site[stats_key]

            ax.hist(d, bins=edges, color=site["color"], rwidth=0.82, edgecolor="none")
            ax.axvline(st["mean"], color="#333333", lw=1.0)
            ax.axvline(st["p95"], color="#333333", lw=1.0, ls=(0, (1, 2)))

            opt = "  (optional marker)" if site.get("optional") else ""
            ax.set_title(
                f"{site['label']}{opt}   mean {st['mean']:.2f}   sd {st['sd']:.2f}"
                f"   p95 {st['p95']:.2f} {unit}   n={st['n']}",
                fontsize=9, loc="left", color="#333333")
            ax.set_yticks([])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#bbbbbb")
            ax.tick_params(axis="x", colors="#555555", labelsize=8)
            ax.grid(axis="x", color="#eeeeee", lw=0.8)
            ax.set_axisbelow(True)

        axes[-1, col].set_xlabel(f"{title} [{unit}]", fontsize=9, color="#333333")

    head = f"{subject}  " if subject else ""
    fig.suptitle(f"{head}placement error over {mc['n_draws']} simulated caps"
                 f"   (solid = mean, dotted = 95th percentile)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(str(out_path), dpi=110)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    # the rotation helpers against things we know
    rng = np.random.default_rng(0)
    a = np.array([0.0, 0.0, 1.0])
    b = np.array([np.sin(np.radians(7.0)), 0.0, np.cos(np.radians(7.0))])
    R = rotation_between(a, b)
    assert np.allclose(R @ a, b), "rotation_between does not take a onto b"
    assert abs(rotation_angle(R) - 7.0) < 1e-9, "rotation_angle is not the tilt"
    assert abs(rotation_angle(rotation_about(b, 5.0)) - 5.0) < 1e-9
    # zero tilt, pure yaw, the total is the yaw
    tilt, total = coil_orientation_error(a, [a], [5.0])
    assert abs(tilt[0]) < 1e-9 and abs(total[0] - 5.0) < 1e-9
    # tilt and yaw about perpendicular axes, the total is bigger than either
    tilt, total = coil_orientation_error(a, [b], [5.0])
    assert abs(tilt[0] - 7.0) < 1e-9 and total[0] > 7.0 and total[0] < 12.0
    print("rotation helpers ok, tilt 7 + yaw 5 gives total", round(float(total[0]), 3), "deg")

    # hand rolled tests against scipy
    x = rng.normal(1.3, 1.7, size=24)
    tt = subject_test(x, mu0=BIAS_MM)
    ref = sps.ttest_1samp(x, BIAS_MM)
    assert abs(tt["t"] - float(ref.statistic)) < 1e-9
    assert abs(tt["p"] - float(ref.pvalue)) < 1e-12
    eq = subject_tost(x, margin=EQ_MARGIN_MM)
    lo, hi = eq["ci90"]
    assert eq["equivalent"] == (lo > -EQ_MARGIN_MM and hi < EQ_MARGIN_MM), \
        "tost verdict and the 90% ci disagree"
    ba = bland_altman(x)
    assert abs(ba["loa_hi"] - ba["loa_lo"] - 2 * 1.96 * ba["sd"]) < 1e-2
    print("t, tost and bland altman ok")
