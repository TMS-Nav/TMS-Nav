# the numbers behind the monte carlo cloud, and the tests that aim 1 reports.
#
# every draw is one cap laid on the same head, so the cloud answers four questions
# about a target. how far is a placement from where the mri says the target is, how
# far apart do two placements of the same target land (session to session), how much
# does the coil face tilt because the mark is off, and how far off is the whole coil
# orientation once the handle is turned too. all four are distributions, and the
# figure here shows them as such.
#
# the statistics live here as well, every one of them a call into statsmodels. each
# helper answers one question and its docstring says which call does the work. they
# run on SUBJECTS, one number per subject per site, never on monte carlo draws. see
# the note above describe for why, it matters.
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import t as t_dist
from statsmodels.stats.descriptivestats import sign_test
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.weightstats import DescrStatsW
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# --- the thresholds aim 1 is judged against, one copy for the whole project -------
# smallest systematic shift worth detecting, the effect the sample size is powered
# for. the paired t test asks whether the eeg minus mri offset is 0, this is how big
# an offset it has to be able to see
BIAS_MM = 1.5
# how close counts as equivalent, mm. the tost margin on each signed offset, and the
# line the average displacement is tested against
EQ_MARGIN_MM = 2.0
# the same two jobs for angles, degrees. signed handle yaw and total orientation
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


def anatomical_axes(outward_normal):
    """(ap, lr, n) at a scalp point: forward and rightward laid flat on the scalp, and out.

    ap is RAS anterior (+y) with its part along the normal taken out, lr = ap x n,
    which points to the subject's right, and n is the outward normal. an offset
    dotted with them reads as mm forward, mm to the right and mm further out, the
    same three numbers whatever the site. works in any frame with x right, y front,
    z up, which covers scanner RAS and the model head
    """

    n = np.asarray(outward_normal, dtype=float)
    n = n / np.linalg.norm(n)
    ap = np.array([0.0, 1.0, 0.0]) - n * n[1]
    if np.linalg.norm(ap) < 1e-6:
        # normal pointing straight forward or back, not a study site, use up instead
        ap = np.array([0.0, 0.0, 1.0]) - n * n[2]
    ap = ap / np.linalg.norm(ap)
    return ap, np.cross(ap, n), n


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


# --- the statistics ------------------------------------------------------------------
# why the tests run on subjects and never on draws. the standard error is s/sqrt(n),
# and on monte carlo draws n is how many times you chose to run the loop, so the p
# value is something you set rather than something you measure. simulate ten times
# longer and p drops by orders of magnitude without a single new head being
# measured. the simulation is there to estimate the spread. the tests run on real
# subjects, one number per subject per site (the mean of their reps), so n is people.

# the 97.5% point of the normal, the number bland and altman round to 1.96
Z95 = 1.959963984540054


def _descr(x):
    # ddof=1 so .std is the sample sd. the tests and cis in DescrStatsW correct for
    # it themselves, so they come out the same either way
    return DescrStatsW(np.asarray(x, dtype=float), ddof=1)


def _round(out, digits):
    if digits is None:
        return out

    def r(v):
        if isinstance(v, (list, tuple)):
            return [r(u) for u in v]
        if isinstance(v, float):
            return round(v, digits)
        return v

    return {k: r(v) for k, v in out.items()}


def describe(x, ci=True, digits=None):
    """What the numbers look like. n, mean, sd, median, rms, p95, max, and the 95% ci.

    statsmodels DescrStatsW: mean, std, quantile, tconfint_mean. the ci is the range
    of true means the data do not rule out, t based. leave it out (ci=False) on
    monte carlo draws, where it only shrinks with the number of draws
    """

    x = np.asarray(x, dtype=float)
    d = _descr(x)
    n = len(x)
    median, p95 = (float(v) for v in d.quantile([0.5, 0.95], return_pandas=False))
    out = {
        "n": int(n),
        "mean": float(d.mean),
        "sd": float(d.std) if n > 1 else 0.0,
        "median": median,
        "rms": float(np.sqrt((x**2).mean())),
        "p95": p95,
        "max": float(x.max()),
    }
    if ci:
        lo, hi = d.tconfint_mean(alpha=0.05) if n > 1 else (out["mean"], out["mean"])
        out["ci95"] = [float(lo), float(hi)]
    return _round(out, digits)


def below_threshold(x, threshold, alpha=0.05):
    """Is the AVERAGE smaller than the threshold? One sided t test.

    statsmodels DescrStatsW.ttest_mean(threshold, alternative="smaller"). the null
    is that the true mean is at or above the threshold, so a small p says it is
    below. for a distance or a total angle, which cannot go negative, this is the
    whole of what an equivalence test can say. the same verdict reads off the 90%
    ci: its upper end is the one sided 95% bound, and the mean is below the line
    exactly when that bound is. note it is about the mean, one placement can still
    land further out, the limits of agreement are what speak to that
    """

    d = _descr(x)
    t, p, df = d.ttest_mean(threshold, alternative="smaller")
    lo, hi = d.tconfint_mean(alpha=2 * alpha)
    return {
        "threshold": float(threshold),
        "t": float(t), "df": float(df), "p": float(p),
        "ci90": [float(lo), float(hi)],
        "below": bool(p < alpha),
    }


def bias_test(diff, alpha=0.05):
    """Is there a systematic eeg minus mri shift? Paired t test of the differences vs 0.

    statsmodels DescrStatsW.ttest_mean(0), two sided. diff is already eeg minus mri
    per subject, so a one sample test on it is the paired t test. p_sign is the
    same question asked by statsmodels sign_test, which only counts how many
    differences are above and below 0, so it does not need the differences to be
    normal. if the two disagree, trust neither on its own and look at the data
    """

    d = _descr(diff)
    t, p, df = d.ttest_mean(0.0)
    _, p_sign = sign_test(np.asarray(diff, dtype=float), mu0=0.0)
    return {"t": float(t), "df": float(df), "p": float(p), "p_sign": float(p_sign),
            "shift": bool(p < alpha)}


def equivalence(diff, margin, alpha=0.05):
    """Is the systematic shift inside +/- margin? Two one sided tests, TOST.

    statsmodels DescrStatsW.ttost_mean(-margin, margin). a plain t test that fails to
    reject says nothing, absence of evidence. tost turns the question round, the
    null is that the shift is at least as big as the margin on one side or the
    other, and rejecting both is a positive claim that it is not. p is the larger of
    the two one sided p values. the same verdict: the 90% ci sits inside the margin
    """

    d = _descr(diff)
    p, _, _ = d.ttost_mean(-margin, margin)
    lo, hi = d.tconfint_mean(alpha=2 * alpha)
    return {"margin": float(margin), "p": float(p), "ci90": [float(lo), float(hi)],
            "equivalent": bool(p < alpha)}


def bland_altman(diff, alpha=0.05, ci=True, digits=None):
    """Bias and 95% limits of agreement of a set of signed differences.

    bias is the mean difference, the limits are bias +/- 1.96 sd, where 95% of single
    differences are expected to land. that makes the limits the answer to "how far
    apart can the two methods be for one person", which is what gets compared with
    the clinical threshold. mean, sd and the bias ci are statsmodels DescrStatsW.

    the ci on each limit is bland and altman 1986, se = sd * sqrt(1/n + 1.96^2/(2(n-1))).
    it needs a t critical value, and statsmodels has no public function for one,
    so that single quantile comes from scipy.stats.t.ppf
    """

    x = np.asarray(diff, dtype=float)
    d = _descr(x)
    n = len(x)
    bias = float(d.mean)
    sd = float(d.std) if n > 1 else 0.0
    out = {
        "n": int(n),
        "bias": bias,
        "sd": sd,
        "loa_lo": bias - Z95 * sd,
        "loa_hi": bias + Z95 * sd,
    }
    if ci and n > 2:
        lo, hi = d.tconfint_mean(alpha=alpha)
        tc = float(t_dist.ppf(1.0 - alpha / 2.0, n - 1))
        half = tc * sd * np.sqrt(1.0 / n + Z95**2 / (2.0 * (n - 1)))
        out["bias_ci"] = [float(lo), float(hi)]
        out["loa_lo_ci"] = [out["loa_lo"] - half, out["loa_lo"] + half]
        out["loa_hi_ci"] = [out["loa_hi"] - half, out["loa_hi"] + half]
    return _round(out, digits)


def fdr(pvals, q=0.05):
    """Benjamini hochberg adjusted p values (q values) and which survive at level q.

    statsmodels multipletests(method="fdr_bh"). sort the p values, the k-th smallest
    is compared with k/m * q, so the more tests there are the stronger each one has
    to be. a q value below 0.05 means the test survives the correction
    """

    p = np.asarray(pvals, dtype=float)
    if len(p) == 0:
        return [], []
    reject, qvals, _, _ = multipletests(p, alpha=q, method="fdr_bh")
    return [float(v) for v in qvals], [bool(v) for v in reject]


def site_mixed_model(df, value, subject="subject", site="site", alpha=0.05):
    """Does the error depend on the target? Linear mixed model, target fixed, subject random.

    statsmodels mixedlm("value ~ 0 + C(site)", groups=subject), fit by reml. each
    subject contributes several rows (sites, reps), and rows from one head are not
    independent, so every subject gets its own random intercept. the fixed part is
    one mean per site. the question "does site matter" is a wald test that all the
    site means are equal. warnings from the fit (a random intercept pinned at 0, no
    convergence) are passed back in the result rather than hidden
    """

    data = pd.DataFrame({
        "subject": df[subject].astype(str),
        "site": df[site].astype(str),
        "value": pd.to_numeric(df[value], errors="coerce"),
    }).dropna()
    sites = sorted(data["site"].unique())
    n_subj = int(data["subject"].nunique())
    if len(sites) < 2 or n_subj < 3:
        return {"note": f"needs 2 sites and 3 subjects, has {len(sites)} and {n_subj}"}

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = smf.mixedlm("value ~ 0 + C(site)", data, groups=data["subject"]).fit(reml=True)

        # all site means equal: mean_i - mean_0 = 0 for every other site. the
        # columns past the fixed effects belong to the random intercept and stay 0.
        # the covariance of that variance term is often nan (it sits on its 0
        # boundary), and 0 * nan is still nan, so those rows are zeroed. the test
        # only ever reads the fixed effect block
        k_fe = len(fit.fe_params)
        R = np.zeros((k_fe - 1, len(fit.params)))
        for i in range(1, k_fe):
            R[i - 1, 0], R[i - 1, i] = -1.0, 1.0
        cov = np.asarray(fit.cov_params(), dtype=float).copy()
        if np.isnan(cov[:k_fe, :k_fe]).any():
            return {"note": "the fixed effect covariance did not come out, the model did not fit"}
        cov[k_fe:, :] = 0.0
        cov[:, k_fe:] = 0.0
        wald = fit.wald_test(R, cov_p=cov, scalar=True, use_f=False)

    ci = fit.conf_int(alpha=alpha)
    means = {}
    for name, est in fit.fe_params.items():
        key = name.split("[", 1)[1].rstrip("]")
        means[key] = {"mean": float(est), "se": float(fit.bse_fe[name]),
                      "ci95": [float(ci.loc[name, 0]), float(ci.loc[name, 1])]}

    p = float(np.squeeze(wald.pvalue))
    return {
        "n_rows": int(len(data)),
        "n_subjects": n_subj,
        "sites": means,
        "subject_sd": float(np.sqrt(max(float(fit.cov_re.iloc[0, 0]), 0.0))),
        "residual_sd": float(np.sqrt(fit.scale)),
        "wald_chi2": float(np.squeeze(wald.statistic)),
        "df": int(k_fe - 1),
        "p": p,
        "site_matters": bool(p < alpha),
        "converged": bool(fit.converged),
        "warnings": sorted({str(w.message).splitlines()[0] for w in caught}),
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

    # the statsmodels answers against the textbook formulas. tests/test_stats.py
    # does this properly, this is the quick look
    x = rng.normal(0.4, 1.7, size=24)
    n, m, s = len(x), x.mean(), x.std(ddof=1)
    bt = bias_test(x)
    assert abs(bt["t"] - m / (s / np.sqrt(n))) < 1e-9, "paired t is not mean / (sd / sqrt n)"
    eq = equivalence(x, EQ_MARGIN_MM)
    lo, hi = eq["ci90"]
    assert eq["equivalent"] == (lo > -EQ_MARGIN_MM and hi < EQ_MARGIN_MM), \
        "tost verdict and the 90% ci disagree"
    ba = bland_altman(x)
    assert abs(ba["loa_hi"] - ba["loa_lo"] - 2 * Z95 * s) < 1e-9
    print("paired t, tost and bland altman ok")
