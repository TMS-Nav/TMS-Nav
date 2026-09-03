# Ali Rad - probabilistic head model for aim 1
#
# monte carlo over the 10-20 cap. draw the landmark placement and the tape reading
# many times, rebuild the whole cap each time, and look at how far each site wanders
# from where a perfect measurement would have put it. that spread is the cap drawing
# error that analysis/sample_size.py currently just guesses at.
#
# the inference lives at the SUBJECT level, not the draw level. see the note above
# subject_test for why, it matters and he will ask.
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.landmark_noise import (  # noqa: E402
    NoiseModel,
    simulate_caps,
    tangent_basis,
)
from src.ten_twenty import HeadDimensions, electrode_positions, fit_ellipsoid  # noqa: E402

# a typical adult head, mm. nasion to inion over the vertex, ear to ear over the
# vertex, and round through nasion and inion
TRUE_DIMS = HeadDimensions(nasion_inion=360.0, lpa_rpa=350.0, circumference=570.0)

# thresholds, straight out of analysis/sample_size.py so the two agree
BIAS = 1.5        # mm, smallest average offset worth detecting
EQ_MARGIN = 2.0   # mm, how close counts as equivalent

# the two study targets. f3 is the dlpfc proxy from herwig 2003, sma is the midline
# point in front of cz used by the mantovani 2010 protocol
STUDY_SITES = ["F4", "SMA"]

# normal sd as a fraction of tangent sd. bone stops you pressing in, skin does not
# stop you sliding, so the error blob is a flattened disc rather than a ball
RATIO = 0.35


# the simulation loop itself lives in src/landmark_noise.py so the viewer export can
# share it. same function, just not duplicated
simulate = simulate_caps


def displacements(samples, truth):
    """How far each realization of a site sits from the noise free site, mm."""

    return np.linalg.norm(samples - truth, axis=1)


def tolerance_region(samples, truth):
    """The 95% region for a SINGLE cap placement, plus how well it actually covers.

    not a confidence region for the mean. a confidence region shrinks like 1/sqrt(n)
    and describes where the average lands, which is not a thing any patient
    experiences. one patient gets one cap drawn on them, so the region that matters
    is the one holding 95% of individual realizations, and that one does not shrink
    however long the simulation runs.
    """

    mean = samples.mean(axis=0)
    cov = np.cov(samples.T)

    eigval, eigvec = np.linalg.eigh(cov)
    eigval = np.clip(eigval, 0.0, None)

    # most sites here do not scatter in all three directions. a reconstructed nasion,
    # for instance, is always at (0, b, 0), so every bit of landmark noise ends up in
    # the fitted b and the cloud is a line, not a blob. the chi squared has to use the
    # rank of the covariance, not a flat 3, or the region badly over covers
    tol = max(eigval.max(), 1e-30) * 1e-6
    rank = max(int((eigval > tol).sum()), 1)
    scale = float(stats.chi2.ppf(0.95, df=rank))

    semi_axes = np.sqrt(eigval * scale)

    # empirical check on that scaling, should land near 0.95
    centred = samples - mean
    keep = eigvec[:, eigval > tol]
    proj = centred @ keep
    mahal2 = ((proj**2) / eigval[eigval > tol]).sum(axis=1)
    coverage = float((mahal2 <= scale).mean())

    d = displacements(samples, truth)
    return {
        "mean": mean,
        "bias_mm": float(np.linalg.norm(mean - truth)),
        "cov": cov,
        "rank": rank,
        "semi_axes_mm": np.sort(semi_axes)[::-1],
        "axes_dirs": eigvec,
        "rms_mm": float(np.sqrt((d**2).mean())),
        "sd_mm": float(d.std(ddof=1)),
        "p95_mm": float(np.percentile(d, 95)),
        "coverage": coverage,
    }


def subject_test(values, mu0=BIAS, alpha=0.05):
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

    t = (mean - mu0) / se
    p = 2.0 * float(stats.t.sf(abs(t), df))
    half = float(stats.t.ppf(1.0 - alpha / 2.0, df)) * se

    return {
        "n": n, "df": df, "mean": mean, "se": se, "t": float(t), "p": p,
        "ci": (mean - half, mean + half),
    }


def subject_tost(values, margin=EQ_MARGIN, alpha=0.05):
    """Two one sided tests. Rejecting BOTH is what buys you 'equivalent'.

    a plain t test that fails to reject says nothing, absence of evidence. tost
    turns the question round, the null is that the difference is at least as big
    as the margin, and rejecting that is a positive claim of equivalence.
    """

    v = np.asarray(values, dtype=float)
    n = len(v)
    df = n - 1
    mean = float(v.mean())
    se = float(v.std(ddof=1)) / np.sqrt(n)

    p_lower = float(stats.t.sf((mean + margin) / se, df))   # H0: mean <= -margin
    p_upper = float(stats.t.cdf((mean - margin) / se, df))  # H0: mean >= +margin
    worst = max(p_lower, p_upper)

    return {"p_lower": p_lower, "p_upper": p_upper, "p": worst, "equivalent": worst < alpha}


def n_from_formula(sigma, effect, z_alpha=1.96, z_beta=0.84):
    """The closed form n from sample_size.py, repeated here so we can check it."""

    return int(np.ceil((z_alpha + z_beta) ** 2 * sigma**2 / effect**2))


def exact_power(sigma, effect, n, alpha=0.05):
    """Exact power of a two sided one sample t test, no normal approximation.

    under the alternative the t statistic is not a t, it is a NONCENTRAL t with
    noncentrality effect*sqrt(n)/sigma. the closed form in sample_size.py quietly
    swaps that for a shifted normal, which is fine once n is large and is not fine
    at n below about 30.
    """

    df = n - 1
    ncp = effect * np.sqrt(n) / sigma
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return float(stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp))


def n_from_t(sigma, effect, alpha=0.05, power=0.80, n_max=10000):
    """Smallest n whose exact power clears the target.

    solving for n directly would need t quantiles that themselves depend on n, and
    iterating on that oscillates between two neighbouring values instead of
    settling. searching upward has no such problem and gives the exact answer.
    """

    for n in range(2, n_max):
        if exact_power(sigma, effect, n, alpha) >= power:
            return n
    raise ValueError(f"no n below {n_max} reaches power {power} at sigma {sigma}")


def power_by_simulation(sigma, effect, n_subjects, rng, n_reps=20000, alpha=0.05):
    """Empirical power of a one sample t test, the check on n_from_formula."""

    draws = rng.normal(effect, sigma, size=(n_reps, n_subjects))
    mean = draws.mean(axis=1)
    se = draws.std(axis=1, ddof=1) / np.sqrt(n_subjects)
    t = mean / se
    p = 2.0 * stats.t.sf(np.abs(t), n_subjects - 1)
    return float((p < alpha).mean())


def main():
    rng = np.random.default_rng(20260819)
    n_draws = 1200

    print("true head:", TRUE_DIMS)
    axes = fit_ellipsoid(TRUE_DIMS)
    print("semi axes a, b, c:", tuple(round(v, 2) for v in axes), "mm")

    # --- sanity, zero noise must leave every site where it started -----------------
    # not bit exact zero. a draw re-measures the arcs by walking a 2001 point polyline
    # and then refits, so the round trip carries the discretization error of that walk,
    # a few times 1e-5 mm. anything above a micron would be a real bug
    quiet = NoiseModel(landmark_tangent=0.0, landmark_normal=0.0,
                       mark_tangent=0.0, mark_normal=0.0, tape=0.0)
    truth, s0 = simulate(TRUE_DIMS, quiet, 3, rng)
    worst = max(float(displacements(s0[k], truth[k]).max()) for k in truth)
    print()
    print("zero noise, largest displacement anywhere:", f"{worst:.2e}",
          "mm (arc walk discretization floor, not real motion)")
    assert worst < 1e-3, "zero noise moved a site, the deterministic path is not clean"

    # --- the real run --------------------------------------------------------------
    noise = NoiseModel()
    print()
    print(f"{n_draws} draws. landmark sd {noise.landmark_tangent} mm tangent /"
          f" {noise.landmark_normal} mm normal,")
    print(f"  mark sd {noise.mark_tangent} mm tangent / {noise.mark_normal} mm normal,"
          f" tape sd {noise.tape} mm")
    truth, samples = simulate(TRUE_DIMS, noise, n_draws, rng)

    print()
    print(f"{'site':6s} {'rms':>7s} {'p95':>7s} {'bias':>7s} {'rank':>5s} {'cover':>7s}"
          "   95% region semi axes")
    for name in sorted(samples):
        r = tolerance_region(samples[name], truth[name])
        ax = ", ".join(f"{v:.2f}" for v in r["semi_axes_mm"][: r["rank"]])
        star = "  <-" if name in STUDY_SITES else ""
        print(f"{name:6s} {r['rms_mm']:7.2f} {r['p95_mm']:7.2f} {r['bias_mm']:7.2f}"
              f" {r['rank']:5d} {r['coverage']:7.3f}   {ax}{star}")
    print("(rms, p95 and bias in mm. rank is how many directions the site actually")
    print(" scatters in, cover is the fraction of draws inside the 95% region and")
    print(" should sit near 0.95 if the chi squared scaling is right)")

    # --- variance split into tangent and normal -------------------------------------
    # this is the check that the anisotropy actually survives the whole pipeline. the
    # scatter is resolved in the local frame of the true head at each site, so sd_t is
    # sliding along the scalp and sd_n is in and out of it
    print()
    print("scatter resolved in the local surface frame (variance in mm^2):")
    print(f"  {'site':6s} {'sd_t1':>7s} {'sd_t2':>7s} {'sd_n':>7s}   {'var_t':>7s}"
          f" {'var_n':>7s}  {'n/t':>5s}")
    true_axes = fit_ellipsoid(TRUE_DIMS)
    for name in STUDY_SITES + ["Cz", "T3"]:
        e1, e2, nvec = tangent_basis(true_axes, truth[name])
        d = samples[name] - truth[name]
        s1, s2, sn = (float((d @ v).std(ddof=1)) for v in (e1, e2, nvec))
        var_t = s1**2 + s2**2
        print(f"  {name:6s} {s1:7.2f} {s2:7.2f} {sn:7.2f}   {var_t:7.2f}"
              f" {sn**2:7.2f}  {sn / max(s1, 1e-9):5.2f}")
    print("  (sd_t1/sd_t2 are the two tangent directions, sd_n is the normal.)")
    print()
    print(f"  the pen alone was told to draw at ratio {RATIO}. F4 comes back near that,")
    print("  but Cz and T3 come back near 1.0, and that is not a bug. getting the head")
    print("  SIZE wrong moves a site straight in or out along its own normal, and at the")
    print("  vertex and the ears that is exactly where the size error points. so the")
    print("  head model piles extra variance onto the normal direction at those sites")
    print("  and the blob rounds out. F4 sits on a slope, so its size error is mostly")
    print("  tangential there and the disc stays flat.")

    # --- does the spread scale the way it should ------------------------------------
    print()
    print("displacement rms at F4 against landmark sd, should be a straight line:")
    for sd in (0.2, 0.5, 1.0, 2.0):
        sweep = NoiseModel(landmark_tangent=sd, landmark_normal=sd * RATIO,
                           mark_tangent=sd, mark_normal=sd * RATIO, tape=0.0)
        _, s = simulate(TRUE_DIMS, sweep, 400, rng)
        r = float(np.sqrt((displacements(s["F4"], truth["F4"]) ** 2).mean()))
        print(f"  landmark sd {sd:4.1f} mm  ->  F4 rms {r:6.2f} mm   ratio {r / sd:6.2f}")

    # --- the number that feeds sample_size.py ----------------------------------------
    print()
    for name in STUDY_SITES:
        r = tolerance_region(samples[name], truth[name])
        print(f"{name}: mean miss {displacements(samples[name], truth[name]).mean():.2f} mm,"
              f" sd {r['sd_mm']:.2f} mm, rms {r['rms_mm']:.2f} mm,"
              f" 95% of draws within {r['p95_mm']:.2f} mm")

    # rms, not sd. sample_size.py combines the two error sources in quadrature,
    # spread = sqrt(capError**2 + navError**2), which is an rms combination, so the
    # thing that belongs in capError is the rms miss and not the scatter about it
    sigma = float(tolerance_region(samples["F4"], truth["F4"])["rms_mm"])
    print()
    print(f"capError to carry into sample_size.py, rms miss at F4: {sigma:.2f} mm")
    print(" this now covers measuring the head AND drawing the mark on it. it lands in")
    print(" the middle of the old guessed sweep [1.0, 2.0, 2.5], so the guess was sound,")
    print(" it just had no evidence under it. still not in here: the coil being held off")
    print(" the mark or tilted, hair and cap slip, and the head not being an ellipsoid.")

    # --- check the closed form n against simulation -----------------------------------
    print()
    print("checking the sample size formula in sample_size.py by simulation")
    print(f"  {'capError':>9s} {'spread':>7s} {'n normal':>9s} {'power':>7s}"
          f" {'n with t':>9s} {'power':>7s}")
    for cap in (sigma, 1.0, 2.0, 2.5):
        spread = np.sqrt(cap**2 + 0.75**2)   # navError = 0.75, as in the script
        n_z = n_from_formula(spread, BIAS)
        n_t = n_from_t(spread, BIAS)
        print(f"  {cap:9.2f} {spread:7.2f} {n_z:9d} {power_by_simulation(spread, BIAS, n_z, rng):7.3f}"
              f" {n_t:9d} {power_by_simulation(spread, BIAS, n_t, rng):7.3f}")
    print(" the normal quantile formula misses 80% power badly at these small n. it")
    print(" assumes you know the sd, but you are estimating it from the same handful of")
    print(" subjects, so the real test uses t and needs a bigger n. sample_size.py now")
    print(" carries both columns, this is the check that the fix in there is right.")

    # --- the subject level test, on stand in subject means so the wiring is visible -----
    print()
    print("subject level test, stand in numbers until there are real subjects")
    per_subject_sd = float(np.sqrt(sigma**2 + 0.75**2))
    subj = rng.normal(1.30, per_subject_sd, size=24)  # 24 subjects, one mean miss each
    tt = subject_test(subj, mu0=BIAS)
    print(f"  n = {tt['n']} subjects, mean miss {tt['mean']:.2f} mm, 95% CI"
          f" [{tt['ci'][0]:.2f}, {tt['ci'][1]:.2f}] mm")
    print(f"  t = {tt['t']:.2f} on {tt['df']} df, p = {tt['p']:.3f}"
          f"   (null: the mean miss equals the {BIAS} mm we care about)")
    eq = subject_tost(subj, margin=EQ_MARGIN)
    print(f"  TOST against +/- {EQ_MARGIN} mm: p = {eq['p']:.4f},"
          f" equivalent = {eq['equivalent']}")
    print("  (TOST is the one that can make a positive claim. a plain t test that")
    print("   fails to reject only means you did not look hard enough)")

    # hand rolled t against scipy, they must agree
    ref = stats.ttest_1samp(subj, BIAS)
    assert abs(tt["t"] - float(ref.statistic)) < 1e-9, "hand rolled t disagrees with scipy"
    assert abs(tt["p"] - float(ref.pvalue)) < 1e-12, "hand rolled p disagrees with scipy"
    print("  hand rolled t and p match scipy.stats.ttest_1samp")


if __name__ == "__main__":
    main()
