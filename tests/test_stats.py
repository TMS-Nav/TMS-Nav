# the statsmodels helpers in src/stats.py checked against the textbook formulas,
# written out here by hand, and against simulation where a formula is not enough.
#
#   python -m pytest tests/          if pytest is installed
#   python tests/test_stats.py       otherwise, the main at the bottom runs them all
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats import (  # noqa: E402
    Z95,
    below_threshold,
    bias_test,
    bland_altman,
    describe,
    equivalence,
    fdr,
    site_mixed_model,
)


def _samples(n_cases=300, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_cases):
        n = int(rng.integers(3, 40))
        yield rng.normal(rng.normal(0.0, 2.0), rng.uniform(0.2, 4.0), n)


def test_describe_is_the_textbook_numbers():
    for x in _samples(100):
        n, m, s = len(x), x.mean(), x.std(ddof=1)
        d = describe(x)
        half = sps.t.ppf(0.975, n - 1) * s / np.sqrt(n)
        assert d["n"] == n
        assert abs(d["mean"] - m) < 1e-12 and abs(d["sd"] - s) < 1e-12
        assert np.allclose(d["ci95"], [m - half, m + half], atol=1e-12)
        assert abs(d["rms"] - np.sqrt((x**2).mean())) < 1e-12
        assert d["max"] == x.max()
        # statsmodels quantiles step through the data, the median matches exactly
        assert abs(d["median"] - np.median(x)) < 1e-12
    # a single value does not break it
    assert describe([2.0])["ci95"] == [2.0, 2.0]


def test_bias_test_is_the_paired_t_test():
    for x in _samples():
        n, m, s = len(x), x.mean(), x.std(ddof=1)
        t = m / (s / np.sqrt(n))
        b = bias_test(x)
        assert abs(b["t"] - t) < 1e-9
        assert abs(b["p"] - 2.0 * sps.t.sf(abs(t), n - 1)) < 1e-12
        assert b["df"] == n - 1
        # and the same as testing one method against the other with a paired test
        y = np.random.default_rng(1).normal(50, 10, n)
        ref = sps.ttest_rel(y + x, y)
        assert abs(b["p"] - ref.pvalue) < 1e-10


def test_sign_test_counts_signs():
    for x in _samples(100):
        pos, neg = int((x > 0).sum()), int((x < 0).sum())
        ref = sps.binomtest(pos, pos + neg, 0.5).pvalue
        assert abs(bias_test(x)["p_sign"] - ref) < 1e-12


def test_below_threshold_is_one_sided_and_agrees_with_the_90_ci():
    for x in _samples():
        n, m, s = len(x), x.mean(), x.std(ddof=1)
        thr = 2.0
        t = (m - thr) / (s / np.sqrt(n))
        b = below_threshold(x, thr)
        assert abs(b["t"] - t) < 1e-9
        assert abs(b["p"] - sps.t.cdf(t, n - 1)) < 1e-12
        assert b["below"] == (b["ci90"][1] < thr)


def test_tost_is_two_one_sided_tests_and_agrees_with_the_90_ci():
    for x in _samples():
        n, m, s = len(x), x.mean(), x.std(ddof=1)
        se = s / np.sqrt(n)
        margin = 2.0
        p_lower = sps.t.sf((m + margin) / se, n - 1)
        p_upper = sps.t.cdf((m - margin) / se, n - 1)
        e = equivalence(x, margin)
        assert abs(e["p"] - max(p_lower, p_upper)) < 1e-12
        lo, hi = e["ci90"]
        assert e["equivalent"] == (lo > -margin and hi < margin)


def test_bland_altman_is_bias_and_1_96_sd():
    for x in _samples(100):
        n, m, s = len(x), x.mean(), x.std(ddof=1)
        b = bland_altman(x)
        assert abs(b["bias"] - m) < 1e-12 and abs(b["sd"] - s) < 1e-12
        assert abs(b["loa_lo"] - (m - 1.959964 * s)) < 1e-5
        assert abs(b["loa_hi"] - (m + 1.959964 * s)) < 1e-5
        # bland and altman 1986, ci on a limit
        half = sps.t.ppf(0.975, n - 1) * s * np.sqrt(1 / n + Z95**2 / (2 * (n - 1)))
        assert np.allclose(b["loa_hi_ci"], [b["loa_hi"] - half, b["loa_hi"] + half], atol=1e-10)
        assert np.allclose(b["bias_ci"], describe(x)["ci95"], atol=1e-12)


def test_fdr_is_benjamini_hochberg():
    rng = np.random.default_rng(3)
    for _ in range(100):
        p = rng.uniform(0, 1, int(rng.integers(1, 30))) ** 3
        m = len(p)
        order = np.argsort(p)
        ranked = p[order] * m / np.arange(1, m + 1)
        adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(0, 1)
        ref = np.empty(m)
        ref[order] = adj
        q, reject = fdr(p)
        assert np.allclose(q, ref, atol=1e-12)
        assert reject == [bool(v <= 0.05) for v in ref]


def test_type_one_error_by_simulation():
    # the tests reject about 5% of the time when their null is true, and no more
    from statsmodels.stats.weightstats import DescrStatsW

    rng = np.random.default_rng(11)
    reps, n = 20000, 12
    x = rng.normal(0.0, 1.7, size=(n, reps))           # no shift at all
    p = DescrStatsW(x).ttest_mean(0.0)[1]
    rate = float((p < 0.05).mean())
    assert abs(rate - 0.05) < 0.006, rate

    y = rng.normal(2.0, 1.7, size=(n, reps))           # shift sitting right on the margin
    p = DescrStatsW(y).ttost_mean(-2.0, 2.0)[0]
    rate = float((p < 0.05).mean())
    assert rate < 0.056, rate

    # and the single sample helpers give the same p as the column version
    for k in range(20):
        assert abs(bias_test(x[:, k])["p"] - DescrStatsW(x[:, k]).ttest_mean(0.0)[1]) < 1e-12


def test_mixed_model_recovers_known_site_means():
    rng = np.random.default_rng(5)
    truth = {"C3": 4.0, "F4": 6.5, "SMA": 3.0}
    rows = []
    for s in range(40):
        u = rng.normal(0.0, 1.0)                      # this head sits a bit high or low
        for site, mu in truth.items():
            for rep in (1, 2):
                rows.append({"subject": f"S{s}", "site": site, "rep": rep,
                             "value": mu + u + rng.normal(0.0, 0.6)})
    df = pd.DataFrame(rows)
    mm = site_mixed_model(df, "value")
    assert mm["converged"] and mm["n_subjects"] == 40 and mm["df"] == 2
    for site, mu in truth.items():
        est = mm["sites"][site]
        assert abs(est["mean"] - mu) < 4 * est["se"], (site, est)
        assert est["ci95"][0] < est["mean"] < est["ci95"][1]
    assert mm["site_matters"] and mm["p"] < 1e-6
    assert 0.6 < mm["subject_sd"] < 1.5 and 0.45 < mm["residual_sd"] < 0.75

    # the wald statistic by hand, off the fitted means and their covariance
    import statsmodels.formula.api as smf
    fit = smf.mixedlm("value ~ 0 + C(site)", df, groups=df["subject"]).fit(reml=True)
    b = fit.fe_params.to_numpy()
    C = np.asarray(fit.cov_params())[:3, :3]
    R = np.array([[-1.0, 1.0, 0.0], [-1.0, 0.0, 1.0]])
    w = float((R @ b) @ np.linalg.solve(R @ C @ R.T, R @ b))
    assert abs(w - mm["wald_chi2"]) < 1e-6 * max(1.0, w)
    assert abs(sps.chi2.sf(w, 2) - mm["p"]) < 1e-12

    # too little data says so instead of fitting
    assert "note" in site_mixed_model(df[df["subject"].isin(["S0", "S1"])], "value")


def test_power_matches_the_noncentral_t():
    # statsmodels TTestPower against the noncentral t walk sample_size.py used to do
    from statsmodels.stats.power import TTestPower

    for spread in (1.25, 1.72, 2.14, 2.61, 5.0):
        n = int(np.ceil(TTestPower().solve_power(effect_size=1.5 / spread, alpha=0.05, power=0.8)))
        for walk in range(2, 10000):
            df = walk - 1
            ncp = 1.5 * np.sqrt(walk) / spread
            crit = sps.t.ppf(0.975, df)
            if sps.nct.sf(crit, df, ncp) + sps.nct.cdf(-crit, df, ncp) >= 0.8:
                break
        assert n == walk, (spread, n, walk)


TESTS = [
    test_describe_is_the_textbook_numbers,
    test_bias_test_is_the_paired_t_test,
    test_sign_test_counts_signs,
    test_below_threshold_is_one_sided_and_agrees_with_the_90_ci,
    test_tost_is_two_one_sided_tests_and_agrees_with_the_90_ci,
    test_bland_altman_is_bias_and_1_96_sd,
    test_fdr_is_benjamini_hochberg,
    test_type_one_error_by_simulation,
    test_mixed_model_recovers_known_site_means,
    test_power_matches_the_noncentral_t,
]

if __name__ == "__main__":
    failed = 0
    for t in TESTS:
        print(f"{t.__name__} ...")
        try:
            t()
            print("  ok")
        except Exception as e:      # keep going so every failure is seen at once
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}")
    print(f"\n{len(TESTS) - failed} of {len(TESTS)} passed")
    sys.exit(1 if failed else 0)
