# Ali Rad - Target Localization Method Comparison Statistics for aim 1

import math

import sys
from pathlib import Path

import numpy as np
from scipy import stats
from statsmodels.stats.power import TTestPower
from statsmodels.stats.weightstats import DescrStatsW

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.stats import BIAS_MM, EQ_MARGIN_MM, NAV_TRE_MM  # noqa: E402

# in mm. the thresholds live in src/stats.py so every script uses the same ones
navError = NAV_TRE_MM # "neuronavigation error, brainsights own localization/registration noise"
# double check navError against the brainsight manual

# thresholds
bias = BIAS_MM # smallest systematic shift worth detecting (for the paired t-test)
eqMargin = EQ_MARGIN_MM # size of difference to be equiavlent (for TOST)
LOAPrec = 1.0 # limit of agreement precision (for Bland-Altman)

# cap drawing error. 1.55 is the rms miss at SMA out of analysis/head_model_mc.py, F4
# comes out at 1.52, so this is the bigger of the two primary sites. the old guesses
# stay in the sweep so they are still comparable
capErrors = [1.0, 1.55, 2.0, 2.5]

# the spread above is measurement noise only, the cap plus brainsight. the real eeg
# minus mri difference also carries how far the 10-20 spot sits from the R01's own
# mri target, and that changes from person to person (herwig 2003 has F3 to dlpfc
# mostly under 20 mm). nobody has measured that for these targets yet, so these rows
# show what n does if the between subject part is a few mm
spreadsWithAnatomy = [3.0, 5.0, 8.0]

alpha = 0.05
powerWant = 0.80      # for the t-test
powerWantTost = 0.80  # for TOST, the proposal asks for at least 80%. matches the 1.28 below


## the closed forms, the ones I derived by hand. still the right thing to reason with,
## but every one of them swaps a t distribution for a normal so that n does not appear
## on both sides. that quietly assumes the sd is KNOWN. it is not, it gets estimated
## from the same handful of subjects, so the real tests are wider and the honest n is
## bigger. the exact versions underneath fix that.

def nTtestClosed(spread):
    return (1.96 + 0.84)**2 * spread**2 / bias**2

def nTostClosed(spread):
    # 1.28 is z(0.90). at a true difference of 0 TOST fails if EITHER side fails, so
    # each side gets half the 20% miss, and z(1 - 0.20/2) = z(0.90). that is 80% power
    return (1.645 + 1.28)**2 * spread**2 / eqMargin**2

def nBlandClosed(spread):
    # 1.71 is sqrt(1 + 1.96**2 / 2), the standard error of a limit of agreement
    return (1.96 * 1.71 * spread / LOAPrec)**2


## the exact versions, now through statsmodels. none of them has a closed form,
## because the critical value needs n before it can tell you n.

def nTtestExact(spread):
    """Smallest n where the two sided paired t test really hits powerWant.

    statsmodels TTestPower. under the alternative the t statistic is not a t, it is
    a NONCENTRAL t with noncentrality bias*sqrt(n)/spread, and TTestPower works with
    exactly that. the closed form replaces it with a shifted normal, which is fine
    at n = 200 and not fine at n = 6. solve_power gives a fractional n, rounding it
    up is the smallest whole number of subjects that clears the power
    """
    analysis = TTestPower()
    effect = bias / spread
    n = math.ceil(analysis.solve_power(effect_size=effect, alpha=alpha, power=powerWant,
                                       alternative="two-sided"))
    return n, analysis.power(effect_size=effect, nobs=n, alpha=alpha, alternative="two-sided")


def nTostExact(spread, reps=40000, seed=0):
    """Smallest n where TOST really hits powerWantTost, at a true difference of zero.

    TOST only rejects when BOTH one sided tests reject, and the two of them share the
    same estimated sd, so they are not independent. that dependence has no clean
    closed form, so this counts rejections directly. statsmodels DescrStatsW runs
    the real test on every simulated study at once, one column per study
    """
    rng = np.random.default_rng(seed)
    for n in range(2, 500):
        x = rng.normal(0.0, spread, size=(n, reps))
        p = DescrStatsW(x).ttost_mean(-eqMargin, eqMargin)[0]
        power = float((p < alpha).mean())
        if power >= powerWantTost:
            return n, power
    raise ValueError("no n reached the target TOST power")


def nBlandExact(spread):
    """Smallest n where the 95% CI on a limit of agreement is within LOAPrec.

    a precision target rather than a power one, but it needs the same two fixes. the
    variance of s divides by n-1 not n, and the CI multiplier is t not 1.96.
    statsmodels has no function for this one, it is a formula and a t quantile
    """
    for n in range(3, 1000000):
        df = n - 1
        se = spread * math.sqrt(1.0 / n + 1.96**2 / (2.0 * df))
        half = stats.t.ppf(1.0 - alpha / 2.0, df) * se
        if half <= LOAPrec:
            return n, half
    raise ValueError("no n reached the target precision")


def checkPower(spread, n, reps=40000, seed=1):
    """Simulate the real paired t test at this n and count how often it actually rejects.

    the true shift is bias, the test is statsmodels DescrStatsW.ttest_mean against 0
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(bias, spread, size=(n, reps))
    p = DescrStatsW(x).ttest_mean(0.0)[1]
    return float((p < alpha).mean())


def row(label, spread, mark=""):
    nClosed = math.ceil(nTtestClosed(spread))   # math.ceil to round these up
    nExact, _ = nTtestExact(spread)
    nTostCl = math.ceil(nTostClosed(spread))
    nTostEx, _ = nTostExact(spread)
    nBlandCl = math.ceil(nBlandClosed(spread))
    nBlandEx, _ = nBlandExact(spread)

    pClosed = checkPower(spread, nClosed)
    pExact = checkPower(spread, nExact)

    print(f"{label:>5s} {spread:7.2f} |{nClosed:9d}{pClosed:7.3f} |{nExact:8d}"
          f"{pExact:7.3f} |{nTostCl:8d}{nTostEx:8d} |{nBlandCl:6d}{nBlandEx:6d}{mark}")


print()
print("target power", powerWant, "for the paired t-test,", powerWantTost, "for TOST")
print("power columns are simulated, 40000 runs of the actual test at that n")
print()
print(f"{'cap':>5s} {'spread':>7s} |{'t closed':>9s}{'power':>7s} |{'t exact':>8s}"
      f"{'power':>7s} |{'TOST cl':>8s}{'TOST ex':>8s} |{'BA cl':>6s}{'BA ex':>6s}")
print("-" * 92)

for capError in capErrors:
    spread = (capError**2 + navError**2) ** 0.5
    mark = "  <- measured" if abs(capError - 1.55) < 1e-9 else ""
    row(f"{capError:.2f}", spread, mark)

print()
print("with a between subject gap between the 10-20 spot and the mri target added in")
for spread in spreadsWithAnatomy:
    row("-", spread)

print()
print("the closed form columns are what I had before. the power beside them is what")
print("those n actually deliver, and it is nowhere near 0.80. the exact columns are the")
print("n that do deliver it. the gap is worst at small n, which is the regime this")
print("study sits in, so this is not an academic correction.")
print()
print("the top block only knows about measurement noise. the real eeg minus mri")
print("difference also depends on how far the 10-20 spot is from each person's own")
print("target, and the bottom block shows how fast n grows if that is a few mm. the")
print("pilot data is what settles which row the study is actually in.")
print()
print("## the exact columns now come from statsmodels (TTestPower, DescrStatsW). the")
print("## closed forms are still mine, I kept them to learn it")
