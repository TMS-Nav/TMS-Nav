# Ali Rad - Target Localization Method Comparison Statistics for aim 1

import math

import numpy as np
from scipy import stats

# in mm
navError = 0.75 # "neuronavigation error, brainsights own localization/registration noise"
# double check navError against the brainsight manual

# thresholds
bias = 1.5 # smallest average offset (for t-test)
eqMargin = 2.0 # size of difference to be equiavlent (for one sided test)
LOAPrec = 1.0 # limit of agreement precision (for Bland-Altman)

# cap drawing error. 1.55 is the measured one out of analysis/head_model_mc.py, the old
# guesses stay in the sweep so they are still comparable
capErrors = [1.0, 1.55, 2.0, 2.5]

alpha = 0.05
powerWant = 0.80      # for the t-test
powerWantTost = 0.90  # for TOST, matches the 1.28 in the closed form below


## the closed forms, the ones I derived by hand. still the right thing to reason with,
## but every one of them swaps a t distribution for a normal so that n does not appear
## on both sides. that quietly assumes the sd is KNOWN. it is not, it gets estimated
## from the same handful of subjects, so the real tests are wider and the honest n is
## bigger. the exact versions underneath fix that.

def nTtestClosed(spread):
    return (1.96 + 0.84)**2 * spread**2 / bias**2

def nTostClosed(spread):
    return (1.645 + 1.28)**2 * spread**2 / eqMargin**2

def nBlandClosed(spread):
    # 1.71 is sqrt(1 + 1.96**2 / 2), the standard error of a limit of agreement
    return (1.96 * 1.71 * spread / LOAPrec)**2


## the exact versions. none of them has a closed form, because the critical value needs
## n before it can tell you n. solving that by iteration oscillates between two
## neighbouring values and never settles, so each one just walks n upward instead.

def nTtestExact(spread):
    """Smallest n where the two sided one sample t test really hits powerWant.

    under the alternative the t statistic is not a t, it is a NONCENTRAL t with
    noncentrality bias*sqrt(n)/spread. the closed form replaces that with a shifted
    normal, which is fine at n = 200 and not fine at n = 6.
    """
    for n in range(2, 10000):
        df = n - 1
        ncp = bias * math.sqrt(n) / spread
        crit = stats.t.ppf(1.0 - alpha / 2.0, df)
        power = stats.nct.sf(crit, df, ncp) + stats.nct.cdf(-crit, df, ncp)
        if power >= powerWant:
            return n, power
    raise ValueError("no n reached the target power")


def nTostExact(spread, reps=40000, seed=0):
    """Smallest n where TOST really hits powerWantTost, at a true difference of zero.

    TOST only rejects when BOTH one sided tests reject, and the two of them share the
    same estimated sd, so they are not independent. that dependence has no clean
    closed form, so this counts rejections directly.
    """
    rng = np.random.default_rng(seed)
    for n in range(2, 500):
        x = rng.normal(0.0, spread, size=(reps, n))
        m = x.mean(axis=1)
        se = x.std(axis=1, ddof=1) / math.sqrt(n)
        tc = stats.t.ppf(1.0 - alpha, n - 1)
        ok = ((m - eqMargin) / se <= -tc) & ((m + eqMargin) / se >= tc)
        power = float(ok.mean())
        if power >= powerWantTost:
            return n, power
    raise ValueError("no n reached the target TOST power")


def nBlandExact(spread):
    """Smallest n where the 95% CI on a limit of agreement is within LOAPrec.

    a precision target rather than a power one, but it needs the same two fixes. the
    variance of s divides by n-1 not n, and the CI multiplier is t not 1.96.
    """
    for n in range(3, 1000000):
        df = n - 1
        se = spread * math.sqrt(1.0 / n + 1.96**2 / (2.0 * df))
        half = stats.t.ppf(1.0 - alpha / 2.0, df) * se
        if half <= LOAPrec:
            return n, half
    raise ValueError("no n reached the target precision")


def checkPower(spread, n, reps=40000, seed=1):
    """Simulate the real t test at this n and count how often it actually rejects."""
    rng = np.random.default_rng(seed)
    x = rng.normal(bias, spread, size=(reps, n))
    t = x.mean(axis=1) / (x.std(axis=1, ddof=1) / math.sqrt(n))
    p = 2.0 * stats.t.sf(np.abs(t), n - 1)
    return float((p < alpha).mean())


print()
print("target power", powerWant, "for the t-test,", powerWantTost, "for TOST")
print("power columns are simulated, 40000 runs of the actual test at that n")
print()
print(f"{'cap':>5s} {'spread':>7s} |{'t closed':>9s}{'power':>7s} |{'t exact':>8s}"
      f"{'power':>7s} |{'TOST cl':>8s}{'TOST ex':>8s} |{'BA cl':>6s}{'BA ex':>6s}")
print("-" * 92)

for capError in capErrors:
    spread = (capError**2 + navError**2) ** 0.5

    nClosed = math.ceil(nTtestClosed(spread))   # math.ceil to round these up
    nExact, _ = nTtestExact(spread)
    nTostCl = math.ceil(nTostClosed(spread))
    nTostEx, _ = nTostExact(spread)
    nBlandCl = math.ceil(nBlandClosed(spread))
    nBlandEx, _ = nBlandExact(spread)

    pClosed = checkPower(spread, nClosed)
    pExact = checkPower(spread, nExact)

    mark = "  <- measured" if abs(capError - 1.55) < 1e-9 else ""
    print(f"{capError:5.2f} {spread:7.2f} |{nClosed:9d}{pClosed:7.3f} |{nExact:8d}"
          f"{pExact:7.3f} |{nTostCl:8d}{nTostEx:8d} |{nBlandCl:6d}{nBlandEx:6d}{mark}")

print()
print("the closed form columns are what I had before. the power beside them is what")
print("those n actually deliver, and it is nowhere near 0.80. the exact columns are the")
print("n that do deliver it. the gap is worst at small n, which is the regime this")
print("study sits in, so this is not an academic correction.")
print()
print("## there's also packages that can do this, but I did it this to way learn it")
