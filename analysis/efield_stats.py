# the aim 2 statistics, on the long table run_efield.py writes.
#
#   python analysis/efield_stats.py                 saves/efield/efield_long.csv
#   python analysis/efield_stats.py <long table csv>
#
# what the proposal asks for and where it is done here, every test through the
# statsmodels helpers in src/stats.py:
#   primary    paired percentage difference in mean roi |E|, eeg vs mri
#              -> tost per site with a 90% ci inside +/- PCT_MARGIN (equivalence)
#   secondary  paired t on the primary for bias, the other paired metrics
#              -> bias_test, benjamini hochberg across the secondary tests (fdr)
#   agreement  bland altman on mean roi |E|, V/m
#   mixed      pct diff ~ site, random intercept per participant (site_mixed_model)
#   sensitivity sign test on the paired differences, for when they are not normal
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

from efield.config import SIM_ROOT  # noqa: E402
from efield.metrics import PAIRED_METRICS, PCT_MARGIN, read_csv, site_stats, subject_means  # noqa: E402
from src.stats import (  # noqa: E402
    bias_test,
    bland_altman,
    describe,
    equivalence,
    fdr,
    site_mixed_model,
)

# --- the report ---------------------------------------------------------------------
def report(rows, margin=PCT_MARGIN):
    sites = sorted({r["site"] for r in rows})
    subjects = sorted({r["subject"] for r in rows})
    print(f"{len(subjects)} subjects, sites {', '.join(sites)}, {len(rows)} rows")
    print(f"equivalence margin on the primary endpoint +/- {margin:g} %  (placeholder, to be prespecified)")

    print("\n--- primary: paired % difference in mean roi |E|, eeg vs mri, one value per subject")
    pct = subject_means(rows, "pct_diff_mean_roi")
    for site in sites:
        v = np.array(list(pct[site].values()))
        tost = equivalence(v, margin)
        tt = bias_test(v)
        lo, hi = tost["ci90"]
        clo, chi = describe(v)["ci95"]
        print(f"  {site:4s} n={len(v)}  mean {v.mean():+.2f} %  sd {v.std(ddof=1):.2f}")
        print(f"       tost      90% ci [{lo:+.2f}, {hi:+.2f}]  ->  {'EQUIVALENT' if tost['equivalent'] else 'not shown equivalent'}  (p {tost['p']:.4f})")
        print(f"       paired t  t {tt['t']:+.2f}  p {tt['p']:.4f}  95% ci [{clo:+.2f}, {chi:+.2f}]")
        print(f"       sign test p {tt['p_sign']:.4f}  (counts differences above and below 0, no normality assumed)")

    print("\n--- agreement: bland altman on mean roi |E|, eeg minus mri, V/m")
    eeg, mri = subject_means(rows, "mean_E_magn", "EEG"), subject_means(rows, "mean_E_magn", "MRI")
    for site in sites:
        common = sorted(set(eeg[site]) & set(mri[site]))
        ba = bland_altman([eeg[site][s] - mri[site][s] for s in common])
        print(f"  {site:4s} bias {ba['bias']:+.2f} [{ba['bias_ci'][0]:+.2f}, {ba['bias_ci'][1]:+.2f}]  "
              f"loa [{ba['loa_lo']:+.2f}, {ba['loa_hi']:+.2f}]  sd {ba['sd']:.2f}")

    print("\n--- secondary endpoints: paired t per site, benjamini hochberg across all of them")
    print("  each metric is tested against the value that means no difference for it.")
    print("  the three differences against 0. dice against 1, perfect overlap, because a")
    print("  dice score cannot be negative and testing it against 0 would always reject.")
    secondary = [m for m in PAIRED_METRICS if m != "pct_diff_mean_roi"]
    tests = []
    for m in secondary:
        by_site = subject_means(rows, m)
        for site in sites:
            if site not in by_site:
                continue
            v = np.array(list(by_site[site].values()))
            mu0 = 1.0 if m.startswith("dice") else 0.0
            tt = bias_test(v - mu0)
            tests.append((m, site, v.mean(), mu0, tt["p"]))
    qvals, reject = fdr([t[4] for t in tests])
    print(f"  {'metric':18s} {'site':4s} {'mean':>8s} {'vs':>4s} {'p':>8s} {'q':>8s}")
    for (m, site, mean, mu0, p), q, rj in zip(tests, qvals, reject):
        print(f"  {m:18s} {site:4s} {mean:+8.3f} {mu0:4.0f} {p:8.4f} {q:8.4f}  {'*' if rj else ''}")
    print(f"  ({sum(reject)} of {len(tests)} survive fdr at q=0.05)")

    print("\n--- mixed effects: pct diff ~ site, random intercept per participant")
    df = pd.DataFrame([r for r in rows if r["metric"] == "pct_diff_mean_roi" and r["method"] == "EEG-MRI"])
    mm = site_mixed_model(df, "value")
    if "note" in mm:
        print(f"  {mm['note']}")
    else:
        for site, v in mm["sites"].items():
            print(f"  {site:4s} {v['mean']:+.2f} %  95% ci [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")
        print(f"  site effect: wald chi2 {mm['wald_chi2']:.2f} on {mm['df']} df, p {mm['p']:.4f}")
        print(f"  subject sd {mm['subject_sd']:.2f}, residual sd {mm['residual_sd']:.2f},"
              f" {mm['n_rows']} rows from {mm['n_subjects']} subjects")
        for w in mm["warnings"]:
            print(f"  fit warning: {w}")

    print("\n--- the same numbers through efield.metrics.site_stats, what run_efield.py prints")
    from efield.metrics import format_site_stats
    print(format_site_stats(site_stats(rows, margin=margin)))


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    path = Path(argv[0]) if argv else SIM_ROOT / "efield_long.csv"
    if not path.exists():
        print(f"no long table at {path}. run_efield.py --run writes it once there are results")
        return 1
    rows = read_csv(path)
    print(f"long table {path}")
    # every test is on subjects, one value each, and none of them work on one
    subjects = {r["subject"] for r in rows}
    if len(subjects) < 2:
        print(f"{len(subjects)} subject in the table, the statistics need at least 2")
        return 1
    report(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
