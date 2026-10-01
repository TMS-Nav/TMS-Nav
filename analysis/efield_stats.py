# the aim 2 statistics, on the long table run_efield.py writes, or on a synthetic
# one when no real table exists yet so the wiring can be seen before there is data.
#
#   python analysis/efield_stats.py                 synthetic table
#   python analysis/efield_stats.py saves/efield/efield_long.csv
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

from efield.metrics import PAIRED_METRICS, PCT_MARGIN, read_csv, site_stats, subject_means  # noqa: E402
from src.stats import (  # noqa: E402
    bias_test,
    bland_altman,
    describe,
    equivalence,
    fdr,
    site_mixed_model,
)

SITES = ("SMA", "F4")
REPS = (1, 2)

# the synthetic effect sizes, so what the report shows is understood before it
# is trusted. mean roi |E| under the mri pose is about 100 V/m at 1 A/us and
# varies +/- 15 between heads. the eeg pose reads a little lower, more at F4
# where the cap is further from the landmarks, with a few percent of placement
# noise per rep on top
SYN_N_SUBJECTS = 20
SYN_MRI_MEAN, SYN_MRI_SD = 100.0, 15.0
SYN_EEG_BIAS_PCT = {"SMA": -3.0, "F4": -6.0}
SYN_REP_NOISE_PCT = 4.0
SYN_DICE = {"SMA": 0.80, "F4": 0.70}

def synthetic_rows(n=SYN_N_SUBJECTS, seed=0):
    """A long table shaped exactly like the real one will be."""

    rng = np.random.default_rng(seed)
    rows = []

    def add(subject, site, method, rep, metric, value):
        rows.append({"subject": subject, "site": site, "method": method, "rep": rep,
                     "metric": metric, "value": float(value)})

    for i in range(n):
        subj = f"S{i + 1:02d}"
        for site in SITES:
            mri_true = rng.normal(SYN_MRI_MEAN, SYN_MRI_SD)
            for rep in REPS:
                mri = mri_true * (1.0 + rng.normal(0.0, 0.01))
                eeg = mri_true * (1.0 + (SYN_EEG_BIAS_PCT[site] + rng.normal(0.0, SYN_REP_NOISE_PCT)) / 100.0)
                for method, v in (("MRI", mri), ("EEG", eeg)):
                    add(subj, site, method, rep, "mean_E_magn", v)
                    add(subj, site, method, rep, "peak_E_magn", v * 1.6)
                    add(subj, site, method, rep, "max_abs_E_normal", v * 0.8)
                add(subj, site, "EEG-MRI", rep, "pct_diff_mean_roi", 100.0 * (eeg - mri) / mri)
                add(subj, site, "EEG-MRI", rep, "diff_peak", (eeg - mri) * 1.6)
                add(subj, site, "EEG-MRI", rep, "diff_normal", (eeg - mri) * 0.8)
                add(subj, site, "EEG-MRI", rep, "diff_at_centre", (eeg - mri) * 1.1)
                d_rel = np.clip(rng.normal(SYN_DICE[site], 0.08), 0, 1)
                add(subj, site, "EEG-MRI", rep, "dice_supra", d_rel)
                # the fixed V/m cut does not move with the weaker pose, so it
                # sees the strength drop as well as the shift and reads lower
                add(subj, site, "EEG-MRI", rep, "dice_supra_abs",
                    float(np.clip(d_rel - abs(SYN_EEG_BIAS_PCT[site]) / 100.0, 0, 1)))
    return rows


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
    if argv and Path(argv[0]).exists():
        rows = read_csv(argv[0])
        print(f"long table {argv[0]}")
    else:
        default = Path("saves") / "efield" / "efield_long.csv"
        if default.exists():
            rows = read_csv(default)
            print(f"long table {default}")
        else:
            rows = synthetic_rows()
            print(f"no long table at {default}, SYNTHETIC data: {SYN_N_SUBJECTS} subjects, "
                  f"eeg bias {SYN_EEG_BIAS_PCT} %, rep noise {SYN_REP_NOISE_PCT} % sd")
    report(rows)


if __name__ == "__main__":
    main()
