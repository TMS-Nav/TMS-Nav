"""
Simple hypothesis-testing demo for the TMS coil-positioning project.

Teaching example:
1) paired t-test for systematic mean bias
2) optional paired TOST equivalence test

All values below are simulated.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.weightstats import DescrStatsW, ttost_paired

# ---------------------------------------------------------------------
# Simulated paired data: mean electric field in the target ROI (V/m)
# ---------------------------------------------------------------------
data = pd.DataFrame({
    "subject": np.arange(1, 13),
    "mri_efield": [82, 76, 91, 88, 79, 84, 95, 73, 86, 90, 81, 87],
    "eeg_efield": [80, 77, 89, 86, 80, 82, 93, 75, 84, 88, 82, 85],
})

# ---------------------------------------------------------------------
# Paired t-test
# H0: mean(EEG - MRI) = 0
# H1: mean(EEG - MRI) != 0
# ---------------------------------------------------------------------
data["difference"] = data["eeg_efield"] - data["mri_efield"]

stats = DescrStatsW(data["difference"])
t_stat, p_value, df = stats.ttest_mean(0, alternative="two-sided")
ci_low, ci_high = stats.tconfint_mean(alpha=0.05)

print("PAIRED T-TEST")
print(f"Mean difference (EEG - MRI): {data['difference'].mean():.2f} V/m")
print(f"95% CI: [{ci_low:.2f}, {ci_high:.2f}] V/m")
print(f"t({int(df)}) = {t_stat:.3f}")
print(f"p = {p_value:.4f}")

if p_value < 0.05:
    print("Conclusion: evidence of a systematic mean difference.")
else:
    print("Conclusion: insufficient evidence of a systematic mean difference.")

# ---------------------------------------------------------------------
# Optional: TOST equivalence test
# Illustrative bounds only; real bounds must be prespecified.
# ---------------------------------------------------------------------
data["pct_difference"] = (
    100 * (data["eeg_efield"] - data["mri_efield"]) / data["mri_efield"]
)

lower_bound = -5
upper_bound = 5

tost_p, lower_test, upper_test = ttost_paired(
    data["pct_difference"],
    np.zeros(len(data)),
    low=lower_bound,
    upp=upper_bound,
)

print("\nTOST EQUIVALENCE TEST")
print(f"Mean percentage difference: {data['pct_difference'].mean():.2f}%")
print(f"Equivalence bounds: [{lower_bound}%, {upper_bound}%]")
print(f"TOST p = {tost_p:.4f}")

if tost_p < 0.05:
    print("Conclusion: evidence supports equivalence within the bounds.")
else:
    print("Conclusion: equivalence has not been demonstrated within the bounds.")
