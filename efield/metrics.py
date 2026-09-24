# from results.json to the aim 2 endpoints. pure python, no simnibs.
#
# the runner leaves one block per pose with the roi numbers in it. here the eeg
# and mri poses at the same site and rep are paired up and the differences the
# proposal names are computed:
#
#   pct_diff_mean_roi   percentage difference in mean roi |E|, eeg vs mri, the
#                       PRIMARY endpoint. (eeg - mri) / mri * 100
#   diff_peak           difference in peak |E| in the roi, V/m
#   diff_normal         difference in max |E_normal| in the roi, V/m
#   dice_supra_abs      overlap of the suprathreshold hotspots at a FIXED V/m
#                       cut, the same for both poses of a pair. the one to
#                       report, see roi_metrics in efield/runner_simnibs.py
#   dice_supra          the same overlap at each pose's own peak fraction. kept
#                       because it is the commoner convention, but a weaker
#                       placement gets a lower bar under it, so it confounds
#                       "the hotspot moved" with "the hotspot got weaker"
#
# plus the raw per pose values. everything comes out as long format rows,
# (subject, site, method, rep, metric, value), one csv for all subjects, which
# is what analysis/efield_stats.py reads.
import csv
import json
from pathlib import Path

import numpy as np

from src.stats import bias_test, bland_altman, equivalence

# the equivalence margin on the primary endpoint, percent. a PLACEHOLDER to be
# prespecified in the analysis plan before any data is looked at
PCT_MARGIN = 10.0

RAW_METRICS = ("mean_E_magn", "peak_E_magn", "max_abs_E_normal", "E_magn_at_centre",
               "supra_area_mm2", "supra_abs_area_mm2")
PAIRED_METRICS = ("pct_diff_mean_roi", "diff_peak", "diff_normal", "dice_supra_abs",
                  "dice_supra", "diff_at_centre")

CSV_COLUMNS = ("subject", "site", "method", "rep", "metric", "value")


def dice(a, b):
    """Dice overlap of two index sets, 1 when identical, 0 when disjoint."""

    a, b = set(int(i) for i in a), set(int(i) for i in b)
    if not a and not b:
        return 1.0
    return 2.0 * len(a & b) / (len(a) + len(b))


def pct_diff(new, ref):
    return 100.0 * (float(new) - float(ref)) / float(ref)


def load_results(path):
    with open(path) as f:
        return json.load(f)


def roi_for_site(entry, site):
    """The roi block for a pose's own site, or the only roi if there is one."""

    rois = entry.get("rois", {})
    for name, block in rois.items():
        if name.startswith(f"{site}_"):
            return block
    if len(rois) == 1:
        return next(iter(rois.values()))
    return None


def pair_poses(results):
    """{(site, rep): {'EEG': block, 'MRI': block}} from a results.json dict."""

    pairs = {}
    for entry in results.get("poses", []):
        block = roi_for_site(entry, entry.get("site", ""))
        if block is None or entry.get("method") not in ("EEG", "MRI"):
            continue
        pairs.setdefault((entry["site"], int(entry.get("rep", 0))), {})[entry["method"]] = block
    return pairs


def paired_metrics(eeg, mri):
    """The endpoint differences for one eeg/mri pair of roi blocks."""

    out = {
        "pct_diff_mean_roi": pct_diff(eeg["mean_E_magn"], mri["mean_E_magn"]),
        "diff_peak": float(eeg["peak_E_magn"]) - float(mri["peak_E_magn"]),
        "diff_normal": float(eeg["max_abs_E_normal"]) - float(mri["max_abs_E_normal"]),
        "diff_at_centre": float(eeg["E_magn_at_centre"]) - float(mri["E_magn_at_centre"]),
        "dice_supra": dice(eeg.get("supra_nodes", []), mri.get("supra_nodes", [])),
    }
    # only present when the runner was given an absolute cut
    if "supra_abs_nodes" in eeg and "supra_abs_nodes" in mri:
        out["dice_supra_abs"] = dice(eeg["supra_abs_nodes"], mri["supra_abs_nodes"])
    return out


def long_rows(results):
    """Long format rows for one subject: raw values per pose, then the paired ones."""

    subject = results["subject"]
    rows = []
    for entry in results.get("poses", []):
        block = roi_for_site(entry, entry.get("site", ""))
        if block is None:
            continue
        for m in RAW_METRICS:
            if m in block:
                rows.append({"subject": subject, "site": entry["site"], "method": entry["method"],
                             "rep": int(entry.get("rep", 0)), "metric": m, "value": float(block[m])})
    for (site, rep), methods in pair_poses(results).items():
        if "EEG" not in methods or "MRI" not in methods:
            continue
        for m, v in paired_metrics(methods["EEG"], methods["MRI"]).items():
            rows.append({"subject": subject, "site": site, "method": "EEG-MRI", "rep": rep,
                         "metric": m, "value": float(v)})
    return rows


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in CSV_COLUMNS})
    return path


def read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["rep"] = int(r["rep"])
        r["value"] = float(r["value"])
    return rows


def subject_means(rows, metric, method="EEG-MRI"):
    """{site: {subject: mean over reps}} for one metric. subjects are the unit."""

    acc = {}
    for r in rows:
        if r["metric"] != metric or r["method"] != method:
            continue
        acc.setdefault(r["site"], {}).setdefault(r["subject"], []).append(r["value"])
    return {site: {s: float(np.mean(v)) for s, v in subs.items()} for site, subs in acc.items()}


def site_stats(rows, margin=PCT_MARGIN):
    """Per site: tost, paired t and bland altman on the subject level primary endpoint.

    the paired difference is already in the rows as pct_diff_mean_roi, one per
    rep, averaged to one per subject first so n is subjects, see src/stats.
    bland altman is on the raw mean |E| so its units are V/m. all three are the
    statsmodels helpers in src/stats.py, the same ones aim 1 uses
    """

    out = {}
    pct = subject_means(rows, "pct_diff_mean_roi")
    eeg = subject_means(rows, "mean_E_magn", "EEG")
    mri = subject_means(rows, "mean_E_magn", "MRI")
    for site, subs in pct.items():
        values = np.array(list(subs.values()))
        block = {"n": int(len(values)), "mean_pct": float(values.mean())}
        if len(values) >= 2:
            block["tost"] = equivalence(values, margin)
            block["t"] = bias_test(values)
            common = sorted(set(eeg.get(site, {})) & set(mri.get(site, {})))
            if len(common) >= 2:
                block["bland_altman_Vm"] = bland_altman([eeg[site][s] - mri[site][s] for s in common])
        out[site] = block
    return out


def format_site_stats(stats):
    lines = []
    for site, b in stats.items():
        lines.append(f"  {site:4s} n={b['n']}  mean pct diff {b['mean_pct']:+.2f} %")
        if "tost" in b:
            t = b["tost"]
            lo, hi = t["ci90"]
            lines.append(f"       tost  90% ci [{lo:+.2f}, {hi:+.2f}] within +/-{t['margin']:g} %: "
                         f"{'equivalent' if t['equivalent'] else 'not shown'}  p {t['p']:.3f}")
            tt = b["t"]
            lines.append(f"       paired t  t {tt['t']:+.2f} p {tt['p']:.3f}")
        if "bland_altman_Vm" in b:
            ba = b["bland_altman_Vm"]
            lines.append(f"       bland altman  bias {ba['bias']:+.2f} V/m  loa [{ba['loa_lo']:+.2f}, {ba['loa_hi']:+.2f}]")
    return "\n".join(lines)
