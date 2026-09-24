# Ali Rad - aim 1 statistics, eeg cap vs mri guided neuronavigation, on the real data
#
#   python analysis/aim1_stats.py              every export in saves/brainsight/
#   python analysis/aim1_stats.py some/folder  every export in that folder
#
# drop the brainsight session sample exports in the folder and run it. src/aim1.py
# turns them into one row per compared pair of coil placements, this script runs the
# statistics the proposal names on those rows and prints the numbers.
#
# the unit is the SUBJECT. each subject's reps at a site are averaged first, so every
# n below is a number of people, never a number of placements. per site:
#
#   how far apart   displacement, tilt, orientation, and each method against itself
#                   (rep 1 vs rep 2). mean, sd, 95% ci. displacement and orientation
#                   also get a one sided test that the average sits under the line
#   which way       the signed eeg minus mri offset, forward, right, out and handle
#                   turn. bland altman (bias and 95% limits of agreement), the paired
#                   t test for a systematic shift with its sign test check, and tost,
#                   is the shift inside +/- the margin
#
# then across sites. benjamini hochberg over all the paired t tests, because each
# site and axis is its own test, and a mixed model per distance with site fixed and
# subject random, which is how the proposal asks whether the error depends on target.
#
# every test is a statsmodels call, see the helpers in src/stats.py.
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from efield.config import BRAINSIGHT_ROOT  # noqa: E402
from src.aim1 import (  # noqa: E402
    OFFSET_METRICS,
    PAIR_METRICS,
    REPEAT_METRICS,
    export_files,
    measurements,
    per_subject,
)
from src.stats import (  # noqa: E402
    EQ_MARGIN_DEG,
    EQ_MARGIN_MM,
    bias_test,
    bland_altman,
    below_threshold,
    describe,
    equivalence,
    fdr,
    site_mixed_model,
)

OUT_DIR = ROOT / "saves" / "aim1"
VIEWER_DATA_DIR = ROOT / "viewer" / "src" / "data"
REGISTRY_PATH = ROOT / "viewer" / "src" / "datasetRegistry.js"

# how each number is labelled, and the line it is judged against. None means it is
# described but not tested against a line
LABELS = {
    "displacement_mm": ("displacement", "mm", EQ_MARGIN_MM),
    "tilt_deg": ("tilt", "deg", None),
    "orientation_deg": ("orientation", "deg", EQ_MARGIN_DEG),
    "repeat_mm": ("repeat", "mm", None),
    "repeat_deg": ("repeat", "deg", None),
    "ap_mm": ("forward", "mm", EQ_MARGIN_MM),
    "lr_mm": ("right", "mm", EQ_MARGIN_MM),
    "depth_mm": ("out", "mm", EQ_MARGIN_MM),
    "yaw_deg": ("handle turn", "deg", EQ_MARGIN_DEG),
}


# --- the numbers ----------------------------------------------------------------------
def site_block(subj, reps, site):
    """Every number for one site, from the per subject tables."""

    rows = subj[subj["site"] == site]
    out = {"site": site, "n": int(len(rows)), "magnitudes": {}, "repeat": {}, "offsets": {}}

    for m in PAIR_METRICS:
        v = rows[m].to_numpy(float)
        block = {"describe": describe(v)}
        line = LABELS[m][2]
        if line is not None and len(v) >= 2:
            block["below"] = below_threshold(v, line)
        out["magnitudes"][m] = block

    for method in sorted(reps["method"].unique()) if len(reps) else []:
        r = reps[(reps["site"] == site) & (reps["method"] == method)]
        if len(r):
            out["repeat"][method] = {m: {"describe": describe(r[m].to_numpy(float))}
                                     for m in REPEAT_METRICS}

    for m in OFFSET_METRICS:
        v = rows[m].to_numpy(float)
        block = {"bland_altman": bland_altman(v)}
        if len(v) >= 2:
            block["bias"] = bias_test(v)
            block["tost"] = equivalence(v, LABELS[m][2])
        out["offsets"][m] = block
    return out


def run(pairs, repeats):
    """The whole analysis as one dict. pairs and repeats come from src/aim1.py."""

    subj = per_subject(pairs, PAIR_METRICS + OFFSET_METRICS)
    reps = per_subject(repeats, REPEAT_METRICS, by=("subject", "site", "method"))
    sites = sorted(subj["site"].unique())
    result = {
        "n_subjects": int(subj["subject"].nunique()),
        "n_pairs": int(len(pairs)),
        "sites": [site_block(subj, reps, s) for s in sites],
        "thresholds": {"mm": EQ_MARGIN_MM, "deg": EQ_MARGIN_DEG},
    }

    # every paired t test is its own question, one per site and axis, so the p
    # values are corrected together
    tests = [(b, m) for b in result["sites"] for m in OFFSET_METRICS if "bias" in b["offsets"][m]]
    qvals, _ = fdr([b["offsets"][m]["bias"]["p"] for b, m in tests])
    for (b, m), q in zip(tests, qvals):
        b["offsets"][m]["bias"]["q"] = q

    # the mixed model runs on every placement, not the subject means, the random
    # intercept is what takes care of reps and sites from one head being related
    result["mixed"] = {m: site_mixed_model(pairs, m) for m in PAIR_METRICS}
    return result


# --- printing -------------------------------------------------------------------------
def _ci(c, f="{:.2f}"):
    return "[" + ", ".join(f.format(v) for v in c) + "]"


def _s(v):
    return f"{v:+.2f}"


def _p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def report(result, notes, folder):
    L = []
    L.append("aim 1, eeg cap vs mri guided neuronavigation")
    L.append(f"exports      {folder}")
    for n in notes:
        L.append(f"  {n}")
    L.append(f"subjects     {result['n_subjects']}   compared placements {result['n_pairs']}")
    L.append("unit         the subject. reps are averaged first, so every n is people")
    L.append(f"lines        {EQ_MARGIN_MM:g} mm and {EQ_MARGIN_DEG:g} deg (src/stats.py, to be prespecified)")

    for b in result["sites"]:
        L.append("")
        L.append(f"=== {b['site']}   n = {b['n']} subjects " + "=" * 40)
        L.append("how far apart, eeg vs mri, one number per subject")
        L.append(f"  {'':18s}{'mean':>7s}{'sd':>7s}   {'95% CI':17s}{'median':>7s}{'max':>7s}   average under the line?")
        for m, blk in b["magnitudes"].items():
            L.append(_mag_line(m, blk))
        for method, blk in b["repeat"].items():
            for m, sub in blk.items():
                L.append(_mag_line(m, sub, prefix=f"{method.lower()} "))
        if b["repeat"]:
            L.append("  (repeat is each method against itself, rep 1 vs rep 2 at the same site)")
        if b["n"] < 2:
            L.append("  (one subject, the tests need at least two)")

        L.append("which way, eeg minus mri (bland altman, paired t, tost)")
        for m, blk in b["offsets"].items():
            name, unit, margin = LABELS[m]
            ba = blk["bland_altman"]
            head = f"  {name + ' ' + unit:17s}"
            line = f"{head}bias {_s(ba['bias'])}"
            if "bias_ci" in ba:
                line += f" {_ci(ba['bias_ci'], '{:+.2f}')}"
            line += f"   sd {ba['sd']:.2f}   95% limits {_s(ba['loa_lo'])} to {_s(ba['loa_hi'])}"
            L.append(line)
            pad = " " * len(head)
            if "loa_lo_ci" in ba:
                L.append(f"{pad}limit cis {_ci(ba['loa_lo_ci'], '{:+.2f}')} and {_ci(ba['loa_hi_ci'], '{:+.2f}')}")
            if "bias" in blk:
                bt, eq = blk["bias"], blk["tost"]
                L.append(f"{pad}shift?  paired t {bt['t']:+.2f} on {bt['df']:.0f} df, p {_p(bt['p'])},"
                         f" fdr q {_p(bt['q'])}, sign test p {_p(bt['p_sign'])}")
                L.append(f"{pad}inside +/-{margin:g}?  90% CI {_ci(eq['ci90'], '{:+.2f}')},"
                         f" tost p {_p(eq['p'])} -> {'equivalent' if eq['equivalent'] else 'not shown'}")

    L.append("")
    L.append("=== does the error depend on the site? mixed model, site fixed, subject random " + "=" * 4)
    for m, mm in result["mixed"].items():
        name, unit, _ = LABELS[m]
        if "note" in mm:
            L.append(f"  {name} {unit}: {mm['note']}")
            continue
        means = "   ".join(f"{s} {v['mean']:.2f} {_ci(v['ci95'])}" for s, v in mm["sites"].items())
        L.append(f"  {name} {unit}: {means}")
        L.append(f"    wald chi2 {mm['wald_chi2']:.2f} on {mm['df']} df, p {_p(mm['p'])}"
                 f" -> {'site matters' if mm['site_matters'] else 'no site effect shown'}")
        L.append(f"    subject sd {mm['subject_sd']:.2f}, residual sd {mm['residual_sd']:.2f},"
                 f" {mm['n_rows']} placements from {mm['n_subjects']} subjects")
        for w in mm["warnings"]:
            L.append(f"    fit warning: {w}")
    return "\n".join(L)


def _mag_line(m, blk, prefix=""):
    name, unit, line = LABELS[m]
    d = blk["describe"]
    ci = _ci(d["ci95"]) if d["n"] > 1 else "-"
    txt = (f"  {(prefix + name + ' ' + unit):18s}{d['mean']:7.2f}{d['sd']:7.2f}   {ci:17s}"
           f"{d['median']:7.2f}{d['max']:7.2f}")
    if "below" in blk:
        bt = blk["below"]
        txt += (f"   < {line:g} {unit}: p {_p(bt['p'])}, 90% CI top {bt['ci90'][1]:.2f}"
                f" -> {'yes' if bt['below'] else 'not shown'}")
    return txt


# --- files ----------------------------------------------------------------------------
def long_table(result):
    """Every number as (site, quantity, statistic, value), for a spreadsheet."""

    rows = []

    def add(site, quantity, prefix, d):
        for k, v in d.items():
            if isinstance(v, (list, tuple)):
                for tag, u in zip(("lo", "hi"), v):
                    rows.append((site, quantity, f"{prefix}{k}_{tag}", u))
            elif isinstance(v, (int, float, bool, np.floating)):
                rows.append((site, quantity, f"{prefix}{k}", float(v)))

    for b in result["sites"]:
        for m, blk in b["magnitudes"].items():
            for part, d in blk.items():
                add(b["site"], m, f"{part}.", d)
        for method, blk in b["repeat"].items():
            for m, sub in blk.items():
                add(b["site"], f"{method.lower()}_{m}", "describe.", sub["describe"])
        for m, blk in b["offsets"].items():
            for part, d in blk.items():
                add(b["site"], m, f"{part}.", d)
    for m, mm in result["mixed"].items():
        if "note" in mm:
            continue
        add("all", m, "mixed.", {k: v for k, v in mm.items() if k not in ("sites", "warnings")})
        for s, v in mm["sites"].items():
            add(s, m, "mixed.", v)
    return pd.DataFrame(rows, columns=["site", "quantity", "statistic", "value"])


def write_viewer(result, viewer_dir, registry_path):
    """The aggregate numbers for the viewer, no subject ids, then refresh the registry."""

    viewer_dir = Path(viewer_dir)
    if not viewer_dir.is_dir():
        return None
    payload = {"source": "brainsight", "generated": date.today().isoformat(), **result}
    path = viewer_dir / "aim1_stats.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    try:
        from src.export_web import write_registry
        write_registry(viewer_dir, registry_path)
    except ValueError as err:
        print(f"viewer registry not refreshed ({err}), run run_pipeline.py first")
    return path


def main(argv=None, out_dir=OUT_DIR, viewer_dir=VIEWER_DATA_DIR, registry_path=REGISTRY_PATH):
    argv = sys.argv[1:] if argv is None else argv
    folder = Path(argv[0]) if argv else BRAINSIGHT_ROOT
    out_dir = Path(out_dir)

    if not export_files(folder):
        print(f"no brainsight exports in {folder}")
        print("drop the session sample exports (sub-<ID>_coil_pose_samples.txt) in there and rerun")
        return 0

    pairs, repeats, notes = measurements(folder)
    if pairs.empty:
        print(f"read {len(export_files(folder))} files in {folder} but found no eeg / mri pair to compare")
        for n in notes:
            print(f"  {n}")
        print("the samples have to be named sub-<ID>_EEGCAP_<SITE>_POSE_REP<n> and"
              " sub-<ID>_MRI_<SITE>_POSE_REP<n>")
        return 1

    result = run(pairs, repeats)
    print(report(result, notes, folder))

    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_dir / "measurements.csv", index=False)
    repeats.to_csv(out_dir / "repeats.csv", index=False)
    long_table(result).to_csv(out_dir / "aim1_stats.csv", index=False)
    print()
    print(f"wrote {out_dir / 'measurements.csv'}  (one row per compared placement, check by hand)")
    print(f"wrote {out_dir / 'repeats.csv'}")
    print(f"wrote {out_dir / 'aim1_stats.csv'}  (every number above, one per row)")
    viewer = write_viewer(result, viewer_dir, registry_path)
    if viewer:
        print(f"wrote {viewer}  (what the viewer shows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
