# the numbers behind the monte carlo cloud. every draw is one cap laid on the same
# head, so the cloud answers two questions about a target. how far is a placement
# from where the mri says the target is, and how far apart do two placements of the
# same target land, ie session to session. both are distributions, not numbers, and
# the figure here shows them as such. aim 1 checkpoint from the meeting notes
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


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


def summary(d):
    """The handful of numbers that describe a distance distribution."""

    d = np.asarray(d, dtype=float)
    return {
        "n": int(len(d)),
        "mean_mm": round(float(d.mean()), 3),
        "sd_mm": round(float(d.std(ddof=1)), 3) if len(d) > 1 else 0.0,
        "median_mm": round(float(np.median(d)), 3),
        "p95_mm": round(float(np.percentile(d, 95)), 3),
        "max_mm": round(float(d.max()), 3),
    }


# the two panels of the figure, json key and what it means
METRICS = [
    ("miss_mm", "miss_stats", "distance to the MRI target"),
    ("pair_mm", "pair_stats", "distance between two placements"),
]


def render_distance_hist(mc, out_path, subject="", bins=30):
    """One row per target, one column per metric, all on shared axes.

    thin bars with a gap of surface between them, a solid line at the mean and a
    dotted one at the 95th percentile, and the numbers in the panel title rather
    than on every bar. same colour per target as the viewer uses
    """

    sites = mc["sites"]
    fig, axes = plt.subplots(len(sites), len(METRICS),
                             figsize=(11, 1.9 * len(sites) + 0.8), sharex="col", squeeze=False)
    fig.patch.set_facecolor("white")

    for col, (key, stats_key, title) in enumerate(METRICS):
        xmax = max(s[stats_key]["max_mm"] for s in sites)
        edges = np.linspace(0.0, np.ceil(xmax * 2.0) / 2.0, bins + 1)

        for row, site in enumerate(sites):
            ax = axes[row, col]
            d = np.asarray(site[key], dtype=float)
            st = site[stats_key]

            ax.hist(d, bins=edges, color=site["color"], rwidth=0.82, edgecolor="none")
            ax.axvline(st["mean_mm"], color="#333333", lw=1.0)
            ax.axvline(st["p95_mm"], color="#333333", lw=1.0, ls=(0, (1, 2)))

            opt = "  (optional marker)" if site.get("optional") else ""
            ax.set_title(
                f"{site['label']}{opt}   mean {st['mean_mm']:.2f} mm   sd {st['sd_mm']:.2f}"
                f"   p95 {st['p95_mm']:.2f}   n={st['n']}",
                fontsize=9, loc="left", color="#333333")
            ax.set_yticks([])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color("#bbbbbb")
            ax.tick_params(axis="x", colors="#555555", labelsize=8)
            ax.grid(axis="x", color="#eeeeee", lw=0.8)
            ax.set_axisbelow(True)

        axes[-1, col].set_xlabel(f"{title} [mm]", fontsize=9, color="#333333")

    head = f"{subject}  " if subject else ""
    fig.suptitle(f"{head}placement error over {mc['n_draws']} simulated caps"
                 f"   (solid = mean, dotted = 95th percentile)", fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(str(out_path), dpi=110)
    plt.close(fig)
    return out_path
