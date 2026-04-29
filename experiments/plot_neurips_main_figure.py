"""
plot_neurips_main_figure.py
===========================

NeurIPS-style main figure:
  A. Precision-sensitive regime needs high resolution
  B. Gray code fixes code-geometry mismatch
  C. Large-batch scaling favors GVLA
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments" / "results"
OUT_DIR = RESULTS / "neurips_main_figure"


def wilson_interval(successes, total, z=1.96):
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    radius = z * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total) / denom
    return center - radius, center + radius


def load_precision_panel():
    data = json.loads((RESULTS / "precision_custom25_200roll_core.json").read_text())
    rows = data["results"]
    labels = [row["label"] for row in rows]
    rates = np.array([row["success_rate"] * 100 for row in rows])
    lows = np.array([(row["success_rate"] - row["ci95_low"]) * 100 for row in rows])
    highs = np.array([(row["ci95_high"] - row["success_rate"]) * 100 for row in rows])
    return labels, rates, np.vstack([lows, highs])


def load_gray_panel():
    data = json.loads((RESULTS / "bc_study" / "eval_results.json").read_text())["rollout_results"]
    lookup = {row["exp_name"]: row for row in data}
    m_values = [128, 1024]
    variants = [
        ("dense", "Dense", "#d1495b"),
        ("gvla", "GVLA", "#2b2d42"),
        ("gvla_gray", "GVLA + Gray", "#2a9d8f"),
    ]
    panel = []
    for m in m_values:
        group = []
        for key, label, color in variants:
            row = lookup["%s_%d" % (key, m)]
            lo, hi = wilson_interval(row["successes"], row["n_rollouts"])
            group.append(
                {
                    "variant": key,
                    "label": label,
                    "color": color,
                    "rate": row["success_rate"] * 100,
                    "err_low": (row["success_rate"] - lo) * 100,
                    "err_high": (hi - row["success_rate"]) * 100,
                }
            )
        panel.append((m, group))
    return panel


def load_scaling_panel():
    data = json.loads((RESULTS / "bc_study" / "latency_batch.json").read_text())
    batch = "1024"
    m_values = [128, 512, 1024, 4096, 16384, 65536]
    dense = np.array([data["dense"][batch][str(m)] for m in m_values])
    gvla = np.array([data["gvla"][batch][str(m)] for m in m_values])
    return m_values, dense, gvla


def plot():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    precision_labels, precision_rates, precision_err = load_precision_panel()
    gray_panel = load_gray_panel()
    scaling_m, scaling_dense, scaling_gvla = load_scaling_panel()

    fig = plt.figure(figsize=(15.5, 5.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.08, 1.05, 1.15], wspace=0.34)

    # Panel A
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(precision_labels))
    colors = ["#6c757d", "#d1495b", "#2a9d8f", "#264653"]
    ax.bar(x, precision_rates, color=colors, width=0.72, zorder=3)
    ax.errorbar(
        x,
        precision_rates,
        yerr=precision_err,
        fmt="none",
        ecolor="black",
        elinewidth=1.4,
        capsize=4,
        zorder=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(precision_labels, fontsize=11)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_ylim(0, 70)
    ax.set_title("A. Precision-Sensitive Control", fontsize=13, loc="left", pad=10)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    for xi, yi, err_hi in zip(x, precision_rates, precision_err[1]):
        ax.text(xi, yi + err_hi + 1.4, "%.1f%%" % yi, ha="center", va="bottom", fontsize=10)

    # Panel B
    ax = fig.add_subplot(gs[0, 1])
    group_centers = np.arange(len(gray_panel))
    width = 0.22
    offsets = [-width, 0.0, width]
    for idx, (m, group) in enumerate(gray_panel):
        for off, row in zip(offsets, group):
            xpos = group_centers[idx] + off
            ax.bar(xpos, row["rate"], width=width * 0.92, color=row["color"], zorder=3)
            ax.errorbar(
                xpos,
                row["rate"],
                yerr=[[row["err_low"]], [row["err_high"]]],
                fmt="none",
                ecolor="black",
                elinewidth=1.2,
                capsize=3,
                zorder=4,
            )
            ax.text(
                xpos,
                row["rate"] + row["err_high"] + 1.2,
                "%.0f%%" % row["rate"],
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xticks(group_centers)
    ax.set_xticklabels(["M=128", "M=1024"], fontsize=11)
    ax.set_ylim(0, 35)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_title("B. Code Geometry Ablation", fontsize=13, loc="left", pad=10)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=row["color"]) for row in gray_panel[0][1]]
    labels = [row["label"] for row in gray_panel[0][1]]
    ax.legend(handles, labels, frameon=False, fontsize=10, loc="upper right")

    # Panel C
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(scaling_m, scaling_dense, "o-", color="#d1495b", lw=2.6, ms=7, label="Dense")
    ax.plot(scaling_m, scaling_gvla, "s-", color="#2a9d8f", lw=2.6, ms=7, label="GVLA")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlim(min(scaling_m) * 0.9, max(scaling_m) * 1.1)
    ax.xaxis.set_major_locator(mticker.FixedLocator(scaling_m))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: "%d" % int(x)))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel("Bins per action dim (M)", fontsize=12)
    ax.set_ylabel("Latency (ms, batch=1024)", fontsize=12)
    ax.set_title("C. Large-Batch Scaling", fontsize=13, loc="left", pad=10)
    ax.grid(True, alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=10, loc="upper left")

    fig.suptitle(
        "Why Large Structured Action Spaces Matter",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    try:
        fig.tight_layout()
    except Exception:
        fig.subplots_adjust(left=0.06, right=0.99, bottom=0.13, top=0.86, wspace=0.34)

    fig.savefig(OUT_DIR / "neurips_main_figure.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT_DIR / "neurips_main_figure_paper.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved → %s" % (OUT_DIR / "neurips_main_figure.png"))


if __name__ == "__main__":
    plot()
