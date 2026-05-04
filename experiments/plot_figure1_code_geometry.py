"""
plot_figure1_code_geometry.py
=============================

Generate the main-paper concept figure:
  Figure 1: Bitwise factorization and code-induced supervision geometry
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "results" / "bc_study" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BLUE = "#155EEF"
GREEN = "#079455"
RED = "#D92D20"
ORANGE = "#F79009"
INK = "#101828"
MUTED = "#667085"
FOG = "#EAECF0"
PANEL_BG = "#FCFCFD"

plt.rcParams.update(
    {
        "font.size": 11.0,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13.0,
        "axes.labelsize": 11.0,
        "axes.edgecolor": "#D0D5DD",
        "axes.linewidth": 1.0,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
    }
)


def add_round_box(ax, xy, width, height, text, fc, ec="#D0D5DD", fontsize=10.5, weight="normal"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
    )


def add_arrow(ax, start, end, color=MUTED):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.6,
            color=color,
        )
    )


def draw_panel_a(ax):
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("A. Dense vs Bitwise Head", loc="left", pad=10)

    add_round_box(ax, (0.06, 0.72), 0.16, 0.14, "latent\n$z$", "#F2F4F7", fontsize=11.5, weight="bold")
    add_round_box(ax, (0.34, 0.70), 0.24, 0.18, "Dense head", "#FDEAD7", ec="#F7B27A", fontsize=11.5, weight="bold")
    add_round_box(ax, (0.73, 0.72), 0.20, 0.14, "$D \\times M$\nlogits", "#FFF6ED", ec="#F7B27A", fontsize=11.0)
    add_arrow(ax, (0.22, 0.79), (0.34, 0.79))
    add_arrow(ax, (0.58, 0.79), (0.73, 0.79))
    ax.text(0.34, 0.62, "per-dimension categorical prediction", fontsize=9.5, color=MUTED)

    add_round_box(ax, (0.06, 0.28), 0.16, 0.14, "latent\n$z$", "#F2F4F7", fontsize=11.5, weight="bold")
    add_round_box(ax, (0.34, 0.26), 0.24, 0.18, "Bitwise head", "#DDF3EA", ec="#84D5AE", fontsize=11.5, weight="bold")
    add_round_box(ax, (0.73, 0.28), 0.20, 0.14, "$D \\times k$\nlogits", "#ECFDF3", ec="#84D5AE", fontsize=11.0)
    add_arrow(ax, (0.22, 0.35), (0.34, 0.35))
    add_arrow(ax, (0.58, 0.35), (0.73, 0.35))
    ax.text(0.34, 0.18, "$k = \\lceil \\log_2 M \\rceil$ bits per dimension", fontsize=9.5, color=MUTED)

    ax.text(
        0.06,
        0.02,
        "Bitwise factorization reduces output size, but it also changes the supervision structure.",
        fontsize=9.5,
        color=INK,
    )


def draw_panel_b(ax):
    ax.set_facecolor(PANEL_BG)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_title("B. Target Code Determines Supervision Geometry", loc="left", pad=10)

    ax.text(0.07, 0.87, "Natural binary", fontsize=11.5, fontweight="bold", color=RED)
    add_round_box(ax, (0.07, 0.62), 0.16, 0.13, "3 = 011", "#FEF3F2", ec="#F97066", fontsize=11.5)
    add_round_box(ax, (0.30, 0.62), 0.16, 0.13, "4 = 100", "#FEF3F2", ec="#F97066", fontsize=11.5)
    add_arrow(ax, (0.23, 0.685), (0.30, 0.685), color=RED)
    ax.text(0.07, 0.52, "adjacent bins can jump across code space", fontsize=9.5, color=MUTED)
    ax.text(0.07, 0.43, "Hamming$(011, 100) = 3$", fontsize=10.5, color=RED, fontweight="bold")

    ax.text(0.57, 0.87, "Gray code", fontsize=11.5, fontweight="bold", color=GREEN)
    gray_rows = [("2", "011"), ("3", "010"), ("4", "110"), ("5", "111")]
    y0 = 0.72
    for idx, (bin_id, code) in enumerate(gray_rows):
        y = y0 - idx * 0.13
        ax.text(0.58, y, bin_id, fontsize=10.5, ha="left", va="center", color=MUTED)
        add_round_box(ax, (0.66, y - 0.045), 0.14, 0.09, code, "#ECFDF3", ec="#84D5AE", fontsize=11.0)
        if idx < len(gray_rows) - 1:
            add_arrow(ax, (0.73, y - 0.05), (0.73, y - 0.08), color=GREEN)
    ax.text(0.57, 0.18, "neighboring bins differ by 1 bit", fontsize=10.5, color=GREEN, fontweight="bold")


def draw_panel_c(ax):
    ax.set_facecolor(PANEL_BG)
    values = np.array([2.7796, 2.2483, 4.0069], dtype=float)
    labels = ["Natural", "Gray", "Random"]
    colors = [RED, GREEN, ORANGE]
    x = np.arange(len(values))

    bars = ax.bar(x, values, color=colors, width=0.62, edgecolor="none")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val + 0.07,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    ax.set_title("C. Demonstration Trajectories", loc="left", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean target bit flips")
    ax.set_ylim(0, 4.5)
    ax.grid(True, axis="y", color=FOG, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.02,
        0.95,
        "actual target-code transitions",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        color=MUTED,
    )


def main():
    fig = plt.figure(figsize=(14.0, 5.2), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.25, 1.0], wspace=0.24)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    draw_panel_a(ax_a)
    draw_panel_b(ax_b)
    draw_panel_c(ax_c)

    fig.suptitle(
        "Figure 1. Bitwise factorization and code-induced supervision geometry",
        x=0.01,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
    )

    fig.subplots_adjust(left=0.05, right=0.995, top=0.88, bottom=0.20, wspace=0.26)
    fig.savefig(OUT_DIR / "figure1_code_geometry.png", dpi=260, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure1_code_geometry_paper.png", dpi=340, bbox_inches="tight")
    fig.savefig(OUT_DIR / "figure1_code_geometry.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
