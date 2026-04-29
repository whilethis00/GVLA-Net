from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "figures"
TABLE_CSV = PROJECT_ROOT / "experiments" / "results" / "appendix_tables" / "action_codebook_proxy_table3.csv"

STYLE = {
    "GVLA-Net (Ours)": {
        "color": "#9f1d35",
        "marker": "o",
        "linewidth": 2.8,
        "linestyle": "-",
        "filled": True,
        "zorder": 5,
    },
    "LSH (Random Projection)": {
        "color": "#5f6368",
        "marker": "s",
        "linewidth": 1.9,
        "linestyle": (0, (4, 3)),
        "filled": False,
        "zorder": 4,
    },
    "PQ": {
        "color": "#c77700",
        "marker": "^",
        "linewidth": 1.9,
        "linestyle": (0, (5, 3)),
        "filled": False,
        "zorder": 4,
    },
}

BIT_MARKER_SIZE = {20: 72, 22: 102, 24: 138}


def load_rows() -> dict[str, list[dict[str, float | int | str]]]:
    grouped: dict[str, list[dict[str, float | int | str]]] = {}
    with TABLE_CSV.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = str(row["method"])
            parsed = {
                "method": method,
                "code_bits": int(row["code_bits"]),
                "unique_code_ratio": float(row["unique_code_ratio"]),
                "total_ms": float(row["total_ms"]),
            }
            grouped.setdefault(method, []).append(parsed)
    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["code_bits"]))
    return grouped


def add_better_region(ax: plt.Axes) -> None:
    ax.annotate(
        "Better",
        xy=(22.5, 0.975),
        xytext=(34.0, 0.92),
        color="#1d6f5f",
        fontsize=11,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color="#1d6f5f", lw=1.4, shrinkA=2, shrinkB=2),
    )


def add_point_labels(ax: plt.Axes, rows: list[dict[str, float | int | str]], method: str) -> None:
    for row in rows:
        x = float(row["total_ms"])
        y = float(row["unique_code_ratio"])
        bits = int(row["code_bits"])
        label = f"{bits}b"

        if method == "GVLA-Net (Ours)":
            dx, dy = 1.025, 0.010
        elif method == "LSH (Random Projection)":
            dx, dy = 1.025, -0.020
        else:
            dx, dy = 1.025, 0.008

        ax.text(
            x * dx,
            y + dy,
            label,
            color=STYLE[method]["color"],
            fontsize=9,
            fontweight="bold" if method == "GVLA-Net (Ours)" else "normal",
        )


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 9.5,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    grouped = load_rows()
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    for method, rows in grouped.items():
        style = STYLE[method]
        x = np.array([float(row["total_ms"]) for row in rows], dtype=float)
        y = np.array([float(row["unique_code_ratio"]) for row in rows], dtype=float)
        sizes = np.array([BIT_MARKER_SIZE[int(row["code_bits"])] for row in rows], dtype=float)

        ax.plot(
            x,
            y,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            zorder=style["zorder"],
        )
        ax.scatter(
            x,
            y,
            s=sizes,
            marker=style["marker"],
            facecolors=style["color"] if style["filled"] else "white",
            edgecolors=style["color"],
            linewidths=1.6,
            zorder=style["zorder"] + 0.1,
        )
        add_point_labels(ax, rows, method)

    add_better_region(ax)

    ax.set_xscale("log")
    ax.set_xlim(18, 360)
    ax.set_ylim(0.33, 1.02)
    ax.set_xticks([20, 40, 80, 160, 320])
    ax.set_xticklabels(["20", "40", "80", "160", "320"])
    ax.set_xlabel("Total Latency (ms, log scale)")
    ax.set_ylabel("Unique Code Ratio")
    ax.set_title("Pareto Frontier: Fast Routing vs. Collision-Free Capacity")

    ax.grid(True, which="major", axis="both", linewidth=0.7, alpha=0.35)
    ax.grid(True, which="minor", axis="x", linewidth=0.4, alpha=0.18)
    ax.text(
        0.99,
        0.02,
        "Up = fewer collisions, Left = faster",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color="#4d4d4d",
    )

    legend_handles = [
        Line2D([0], [0], color=STYLE["GVLA-Net (Ours)"]["color"], linestyle=STYLE["GVLA-Net (Ours)"]["linestyle"], linewidth=STYLE["GVLA-Net (Ours)"]["linewidth"], marker=STYLE["GVLA-Net (Ours)"]["marker"], markersize=7.0, markerfacecolor=STYLE["GVLA-Net (Ours)"]["color"], markeredgecolor=STYLE["GVLA-Net (Ours)"]["color"], label="GVLA-Net (Ours)"),
        Line2D([0], [0], color=STYLE["LSH (Random Projection)"]["color"], linestyle=STYLE["LSH (Random Projection)"]["linestyle"], linewidth=STYLE["LSH (Random Projection)"]["linewidth"], marker=STYLE["LSH (Random Projection)"]["marker"], markersize=6.8, markerfacecolor="white", markeredgecolor=STYLE["LSH (Random Projection)"]["color"], label="LSH (Random Projection)"),
        Line2D([0], [0], color=STYLE["PQ"]["color"], linestyle=STYLE["PQ"]["linestyle"], linewidth=STYLE["PQ"]["linewidth"], marker=STYLE["PQ"]["marker"], markersize=7.2, markerfacecolor="white", markeredgecolor=STYLE["PQ"]["color"], label="Product Quantization (PQ)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="#666666", markersize=5.5, label="20b"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="#666666", markersize=7.0, label="22b"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="#666666", markersize=8.2, label="24b"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=True, fancybox=False, edgecolor="#cccccc")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf_path = OUT_DIR / "fig_pareto_efficiency.pdf"
    png_path = OUT_DIR / "fig_pareto_efficiency.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")


if __name__ == "__main__":
    main()
