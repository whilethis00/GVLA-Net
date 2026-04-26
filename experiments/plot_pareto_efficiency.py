import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "experiments" / "results" / "figures"


def main() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )

    data = {
        "GVLA-Net (Ours)": {
            "bits": ["20b", "22b", "24b"],
            "latency_ms": [12.78, 21.52, 24.07],
            "unique_ratio": [0.6325, 0.8852, 0.9695],
            "color": "#d1495b",
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.6,
            "marker_sizes": [6.4, 7.8, 9.2],
            "zorder": 4,
        },
        "LSH (Random Projection)": {
            "bits": ["20b", "22b", "24b"],
            "latency_ms": [24.78, 15.51, 21.25],
            "unique_ratio": [0.3748, 0.5405, 0.7649],
            "color": "#6c757d",
            "linestyle": (0, (4, 3)),
            "marker": "o",
            "linewidth": 1.9,
            "marker_sizes": [6.4, 7.8, 9.2],
            "zorder": 3,
        },
        "Product Quantization (PQ)": {
            "bits": ["20b", "24b"],
            "latency_ms": [208.05, 262.38],
            "unique_ratio": [0.6306, 0.9691],
            "color": "#f4a261",
            "linestyle": (0, (5, 3)),
            "marker": "o",
            "linewidth": 1.9,
            "marker_sizes": [6.4, 9.2],
            "zorder": 3,
        },
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    x_min, x_max = 9.0, 320.0
    y_min, y_max = 0.0, 0.68

    for method, spec in data.items():
        x = np.array(spec["latency_ms"], dtype=float)
        y = 1.0 - np.array(spec["unique_ratio"], dtype=float)
        ax.plot(
            x,
            y,
            color=spec["color"],
            linestyle=spec["linestyle"],
            linewidth=spec["linewidth"],
            label=method,
            zorder=spec["zorder"],
        )
        ax.scatter(
            x,
            y,
            s=np.square(np.array(spec["marker_sizes"]) * 1.9),
            marker=spec["marker"],
            facecolors="white" if method != "GVLA-Net (Ours)" else spec["color"],
            edgecolors=spec["color"],
            linewidths=1.5,
            zorder=spec["zorder"] + 0.1,
        )

    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([10, 20, 50, 100, 200])
    ax.set_xticklabels(["10", "20", "50", "100", "200"])
    ax.set_xlabel("Encode Latency (ms)")
    ax.set_ylabel("Collision Rate")

    ax.grid(True, which="major", axis="both", linewidth=0.7, alpha=0.35)
    ax.grid(True, which="minor", axis="x", linewidth=0.4, alpha=0.18)
    legend_handles = [
        Line2D([0], [0], color="#d1495b", linestyle="-", linewidth=2.6, marker="o", markersize=6.6, markerfacecolor="#d1495b", markeredgecolor="#d1495b", label="GVLA-Net (Ours)"),
        Line2D([0], [0], color="#6c757d", linestyle=(0, (4, 3)), linewidth=1.9, marker="o", markersize=6.6, markerfacecolor="white", markeredgecolor="#6c757d", label="LSH (Random Projection)"),
        Line2D([0], [0], color="#f4a261", linestyle=(0, (5, 3)), linewidth=1.9, marker="o", markersize=6.6, markerfacecolor="white", markeredgecolor="#f4a261", label="Product Quantization (PQ)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="#666666", markersize=6.4, label="20b"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="#666666", markersize=7.8, label="22b"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666", markeredgecolor="#666666", markersize=9.2, label="24b"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        frameon=True,
        fancybox=False,
        edgecolor="#cccccc",
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

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
