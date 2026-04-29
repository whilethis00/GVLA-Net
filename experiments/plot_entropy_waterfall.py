import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a 3D entropy waterfall for bit-conditioned action posterior collapse."
    )
    parser.add_argument("--total-bits", type=int, default=24)
    parser.add_argument("--num-points", type=int, default=2049)
    parser.add_argument("--sigma", type=float, default=0.28)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-name", type=str, default="fig_entropy_waterfall_3d")
    return parser.parse_args()


def truncated_posterior_grid(x: np.ndarray, sigma: float, bit: int) -> np.ndarray:
    base = np.exp(-0.5 * (x / sigma) ** 2)
    interval_half_width = 2.0 ** (-bit)
    mask = np.abs(x) <= interval_half_width
    probs = base * mask.astype(np.float64)
    if np.count_nonzero(mask) <= 1 or probs.sum() == 0.0:
        probs[:] = 0.0
        probs[np.argmin(np.abs(x))] = 1.0
        return probs
    probs /= probs.sum()
    return probs


def write_summary_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    fieldnames = ["bit", "entropy_nats", "entropy_bits", "peak_probability", "support_fraction"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_waterfall(x: np.ndarray, distributions: list[np.ndarray], summary_rows: list[dict[str, float]], output_png: Path, output_pdf: Path) -> None:
    fig = plt.figure(figsize=(13, 8.5))
    ax = fig.add_subplot(111, projection="3d")

    cmap = plt.get_cmap("inferno")
    x_plot = x
    bit_values = np.arange(1, len(distributions) + 1)

    for i, probs in enumerate(distributions):
        color = cmap(0.08 + 0.85 * i / max(1, len(distributions) - 1))
        y = np.full_like(x_plot, bit_values[i], dtype=np.float64)
        ax.plot(x_plot, y, probs, color=color, linewidth=1.9, alpha=0.98)
        ax.plot(x_plot, y, np.zeros_like(probs), color=color, linewidth=0.4, alpha=0.10)

    peak_rows = [1, 4, 8, 12, 16, 20, 24]
    for bit in peak_rows:
        probs = distributions[bit - 1]
        peak_idx = int(np.argmax(probs))
        peak_x = x_plot[peak_idx]
        peak_z = probs[peak_idx]
        color = cmap(0.08 + 0.85 * (bit - 1) / max(1, len(distributions) - 1))
        ax.scatter([peak_x], [bit], [peak_z], color=color, s=22, depthshade=False)

    ax.set_title("The Entropy Waterfall: Posterior Collapse Under Bit Expansion", pad=20, fontsize=18)
    ax.set_xlabel("Candidate Offset Around Target", labelpad=12)
    ax.set_ylabel("Routing Bits", labelpad=10)
    ax.set_zlabel("Normalized Posterior Probability", labelpad=10)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(1, len(distributions))
    ax.set_yticks([1, 4, 8, 12, 16, 20, 24])
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticklabels(["far left", "", "target", "", "far right"])
    ax.view_init(elev=28, azim=-63)
    ax.xaxis.pane.set_alpha(0.03)
    ax.yaxis.pane.set_alpha(0.03)
    ax.zaxis.pane.set_alpha(0.04)
    ax.grid(True, alpha=0.15)

    entropy_bits = np.array([row["entropy_bits"] for row in summary_rows], dtype=np.float64)
    peak_prob = np.array([row["peak_probability"] for row in summary_rows], dtype=np.float64)
    fig.text(
        0.79,
        0.82,
        f"Entropy: {entropy_bits[0]:.2f}b → {entropy_bits[-1]:.2f}b\nPeak prob: {peak_prob[0]:.3f} → {peak_prob[-1]:.3f}",
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.92},
    )

    fig.tight_layout()
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-1.0, 1.0, args.num_points, dtype=np.float64)
    distributions: list[np.ndarray] = []
    summary_rows: list[dict[str, float]] = []

    for bit in range(1, args.total_bits + 1):
        probs = truncated_posterior_grid(x, sigma=args.sigma, bit=bit)
        entropy_nats = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum())
        summary_rows.append(
            {
                "bit": float(bit),
                "entropy_nats": entropy_nats,
                "entropy_bits": entropy_nats / np.log(2.0),
                "peak_probability": float(probs.max()),
                "support_fraction": float(np.count_nonzero(probs > 0) / probs.size),
            }
        )
        distributions.append(probs)

    output_png = args.output_dir / f"{args.figure_name}.png"
    output_pdf = args.output_dir / f"{args.figure_name}.pdf"
    output_csv = args.output_dir / f"{args.figure_name}.csv"

    write_summary_csv(summary_rows, output_csv)
    plot_waterfall(x, distributions, summary_rows, output_png, output_pdf)

    print(f"Summary written to: {output_csv}")
    print(f"Figure written to: {output_png}")
    print(f"Figure written to: {output_pdf}")


if __name__ == "__main__":
    main()
