import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure bit-conditioned posterior collapse on a real 24-bit scalar action lattice."
    )
    parser.add_argument("--total-bits", type=int, default=24)
    parser.add_argument("--target", type=float, default=0.173)
    parser.add_argument("--sigma", type=float, default=0.18)
    parser.add_argument("--display-points", type=int, default=1024)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-name", type=str, default="fig_entropy_waterfall_measured")
    return parser.parse_args()


def stable_probs(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values)


def write_csv(rows: list[dict[str, float]], output_path: Path) -> None:
    fieldnames = [
        "bit",
        "candidate_count",
        "support_fraction",
        "entropy_nats",
        "entropy_bits",
        "peak_probability",
        "target_bucket_start",
        "target_bucket_end",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_waterfall(
    sampled_offsets: list[np.ndarray],
    sampled_probs: list[np.ndarray],
    summary_rows: list[dict[str, float]],
    output_png: Path,
    output_pdf: Path,
) -> None:
    fig = plt.figure(figsize=(15.4, 8.6))
    gs = gridspec.GridSpec(
        2,
        2,
        width_ratios=[2.35, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.08,
        hspace=0.18,
    )
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ax_entropy = fig.add_subplot(gs[0, 1])
    ax_peak = fig.add_subplot(gs[1, 1])
    cmap = plt.get_cmap("inferno")

    verts = []
    facecolors = []
    edge_x = []
    edge_y = []
    edge_z = []
    for idx, (x, z) in enumerate(zip(sampled_offsets, sampled_probs), start=1):
        color = cmap(0.08 + 0.86 * (idx - 1) / max(1, len(sampled_offsets) - 1))
        polygon = [(float(xi), float(zi)) for xi, zi in zip(x, z)]
        polygon = [(float(x[0]), 0.0)] + polygon + [(float(x[-1]), 0.0)]
        verts.append(polygon)
        facecolors.append((color[0], color[1], color[2], 0.42))
        edge_x.append(x)
        edge_y.append(np.full_like(x, idx, dtype=np.float64))
        edge_z.append(z)

    collection = PolyCollection(verts, facecolors=facecolors, edgecolors="none")
    ax.add_collection3d(collection, zs=np.arange(1, len(sampled_offsets) + 1), zdir="y")

    for idx, (x, y, z) in enumerate(zip(edge_x, edge_y, edge_z), start=1):
        color = cmap(0.08 + 0.86 * (idx - 1) / max(1, len(sampled_offsets) - 1))
        ax.plot(x, y, z, color=color, linewidth=1.5, alpha=0.95)
        ax.plot([x[0], x[-1]], [idx, idx], [0.0, 0.0], color=(0, 0, 0, 0.08), linewidth=0.6)

    for bit in [1, 4, 8, 12, 16, 20, 24]:
        row = summary_rows[bit - 1]
        color = cmap(0.12 + 0.82 * (bit - 1) / max(1, len(sampled_offsets) - 1))
        ax.scatter([0.0], [bit], [row["peak_probability"]], color=color, s=26, depthshade=False)

    ax.set_title("Measured Entropy Waterfall on a 24-bit Action Lattice", pad=18, fontsize=18)
    ax.set_xlabel("Normalized Offset From Target Action", labelpad=10)
    ax.set_ylabel("Routing Bits", labelpad=10)
    ax.set_zlabel("Posterior Probability", labelpad=10)
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(1, len(sampled_offsets))
    ax.set_yticks([1, 4, 8, 12, 16, 20, 24])
    ax.set_zlim(0.0, 1.04)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_xticklabels(["far left", "", "target", "", "far right"])
    ax.view_init(elev=24, azim=-57)
    ax.grid(True, alpha=0.10)
    ax.xaxis.pane.set_alpha(0.02)
    ax.yaxis.pane.set_alpha(0.02)
    ax.zaxis.pane.set_alpha(0.03)
    ax.xaxis.line.set_color((0, 0, 0, 0.35))
    ax.yaxis.line.set_color((0, 0, 0, 0.35))
    ax.zaxis.line.set_color((0, 0, 0, 0.35))

    entropy_bits = np.array([row["entropy_bits"] for row in summary_rows], dtype=np.float64)
    candidate_counts = np.array([row["candidate_count"] for row in summary_rows], dtype=np.float64)
    peak_probs = np.array([row["peak_probability"] for row in summary_rows], dtype=np.float64)
    bit_axis = np.arange(1, len(summary_rows) + 1, dtype=np.float64)

    gvla_color = "#b73a52"
    accent_color = "#d68c1f"
    dark_color = "#24292f"
    for axis in (ax_entropy, ax_peak):
        axis.grid(True, alpha=0.22, linewidth=0.8)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.set_xlim(1, len(summary_rows))
        axis.set_xticks([1, 4, 8, 12, 16, 20, 24])

    ax_entropy.plot(bit_axis, entropy_bits, color=gvla_color, linewidth=2.6, marker="o", markersize=4.8)
    ax_entropy.fill_between(bit_axis, entropy_bits, color=gvla_color, alpha=0.14)
    ax_entropy.set_title("Entropy Collapse")
    ax_entropy.set_ylabel("Posterior Entropy (bits)")
    ax_entropy.set_xticklabels([])
    ax_entropy.annotate(
        "near-log2 support",
        xy=(6, entropy_bits[5]),
        xytext=(8.5, entropy_bits[5] + 1.7),
        fontsize=9.5,
        color=dark_color,
        arrowprops={"arrowstyle": "->", "color": dark_color, "lw": 1.1},
    )
    ax_entropy.annotate(
        "single-cell identification",
        xy=(24, entropy_bits[-1]),
        xytext=(14.5, 5.4),
        fontsize=9.5,
        color=accent_color,
        arrowprops={"arrowstyle": "->", "color": accent_color, "lw": 1.1},
    )

    ax_peak.plot(bit_axis, peak_probs, color=accent_color, linewidth=2.6, marker="o", markersize=4.8)
    ax_peak.fill_between(bit_axis, peak_probs, color=accent_color, alpha=0.16)
    ax_peak.set_title("Confidence Sharpening")
    ax_peak.set_xlabel("Routing Bits")
    ax_peak.set_ylabel("Peak Posterior Probability")
    ax_peak.set_ylim(0.0, 1.03)
    ax_peak.annotate(
        "delta-like spike",
        xy=(24, peak_probs[-1]),
        xytext=(15.2, 0.68),
        fontsize=9.5,
        color=accent_color,
        arrowprops={"arrowstyle": "->", "color": accent_color, "lw": 1.1},
    )

    fig.text(
        0.77,
        0.86,
        (
            f"Candidates: {int(candidate_counts[0]):,} → {int(candidate_counts[-1])}\n"
            f"Entropy: {entropy_bits[0]:.2f}b → {entropy_bits[-1]:.2f}b\n"
            f"Peak prob: {peak_probs[0]:.4f} → {peak_probs[-1]:.4f}"
        ),
        ha="left",
        va="top",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.94},
    )

    fig.subplots_adjust(left=0.03, right=0.97, top=0.93, bottom=0.08)
    fig.savefig(output_png, dpi=240, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    num_actions = 1 << args.total_bits
    actions = np.linspace(-1.0, 1.0, num_actions, dtype=np.float32)
    logits = -0.5 * ((actions - args.target) / args.sigma) ** 2
    target_index = int(np.argmin(np.abs(actions - args.target)))

    summary_rows: list[dict[str, float]] = []
    sampled_offsets: list[np.ndarray] = []
    sampled_probs: list[np.ndarray] = []

    for bit in range(1, args.total_bits + 1):
        bucket_size = max(1, num_actions >> bit)
        bucket_start = (target_index // bucket_size) * bucket_size
        bucket_end = min(num_actions, bucket_start + bucket_size)

        local_logits = logits[bucket_start:bucket_end].astype(np.float64, copy=False)
        probs = stable_probs(local_logits)

        entropy_nats = float(-(probs * np.log(probs + 1e-32)).sum())
        peak_probability = float(probs.max())

        if probs.size > args.display_points:
            sample_idx = np.linspace(0, probs.size - 1, args.display_points, dtype=np.int64)
            local_probs = probs[sample_idx]
            local_actions = actions[bucket_start:bucket_end][sample_idx]
        else:
            local_probs = probs
            local_actions = actions[bucket_start:bucket_end]

        offsets = np.linspace(-1.0, 1.0, local_probs.shape[0], dtype=np.float64)
        sampled_offsets.append(offsets)
        sampled_probs.append(local_probs)

        summary_rows.append(
            {
                "bit": float(bit),
                "candidate_count": float(bucket_end - bucket_start),
                "support_fraction": float((bucket_end - bucket_start) / num_actions),
                "entropy_nats": entropy_nats,
                "entropy_bits": entropy_nats / np.log(2.0),
                "peak_probability": peak_probability,
                "target_bucket_start": float(bucket_start),
                "target_bucket_end": float(bucket_end),
            }
        )

    output_png = args.output_dir / f"{args.figure_name}.png"
    output_pdf = args.output_dir / f"{args.figure_name}.pdf"
    output_csv = args.output_dir / f"{args.figure_name}.csv"

    write_csv(summary_rows, output_csv)
    plot_waterfall(sampled_offsets, sampled_probs, summary_rows, output_png, output_pdf)

    print(f"Summary written to: {output_csv}")
    print(f"Figure written to: {output_png}")
    print(f"Figure written to: {output_pdf}")
    print(f"Measured action space size: {num_actions}")
    print(f"Target action index: {target_index}")


if __name__ == "__main__":
    main()
