"""
synthetic_code_geometry.py
==========================

Visualise how natural binary and Gray code preserve locality over a 1D bin axis.
Produces:
  - adjacency Hamming distance comparison
  - distance-vs-distance scatter / trend
  - JSON summary with simple locality statistics
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "experiments" / "results" / "synthetic_code_geometry"


def int_to_gray(n: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(n, np.right_shift(n, 1))


def to_bits(values: np.ndarray, n_bits: int) -> np.ndarray:
    shifts = np.arange(n_bits - 1, -1, -1, dtype=np.int64)
    return ((values[:, None] >> shifts[None, :]) & 1).astype(np.int8)


def hamming_matrix(bits: np.ndarray) -> np.ndarray:
    return np.not_equal(bits[:, None, :], bits[None, :, :]).sum(axis=-1)


def distance_profile(hmat: np.ndarray) -> dict[str, list[float]]:
    n = hmat.shape[0]
    max_delta = n - 1
    mean_by_delta = []
    std_by_delta = []
    for delta in range(1, max_delta + 1):
        vals = np.diag(hmat, k=delta)
        mean_by_delta.append(float(vals.mean()))
        std_by_delta.append(float(vals.std()))
    return {"mean": mean_by_delta, "std": std_by_delta}


def summarize(bits: np.ndarray, label: str) -> dict[str, float | list[float] | str]:
    hmat = hamming_matrix(bits)
    adjacency = np.diag(hmat, k=1).astype(float)
    profile = distance_profile(hmat)
    return {
        "label": label,
        "n_bins": int(bits.shape[0]),
        "n_bits": int(bits.shape[1]),
        "adjacent_mean_hamming": float(adjacency.mean()),
        "adjacent_std_hamming": float(adjacency.std()),
        "adjacent_unique_hamming": sorted({int(v) for v in adjacency.tolist()}),
        "distance_profile_mean": profile["mean"],
        "distance_profile_std": profile["std"],
    }


def make_plot(n_bins: int, natural_bits: np.ndarray, gray_bits: np.ndarray, out_path: Path) -> None:
    natural_hmat = hamming_matrix(natural_bits)
    gray_hmat = hamming_matrix(gray_bits)

    deltas = np.arange(1, n_bins)
    natural_adj = np.diag(natural_hmat, k=1)
    gray_adj = np.diag(gray_hmat, k=1)
    natural_profile = distance_profile(natural_hmat)
    gray_profile = distance_profile(gray_hmat)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    ax = axes[0]
    idx = np.arange(n_bins - 1)
    ax.plot(idx, natural_adj, color="#c0392b", lw=1.8, label="Natural binary")
    ax.plot(idx, gray_adj, color="#0f8b8d", lw=1.8, label="Gray code")
    ax.set_title("Adjacent Bin Hamming Distance")
    ax.set_xlabel("Bin index i  for pair (i, i+1)")
    ax.set_ylabel("Hamming distance")
    ax.set_ylim(0.8, max(natural_adj.max(), gray_adj.max()) + 0.3)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[1]
    ax.plot(deltas, natural_profile["mean"], color="#c0392b", lw=2.0, label="Natural binary")
    ax.fill_between(
        deltas,
        np.array(natural_profile["mean"]) - np.array(natural_profile["std"]),
        np.array(natural_profile["mean"]) + np.array(natural_profile["std"]),
        color="#c0392b",
        alpha=0.15,
    )
    ax.plot(deltas, gray_profile["mean"], color="#0f8b8d", lw=2.0, label="Gray code")
    ax.fill_between(
        deltas,
        np.array(gray_profile["mean"]) - np.array(gray_profile["std"]),
        np.array(gray_profile["mean"]) + np.array(gray_profile["std"]),
        color="#0f8b8d",
        alpha=0.15,
    )
    ax.set_title("Code Distance vs Bin Distance")
    ax.set_xlabel("|i - j| along 1D output axis")
    ax.set_ylabel("Mean Hamming distance")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)

    ax = axes[2]
    pair_limit = min(n_bins, 64)
    natural_img = natural_hmat[:pair_limit, :pair_limit]
    gray_img = gray_hmat[:pair_limit, :pair_limit]
    combined = np.concatenate([natural_img, gray_img], axis=1)
    im = ax.imshow(combined, cmap="viridis", aspect="auto")
    ax.axvline(pair_limit - 0.5, color="white", lw=2)
    ax.set_title("Hamming Distance Matrix\nleft: natural, right: Gray")
    ax.set_xlabel("Bin index")
    ax.set_ylabel("Bin index")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Synthetic Code Geometry for {n_bins} bins ({natural_bits.shape[1]} bits)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    fig.savefig(out_path.with_name(out_path.stem + "_paper.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    n_bins = 128
    n_bits = int(np.ceil(np.log2(n_bins)))
    values = np.arange(n_bins, dtype=np.int64)

    natural_bits = to_bits(values, n_bits)
    gray_bits = to_bits(int_to_gray(values), n_bits)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = {
        "natural_binary": summarize(natural_bits, "natural_binary"),
        "gray_code": summarize(gray_bits, "gray_code"),
    }

    out_json = OUT_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    make_plot(n_bins, natural_bits, gray_bits, OUT_DIR / "synthetic_code_geometry.png")
    print(f"Saved summary → {out_json}")
    print(f"Saved figure  → {OUT_DIR / 'synthetic_code_geometry.png'}")


if __name__ == "__main__":
    main()
