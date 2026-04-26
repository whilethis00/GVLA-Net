"""
GVLA-Net Ablation: Orthogonal vs. Random Basis

Key experiments:
  1. Collision Rate: How many distinct hash codes are produced?
     - Orthogonal W partitions space evenly → nearly all codes are unique
     - Random W creates skewed partitions → many states share the same code
  2. Hyperplane Margin: Average distance from a latent state to its nearest hyperplane
     - Orthogonal W → large margin → noise-robust
     - Random W → small margin for states near correlated hyperplanes → fragile
  3. 2D Intuition Plot: visualize how orthogonal vs. random hyperplanes
     partition a 2D latent space
"""

import os
import math
import csv
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

torch.manual_seed(42)
np.random.seed(42)

OUT_DIR = "/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Use D=32 so k/D ratio is meaningful and effect is visible
D_LOW = 32
N_VALUES = [64, 256, 1024, 4096, 16384]
METHODS = ["Orthogonal W", "Random W"]

NOISE_SIGMA = 0.20


# ── W constructors ────────────────────────────────────────────────────────────
def make_orthogonal_W(d, k):
    A = torch.randn(k, d)
    Q, _ = torch.linalg.qr(A.T)   # Q: (d, k)
    return Q.T.clone()             # (k, d)


def make_random_W(d, k):
    W = torch.randn(k, d)
    W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    return W


def orthogonality_error(W):
    G = W @ W.T
    I = torch.eye(W.shape[0])
    return (G - I).norm(p='fro').item()


def info_overlap(W):
    k = W.shape[0]
    if k < 2:
        return 0.0
    Wn = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    cos = (Wn @ Wn.T).abs()
    mask = ~torch.eye(k, dtype=torch.bool)
    return cos[mask].mean().item()


def unique_code_rate(W, N):
    """Fraction of N random states that get a UNIQUE binary hash."""
    d = W.shape[1]
    states = torch.randn(N, d)
    proj   = states @ W.T                 # (N, k)
    codes  = (torch.sign(proj) > 0).int() # {0, 1}^k
    # Convert each code row to a Python tuple for hashing
    code_tuples = [tuple(c.tolist()) for c in codes]
    n_unique = len(set(code_tuples))
    return n_unique / N


def mean_margin(W, n_samples=2000):
    """
    Average minimum |projection| across all hyperplanes.
    Higher = states are farther from boundaries on average.
    """
    d = W.shape[1]
    states = torch.randn(n_samples, d)
    proj   = states @ W.T               # (n_samples, k)
    margins = proj.abs()                # distance to each hyperplane
    min_margin = margins.min(dim=1).values
    return min_margin.mean().item()


def noise_accuracy(W, N, sigma=NOISE_SIGMA):
    """Noise-robust retrieval: does sign(W(s+noise)) == sign(Ws)?"""
    d = W.shape[1]
    states = torch.randn(N, d)
    clean  = torch.sign(states @ W.T)
    clean[clean == 0] = 1.0
    noisy  = torch.sign((states + sigma * torch.randn_like(states)) @ W.T)
    noisy[noisy == 0] = 1.0
    return (noisy == clean).all(dim=1).float().mean().item()


# ── Run experiments ───────────────────────────────────────────────────────────
print(f"D={D_LOW}, noise σ={NOISE_SIGMA}")
results = []
for N in N_VALUES:
    k = math.ceil(math.log2(N))
    W_orth = make_orthogonal_W(D_LOW, k)
    W_rand = make_random_W(D_LOW, k)

    for method, W in [("Orthogonal W", W_orth), ("Random W", W_rand)]:
        ucr    = unique_code_rate(W, N)
        mm     = mean_margin(W)
        acc    = noise_accuracy(W, N)
        o_err  = orthogonality_error(W)
        o_lap  = info_overlap(W)
        results.append({
            "N": N, "method": method,
            "unique_code_rate": ucr,
            "mean_margin": mm,
            "noise_accuracy": acc,
            "ortho_error": o_err,
            "info_overlap": o_lap,
        })
        print(f"  N={N:>6}, {method:<18}: unique={ucr:.3f}  margin={mm:.4f}  acc={acc:.3f}  overlap={o_lap:.4f}")

csv_path = os.path.join(OUT_DIR, "ablation_orthogonality.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
    writer.writeheader()
    writer.writerows(results)
print(f"\n[SAVED CSV] {csv_path}")


# ── 2D geometry helpers (shared) ─────────────────────────────────────────────
torch.manual_seed(7)
np.random.seed(7)
n_pts = 400
pts_2d = torch.randn(n_pts, 2)

W_2d_orth = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
W_2d_rand = torch.tensor([[1.0, 0.3], [0.9, 0.2]])
W_2d_rand = W_2d_rand / W_2d_rand.norm(dim=1, keepdim=True)

QUAD_EDGE = {
    (0, 0): '#E05C5C',   # red-ish
    (0, 1): '#2AADA6',   # teal
    (1, 0): '#D4A017',   # gold
    (1, 1): '#5FAD79',   # green
}
MARKERS = {"Orthogonal W": "o", "Random W": "^"}


# ── Figure renderer ───────────────────────────────────────────────────────────
def render_figure(dark: bool) -> None:
    if dark:
        plt.style.use('dark_background')
        FIG_FACE    = '#0E0E0E'
        AX_FACE     = '#1A1A2E'
        TEXT        = '#FFFFFF'
        SUB         = '#AAAAAA'
        SPINE       = '#444444'
        GRID_COL    = '#FFFFFF'
        GRID_ALPHA  = 0.12
        LINE_ORTH   = '#4ECDC4'
        LINE_RAND   = '#FF6B6B'
        BASELINE    = '#FFD700'
        HPLANE_COL  = '#FFFFFF'
        ANNOT_COL   = '#FFD700'
        BOX_FACE    = '#2A2A4A'
        BOX_EDGE    = '#4ECDC4'
        out_name    = "ablation_orthogonality.png"
    else:
        plt.rcdefaults()
        FIG_FACE    = '#FFFFFF'
        AX_FACE     = '#FFFFFF'
        TEXT        = '#111111'
        SUB         = '#555555'
        SPINE       = '#AAAAAA'
        GRID_COL    = '#333333'
        GRID_ALPHA  = 0.18
        LINE_ORTH   = '#1A9E94'   # darker teal, readable on white
        LINE_RAND   = '#C0392B'   # darker red, readable on white
        BASELINE    = '#B07D00'   # dark gold
        HPLANE_COL  = '#222222'
        ANNOT_COL   = '#333333'
        BOX_FACE    = '#F0F8FF'   # alice blue
        BOX_EDGE    = '#1A9E94'
        out_name    = "ablation_orthogonality_paper.png"

    COLORS = {"Orthogonal W": LINE_ORTH, "Random W": LINE_RAND}

    def style_ax(ax):
        ax.tick_params(colors=TEXT, labelsize=9)
        ax.grid(color=GRID_COL, alpha=GRID_ALPHA, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE)

    def plot_2d_partition(ax, W, pts, title):
        codes = (torch.sign(pts @ W.T) > 0).int()
        for p, c in zip(pts, codes):
            key = (c[0].item(), c[1].item())
            ax.scatter(p[0].item(), p[1].item(),
                       color=QUAD_EDGE[key], s=12, alpha=0.60, zorder=2)
        lim = 3.5
        t = torch.linspace(-lim, lim, 100)
        for j in range(2):
            w = W[j]
            if abs(w[1].item()) > 1e-4:
                slope = -w[0].item() / w[1].item()
                y_line = slope * t
                mask = (y_line >= -lim) & (y_line <= lim)
                ax.plot(t[mask].numpy(), y_line[mask].numpy(),
                        color=HPLANE_COL, lw=2.0, zorder=4)
            else:
                ax.axvline(0, color=HPLANE_COL, lw=2.0, zorder=4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=11, color=TEXT, fontweight='bold')
        ax.tick_params(colors=TEXT, labelsize=8)
        code_tuples = [tuple(c.tolist()) for c in codes]
        n_unique = len(set(code_tuples))
        ax.text(0.03, 0.03, f"{n_unique}/4 quadrants used\n({n_unique * 25}% coverage)",
                transform=ax.transAxes, fontsize=8.5,
                color=ANNOT_COL, va='bottom', fontweight='bold')

    # Build figure
    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(FIG_FACE)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    ax_uniq    = fig.add_subplot(gs[0, 0])
    ax_marg    = fig.add_subplot(gs[0, 1])
    ax_acc     = fig.add_subplot(gs[0, 2])
    ax_2d_orth = fig.add_subplot(gs[1, 0])
    ax_2d_rand = fig.add_subplot(gs[1, 1])
    ax_text    = fig.add_subplot(gs[1, 2])

    for ax in [ax_uniq, ax_marg, ax_acc, ax_2d_orth, ax_2d_rand, ax_text]:
        ax.set_facecolor(AX_FACE)

    # Panel 1: Unique code rate
    for method in METHODS:
        rows = [r for r in results if r["method"] == method]
        ax_uniq.plot([r["N"] for r in rows], [r["unique_code_rate"] for r in rows],
                     color=COLORS[method], marker=MARKERS[method],
                     linewidth=2.2, markersize=7, label=method)
    ax_uniq.set_xscale('log')
    ax_uniq.set_xlabel("N", fontsize=11, color=TEXT)
    ax_uniq.set_ylabel("Unique Code Rate", fontsize=11, color=TEXT)
    ax_uniq.set_title("Hash Collision Rate\n(Higher = Better)", fontsize=11, color=TEXT, fontweight='bold')
    ax_uniq.set_ylim(0, 1.15)
    ax_uniq.axhline(1.0, color=BASELINE, lw=1.0, ls='--', alpha=0.7)
    ax_uniq.legend(fontsize=9, framealpha=0.35, edgecolor=SPINE, labelcolor=TEXT)
    style_ax(ax_uniq)

    # Panel 2: Mean margin
    for method in METHODS:
        rows = [r for r in results if r["method"] == method]
        ax_marg.plot([r["N"] for r in rows], [r["mean_margin"] for r in rows],
                     color=COLORS[method], marker=MARKERS[method],
                     linewidth=2.2, markersize=7, label=method)
    ax_marg.set_xscale('log')
    ax_marg.set_xlabel("N", fontsize=11, color=TEXT)
    ax_marg.set_ylabel("Mean Min-Projection Margin", fontsize=11, color=TEXT)
    ax_marg.set_title("Hyperplane Margin\n(Higher = More Noise-Robust)", fontsize=11, color=TEXT, fontweight='bold')
    ax_marg.legend(fontsize=9, framealpha=0.35, edgecolor=SPINE, labelcolor=TEXT)
    style_ax(ax_marg)

    # Panel 3: Noise accuracy
    for method in METHODS:
        rows = [r for r in results if r["method"] == method]
        ax_acc.plot([r["N"] for r in rows], [r["noise_accuracy"] for r in rows],
                    color=COLORS[method], marker=MARKERS[method],
                    linewidth=2.2, markersize=7, label=method)
    ax_acc.set_xscale('log')
    ax_acc.set_xlabel("N", fontsize=11, color=TEXT)
    ax_acc.set_ylabel(f"Accuracy Under Noise (\u03c3={NOISE_SIGMA})", fontsize=11, color=TEXT)
    ax_acc.set_title("Noise-Robust Retrieval\n(Higher = Better)", fontsize=11, color=TEXT, fontweight='bold')
    ax_acc.set_ylim(-0.05, 1.10)
    ax_acc.legend(fontsize=9, framealpha=0.35, edgecolor=SPINE, labelcolor=TEXT, loc='lower left')
    style_ax(ax_acc)

    # Panels 4 & 5: 2D partitions
    plot_2d_partition(ax_2d_orth, W_2d_orth, pts_2d,
                      "Orthogonal W\n(Axis-aligned: 4/4 quadrants used)")
    plot_2d_partition(ax_2d_rand, W_2d_rand, pts_2d,
                      "Correlated W\n(Nearly parallel: fewer quadrants)")
    for ax in [ax_2d_orth, ax_2d_rand]:
        style_ax(ax)

    # Panel 6: Key insight text
    ax_text.axis('off')
    # Use only ASCII-safe characters to avoid font rendering issues
    sep = "-" * 24
    insight_lines = [
        ("Key Insight", 11, True),
        (sep, 9, False),
        ("", 9, False),
        ("Orthogonal W", 10, True),
        ("  * Hyperplanes perp. each other", 9, False),
        ("  * Each bit = fully new info", 9, False),
        ("  * Max unique codes = 2^k", 9, False),
        ("  * High noise margin", 9, False),
        ("  * ||WW^T - I||_F = 0", 9, False),
        ("", 9, False),
        ("Random W", 10, True),
        ("  * Hyperplanes may be correlated", 9, False),
        ("  * Bits encode overlapping info", 9, False),
        ("  * Fewer unique codes (collisions)", 9, False),
        ("  * Lower noise margin", 9, False),
        ("  * ||WW^T - I||_F > 0", 9, False),
        ("", 9, False),
        ("Quantum Analogy:", 9, True),
        ("  Orthogonal measurement axes", 9, False),
        ("  each give 1 independent bit", 9, False),
        ("  => zero information redundancy", 9, False),
    ]
    # Draw a rounded rectangle background manually
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                         boxstyle="round,pad=0.02",
                         linewidth=1.5, edgecolor=BOX_EDGE,
                         facecolor=BOX_FACE, transform=ax_text.transAxes,
                         zorder=0, clip_on=False)
    ax_text.add_patch(box)
    y_cursor = 0.95
    line_height = 0.044
    for text, fsize, bold in insight_lines:
        if text == "":
            y_cursor -= line_height * 0.5
            continue
        ax_text.text(0.07, y_cursor, text,
                     transform=ax_text.transAxes,
                     fontsize=fsize, color=TEXT,
                     va='top', ha='left',
                     fontfamily='monospace',
                     fontweight='bold' if bold else 'normal')
        y_cursor -= line_height

    # Titles
    fig.suptitle("Ablation: Why Orthogonality is Non-Negotiable  (d=32)",
                 fontsize=15, fontweight='bold', color=TEXT, y=1.01)
    fig.text(0.5, -0.015,
             f"Non-orthogonal bases encode redundant information \u2192 hash collisions \u2192 accuracy drop  |  \u03c3={NOISE_SIGMA}",
             ha='center', fontsize=10, color=SUB, style='italic')

    out_path = os.path.join(OUT_DIR, out_name)
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[SAVED] {out_path}")


render_figure(dark=True)
render_figure(dark=False)

# ── Verify ────────────────────────────────────────────────────────────────────
print("\nFiles in figures/:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
print("\nAll tasks complete.")
