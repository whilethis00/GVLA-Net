"""
GVLA-Net: Geometric Visualization Suite
Figure 1 — Weight Orthogonality Heatmap  (WW^T)
Figure 2 — Bit Independence & Entropy Analysis
Figure 3 — 3D Latent Space Partitioning
"""

import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.ticker as mticker

torch.manual_seed(42)
np.random.seed(42)

OUT_DIR = "/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── W constructors ─────────────────────────────────────────────────────────────
def make_orthogonal_W(d, k):
    A = torch.randn(k, d)
    Q, _ = torch.linalg.qr(A.T)
    return Q.T.clone()

def make_random_W(d, k):
    W = torch.randn(k, d)
    W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    return W


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Weight Orthogonality Heatmap
# ══════════════════════════════════════════════════════════════════════════════
def make_fig1(dark: bool):
    k, d = 16, 128
    W_orth = make_orthogonal_W(d, k)
    W_rand = make_random_W(d, k)

    gram_orth = (W_orth @ W_orth.T).numpy()
    gram_rand = (W_rand @ W_rand.T).numpy()

    if dark:
        plt.style.use('dark_background')
        FIG_FACE = '#0E0E0E'; AX_FACE = '#1A1A2E'
        TEXT = '#FFFFFF'; SUB = '#AAAAAA'; SPINE = '#444444'
        cmap = 'RdBu_r'
        suffix = ""
    else:
        plt.rcdefaults()
        FIG_FACE = '#FFFFFF'; AX_FACE = '#FFFFFF'
        TEXT = '#111111'; SUB = '#555555'; SPINE = '#AAAAAA'
        cmap = 'RdBu_r'
        suffix = "_paper"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor(FIG_FACE)

    titles = [
        r"Orthogonal $W$:  $WW^\top \approx I$" + "\n(zero off-diagonal noise)",
        r"Random $W$:  $WW^\top \neq I$" + "\n(correlated hyperplanes)",
    ]
    grams  = [gram_orth, gram_rand]
    labels = ["Orthogonal W", "Random W"]

    for ax, gram, title, label in zip(axes, grams, titles, labels):
        ax.set_facecolor(AX_FACE)
        im = ax.imshow(gram, cmap=cmap, vmin=-1.0, vmax=1.0,
                       interpolation='nearest', aspect='auto')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=9, colors=TEXT)
        cbar.set_label(r"$[WW^\top]_{ij}$", fontsize=10, color=TEXT)
        cbar.outline.set_edgecolor(SPINE)

        # Annotate diagonal mean and off-diag std
        diag_val = np.diag(gram).mean()
        mask_off = ~np.eye(k, dtype=bool)
        off_std  = gram[mask_off].std()
        off_mean = gram[mask_off].mean()

        ax.set_title(title, fontsize=12, color=TEXT, fontweight='bold', pad=10)
        ax.set_xlabel(f"Basis index  (k={k})", fontsize=11, color=TEXT)
        ax.set_ylabel(f"Basis index  (k={k})", fontsize=11, color=TEXT)
        ax.tick_params(colors=TEXT, labelsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE)

        stats = (f"diag mean = {diag_val:.4f}\n"
                 f"off-diag mean = {off_mean:.4f}\n"
                 f"off-diag std  = {off_std:.4f}")
        ax.text(0.98, 0.02, stats,
                transform=ax.transAxes, fontsize=8.5,
                color=TEXT, va='bottom', ha='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.4',
                          facecolor='#2A2A4A' if dark else '#EEF4FF',
                          edgecolor='#4ECDC4' if dark else '#1A9E94',
                          linewidth=1.2, alpha=0.9))

    fig.suptitle(r"Weight Orthogonality: $WW^\top$ Gram Matrix Heatmap",
                 fontsize=14, fontweight='bold', color=TEXT, y=1.02)
    fig.text(0.5, -0.02,
             r"Perfect orthogonality $\Rightarrow$ identity matrix  |  "
             r"Off-diagonal = information redundancy between hyperplanes",
             ha='center', fontsize=10, color=SUB, style='italic')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"fig1_orthogonality_heatmap{suffix}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=FIG_FACE)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Bit Independence & Entropy
# ══════════════════════════════════════════════════════════════════════════════
def make_fig2(dark: bool):
    k, d = 20, 256
    N_SAMPLES = 50_000
    W_orth = make_orthogonal_W(d, k)
    W_rand = make_random_W(d, k)

    states = torch.randn(N_SAMPLES, d)

    def get_bits(W):
        proj  = states @ W.T          # (N, k)
        bits  = (proj > 0).float()   # {0,1}
        return bits.numpy()

    bits_orth = get_bits(W_orth)
    bits_rand = get_bits(W_rand)

    def per_bit_entropy(bits):
        p = bits.mean(axis=0)        # P(bit=1) per dimension
        p_clipped = np.clip(p, 1e-10, 1 - 1e-10)
        h = -(p_clipped * np.log2(p_clipped)
              + (1 - p_clipped) * np.log2(1 - p_clipped))
        return p, h

    def mutual_information_matrix(bits, n_bins=2):
        """Pairwise MI between all bit pairs (sampled for speed)."""
        k = bits.shape[1]
        mi = np.zeros((k, k))
        for i in range(k):
            for j in range(i, k):
                if i == j:
                    mi[i, j] = 1.0   # placeholder
                    continue
                # Joint distribution over {0,1}x{0,1}
                joint = np.zeros((2, 2))
                for bi in range(2):
                    for bj in range(2):
                        joint[bi, bj] = ((bits[:, i] == bi) & (bits[:, j] == bj)).mean()
                px  = joint.sum(axis=1)
                py  = joint.sum(axis=0)
                for bi in range(2):
                    for bj in range(2):
                        if joint[bi, bj] > 1e-10:
                            mi[i, j] += joint[bi, bj] * np.log2(
                                joint[bi, bj] / (px[bi] * py[bj] + 1e-10))
                mi[j, i] = mi[i, j]
        return mi

    p_orth, h_orth = per_bit_entropy(bits_orth)
    p_rand, h_rand = per_bit_entropy(bits_rand)
    mi_orth = mutual_information_matrix(bits_orth)
    mi_rand = mutual_information_matrix(bits_rand)
    # Zero diagonal for display
    np.fill_diagonal(mi_orth, 0)
    np.fill_diagonal(mi_rand, 0)

    if dark:
        plt.style.use('dark_background')
        FIG_FACE = '#0E0E0E'; AX_FACE = '#1A1A2E'
        TEXT = '#FFFFFF'; SUB = '#AAAAAA'; SPINE = '#444444'
        C_ORTH = '#4ECDC4'; C_RAND = '#FF6B6B'
        MI_CMAP = 'YlOrRd'
        suffix = ""
    else:
        plt.rcdefaults()
        FIG_FACE = '#FFFFFF'; AX_FACE = '#FFFFFF'
        TEXT = '#111111'; SUB = '#555555'; SPINE = '#AAAAAA'
        C_ORTH = '#1A9E94'; C_RAND = '#C0392B'
        MI_CMAP = 'YlOrRd'
        suffix = "_paper"

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(FIG_FACE)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)
    ax_p_orth  = fig.add_subplot(gs[0, 0])
    ax_p_rand  = fig.add_subplot(gs[0, 1])
    ax_h_comp  = fig.add_subplot(gs[0, 2])
    ax_mi_orth = fig.add_subplot(gs[1, 0])
    ax_mi_rand = fig.add_subplot(gs[1, 1])
    ax_summary = fig.add_subplot(gs[1, 2])

    for ax in [ax_p_orth, ax_p_rand, ax_h_comp,
               ax_mi_orth, ax_mi_rand, ax_summary]:
        ax.set_facecolor(AX_FACE)
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE)

    x = np.arange(k)
    bar_kw = dict(width=0.7, edgecolor=TEXT, linewidth=0.4, alpha=0.85)

    # ── P(bit=1) bars: Orthogonal ────────────────────────────────────────────
    bars = ax_p_orth.bar(x, p_orth, color=C_ORTH, **bar_kw)
    ax_p_orth.axhline(0.5, color='#FFD700' if dark else '#B07D00',
                      lw=1.5, ls='--', label='Ideal = 0.5')
    ax_p_orth.set_ylim(0, 1.05)
    ax_p_orth.set_xlabel("Bit index", fontsize=10, color=TEXT)
    ax_p_orth.set_ylabel(r"$P(\mathrm{bit}_i = 1)$", fontsize=10, color=TEXT)
    ax_p_orth.set_title("Orthogonal W\nBit Activation Probability",
                        fontsize=11, color=TEXT, fontweight='bold')
    ax_p_orth.tick_params(colors=TEXT, labelsize=8)
    ax_p_orth.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT)
    ax_p_orth.grid(axis='y', alpha=0.15, linestyle='--')
    ax_p_orth.text(0.98, 0.96, f"mean = {p_orth.mean():.3f}\nstd = {p_orth.std():.4f}",
                   transform=ax_p_orth.transAxes, fontsize=8.5, color=TEXT,
                   ha='right', va='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='#2A2A4A' if dark else '#EEF4FF',
                             edgecolor=C_ORTH, alpha=0.9))

    # ── P(bit=1) bars: Random ────────────────────────────────────────────────
    ax_p_rand.bar(x, p_rand, color=C_RAND, **bar_kw)
    ax_p_rand.axhline(0.5, color='#FFD700' if dark else '#B07D00',
                      lw=1.5, ls='--', label='Ideal = 0.5')
    ax_p_rand.set_ylim(0, 1.05)
    ax_p_rand.set_xlabel("Bit index", fontsize=10, color=TEXT)
    ax_p_rand.set_ylabel(r"$P(\mathrm{bit}_i = 1)$", fontsize=10, color=TEXT)
    ax_p_rand.set_title("Random W\nBit Activation Probability",
                        fontsize=11, color=TEXT, fontweight='bold')
    ax_p_rand.tick_params(colors=TEXT, labelsize=8)
    ax_p_rand.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT)
    ax_p_rand.grid(axis='y', alpha=0.15, linestyle='--')
    ax_p_rand.text(0.98, 0.96, f"mean = {p_rand.mean():.3f}\nstd = {p_rand.std():.4f}",
                   transform=ax_p_rand.transAxes, fontsize=8.5, color=TEXT,
                   ha='right', va='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='#2A2A4A' if dark else '#EEF4FF',
                             edgecolor=C_RAND, alpha=0.9))

    # ── Entropy comparison ───────────────────────────────────────────────────
    w = 0.35
    bar_kw_w = dict(edgecolor=TEXT, linewidth=0.4, alpha=0.85)
    ax_h_comp.bar(x - w/2, h_orth, width=w, color=C_ORTH,
                  label='Orthogonal W', **bar_kw_w)
    ax_h_comp.bar(x + w/2, h_rand, width=w, color=C_RAND,
                  label='Random W', **bar_kw_w)
    ax_h_comp.axhline(1.0, color='#FFD700' if dark else '#B07D00',
                      lw=1.5, ls='--', label='Max entropy = 1 bit')
    ax_h_comp.set_ylim(0, 1.15)
    ax_h_comp.set_xlabel("Bit index", fontsize=10, color=TEXT)
    ax_h_comp.set_ylabel("Shannon Entropy (bits)", fontsize=10, color=TEXT)
    ax_h_comp.set_title("Per-bit Entropy\n(Higher = More Independent)",
                        fontsize=11, color=TEXT, fontweight='bold')
    ax_h_comp.tick_params(colors=TEXT, labelsize=8)
    ax_h_comp.legend(fontsize=8, framealpha=0.3, labelcolor=TEXT, loc='lower right')
    ax_h_comp.grid(axis='y', alpha=0.15, linestyle='--')
    ax_h_comp.text(0.02, 0.04,
                   f"Orth mean H = {h_orth.mean():.4f} bits\n"
                   f"Rand mean H = {h_rand.mean():.4f} bits",
                   transform=ax_h_comp.transAxes, fontsize=8.5, color=TEXT,
                   ha='left', va='bottom', fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=0.3',
                             facecolor='#2A2A4A' if dark else '#EEF4FF',
                             edgecolor='#888888', alpha=0.9))

    # ── MI heatmap: Orthogonal ───────────────────────────────────────────────
    im1 = ax_mi_orth.imshow(mi_orth, cmap='YlOrRd', vmin=0, vmax=0.05,
                             interpolation='nearest', aspect='auto')
    fig.colorbar(im1, ax=ax_mi_orth, fraction=0.046, pad=0.04).ax.tick_params(
        colors=TEXT, labelsize=8)
    ax_mi_orth.set_title("Orthogonal W\nPairwise Mutual Information",
                          fontsize=11, color=TEXT, fontweight='bold')
    ax_mi_orth.set_xlabel("Bit index", fontsize=10, color=TEXT)
    ax_mi_orth.set_ylabel("Bit index", fontsize=10, color=TEXT)
    ax_mi_orth.tick_params(colors=TEXT, labelsize=8)
    off_mask = ~np.eye(k, dtype=bool)
    ax_mi_orth.text(0.98, 0.02,
                    f"off-diag MI\nmean={mi_orth[off_mask].mean():.5f}",
                    transform=ax_mi_orth.transAxes, fontsize=8.5, color=TEXT,
                    ha='right', va='bottom', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='#2A2A4A' if dark else '#EEF4FF',
                              edgecolor=C_ORTH, alpha=0.9))

    # ── MI heatmap: Random ───────────────────────────────────────────────────
    im2 = ax_mi_rand.imshow(mi_rand, cmap='YlOrRd', vmin=0, vmax=0.05,
                             interpolation='nearest', aspect='auto')
    fig.colorbar(im2, ax=ax_mi_rand, fraction=0.046, pad=0.04).ax.tick_params(
        colors=TEXT, labelsize=8)
    ax_mi_rand.set_title("Random W\nPairwise Mutual Information",
                          fontsize=11, color=TEXT, fontweight='bold')
    ax_mi_rand.set_xlabel("Bit index", fontsize=10, color=TEXT)
    ax_mi_rand.set_ylabel("Bit index", fontsize=10, color=TEXT)
    ax_mi_rand.tick_params(colors=TEXT, labelsize=8)
    ax_mi_rand.text(0.98, 0.02,
                    f"off-diag MI\nmean={mi_rand[off_mask].mean():.5f}",
                    transform=ax_mi_rand.transAxes, fontsize=8.5, color=TEXT,
                    ha='right', va='bottom', fontfamily='monospace',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='#2A2A4A' if dark else '#EEF4FF',
                              edgecolor=C_RAND, alpha=0.9))

    # ── Summary text ─────────────────────────────────────────────────────────
    ax_summary.axis('off')
    lines = [
        ("Bit Independence Summary", 11, True),
        ("-" * 30, 9, False),
        ("", 0, False),
        ("Orthogonal W:", 10, True),
        (f"  P(bit=1) mean  = {p_orth.mean():.4f}", 9, False),
        (f"  P(bit=1) std   = {p_orth.std():.5f}", 9, False),
        (f"  Entropy mean   = {h_orth.mean():.6f} bits", 9, False),
        (f"  Max entropy    = 1.000000 bits", 9, False),
        (f"  MI (off-diag)  = {mi_orth[off_mask].mean():.6f}", 9, False),
        ("", 0, False),
        ("Random W:", 10, True),
        (f"  P(bit=1) mean  = {p_rand.mean():.4f}", 9, False),
        (f"  P(bit=1) std   = {p_rand.std():.5f}", 9, False),
        (f"  Entropy mean   = {h_rand.mean():.6f} bits", 9, False),
        (f"  MI (off-diag)  = {mi_rand[off_mask].mean():.6f}", 9, False),
        ("", 0, False),
        ("Interpretation:", 10, True),
        ("  Each orthogonal bit achieves", 9, False),
        ("  maximum entropy (=1 bit) and", 9, False),
        ("  near-zero MI with all others.", 9, False),
        ("  => Zero information redundancy.", 9, False),
    ]
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                         boxstyle="round,pad=0.02", linewidth=1.5,
                         edgecolor=C_ORTH,
                         facecolor='#2A2A4A' if dark else '#EEF4FF',
                         transform=ax_summary.transAxes, zorder=0)
    ax_summary.add_patch(box)
    y = 0.95
    lh = 0.043
    for text, fsize, bold in lines:
        if fsize == 0:
            y -= lh * 0.5; continue
        ax_summary.text(0.07, y, text, transform=ax_summary.transAxes,
                        fontsize=fsize, color=TEXT, va='top', ha='left',
                        fontfamily='monospace',
                        fontweight='bold' if bold else 'normal')
        y -= lh

    fig.suptitle("Bit Independence & Entropy Analysis",
                 fontsize=14, fontweight='bold', color=TEXT, y=1.01)
    fig.text(0.5, -0.01,
             r"Ideal: $P(\mathrm{bit}_i=1)=0.5$, $H(\mathrm{bit}_i)=1$ bit, $\mathrm{MI}(\mathrm{bit}_i, \mathrm{bit}_j)=0$  for all $i \neq j$",
             ha='center', fontsize=10, color=SUB, style='italic')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"fig2_bit_entropy{suffix}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=FIG_FACE)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — 3D Latent Space Partitioning
# ══════════════════════════════════════════════════════════════════════════════
def make_fig3(dark: bool):
    """
    3D visualization: 3 orthogonal hyperplanes slice the unit sphere into 8 cells.
    Points are colored by their 3-bit code.
    Projection arrows show how a query is routed to its cell.
    """
    if dark:
        plt.style.use('dark_background')
        FIG_FACE = '#0E0E0E'
        TEXT     = '#FFFFFF'; SUB = '#AAAAAA'
        PANE_COL = '#1A1A2E'; GRID_COL = '#333333'
        suffix   = ""
    else:
        plt.rcdefaults()
        FIG_FACE = '#FFFFFF'
        TEXT     = '#111111'; SUB = '#555555'
        PANE_COL = '#F0F0F0'; GRID_COL = '#CCCCCC'
        suffix   = "_paper"

    # 3 orthogonal planes in 3D: xy, xz, yz planes (normals = z, y, x)
    W3 = torch.eye(3)  # perfectly orthogonal in 3D

    # Cell color palette (8 cells = 3 bits)
    CELL_COLORS = {
        (0, 0, 0): '#E74C3C',   # red
        (0, 0, 1): '#E67E22',   # orange
        (0, 1, 0): '#2ECC71',   # green
        (0, 1, 1): '#1ABC9C',   # teal
        (1, 0, 0): '#3498DB',   # blue
        (1, 0, 1): '#9B59B6',   # purple
        (1, 1, 0): '#F1C40F',   # yellow
        (1, 1, 1): '#EC407A',   # pink
    }

    torch.manual_seed(99)
    N_PTS = 800
    pts = torch.randn(N_PTS, 3)
    pts = pts / (pts.norm(dim=1, keepdim=True) * 1.2)  # roughly on/near sphere

    codes = tuple(map(tuple, ((pts @ W3.T) > 0).int().tolist()))
    colors = [CELL_COLORS[c] for c in codes]

    fig = plt.figure(figsize=(16, 6))
    fig.patch.set_facecolor(FIG_FACE)

    # ── Left panel: full partitioning with 3 hyperplanes ─────────────────────
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_facecolor(PANE_COL)
    ax1.scatter(pts[:, 0].numpy(), pts[:, 1].numpy(), pts[:, 2].numpy(),
                c=colors, s=10, alpha=0.7, zorder=3)

    # Draw 3 hyperplane surfaces (semi-transparent)
    lim = 1.6
    g   = np.linspace(-lim, lim, 30)
    G1, G2 = np.meshgrid(g, g)
    plane_kw = dict(alpha=0.12, shade=False)

    # xy-plane (z=0, normal=z-axis)
    Z0 = np.zeros_like(G1)
    ax1.plot_surface(G1, G2, Z0, color='#4ECDC4' if dark else '#1A9E94', **plane_kw)
    # xz-plane (y=0, normal=y-axis)
    ax1.plot_surface(G1, Z0, G2, color='#FF6B6B' if dark else '#C0392B', **plane_kw)
    # yz-plane (x=0, normal=x-axis)
    ax1.plot_surface(Z0, G1, G2, color='#FFD700' if dark else '#B07D00', **plane_kw)

    ax1.set_xlim(-lim, lim); ax1.set_ylim(-lim, lim); ax1.set_zlim(-lim, lim)
    ax1.set_xlabel("$x$", color=TEXT, fontsize=10)
    ax1.set_ylabel("$y$", color=TEXT, fontsize=10)
    ax1.set_zlabel("$z$", color=TEXT, fontsize=10)
    ax1.set_title("3 Orthogonal Hyperplanes\nPartition Space into 8 Cells",
                  fontsize=11, color=TEXT, fontweight='bold', pad=8)
    ax1.tick_params(colors=TEXT, labelsize=7)
    ax1.xaxis.pane.fill = True; ax1.xaxis.pane.set_facecolor(PANE_COL)
    ax1.yaxis.pane.fill = True; ax1.yaxis.pane.set_facecolor(PANE_COL)
    ax1.zaxis.pane.fill = True; ax1.zaxis.pane.set_facecolor(PANE_COL)

    # ── Middle panel: projection arrows ──────────────────────────────────────
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_facecolor(PANE_COL)

    # Show fewer points with arrows pointing to octant centers
    torch.manual_seed(7)
    demo_pts = torch.randn(40, 3)
    demo_pts = demo_pts / (demo_pts.norm(dim=1, keepdim=True) + 1e-8)
    demo_codes = ((demo_pts @ W3.T) > 0).int()
    demo_colors = [CELL_COLORS[tuple(c.tolist())] for c in demo_codes]

    # Octant centers (unit vectors in each octant)
    octant_centers = {
        (0, 0, 0): torch.tensor([-1., -1., -1.]) / 3**0.5,
        (0, 0, 1): torch.tensor([-1., -1.,  1.]) / 3**0.5,
        (0, 1, 0): torch.tensor([-1.,  1., -1.]) / 3**0.5,
        (0, 1, 1): torch.tensor([-1.,  1.,  1.]) / 3**0.5,
        (1, 0, 0): torch.tensor([ 1., -1., -1.]) / 3**0.5,
        (1, 0, 1): torch.tensor([ 1., -1.,  1.]) / 3**0.5,
        (1, 1, 0): torch.tensor([ 1.,  1., -1.]) / 3**0.5,
        (1, 1, 1): torch.tensor([ 1.,  1.,  1.]) / 3**0.5,
    }

    ax2.scatter(demo_pts[:, 0], demo_pts[:, 1], demo_pts[:, 2],
                c=demo_colors, s=30, alpha=0.85, zorder=4)

    arrow_col = '#FFFFFF' if dark else '#333333'
    for p, code in zip(demo_pts, demo_codes):
        target = octant_centers[tuple(code.tolist())] * 0.72
        dp = target - p
        ax2.quiver(p[0].item(), p[1].item(), p[2].item(),
                   dp[0].item(), dp[1].item(), dp[2].item(),
                   length=0.85, normalize=False,
                   color=arrow_col, alpha=0.35, linewidth=0.6,
                   arrow_length_ratio=0.25)

    # Draw hyperplanes lightly
    ax2.plot_surface(G1, G2, Z0, color='#4ECDC4' if dark else '#1A9E94', **plane_kw)
    ax2.plot_surface(G1, Z0, G2, color='#FF6B6B' if dark else '#C0392B', **plane_kw)
    ax2.plot_surface(Z0, G1, G2, color='#FFD700' if dark else '#B07D00', **plane_kw)

    ax2.set_xlim(-lim, lim); ax2.set_ylim(-lim, lim); ax2.set_zlim(-lim, lim)
    ax2.set_xlabel("$x$", color=TEXT, fontsize=10)
    ax2.set_ylabel("$y$", color=TEXT, fontsize=10)
    ax2.set_zlabel("$z$", color=TEXT, fontsize=10)
    ax2.set_title("Geometric Routing:\nEach Point Projected to its Cell",
                  fontsize=11, color=TEXT, fontweight='bold', pad=8)
    ax2.tick_params(colors=TEXT, labelsize=7)
    ax2.xaxis.pane.fill = True; ax2.xaxis.pane.set_facecolor(PANE_COL)
    ax2.yaxis.pane.fill = True; ax2.yaxis.pane.set_facecolor(PANE_COL)
    ax2.zaxis.pane.fill = True; ax2.zaxis.pane.set_facecolor(PANE_COL)

    # ── Right panel: scaling intuition (k=20 → 1M cells) ────────────────────
    ax3 = fig.add_subplot(133)
    ax3.set_facecolor('#1A1A2E' if dark else '#FFFFFF')
    for spine in ax3.spines.values():
        spine.set_edgecolor('#444444' if dark else '#AAAAAA')

    k_vals = np.arange(1, 21)
    n_cells = 2 ** k_vals
    ax3.semilogy(k_vals, n_cells,
                 color='#4ECDC4' if dark else '#1A9E94',
                 linewidth=2.5, marker='o', markersize=5, zorder=3)
    ax3.fill_between(k_vals, 1, n_cells,
                     alpha=0.12,
                     color='#4ECDC4' if dark else '#1A9E94')

    # Highlight key points
    highlights = [(10, 1024, "N=1k\n(10-bit)"),
                  (15, 32768, "N=32k\n(15-bit)"),
                  (20, 1048576, "N=1M\n(20-bit)")]
    for kv, nv, label in highlights:
        ax3.scatter([kv], [nv], s=80, color='#FFD700' if dark else '#B07D00',
                    zorder=5)
        ax3.annotate(label, xy=(kv, nv), xytext=(kv - 3.5, nv * 1.8),
                     fontsize=8.5, color='#FFD700' if dark else '#B07D00',
                     fontweight='bold',
                     arrowprops=dict(arrowstyle='->', lw=1.2,
                                     color='#FFD700' if dark else '#B07D00'))

    ax3.set_xlabel("Number of hyperplanes $k$", fontsize=11, color=TEXT)
    ax3.set_ylabel(r"Number of cells = $2^k$", fontsize=11, color=TEXT)
    ax3.set_title(r"Exponential Capacity: $2^k$ Cells" + "\nfrom $k$ Hyperplanes",
                  fontsize=11, color=TEXT, fontweight='bold')
    ax3.tick_params(colors=TEXT, labelsize=9)
    ax3.grid(color='#FFFFFF' if dark else '#CCCCCC',
             alpha=0.15, linestyle='--', linewidth=0.6)
    ax3.set_xlim(0.5, 20.5)

    ax3.text(0.97, 0.05,
             "k=20 hyperplanes\n=> 1,048,576 cells\n(sub-millimeter precision)",
             transform=ax3.transAxes, fontsize=8.5, color=TEXT,
             ha='right', va='bottom', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4',
                       facecolor='#2A2A4A' if dark else '#EEF4FF',
                       edgecolor='#4ECDC4' if dark else '#1A9E94',
                       alpha=0.9))

    fig.suptitle(r"Latent Space Partitioning: $k$ Orthogonal Hyperplanes $\Rightarrow$ $2^k$ Geometric Cells",
                 fontsize=13, fontweight='bold', color=TEXT, y=1.02)
    fig.text(0.5, -0.02,
             "Each hyperplane bisects the space → exponential growth in discriminable actions "
             r"→ O$(\log N)$ routing with $k = \lceil\log_2 N\rceil$ questions",
             ha='center', fontsize=10, color=SUB, style='italic')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"fig3_latent_partitioning{suffix}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=FIG_FACE)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Correlation Sensitivity Sweep  "The Melting Space"
# W_test(ρ) = (1-ρ)*W_ortho + ρ*W_parallel
# ══════════════════════════════════════════════════════════════════════════════
def make_fig4(dark: bool):
    k, d = 16, 256
    N_SAMPLES = 30_000
    N_ACTION  = 65536   # 2^16

    W_ortho    = make_orthogonal_W(d, k)
    # W_parallel: all rows identical to the first orthogonal row (worst case)
    W_parallel = W_ortho[0:1].expand(k, -1).clone()

    states = torch.randn(N_SAMPLES, d)

    rho_values = np.linspace(0, 1.0, 41)

    def make_W_test(rho):
        W = (1 - rho) * W_ortho + rho * W_parallel
        # Re-normalise rows so magnitudes stay unit
        W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
        return W

    def unique_code_rate(W):
        proj  = states @ W.T
        codes = (proj > 0).int()
        tuples = [tuple(r.tolist()) for r in codes]
        return len(set(tuples)) / N_SAMPLES

    def gram_matrix(W):
        return (W @ W.T).numpy()

    # Compute random W's effective ρ: find ρ that gives the same off-diag std
    W_rand  = make_random_W(d, k)
    rand_offstd = gram_matrix(W_rand)[~np.eye(k, dtype=bool)].std()

    # Sweep
    ucr_vals   = []
    ortho_errs = []
    for rho in rho_values:
        W = make_W_test(rho)
        ucr_vals.append(unique_code_rate(W))
        G = W @ W.T
        I = torch.eye(k)
        ortho_errs.append((G - I).norm(p='fro').item())
    ucr_vals   = np.array(ucr_vals)
    ortho_errs = np.array(ortho_errs)

    # Find where random W sits on the ρ axis (by ortho error matching)
    rand_ortho_err = gram_matrix(W_rand)[~np.eye(k, dtype=bool)].std() * math.sqrt(k*(k-1))
    # approximate: find closest rho
    closest_rho_idx = int(np.argmin(np.abs(ortho_errs - rand_ortho_err)))
    rand_rho_approx = rho_values[closest_rho_idx]
    rand_ucr        = unique_code_rate(W_rand)

    # Heatmap snapshots at ρ = 0, 0.3, 0.7, 0.9
    snap_rhos = [0.0, 0.3, 0.7, 0.9]

    if dark:
        plt.style.use('dark_background')
        FIG_FACE = '#0E0E0E'; AX_FACE = '#1A1A2E'
        TEXT = '#FFFFFF'; SUB = '#AAAAAA'; SPINE = '#444444'
        LINE_COL = '#4ECDC4'; RAND_COL = '#FF6B6B'
        suffix = ""
    else:
        plt.rcdefaults()
        FIG_FACE = '#FFFFFF'; AX_FACE = '#FFFFFF'
        TEXT = '#111111'; SUB = '#555555'; SPINE = '#AAAAAA'
        LINE_COL = '#1A9E94'; RAND_COL = '#C0392B'
        suffix = "_paper"

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(FIG_FACE)
    gs = gridspec.GridSpec(2, 5, figure=fig,
                           hspace=0.48, wspace=0.38,
                           width_ratios=[1, 1, 1, 1, 1.6])

    # ── Top row: 4 heatmap snapshots ─────────────────────────────────────────
    for col, rho in enumerate(snap_rhos):
        ax = fig.add_subplot(gs[0, col])
        ax.set_facecolor(AX_FACE)
        G = gram_matrix(make_W_test(rho))
        im = ax.imshow(G, cmap='RdBu_r', vmin=-1, vmax=1,
                       interpolation='nearest', aspect='auto')
        off_std = G[~np.eye(k, dtype=bool)].std()
        ax.set_title(f"ρ = {rho:.1f}\noff-diag std={off_std:.3f}",
                     fontsize=10, color=TEXT, fontweight='bold', pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(SPINE)
        # Subtle color border showing how "melted" it is
        border_col = LINE_COL if rho == 0.0 else RAND_COL
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(2.0)

    # Shared colorbar for heatmaps
    cbar_ax = fig.add_axes([0.74, 0.57, 0.008, 0.32])
    sm = plt.cm.ScalarMappable(cmap='RdBu_r',
                                norm=plt.Normalize(vmin=-1, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, cax=cbar_ax)
    cb.ax.tick_params(labelsize=8, colors=TEXT)
    cb.set_label(r"$[WW^\top]_{ij}$", fontsize=9, color=TEXT)
    cb.outline.set_edgecolor(SPINE)

    # ── Bottom-left span (4 cols): performance decay curve ───────────────────
    ax_decay = fig.add_subplot(gs[1, :4])
    ax_decay.set_facecolor(AX_FACE)
    for spine in ax_decay.spines.values():
        spine.set_edgecolor(SPINE)

    # Unique code count (scale to N_ACTION)
    unique_count = ucr_vals * N_SAMPLES
    # Clip to realistic max = 2^k
    unique_count = np.minimum(unique_count, 2**k)

    ax_decay.semilogy(rho_values, unique_count,
                      color=LINE_COL, linewidth=2.5, zorder=3, label='Unique codes')
    ax_decay.fill_between(rho_values, 1, unique_count,
                          alpha=0.12, color=LINE_COL)

    # Mark random W
    ax_decay.scatter([rand_rho_approx], [rand_ucr * N_SAMPLES],
                     s=120, color=RAND_COL, zorder=6,
                     label=f'Random W  (ρ≈{rand_rho_approx:.2f})', marker='*')
    ax_decay.annotate(f"Random W\nρ ≈ {rand_rho_approx:.2f}",
                      xy=(rand_rho_approx, rand_ucr * N_SAMPLES),
                      xytext=(rand_rho_approx + 0.07,
                              rand_ucr * N_SAMPLES * 3),
                      fontsize=9, color=RAND_COL, fontweight='bold',
                      arrowprops=dict(arrowstyle='->', color=RAND_COL, lw=1.5))

    # Reference lines
    ax_decay.axhline(2**k, color='#FFD700' if dark else '#B07D00',
                     lw=1.2, ls='--', alpha=0.7,
                     label=f'Max = 2^k = {2**k:,}')
    ax_decay.axhline(2**(k//2), color='#888888', lw=1.0, ls=':',
                     alpha=0.6, label=f'Half capacity = {2**(k//2):,}')

    # Shade collapse zone
    ax_decay.axvspan(0.6, 1.0, alpha=0.07,
                     color=RAND_COL, label='Collapse zone (ρ > 0.6)')
    ax_decay.text(0.75, 1.5,
                  "Collapse\nZone", fontsize=9, color=RAND_COL,
                  ha='center', fontweight='bold')

    ax_decay.set_xlabel("Correlation coefficient ρ  (0 = perfect orthogonal, 1 = fully parallel)",
                        fontsize=11, color=TEXT)
    ax_decay.set_ylabel("Effective unique codes\n(log scale)", fontsize=11, color=TEXT)
    ax_decay.set_title("Performance Decay as Hyperplanes Become Correlated",
                       fontsize=12, color=TEXT, fontweight='bold')
    ax_decay.set_xlim(-0.02, 1.02)
    ax_decay.tick_params(colors=TEXT, labelsize=9)
    ax_decay.grid(color='#FFFFFF' if dark else '#CCCCCC',
                  alpha=0.15, linestyle='--', linewidth=0.6)
    ax_decay.legend(fontsize=9, framealpha=0.35, edgecolor=SPINE,
                    labelcolor=TEXT, loc='lower left')

    # ── Bottom-right: summary box ─────────────────────────────────────────────
    ax_sum = fig.add_subplot(gs[1, 4])
    ax_sum.axis('off')
    from matplotlib.patches import FancyBboxPatch
    box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                         boxstyle="round,pad=0.02", linewidth=1.5,
                         edgecolor=LINE_COL,
                         facecolor='#2A2A4A' if dark else '#EEF4FF',
                         transform=ax_sum.transAxes, zorder=0)
    ax_sum.add_patch(box)

    lines = [
        ("Correlation Sweep", 10, True),
        ("-" * 24, 8, False),
        ("", 0, False),
        (f"W(rho) = (1-rho)*W_ortho", 8, False),
        (f"       + rho*W_parallel", 8, False),
        ("", 0, False),
        ("rho=0.0: perfect ortho.", 9, True),
        (f"  unique codes ~ {int(ucr_vals[0]*N_SAMPLES):,}", 8, False),
        ("", 0, False),
        ("rho=0.5:", 9, False),
        (f"  unique codes ~ {int(ucr_vals[20]*N_SAMPLES):,}", 8, False),
        ("", 0, False),
        ("rho=1.0: all parallel.", 9, True),
        (f"  unique codes ~ {int(ucr_vals[-1]*N_SAMPLES):,}", 8, False),
        ("", 0, False),
        (f"Random W sits at", 8, False),
        (f"  rho ~ {rand_rho_approx:.2f}", 8, False),
        ("", 0, False),
        ("Key takeaway:", 9, True),
        ("  Even small correlation", 8, False),
        ("  exponentially reduces", 8, False),
        ("  discriminable actions.", 8, False),
    ]
    y = 0.97; lh = 0.042
    for text, fsize, bold in lines:
        if fsize == 0:
            y -= lh * 0.45; continue
        ax_sum.text(0.07, y, text, transform=ax_sum.transAxes,
                    fontsize=fsize, color=TEXT, va='top', ha='left',
                    fontfamily='monospace',
                    fontweight='bold' if bold else 'normal')
        y -= lh

    fig.suptitle(
        r"Correlation Sensitivity Sweep: $W_{\rho} = (1-\rho)\,W_{\mathrm{ortho}} + \rho\,W_{\mathrm{parallel}}$",
        fontsize=13, fontweight='bold', color=TEXT, y=1.01)
    fig.text(0.5, -0.01,
             r"As $\rho \to 1$, hyperplanes become parallel $\Rightarrow$ "
             r"exponential collapse in unique codes $\Rightarrow$ "
             r"action resolution degrades catastrophically",
             ha='center', fontsize=10, color=SUB, style='italic')

    plt.tight_layout(rect=[0, 0, 0.96, 1.0])
    out = os.path.join(OUT_DIR, f"fig4_correlation_sweep{suffix}.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=FIG_FACE)
    plt.close(fig)
    print(f"[SAVED] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# RUN ALL
# ══════════════════════════════════════════════════════════════════════════════
print("=== Figure 1: Weight Orthogonality Heatmap ===")
make_fig1(dark=False);  make_fig1(dark=True)

print("\n=== Figure 2: Bit Independence & Entropy ===")
make_fig2(dark=False);  make_fig2(dark=True)

print("\n=== Figure 3: 3D Latent Space Partitioning ===")
make_fig3(dark=False);  make_fig3(dark=True)

print("\n=== Figure 4: Correlation Sensitivity Sweep ===")
make_fig4(dark=False);  make_fig4(dark=True)

print("\nFiles in figures/:")
for f in sorted(os.listdir(OUT_DIR)):
    print(f"  {f}")
print("\nDone.")
