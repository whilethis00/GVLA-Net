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
# FIGURE 2 — Codebook Geometric Analysis (3-part)
#   Row 1: Hash Collision Distribution (Orth vs Rand, Poisson ref, cell breakdown)
#   Row 2: Valid Code Rate vs N  |  Codebook Alignment vs d  |  Summary
# ══════════════════════════════════════════════════════════════════════════════
def make_fig2(dark: bool):
    from math import factorial, exp as mexp
    from matplotlib.patches import FancyBboxPatch

    torch.manual_seed(42); np.random.seed(42)

    # ── PART 1: Collision Distribution  k=17, d=32 ───────────────────────────
    k_c, d_c = 17, 32
    N_C = 2 ** k_c   # 131,072

    W_co = make_orthogonal_W(d_c, k_c)
    W_cr = make_random_W(d_c, k_c)

    def bin_occ(W, k):
        N = 2 ** k
        s = torch.randn(N, W.shape[1])
        pw = (2 ** torch.arange(k)).long()
        h = ((s @ W.T > 0).int().long() * pw).sum(1).numpy()
        return np.bincount(h, minlength=N)

    print("  [fig2] collision: orth...")
    occ_o = bin_occ(W_co, k_c)
    print("  [fig2] collision: rand...")
    occ_r = bin_occ(W_cr, k_c)

    def cell_stats(occ, N):
        return dict(empty=(occ == 0).sum(), single=(occ == 1).sum(),
                    collision=(occ >= 2).sum(), max_occ=int(occ.max()),
                    valid_rate=(occ > 0).sum() / N)

    st_o = cell_stats(occ_o, N_C)
    st_r = cell_stats(occ_r, N_C)

    lam = 1.0
    max_show = min(int(max(occ_o.max(), occ_r.max())) + 1, 10)
    x_occ = np.arange(0, max_show + 1)
    poisson_ref = np.array([N_C * mexp(-lam) * lam**v / factorial(v) for v in x_occ])
    y_o = np.array([(occ_o == v).sum() for v in x_occ], dtype=float)
    y_r = np.array([(occ_r == v).sum() for v in x_occ], dtype=float)

    # ── PART 2: Valid Code Rate vs N  (k=16, d=32 fixed) ─────────────────────
    k_v, d_v = 16, 32
    N_cells_v = 2 ** k_v   # 65,536
    N_vals = [200, 500, 1000, 2000, 5000, 10000, 20000, 65536, 131072, 262144]

    W_vo = make_orthogonal_W(d_v, k_v)
    W_vr = make_random_W(d_v, k_v)
    pw_v = (2 ** torch.arange(k_v)).long()

    vcr_o, vcr_r = [], []
    for Nv in N_vals:
        ro, rr = [], []
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            s = torch.randn(Nv, d_v)
            for W, lst in [(W_vo, ro), (W_vr, rr)]:
                h = ((s @ W.T > 0).int().long() * pw_v).sum(1).numpy()
                occ = np.bincount(h, minlength=N_cells_v)
                lst.append((occ > 0).sum() / N_cells_v)
        vcr_o.append(np.mean(ro)); vcr_r.append(np.mean(rr))
    vcr_o = np.array(vcr_o); vcr_r = np.array(vcr_r)

    N_theory = np.logspace(np.log10(200), np.log10(262144), 120)
    vcr_theory = 1 - np.exp(-N_theory / N_cells_v)

    print("  [fig2] vcr computed")

    # ── PART 3: Codebook Alignment vs d  (k=16 fixed, d: 16→512) ─────────────
    k_a = 16
    d_list = [16, 32, 64, 128, 256, 512]
    aln_o, aln_r = [], []
    for d_a in d_list:
        for make_fn, lst in [(make_orthogonal_W, aln_o), (make_random_W, aln_r)]:
            W = make_fn(d_a, k_a)
            G = (W @ W.T).numpy()
            off_mask = ~np.eye(k_a, dtype=bool)
            lst.append(np.abs(G[off_mask]).mean())
    aln_theory = [np.sqrt(2 / (np.pi * d_a)) for d_a in d_list]

    print("  [fig2] alignment computed")

    # ── THEME ──────────────────────────────────────────────────────────────────
    if dark:
        plt.style.use('dark_background')
        FIG_FACE = '#0E0E0E'; AX_FACE = '#1A1A2E'
        TEXT = '#FFFFFF';  SUB = '#AAAAAA';  SPINE = '#444444'
        C_ORTH = '#4ECDC4'; C_RAND = '#FF6B6B'
        GOLD   = '#FFD700'; BOX_BG = '#2A2A4A'; C_TH = '#AAAAFF'
        suffix = ""
    else:
        plt.rcdefaults()
        FIG_FACE = '#FFFFFF'; AX_FACE = '#FFFFFF'
        TEXT = '#111111';  SUB = '#555555';  SPINE = '#AAAAAA'
        C_ORTH = '#1A9E94'; C_RAND = '#C0392B'
        GOLD   = '#B07D00'; BOX_BG = '#EEF4FF'; C_TH = '#6655BB'
        suffix = "_paper"

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(FIG_FACE)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.50, wspace=0.38)

    ax_ho  = fig.add_subplot(gs[0, 0])   # collision hist — orth
    ax_hr  = fig.add_subplot(gs[0, 1])   # collision hist — rand
    ax_hc  = fig.add_subplot(gs[0, 2])   # cell breakdown
    ax_vcr = fig.add_subplot(gs[1, 0])   # valid code rate vs N
    ax_aln = fig.add_subplot(gs[1, 1])   # alignment vs d
    ax_sum = fig.add_subplot(gs[1, 2])   # summary box

    for ax in [ax_ho, ax_hr, ax_hc, ax_vcr, ax_aln, ax_sum]:
        ax.set_facecolor(AX_FACE)
        for sp in ax.spines.values():
            sp.set_edgecolor(SPINE)

    bar_kw = dict(edgecolor=TEXT, linewidth=0.5, alpha=0.88, zorder=3)

    # ── Panel 1: Orthogonal W collision histogram ─────────────────────────────
    ax_ho.bar(x_occ, np.where(y_o > 0, y_o, np.nan), color=C_ORTH, **bar_kw)
    ax_ho.plot(x_occ, poisson_ref, 'o--', color=GOLD, lw=1.5, ms=5,
               label='Poisson(λ=1)', zorder=5)
    ax_ho.set_yscale('log')
    ax_ho.set_xlabel("Actions per cell (occupancy)", fontsize=10, color=TEXT)
    ax_ho.set_ylabel("Number of cells (log)", fontsize=10, color=TEXT)
    ax_ho.set_title("Orthogonal W\nHash Collision Distribution",
                    fontsize=11, color=TEXT, fontweight='bold')
    ax_ho.set_xticks(x_occ)
    ax_ho.tick_params(colors=TEXT, labelsize=8)
    ax_ho.grid(axis='y', alpha=0.15, ls='--')
    ax_ho.legend(fontsize=8.5, framealpha=0.35, labelcolor=TEXT)
    ax_ho.text(0.97, 0.97,
               f"Valid: {st_o['valid_rate']:.1%}\n"
               f"Collisions: {st_o['collision']:,}",
               transform=ax_ho.transAxes, fontsize=8, color=TEXT,
               ha='right', va='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=BOX_BG,
                         edgecolor=C_ORTH, alpha=0.92))

    # ── Panel 2: Random W collision histogram ─────────────────────────────────
    ax_hr.bar(x_occ, np.where(y_r > 0, y_r, np.nan), color=C_RAND, **bar_kw)
    ax_hr.plot(x_occ, poisson_ref, 'o--', color=GOLD, lw=1.5, ms=5,
               label='Poisson(λ=1)', zorder=5)
    ax_hr.set_yscale('log')
    ax_hr.set_xlabel("Actions per cell (occupancy)", fontsize=10, color=TEXT)
    ax_hr.set_ylabel("Number of cells (log)", fontsize=10, color=TEXT)
    ax_hr.set_title("Random W\nHash Collision Distribution",
                    fontsize=11, color=TEXT, fontweight='bold')
    ax_hr.set_xticks(x_occ)
    ax_hr.tick_params(colors=TEXT, labelsize=8)
    ax_hr.grid(axis='y', alpha=0.15, ls='--')
    ax_hr.legend(fontsize=8.5, framealpha=0.35, labelcolor=TEXT)
    ax_hr.text(0.97, 0.97,
               f"Valid: {st_r['valid_rate']:.1%}\n"
               f"Collisions: {st_r['collision']:,}",
               transform=ax_hr.transAxes, fontsize=8, color=TEXT,
               ha='right', va='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=BOX_BG,
                         edgecolor=C_RAND, alpha=0.92))

    # ── Panel 3: Cell usage stacked bars ──────────────────────────────────────
    cats    = ['Empty\n(wasted)', 'Single\n(ideal)', 'Collision\n(>=2)']
    v_o_c   = [st_o['empty'], st_o['single'], st_o['collision']]
    v_r_c   = [st_r['empty'], st_r['single'], st_r['collision']]
    cat_col = ['#888888', C_ORTH, '#E74C3C']
    x_pos   = np.array([0.22, 0.72]); bw = 0.38
    bot_o = bot_r = 0
    for label, co, cr, cc in zip(cats, v_o_c, v_r_c, cat_col):
        fo = co / N_C; fr = cr / N_C
        ax_hc.bar(x_pos[0], fo, width=bw, bottom=bot_o,
                  color=cc, edgecolor=TEXT, lw=0.4, alpha=0.85, zorder=3)
        ax_hc.bar(x_pos[1], fr, width=bw, bottom=bot_r,
                  color=cc, edgecolor=TEXT, lw=0.4, alpha=0.85, zorder=3,
                  label=label)
        if fo > 0.03:
            ax_hc.text(x_pos[0], bot_o + fo/2, f"{fo:.1%}", ha='center', va='center',
                       fontsize=8, color='white' if fo > 0.10 else TEXT, fontweight='bold')
        if fr > 0.03:
            ax_hc.text(x_pos[1], bot_r + fr/2, f"{fr:.1%}", ha='center', va='center',
                       fontsize=8, color='white' if fr > 0.10 else TEXT, fontweight='bold')
        bot_o += fo; bot_r += fr
    ax_hc.set_xticks(x_pos)
    ax_hc.set_xticklabels(['Orthogonal W', 'Random W'], fontsize=9,
                          color=TEXT, fontweight='bold')
    ax_hc.set_ylim(0, 1.12)
    ax_hc.set_ylabel("Fraction of cells", fontsize=10, color=TEXT)
    ax_hc.set_title("Cell Usage Breakdown\n(N=131k, 131k cells)",
                    fontsize=11, color=TEXT, fontweight='bold')
    ax_hc.tick_params(colors=TEXT, labelsize=8)
    ax_hc.legend(fontsize=8, framealpha=0.35, labelcolor=TEXT,
                 loc='upper right', bbox_to_anchor=(1.0, 0.46))
    ax_hc.grid(axis='y', alpha=0.12, ls='--')

    # ── Panel 4: Valid Code Rate vs N ─────────────────────────────────────────
    ax_vcr.semilogx(N_theory, vcr_theory, '--', color=GOLD, lw=1.5,
                    label='Poisson ideal', zorder=2)
    ax_vcr.semilogx(N_vals, vcr_o, 'o-', color=C_ORTH, lw=2.2, ms=6,
                    label='Orthogonal W', zorder=4)
    ax_vcr.semilogx(N_vals, vcr_r, 's-', color=C_RAND, lw=2.2, ms=6,
                    label='Random W', zorder=4)
    ax_vcr.axvline(N_cells_v, color=SPINE, lw=1.0, ls=':', alpha=0.7)
    ax_vcr.text(N_cells_v * 1.15, 0.04,
                f'N=2^{k_v}\n(λ=1)', fontsize=8, color=SUB, va='bottom')
    ax_vcr.set_xlabel("Number of actions N  (log scale)", fontsize=10, color=TEXT)
    ax_vcr.set_ylabel("Valid Code Rate\n(fraction of cells occupied)", fontsize=10, color=TEXT)
    ax_vcr.set_title(f"Valid Code Rate vs N\n(k={k_v} bits, d={d_v}, {N_cells_v:,} cells)",
                     fontsize=11, color=TEXT, fontweight='bold')
    ax_vcr.tick_params(colors=TEXT, labelsize=8)
    ax_vcr.grid(alpha=0.15, ls='--')
    ax_vcr.set_ylim(0, 1.05)
    ax_vcr.legend(fontsize=8.5, framealpha=0.35, labelcolor=TEXT)

    # ── Panel 5: Codebook Alignment vs d ──────────────────────────────────────
    ax_aln.plot(d_list, aln_theory, '--', color=GOLD, lw=1.5,
                label=r'$\sqrt{2/\pi d}$ (theory)', zorder=2)
    ax_aln.plot(d_list, aln_o, 'o-', color=C_ORTH, lw=2.2, ms=6,
                label='Orthogonal W', zorder=4)
    ax_aln.plot(d_list, aln_r, 's-', color=C_RAND, lw=2.2, ms=6,
                label='Random W', zorder=4)
    ax_aln.set_yscale('log')
    ax_aln.set_xscale('log')
    ax_aln.set_xticks(d_list)
    ax_aln.get_xaxis().set_major_formatter(mticker.ScalarFormatter())
    ax_aln.set_xlabel("Embedding dimension d  (log scale)", fontsize=10, color=TEXT)
    ax_aln.set_ylabel(r"Mean |off-diagonal| of $WW^\top$" + "\n(lower = better alignment)",
                      fontsize=10, color=TEXT)
    ax_aln.set_title(f"Codebook Alignment vs d\n(k={k_a} bits fixed)",
                     fontsize=11, color=TEXT, fontweight='bold')
    ax_aln.tick_params(colors=TEXT, labelsize=8)
    ax_aln.grid(alpha=0.15, ls='--')
    ax_aln.legend(fontsize=8.5, framealpha=0.35, labelcolor=TEXT)

    # ── Panel 6: Summary box ──────────────────────────────────────────────────
    ax_sum.axis('off')
    box = FancyBboxPatch((0.02, 0.02), 0.96, 0.96,
                         boxstyle="round,pad=0.02", linewidth=1.5,
                         edgecolor=C_ORTH, facecolor=BOX_BG,
                         transform=ax_sum.transAxes, zorder=0)
    ax_sum.add_patch(box)

    idx_Nk   = N_vals.index(N_cells_v)
    idx_d32  = d_list.index(32)
    delta_vr = st_o['valid_rate'] - st_r['valid_rate']
    lines = [
        ("Geometric Summary", 10, True),
        ("-" * 22, 8, False),
        ("", 0, False),
        ("Collision (k=17, d=32):", 9, True),
        (f"  Orth valid: {st_o['valid_rate']:.1%}", 8, False),
        (f"  Rand valid: {st_r['valid_rate']:.1%}", 8, False),
        (f"  Delta:      {delta_vr:+.1%}", 8, False),
        ("", 0, False),
        (f"Code Rate at N=2^{k_v} (d={d_v}):", 9, True),
        (f"  Orth:  {vcr_o[idx_Nk]:.3f}", 8, False),
        (f"  Rand:  {vcr_r[idx_Nk]:.3f}", 8, False),
        (f"  Ideal: ~0.632 (Poisson)", 8, False),
        ("", 0, False),
        (f"Alignment d=32, k={k_a}:", 9, True),
        (f"  Orth: {aln_o[idx_d32]:.2e}", 8, False),
        (f"  Rand: {aln_r[idx_d32]:.4f}", 8, False),
        ("", 0, False),
        ("Key insight:", 9, True),
        ("  Orth W achieves near-", 8, False),
        ("  Poisson hashing AND", 8, False),
        ("  zero off-diag overlap", 8, False),
        ("  across all N and d.", 8, False),
        ("", 0, False),
        ("  Random W converges", 8, False),
        ("  slowly as d grows.", 8, False),
    ]
    yp = 0.97; lh = 0.038
    for txt, fsize, bold in lines:
        if fsize == 0:
            yp -= lh * 0.4; continue
        ax_sum.text(0.06, yp, txt, transform=ax_sum.transAxes,
                    fontsize=fsize, color=TEXT, va='top', ha='left',
                    fontfamily='monospace',
                    fontweight='bold' if bold else 'normal')
        yp -= lh

    # ── Titles ────────────────────────────────────────────────────────────────
    fig.suptitle(
        "Codebook Geometric Analysis: "
        "Collision Distribution  |  Valid Code Rate vs N  |  Alignment vs d",
        fontsize=13, fontweight='bold', color=TEXT, y=1.01)
    fig.text(
        0.5, -0.01,
        r"Row 1: Hash collision (k=17, d=32, $N=2^{17}$ actions $\to 2^{17}$ cells)  "
        r"$|$  Row 2: Collective codebook performance curves",
        ha='center', fontsize=9.5, color=SUB, style='italic')

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"fig2_collision{suffix}.png")
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
