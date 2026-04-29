"""
fig_concept.py
==============
Concept figure: problem → method → result

Panel 1 (Left)   — "문제": Dense head, M개 화살표 → 복잡하고 무거움
Panel 2 (Middle) — "우리 방법": k개 hyperplane + 점 → cell routing 화살표
Panel 3 (Right)  — "결과": k=3, 8개 cell 이쁘게 색으로 구분

Usage:
    python experiments/fig_concept.py
"""

import os
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

torch.manual_seed(42)
np.random.seed(42)

OUT_DIR = "/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── 색상 팔레트 ──────────────────────────────────────────────────────────────
CELL_COLORS = [
    '#E74C3C', '#E67E22', '#2ECC71', '#1ABC9C',
    '#3498DB', '#9B59B6', '#F1C40F', '#EC407A',
]
C_DENSE  = '#C0392B'
C_GVLA   = '#1A9E94'
C_ARROW  = '#555555'
C_HYPERPLANE = ['#1A9E94', '#C0392B', '#B07D00']

W3 = torch.eye(3)   # 3 orthogonal hyperplanes in 3D


# ── Panel 1: Dense — M개 화살표 ──────────────────────────────────────────────
def draw_dense(ax, M=24):
    """2D: latent node z → M output nodes, fan of arrows"""
    ax.set_xlim(-0.3, 3.5)
    ax.set_ylim(-1.2, 1.2)
    ax.axis('off')

    # latent node
    circ_z = plt.Circle((0, 0), 0.18, color='#34495E', zorder=5)
    ax.add_patch(circ_z)
    ax.text(0, 0, '$z$', ha='center', va='center',
            fontsize=13, color='white', fontweight='bold', zorder=6)

    # M output nodes evenly spaced vertically
    y_positions = np.linspace(-1.05, 1.05, M)
    for i, y in enumerate(y_positions):
        # arrow z → node
        alpha = 0.25 + 0.15 * abs(math.sin(i))
        ax.annotate('', xy=(2.7, y), xytext=(0.18, 0),
                    arrowprops=dict(arrowstyle='->', color=C_DENSE,
                                    lw=0.9, alpha=alpha))
        # output node
        c = plt.Circle((2.85, y), 0.055,
                        color=C_DENSE, alpha=0.55 + 0.3 * (i % 3 == 0), zorder=4)
        ax.add_patch(c)

    # labels
    ax.text(0, -1.15, 'latent $z$\n($d$-dim)', ha='center', va='top',
            fontsize=9, color='#333333')
    ax.text(3.1, 0, f'$M$ logits\n({M} shown)', ha='left', va='center',
            fontsize=9, color=C_DENSE)
    ax.text(1.4, 1.18, f'$M$ operations\n→ score every class',
            ha='center', va='bottom', fontsize=9.5, color='#555555',
            style='italic')

    # complexity badge
    ax.text(1.35, -1.15, r'$\mathcal{O}(dM)$',
            ha='center', va='top', fontsize=13, color=C_DENSE,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FDECEA',
                      edgecolor=C_DENSE, linewidth=1.5))

    ax.set_title('Dense Head\n(softmax over $M$ bins)',
                 fontsize=12, color='#111111', fontweight='bold', pad=10)


# ── Panel 2: Geometric routing — hyperplane + 화살표 ─────────────────────────
def draw_routing(ax):
    """3D: k=3 hyperplanes, query points → routing arrows to octant centers"""
    lim = 1.5
    g   = np.linspace(-lim, lim, 25)
    G1, G2 = np.meshgrid(g, g)
    Z0 = np.zeros_like(G1)

    plane_alpha = 0.13
    ax.plot_surface(G1, G2, Z0,
                    color=C_HYPERPLANE[0], alpha=plane_alpha, shade=False)
    ax.plot_surface(G1, Z0, G2,
                    color=C_HYPERPLANE[1], alpha=plane_alpha, shade=False)
    ax.plot_surface(Z0, G1, G2,
                    color=C_HYPERPLANE[2], alpha=plane_alpha, shade=False)

    # Hyperplane normal vectors (small arrows at origin)
    for i, (dx, dy, dz) in enumerate([(0, 0, 0.7), (0, 0.7, 0), (0.7, 0, 0)]):
        ax.quiver(0, 0, 0, dx, dy, dz,
                  color=C_HYPERPLANE[i], linewidth=2.0,
                  arrow_length_ratio=0.3, alpha=0.85)

    # query points — 선별해서 8개 (각 octant 하나씩)
    torch.manual_seed(17)
    raw = torch.randn(200, 3)
    raw = raw / raw.norm(dim=1, keepdim=True)
    codes_all = ((raw @ W3.T) > 0).int()
    # pick one representative per octant
    query_pts, query_codes = [], []
    for c0 in range(2):
        for c1 in range(2):
            for c2 in range(2):
                target = torch.tensor([c0, c1, c2])
                for i in range(len(codes_all)):
                    if (codes_all[i] == target).all():
                        query_pts.append(raw[i])
                        query_codes.append(codes_all[i])
                        break

    octant_center = lambda c: torch.tensor([
        0.75 if c[0] else -0.75,
        0.75 if c[1] else -0.75,
        0.75 if c[2] else -0.75,
    ])

    for pt, code in zip(query_pts, query_codes):
        idx = code[0] * 4 + code[1] * 2 + code[2]
        col = CELL_COLORS[idx]
        ax.scatter(*pt.tolist(), color=col, s=55, zorder=5, alpha=0.95,
                   edgecolors='white', linewidths=0.6)
        center = octant_center(code)
        dp = center * 0.72 - pt
        ax.quiver(*pt.tolist(), *dp.tolist(),
                  color=col, alpha=0.75, linewidth=1.5,
                  arrow_length_ratio=0.28, normalize=False)

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('$x$', fontsize=9, color='#333333', labelpad=0)
    ax.set_ylabel('$y$', fontsize=9, color='#333333', labelpad=0)
    ax.set_zlabel('$z$', fontsize=9, color='#333333', labelpad=0)
    ax.tick_params(labelsize=6, colors='#666666')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor('#F7F7F7')
    ax.set_title('GVLA: $k$ Binary Questions\n(each hyperplane asks 1 bit)',
                 fontsize=12, color='#111111', fontweight='bold', pad=8)

    ax.text2D(0.5, -0.04,
              r'$\mathcal{O}(d \log M)$',
              transform=ax.transAxes, ha='center', fontsize=13,
              color=C_GVLA, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.35', facecolor='#E8F8F5',
                        edgecolor=C_GVLA, linewidth=1.5))


# ── Panel 3: 결과 — 이쁘게 쪼개진 8개 octant ────────────────────────────────
def draw_cells(ax):
    """3D: 8 octants filled with colored volumes"""
    lim = 1.4
    delta = 0.015   # small gap between cells

    octants = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                # vertices of the octant cube
                x0 = delta if sx > 0 else -lim
                x1 = lim   if sx > 0 else -delta
                y0 = delta if sy > 0 else -lim
                y1 = lim   if sy > 0 else -delta
                z0 = delta if sz > 0 else -lim
                z1 = lim   if sz > 0 else -delta
                octants.append(((x0, x1), (y0, y1), (z0, z1)))

    for i, ((x0,x1),(y0,y1),(z0,z1)) in enumerate(octants):
        col = CELL_COLORS[i]
        # 6 faces
        faces = [
            [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],  # bottom
            [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],  # top
            [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],  # front
            [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],  # back
            [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],  # left
            [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],  # right
        ]
        poly = Poly3DCollection(faces, alpha=0.38, linewidth=0.4,
                                edgecolor='white')
        poly.set_facecolor(col)
        ax.add_collection3d(poly)

        # cell center dot + bit label
        cx = (x0+x1)/2; cy = (y0+y1)/2; cz = (z0+z1)/2
        ax.scatter(cx, cy, cz, color=col, s=40, zorder=6,
                   edgecolors='white', linewidths=0.8, alpha=0.95)
        bits = format(i, '03b')
        ax.text(cx, cy, cz + 0.18, bits,
                ha='center', va='bottom', fontsize=7.5,
                color='#222222', fontweight='bold')

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel('$x$', fontsize=9, color='#333333', labelpad=0)
    ax.set_ylabel('$y$', fontsize=9, color='#333333', labelpad=0)
    ax.set_zlabel('$z$', fontsize=9, color='#333333', labelpad=0)
    ax.tick_params(labelsize=6, colors='#666666')
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor('#F7F7F7')
    ax.set_title('Result: $2^k$ Clean Cells\n($k=3$ → 8 distinct regions)',
                 fontsize=12, color='#111111', fontweight='bold', pad=8)

    ax.text2D(0.5, -0.04,
              '$k$ bits → $2^k$ cells\n(exponential capacity)',
              transform=ax.transAxes, ha='center', fontsize=9.5,
              color='#333333', style='italic')


# ── Main ─────────────────────────────────────────────────────────────────────
def make_fig(suffix='_paper', dpi=300):
    plt.rcdefaults()
    fig = plt.figure(figsize=(17, 6))
    fig.patch.set_facecolor('#FFFFFF')

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    ax2.view_init(elev=22, azim=35)
    ax3.view_init(elev=22, azim=35)

    draw_dense(ax1)
    draw_routing(ax2)
    draw_cells(ax3)

    # ── 화살표로 패널 연결 ─────────────────────────────────────────────────
    for x in [0.345, 0.665]:
        fig.text(x, 0.52, '→', fontsize=28, color='#AAAAAA',
                 ha='center', va='center', fontweight='bold')

    fig.suptitle(
        'GVLA: From Exhaustive Scoring to Geometric Routing',
        fontsize=14, fontweight='bold', color='#111111', y=1.02,
    )
    fig.text(
        0.5, -0.04,
        r'Dense scores all $M$ classes ($\mathcal{O}(dM)$)  '
        r'$\;\longrightarrow\;$  '
        r'GVLA asks $k = \lceil\log_2 M\rceil$ binary questions ($\mathcal{O}(d\log M)$)  '
        r'$\;\longrightarrow\;$  '
        r'$2^k$ clean geometric cells',
        ha='center', fontsize=10, color='#555555', style='italic',
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    out = os.path.join(OUT_DIR, f'fig_concept{suffix}.png')
    fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close(fig)
    print(f'[SAVED] {out}')


make_fig(suffix='_paper', dpi=300)
make_fig(suffix='',       dpi=150)
print('Done.')
