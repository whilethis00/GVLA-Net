"""
fig_concept.py
==============
Concept figure: problem → method → result

Panel 1 (Left)   — Dense head: z → M arrows, O(dM), 복잡하고 무거움
Panel 2 (Middle) — GVLA routing: 점 잔뜩 + k hyperplane + 색 화살표로 각 셀로 수렴
Panel 3 (Right)  — 결과: 같은 점들이 8개 셀에 이쁘게 정렬됨

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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

torch.manual_seed(42)
np.random.seed(42)

OUT_DIR = "/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

CELL_COLORS = [
    '#E74C3C', '#E67E22', '#2ECC71', '#1ABC9C',
    '#3498DB', '#9B59B6', '#F1C40F', '#EC407A',
]
C_DENSE     = '#C0392B'
C_GVLA      = '#1A9E94'
C_HYPERPLANE = ['#1A9E94', '#C0392B', '#B07D00']
W3 = torch.eye(3)


# ── 공용 점 (패널 2, 3 동일 사용) ────────────────────────────────────────────
def make_points(n=500):
    torch.manual_seed(99)
    pts = torch.randn(n, 3)
    pts = pts / (pts.norm(dim=1, keepdim=True) * 1.15)
    codes = ((pts @ W3.T) > 0).int()
    colors = [CELL_COLORS[int(c[0])*4 + int(c[1])*2 + int(c[2])] for c in codes]
    return pts, codes, colors

SHARED_PTS, SHARED_CODES, SHARED_COLORS = make_points(500)


# ── Panel 1: Dense ───────────────────────────────────────────────────────────
def draw_dense(ax, M=28):
    ax.set_xlim(-0.3, 3.6)
    ax.set_ylim(-1.25, 1.25)
    ax.axis('off')

    # latent node
    ax.add_patch(plt.Circle((0, 0), 0.18, color='#34495E', zorder=5))
    ax.text(0, 0, '$z$', ha='center', va='center',
            fontsize=13, color='white', fontweight='bold', zorder=6)

    # M output nodes + arrows
    y_positions = np.linspace(-1.1, 1.1, M)
    for i, y in enumerate(y_positions):
        alpha = 0.20 + 0.18 * abs(math.sin(i * 0.7))
        ax.annotate('', xy=(2.72, y), xytext=(0.18, 0),
                    arrowprops=dict(arrowstyle='->', color=C_DENSE,
                                    lw=0.85, alpha=alpha))
        ax.add_patch(plt.Circle((2.86, y), 0.052,
                                color=C_DENSE,
                                alpha=0.45 + 0.35*(i % 4 == 0), zorder=4))

    ax.text(0, -1.2, 'latent $z$', ha='center', va='top',
            fontsize=9, color='#444')
    ax.text(3.22, 0, f'$M$ logits', ha='left', va='center',
            fontsize=9, color=C_DENSE)
    ax.text(1.38, 1.22, 'score every class — $O(dM)$',
            ha='center', va='bottom', fontsize=9, color='#666', style='italic')

    ax.text(1.38, -1.22, r'$\mathcal{O}(dM)$',
            ha='center', va='top', fontsize=14, color=C_DENSE, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.35', facecolor='#FDECEA',
                      edgecolor=C_DENSE, linewidth=1.5))

    ax.set_title('Dense Head\n(softmax over $M$ bins)',
                 fontsize=12, color='#111', fontweight='bold', pad=10)


# ── Panel 2: GVLA routing ────────────────────────────────────────────────────
def draw_routing(ax):
    lim = 1.55
    g   = np.linspace(-lim, lim, 26)
    G1, G2 = np.meshgrid(g, g)
    Z0 = np.zeros_like(G1)

    # hyperplanes
    for col, surf in zip(C_HYPERPLANE,
                         [(G1,G2,Z0),(G1,Z0,G2),(Z0,G1,G2)]):
        ax.plot_surface(*surf, color=col, alpha=0.10, shade=False)

    # hyperplane 법선
    for i,(dx,dy,dz) in enumerate([(0,0,0.78),(0,0.78,0),(0.78,0,0)]):
        ax.quiver(0,0,0,dx,dy,dz, color=C_HYPERPLANE[i],
                  linewidth=2.3, arrow_length_ratio=0.28, alpha=0.92)

    pts = SHARED_PTS
    # 전체 점 — 회색 (아직 정렬 전)
    ax.scatter(pts[:,0].numpy(), pts[:,1].numpy(), pts[:,2].numpy(),
               c='#BBBBBB', s=7, alpha=0.30, zorder=2)

    # 각 octant에서 선택된 점만 컬러 화살표
    octant_center = lambda c0,c1,c2: np.array([
        0.85 if c0 else -0.85,
        0.85 if c1 else -0.85,
        0.85 if c2 else -0.85,
    ], dtype=float)

    np.random.seed(7)
    for cell_idx in range(8):
        c0 = cell_idx // 4
        c1 = (cell_idx // 2) % 2
        c2 = cell_idx % 2
        mask = np.array([
            (int(SHARED_CODES[i][0])==c0 and
             int(SHARED_CODES[i][1])==c1 and
             int(SHARED_CODES[i][2])==c2)
            for i in range(len(SHARED_CODES))
        ])
        idxs = np.where(mask)[0]
        chosen = np.random.choice(idxs, size=min(12, len(idxs)), replace=False)
        col = CELL_COLORS[cell_idx]
        center = octant_center(c0, c1, c2)

        for i in chosen:
            pt = pts[i].numpy()
            dp = center * 0.80 - pt
            ax.scatter(*pt, color=col, s=20, alpha=0.88,
                       zorder=5, edgecolors='white', linewidths=0.3)
            ax.quiver(*pt, *dp, color=col, alpha=0.55,
                      linewidth=0.9, arrow_length_ratio=0.22, normalize=False)

    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    for axis, label in zip([ax.xaxis,ax.yaxis,ax.zaxis],['$x$','$y$','$z$']):
        axis.pane.fill = True
        axis.pane.set_facecolor('#F4F4F4')
    ax.set_xlabel('$x$', fontsize=9, color='#444', labelpad=0)
    ax.set_ylabel('$y$', fontsize=9, color='#444', labelpad=0)
    ax.set_zlabel('$z$', fontsize=9, color='#444', labelpad=0)
    ax.tick_params(labelsize=6, colors='#888')

    ax.set_title('GVLA: $k$ Binary Questions\n(each hyperplane → 1 bit)',
                 fontsize=12, color='#111', fontweight='bold', pad=8)
    ax.text2D(0.5, -0.04, r'$\mathcal{O}(d\log M)$',
              transform=ax.transAxes, ha='center', fontsize=14,
              color=C_GVLA, fontweight='bold',
              bbox=dict(boxstyle='round,pad=0.35', facecolor='#E8F8F5',
                        edgecolor=C_GVLA, linewidth=1.5))


# ── Panel 3: 결과 ─────────────────────────────────────────────────────────────
def draw_cells(ax):
    lim = 1.48
    delta = 0.02

    # 반투명 octant 블록
    for cell_idx in range(8):
        c0 = cell_idx // 4
        c1 = (cell_idx // 2) % 2
        c2 = cell_idx % 2
        sx,sy,sz = (1 if c0 else -1),(1 if c1 else -1),(1 if c2 else -1)
        x0 = delta if sx>0 else -lim;  x1 = lim if sx>0 else -delta
        y0 = delta if sy>0 else -lim;  y1 = lim if sy>0 else -delta
        z0 = delta if sz>0 else -lim;  z1 = lim if sz>0 else -delta
        col = CELL_COLORS[cell_idx]
        faces = [
            [[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0]],
            [[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]],
            [[x0,y0,z0],[x1,y0,z0],[x1,y0,z1],[x0,y0,z1]],
            [[x0,y1,z0],[x1,y1,z0],[x1,y1,z1],[x0,y1,z1]],
            [[x0,y0,z0],[x0,y1,z0],[x0,y1,z1],[x0,y0,z1]],
            [[x1,y0,z0],[x1,y1,z0],[x1,y1,z1],[x1,y0,z1]],
        ]
        poly = Poly3DCollection(faces, alpha=0.20, linewidth=0.3,
                                edgecolor='white')
        poly.set_facecolor(col)
        ax.add_collection3d(poly)

    # 같은 점들 — 이제 컬러로 뚜렷하게
    pts = SHARED_PTS
    ax.scatter(pts[:,0].numpy(), pts[:,1].numpy(), pts[:,2].numpy(),
               c=SHARED_COLORS, s=14, alpha=0.85,
               zorder=5, edgecolors='white', linewidths=0.2)

    # bit label
    for cell_idx in range(8):
        c0 = cell_idx // 4
        c1 = (cell_idx // 2) % 2
        c2 = cell_idx % 2
        cx = 0.74*(1 if c0 else -1)
        cy = 0.74*(1 if c1 else -1)
        cz = 0.74*(1 if c2 else -1)
        ax.text(cx, cy, cz+0.22, format(cell_idx,'03b'),
                ha='center', va='bottom', fontsize=8,
                color='#222', fontweight='bold', zorder=6)

    ax.set_xlim(-lim,lim); ax.set_ylim(-lim,lim); ax.set_zlim(-lim,lim)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.pane.fill = True
        axis.pane.set_facecolor('#F4F4F4')
    ax.set_xlabel('$x$', fontsize=9, color='#444', labelpad=0)
    ax.set_ylabel('$y$', fontsize=9, color='#444', labelpad=0)
    ax.set_zlabel('$z$', fontsize=9, color='#444', labelpad=0)
    ax.tick_params(labelsize=6, colors='#888')

    ax.set_title('Result: $2^k$ Clean Cells\n(same 500 points, now perfectly sorted)',
                 fontsize=12, color='#111', fontweight='bold', pad=8)
    ax.text2D(0.5, -0.04,
              '$k$ bits → $2^k$ cells  (exponential capacity)',
              transform=ax.transAxes, ha='center', fontsize=9.5,
              color='#444', style='italic')


# ── Main ─────────────────────────────────────────────────────────────────────
def make_fig(suffix='_paper', dpi=300):
    plt.rcdefaults()
    fig = plt.figure(figsize=(17, 6))
    fig.patch.set_facecolor('#FFFFFF')

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132, projection='3d')
    ax3 = fig.add_subplot(133, projection='3d')

    ax2.view_init(elev=22, azim=38)
    ax3.view_init(elev=22, azim=38)

    draw_dense(ax1)
    draw_routing(ax2)
    draw_cells(ax3)

    for x in [0.345, 0.665]:
        fig.text(x, 0.50, '→', fontsize=30, color='#BBBBBB',
                 ha='center', va='center', fontweight='bold')

    fig.suptitle('GVLA: From Exhaustive Scoring to Geometric Routing',
                 fontsize=14, fontweight='bold', color='#111', y=1.02)
    fig.text(
        0.5, -0.04,
        r'Dense scores all $M$ classes $(\mathcal{O}(dM))$'
        r'$\;\longrightarrow\;$ GVLA asks $k=\lceil\log_2 M\rceil$ binary questions'
        r'$\;(\mathcal{O}(d\log M))$'
        r'$\;\longrightarrow\;$ $2^k$ clean geometric cells',
        ha='center', fontsize=10, color='#555', style='italic',
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    out = os.path.join(OUT_DIR, f'fig_concept{suffix}.png')
    fig.savefig(out, dpi=dpi, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close(fig)
    print(f'[SAVED] {out}')


make_fig(suffix='_paper', dpi=300)
make_fig(suffix='',       dpi=150)
print('Done.')
