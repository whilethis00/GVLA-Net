"""
GVLA-Net: FLOPs Reduction Bar Chart
Publication-quality grouped bar chart comparing FLOPs reduction across VLA models and action space sizes.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker

OUT_DIR = "/home/introai11/.agile/users/hsjung/projects/GVLA-Net/experiments/results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

models = ['Octo-Base', 'OpenVLA-7B', 'RT-2-X', 'pi0.5']
n_labels = ['N=1k\n(10-bit)', 'N=32k\n(15-bit)', 'N=1M\n(20-bit)']

flops_reduction = {
    'Octo-Base':   [102.6,   2188.8,  52531.1],
    'OpenVLA-7B':  [102.4,   2185.3,  52448.0],
    'RT-2-X':      [102.4,   2185.3,  52448.0],
    'pi0.5':       [102.5,   2187.7,  52505.5],
}

colors = {
    'Octo-Base':   '#FF6B6B',
    'OpenVLA-7B':  '#4ECDC4',
    'RT-2-X':      '#45B7D1',
    'pi0.5':       '#96CEB4',
}

def format_val(v):
    iv = int(round(v))
    return f"{iv:,}×"


def make_chart(dark=True):
    if dark:
        plt.style.use('dark_background')
        text_color     = '#FFFFFF'
        sub_color      = '#AAAAAA'
        baseline_color = '#FFD700'
        fig_face       = '#0E0E0E'
        ax_face        = '#1A1A2E'
        grid_color     = '#FFFFFF'
        spine_color    = '#444444'
    else:
        plt.rcdefaults()
        text_color     = '#111111'
        sub_color      = '#555555'
        baseline_color = '#B8860B'
        fig_face       = '#FAFAFA'
        ax_face        = '#FFFFFF'
        grid_color     = '#333333'
        spine_color    = '#CCCCCC'

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor(fig_face)
    ax.set_facecolor(ax_face)

    n_models  = len(models)
    n_groups  = len(n_labels)
    bar_width = 0.17
    group_gap = 0.04
    total_width = n_models * bar_width + group_gap

    x = np.arange(n_groups) * (total_width + 0.30)

    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2.0) * bar_width
        vals   = flops_reduction[model]
        col    = colors[model]

        bars = ax.bar(
            x + offset, vals,
            width=bar_width,
            color=col,
            alpha=0.88,
            edgecolor=text_color,
            linewidth=0.5,
            zorder=3,
            label=model,
        )

        for bar, v in zip(bars, vals):
            fsize = 7.0 if v > 10000 else 8.0
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() * 1.12,
                format_val(v),
                ha='center', va='bottom',
                fontsize=fsize, fontweight='bold',
                color=text_color, rotation=55,
            )

    # Baseline
    x_min = x[0] - total_width * 0.7
    x_max = x[-1] + total_width * 0.7
    ax.axhline(y=1, color=baseline_color, linewidth=1.8,
               linestyle='--', zorder=2, alpha=0.9)
    ax.text(x_max + 0.08, 1.0, 'Baseline (1×)',
            va='center', ha='left', fontsize=9,
            color=baseline_color, fontweight='bold')

    # Annotation arrow
    ax.annotate(
        "~52,000× reduction\nat N=1M",
        xy=(x[2], 52531.1),
        xytext=(x[2] + 0.60, 52531.1 * 5.5),
        fontsize=9.5, fontweight='bold',
        color=baseline_color,
        arrowprops=dict(
            arrowstyle='->', color=baseline_color,
            lw=1.8,
            connectionstyle='arc3,rad=-0.25',
        ),
        ha='left',
    )

    ax.set_yscale('log')
    ax.set_xlim(x_min - 0.1, x_max + 0.6)
    ax.set_ylim(0.4, 52531.1 * 40)

    ax.set_xticks(x)
    ax.set_xticklabels(n_labels, fontsize=12, color=text_color, fontweight='bold')
    ax.tick_params(axis='y', labelsize=10, colors=text_color)
    ax.tick_params(axis='x', colors=text_color)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{int(v):,}×" if v >= 1 else f"{v:.1f}×"
        )
    )

    ax.set_ylabel("FLOPs Reduction (×)", fontsize=13, color=text_color, labelpad=12)
    ax.set_xlabel("Action Space Size", fontsize=13, color=text_color, labelpad=10)

    ax.grid(axis='y', color=grid_color, alpha=0.15, linestyle='--', linewidth=0.6, zorder=1)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_edgecolor(spine_color)
        spine.set_linewidth(0.8)

    legend = ax.legend(
        loc='upper left',
        fontsize=10,
        framealpha=0.35 if dark else 0.75,
        edgecolor='#888888',
        labelcolor=text_color,
        title="VLA Model",
        title_fontsize=10,
    )
    legend.get_title().set_color(text_color)

    fig.suptitle(
        "FLOPs Reduction: GVLA-Net Head vs. Dense Action Head\n"
        "(Per VLA Model  ×  Action Space Size)",
        fontsize=15, fontweight='bold', color=text_color, y=0.98,
    )
    ax.annotate(
        "Higher = Better  |  Log Scale  |  Measured on A100",
        xy=(0.5, -0.11), xycoords='axes fraction',
        ha='center', va='top',
        fontsize=10, color=sub_color, style='italic',
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    return fig


fig_dark = make_chart(dark=True)
out_dark = os.path.join(OUT_DIR, "flops_reduction_chart.png")
fig_dark.savefig(out_dark, dpi=200, bbox_inches='tight',
                 facecolor=fig_dark.get_facecolor())
plt.close(fig_dark)
print(f"[SAVED] {out_dark}")

fig_light = make_chart(dark=False)
out_light = os.path.join(OUT_DIR, "flops_reduction_chart_light.png")
fig_light.savefig(out_light, dpi=200, bbox_inches='tight',
                  facecolor=fig_light.get_facecolor())
plt.close(fig_light)
print(f"[SAVED] {out_light}")

print("\nTask 1 complete.")
