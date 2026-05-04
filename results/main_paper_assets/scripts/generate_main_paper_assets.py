"""
Generate main-paper figures and tables for the final NeurIPS framing.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "experiments" / "results"
ASSET_ROOT = ROOT / "results" / "main_paper_assets"
FIG_DIR = ASSET_ROOT / "figures"
TABLE_DIR = ASSET_ROOT / "tables"
for path in [FIG_DIR, TABLE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

BLUE = "#155EEF"
GREEN = "#079455"
RED = "#D92D20"
ORANGE = "#F79009"
INK = "#101828"
MUTED = "#667085"
FOG = "#EAECF0"
PANEL_BG = "#FCFCFD"
GRID = "#D0D5DD"

plt.rcParams.update(
    {
        "font.size": 11.0,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13.0,
        "axes.labelsize": 11.0,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "legend.fontsize": 10.0,
    }
)


def load_json(path):
    return json.loads(path.read_text())


def wilson_interval(successes, total, z=1.96):
    if total <= 0:
        return 0.0, 0.0
    phat = float(successes) / float(total)
    denom = 1.0 + z * z / float(total)
    center = (phat + z * z / (2.0 * total)) / denom
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denom
    return center - radius, center + radius


def natural_bits(i, k):
    return format(i, "0%db" % k)


def gray_bits(i, k):
    return format(i ^ (i >> 1), "0%db" % k)


def hamming(a, b):
    return sum(x != y for x, y in zip(a, b))


def add_round_box(ax, xy, width, height, text, fc, ec="#D0D5DD", fontsize=10.5, weight="normal"):
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.0,
        facecolor=fc,
        edgecolor=ec,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2.0,
        xy[1] + height / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
    )


def add_arrow(ax, start, end, color=MUTED, lw=1.6):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=lw,
            color=color,
        )
    )


def save_figure(fig, stem):
    fig.savefig(FIG_DIR / (stem + ".png"), dpi=260, bbox_inches="tight")
    fig.savefig(FIG_DIR / (stem + ".pdf"), bbox_inches="tight")
    fig.savefig(FIG_DIR / (stem + "_paper.png"), dpi=340, bbox_inches="tight")
    plt.close(fig)


def make_figure1():
    fig = plt.figure(figsize=(14.2, 4.8), facecolor="white")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.1, 1.0], wspace=0.24)

    ax = fig.add_subplot(gs[0, 0])
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("A. Dense vs Bitwise Head", loc="left", pad=10)
    add_round_box(ax, (0.06, 0.67), 0.16, 0.16, "latent\n$z$", "#F2F4F7", fontsize=12, weight="bold")
    add_round_box(ax, (0.35, 0.64), 0.24, 0.20, "Dense head", "#FDEAD7", ec="#F7B27A", fontsize=12, weight="bold")
    add_round_box(ax, (0.77, 0.67), 0.17, 0.16, "$D \\times M$\nlogits", "#FFF6ED", ec="#F7B27A", fontsize=11)
    add_arrow(ax, (0.22, 0.75), (0.35, 0.75))
    add_arrow(ax, (0.59, 0.75), (0.77, 0.75))
    ax.text(0.35, 0.54, "per-dimension categorical prediction", fontsize=10, color=MUTED)

    add_round_box(ax, (0.06, 0.26), 0.16, 0.16, "latent\n$z$", "#F2F4F7", fontsize=12, weight="bold")
    add_round_box(ax, (0.35, 0.23), 0.24, 0.20, "Bitwise head", "#DDF3EA", ec="#84D5AE", fontsize=12, weight="bold")
    add_round_box(ax, (0.77, 0.26), 0.17, 0.16, "$D \\times k$\nlogits", "#ECFDF3", ec="#84D5AE", fontsize=11)
    add_arrow(ax, (0.22, 0.34), (0.35, 0.34))
    add_arrow(ax, (0.59, 0.34), (0.77, 0.34))
    ax.text(0.35, 0.13, "$k = \\lceil \\log_2 M \\rceil$ bits per dimension", fontsize=10, color=MUTED)

    ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("B. Natural vs Gray Code", loc="left", pad=10)
    ax.text(0.10, 0.82, "Natural binary", color=RED, fontsize=12, fontweight="bold")
    natural_rows = [("2", "010"), ("3", "011"), ("4", "100"), ("5", "101")]
    y0 = 0.68
    for idx, (idx_label, code) in enumerate(natural_rows):
        y = y0 - idx * 0.13
        ax.text(0.12, y + 0.005, idx_label, ha="right", va="center", fontsize=10, color=MUTED)
        add_round_box(ax, (0.16, y - 0.05), 0.15, 0.10, code, "#FEF3F2", ec="#F97066", fontsize=11)
        if idx < len(natural_rows) - 1:
            add_arrow(ax, (0.235, y - 0.055), (0.235, y - 0.085), color=RED)
    ax.text(0.04, 0.11, "carry boundaries trigger\nmulti-bit jumps", color=RED, fontsize=10, fontweight="bold", ha="left")

    ax.text(0.67, 0.82, "Gray code", color=GREEN, fontsize=12, fontweight="bold")
    rows = [("2", "011"), ("3", "010"), ("4", "110"), ("5", "111")]
    for idx, (idx_label, code) in enumerate(rows):
        y = y0 - idx * 0.13
        ax.text(0.69, y + 0.005, idx_label, ha="right", va="center", fontsize=10, color=MUTED)
        add_round_box(ax, (0.73, y - 0.05), 0.15, 0.10, code, "#ECFDF3", ec="#84D5AE", fontsize=11)
        if idx < len(rows) - 1:
            add_arrow(ax, (0.805, y - 0.055), (0.805, y - 0.085), color=GREEN)
    ax.text(0.57, 0.11, "neighboring bins\ndiffer by 1 bit", color=GREEN, fontsize=10, fontweight="bold", ha="left")

    ax = fig.add_subplot(gs[0, 2])
    ax.set_facecolor(PANEL_BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("C. Bit-wise BCE Sees Code Geometry", loc="left", pad=10)
    add_round_box(ax, (0.18, 0.71), 0.62, 0.14, "neighboring-bin transition", "#F2F4F7", fontsize=12, weight="bold")
    add_round_box(ax, (0.18, 0.43), 0.62, 0.14, "target code change", "#FEF3F2", ec="#F97066", fontsize=12, weight="bold")
    add_round_box(ax, (0.18, 0.15), 0.62, 0.14, "bit-wise BCE supervision", "#ECFDF3", ec="#84D5AE", fontsize=12, weight="bold")
    add_arrow(ax, (0.49, 0.71), (0.49, 0.57), color=INK, lw=1.8)
    add_arrow(ax, (0.49, 0.43), (0.49, 0.29), color=INK, lw=1.8)
    ax.text(0.18, 0.02, "The target code changes the local loss geometry.", fontsize=10, color=MUTED)

    fig.subplots_adjust(left=0.04, right=0.995, top=0.91, bottom=0.12, wspace=0.24)
    save_figure(fig, "figure1_concept_method")


def make_figure2():
    fig = plt.figure(figsize=(16.2, 4.6), facecolor="white")
    outer = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.15, 1.05], wspace=0.28)

    m = 16
    k = int(math.ceil(math.log(m, 2)))
    bins = np.arange(m)
    natural_adj = np.array([hamming(natural_bits(i, k), natural_bits(i + 1, k)) for i in range(m - 1)])
    gray_adj = np.array([hamming(gray_bits(i, k), gray_bits(i + 1, k)) for i in range(m - 1)])

    ax = fig.add_subplot(outer[0, 0])
    ax.set_facecolor(PANEL_BG)
    ax.plot(bins[:-1], natural_adj, color=RED, marker="o", lw=2.0, ms=5, label="Natural")
    ax.plot(bins[:-1], gray_adj, color=GREEN, marker="s", lw=2.0, ms=4.5, label="Gray")
    ax.set_title("A. Adjacent Hamming Distance", loc="left", pad=10)
    ax.set_xlabel("transition  $i \\rightarrow i+1$  for  $M=16$")
    ax.set_ylabel("Hamming distance")
    ax.set_ylim(0.75, 4.25)
    ax.set_yticks([1, 2, 3, 4])
    ax.grid(True, axis="y", color=FOG, linewidth=0.8)
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.02, 0.03, "Gray stays at 1; Natural spikes at binary carry boundaries.", transform=ax.transAxes, fontsize=9.5, color=MUTED)

    inner = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0, 1], wspace=0.18)
    labels = ["Natural", "Gray"]
    encoders = [natural_bits, gray_bits]
    colors = [RED, GREEN]
    for idx in range(2):
        ax = fig.add_subplot(inner[0, idx])
        ax.set_facecolor(PANEL_BG)
        codes = [encoders[idx](i, k) for i in range(m)]
        mat = np.zeros((m, m), dtype=float)
        for i in range(m):
            for j in range(m):
                mat[i, j] = hamming(codes[i], codes[j])
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=k, interpolation="nearest")
        ax.set_title("%s code" % labels[idx], color=colors[idx], fontsize=12, pad=8)
        ax.set_xlabel("bin index")
        if idx == 0:
            ax.set_ylabel("bin index")
        ax.set_xticks([0, 5, 10, 15])
        ax.set_yticks([0, 5, 10, 15])
        for spine in ax.spines.values():
            spine.set_color(GRID)
    cb = fig.colorbar(im, ax=[fig.axes[1], fig.axes[2]], fraction=0.046, pad=0.04)
    cb.set_label("Hamming distance")
    fig.text(0.36, 0.92, "B. Synthetic Hamming Distance Matrix", fontsize=13, ha="left", color=INK)

    ax = fig.add_subplot(outer[0, 2])
    ax.set_facecolor(PANEL_BG)
    vals = np.array([2.7796, 2.2483, 4.0069], dtype=float)
    x = np.arange(3)
    bars = ax.bar(x, vals, color=[RED, GREEN, ORANGE], width=0.62)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2.0, val + 0.07, "%.4f" % val, ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["Natural", "Gray", "Random"])
    ax.set_ylabel("Mean target bit flips")
    ax.set_ylim(0, 4.5)
    ax.grid(True, axis="y", color=FOG, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("C. Demonstration Trajectories", loc="left", pad=10)

    fig.subplots_adjust(left=0.05, right=0.96, top=0.83, bottom=0.16)
    save_figure(fig, "figure2_code_geometry_diagnostic")


def make_figure3():
    data = load_json(RESULTS / "precision_custom25_200roll_core.json")
    rows = data["results"]
    labels = [row["label"] for row in rows]
    rates = np.array([row["success_rate"] * 100.0 for row in rows], dtype=float)
    lows = np.array([(row["success_rate"] - row["ci95_low"]) * 100.0 for row in rows], dtype=float)
    highs = np.array([(row["ci95_high"] - row["success_rate"]) * 100.0 for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8), facecolor="white")
    ax.set_facecolor(PANEL_BG)
    x = np.arange(len(labels))
    bars = ax.bar(x, rates, color=["#667085", RED, GREEN, BLUE], width=0.68, zorder=3)
    ax.errorbar(x, rates, yerr=np.vstack([lows, highs]), fmt="none", ecolor=INK, elinewidth=1.2, capsize=4, zorder=4)
    for bar, rate, hi in zip(bars, rates, highs):
        ax.text(bar.get_x() + bar.get_width() / 2.0, rate + hi + 1.3, "%.1f%%" % rate, ha="center", va="bottom", fontsize=9.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 70)
    ax.grid(True, axis="y", color=FOG, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title("Resolution study", loc="left", pad=10, fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "figure3_resolution_matters")


def make_figure4():
    payload = load_json(RESULTS / "bc_study" / "rollout_robustness" / "rollout_robustness.json")
    rows = payload["results"]
    lookup = {}
    for row in rows:
        lookup[row["exp_name"]] = row
    ms = [1024, 2048]
    variants = [("gvla", "Natural", RED), ("dense", "Dense", BLUE), ("gvla_gray", "Gray", GREEN)]

    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor="white")
    ax.set_facecolor(PANEL_BG)
    x = np.arange(len(ms))
    width = 0.22
    for idx, (prefix, label, color) in enumerate(variants):
        vals = []
        ylow = []
        yhigh = []
        for m in ms:
            row = lookup["%s_%d" % (prefix, m)]
            rate = float(row["success_rate"]) * 100.0
            lo, hi = float(row["wilson_95"][0]) * 100.0, float(row["wilson_95"][1]) * 100.0
            vals.append(rate)
            ylow.append(rate - lo)
            yhigh.append(hi - rate)
        vals = np.array(vals, dtype=float)
        bars = ax.bar(x + (idx - 1) * width, vals, width=width, color=color, label=label, zorder=3)
        ax.errorbar(x + (idx - 1) * width, vals, yerr=np.vstack([ylow, yhigh]), fmt="none", ecolor=INK, elinewidth=1.0, capsize=3, zorder=4)
        for bar, val, hi in zip(bars, vals, yhigh):
            ax.text(bar.get_x() + bar.get_width() / 2.0, val + hi + 0.8, "%.1f%%" % val, ha="center", va="bottom", fontsize=9.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["M=1024", "M=2048"])
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 30)
    ax.grid(True, axis="y", color=FOG, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("200-rollout robustness", loc="left", pad=10, fontsize=14, fontweight="bold")
    ax.text(x[0], 28.0, "Natural vs Gray:\np = 6.55e-08", ha="center", va="top", fontsize=9.5, color=INK)
    ax.text(x[1], 28.0, "Natural vs Gray:\np = 1.95e-04", ha="center", va="top", fontsize=9.5, color=INK)
    fig.tight_layout()
    save_figure(fig, "figure4_robustness_200roll")


def make_figure5():
    rows = load_json(RESULTS / "bc_study" / "validation_metrics" / "validation_metrics.json")
    selected = [row for row in rows if row["exp_name"] in {"gvla_128", "gvla_gray_128", "gvla_256", "gvla_gray_256", "gvla_1024", "gvla_gray_1024", "gvla_2048", "gvla_gray_2048"}]
    natural = sorted([row for row in selected if row["encoding"] == "Natural"], key=lambda row: row["n_bins"])
    gray = sorted([row for row in selected if row["encoding"] == "Gray"], key=lambda row: row["n_bins"])
    ms = np.array([row["n_bins"] for row in natural], dtype=int)
    xpos = np.arange(len(ms), dtype=float)
    xlabels = [str(m) for m in ms]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5), facecolor="white")
    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.grid(True, axis="y", color=FOG, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabels)
        ax.set_xlabel("Bins per dimension  $M$")

    axes[0].plot(xpos, [row["mean_action_l1"] for row in natural], color=RED, marker="o", lw=2.2, ms=6, label="Natural")
    axes[0].plot(xpos, [row["mean_action_l1"] for row in gray], color=GREEN, marker="s", lw=2.2, ms=5.5, label="Gray")
    axes[0].set_title("A. Action L1", loc="left")
    axes[0].set_ylabel("Mean action L1")
    axes[0].legend(frameon=False, loc="upper left")

    axes[1].plot(xpos, [row["mean_bin_error"] / float(row["n_bins"]) for row in natural], color=RED, marker="o", lw=2.2, ms=6)
    axes[1].plot(xpos, [row["mean_bin_error"] / float(row["n_bins"]) for row in gray], color=GREEN, marker="s", lw=2.2, ms=5.5)
    axes[1].set_title("B. Normalized Bin Error", loc="left")
    axes[1].set_ylabel("Bin error / $M$")

    axes[2].plot(xpos, [row["mean_hamming_error"] for row in natural], color=RED, marker="o", lw=2.2, ms=6)
    axes[2].plot(xpos, [row["mean_hamming_error"] for row in gray], color=GREEN, marker="s", lw=2.2, ms=5.5)
    axes[2].set_title("C. Hamming Error", loc="left")
    axes[2].set_ylabel("Mean Hamming error")

    fig.tight_layout()
    save_figure(fig, "figure5_validation_across_m")


def write_table_file(name, title, caption, source_lines, table_lines, notes=None):
    lines = ["# %s" % title, "", caption, "", "## Sources", ""]
    for src in source_lines:
        lines.append("- `%s`" % src)
    lines.extend(["", "## Table", ""])
    lines.extend(table_lines)
    if notes:
        lines.extend(["", "## Notes", ""])
        for note in notes:
            lines.append("- %s" % note)
    (TABLE_DIR / name).write_text("\n".join(lines) + "\n")


def make_tables():
    write_table_file(
        "table1_experimental_setup.md",
        "Table 1. Experimental Setup",
        "Main-paper setup summary for the controlled behavior-cloning comparison.",
        [
            "results/validation_split_lock.md",
            "results/comparison_lock.md",
        ],
        [
            "| Item | Setting |",
            "| --- | --- |",
            "| Environment | robosuite / RoboMimic Lift low-dimensional behavior cloning |",
            "| Task | Lift / precision-sensitive manipulation variant |",
            "| Dataset | RoboMimic Lift PH low-dimensional demonstrations |",
            "| Validation split | 10%, seed 20260503 |",
            "| Action | 7D continuous action, discretized per dimension |",
            "| Policy | Behavior cloning with shared MLP backbone |",
            "| Dense baseline | per-dimension M-way categorical head |",
            "| Bitwise heads | Natural, Gray, Random code |",
            "| Metrics | rollout success, action L1/L2, bin error, Hamming error |",
        ],
        notes=[
            "Main comparison rows use the matched BC contract from `experiments/bc_train.py` and `experiments/bc_eval.py`.",
            "The main paper should not mix these rows with VLA-backbone or synthetic pilot results.",
        ],
    )

    write_table_file(
        "table2_main_rollout_sweep.md",
        "Table 2. Main Rollout Success Sweep",
        "All entries are 50-rollout success rates. The `M=2048` results come from a separate later run. The central comparison is Gray-coded bitwise vs natural-binary bitwise; Dense remains a mixed expressive baseline.",
        [
            "results/final_submission_results.md",
            "experiments/results/bc_study/eval_results.json",
            "experiments/results/bc_study/eval_results_2048.json",
        ],
        [
            "| Head | Encoding | M=128 | M=256 | M=1024 | M=2048 |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
            "| Dense categorical | N/A | 10.0% (5/50) | 16.0% (8/50) | 4.0% (2/50) | 16.0% (8/50) |",
            "| Bitwise | Natural | 2.0% (1/50) | 4.0% (2/50) | 2.0% (1/50) | 4.0% (2/50) |",
            "| Bitwise | Gray | 16.0% (8/50) | 10.0% (5/50) | 18.0% (9/50) | 24.0% (12/50) |",
        ],
        notes=[
            "The main claim is `Gray > Natural`, not `Gray > Dense`.",
            "Do not mix 50-rollout rows with the separate 200-rollout robustness study in this table.",
        ],
    )

    write_table_file(
        "table3_reviewer_defense_controls.md",
        "Table 3. Reviewer-Defense Controls at M=1024",
        "Validation metrics used to answer the main reviewer attacks: single seed, orthogonality dependence, and arbitrary-code alternatives.",
        [
            "results/final_submission_results.md",
            "experiments/results/bc_study/reviewer_defense_metrics/validation_metrics.json",
        ],
        [
            "| Head / Code | L1 ↓ | Bin Err ↓ | Hamming ↓ | Exact Match ↑ |",
            "| --- | ---: | ---: | ---: | ---: |",
            "| Bitwise / Natural | 0.0554 | 28.2745 | 0.1938 | 0.2393 |",
            "| Bitwise / Natural (seed2) | 0.0530 | 27.0196 | 0.1890 | 0.2408 |",
            "| Bitwise / Gray | 0.0304 | 15.4692 | 0.1498 | 0.2946 |",
            "| Bitwise / Gray (no-orth) | 0.0294 | 14.9637 | 0.1497 | 0.2934 |",
            "| Bitwise / Random code | 0.4020 | 205.7474 | 0.2840 | 0.1819 |",
        ],
        notes=[
            "Natural seed2 stays close to Natural, reducing the single-seed artifact concern.",
            "Gray no-orth stays close to Gray, so the main effect is not explained by orthogonality regularization.",
            "Random degrades sharply, supporting the code-geometry interpretation.",
        ],
    )

    write_table_file(
        "table4_latency_boundary.md",
        "Table 4. Latency / Efficiency Boundary",
        "Small main-paper summary that shows the paper is not hiding unfavorable latency evidence.",
        [
            "results/final_submission_results.md",
            "experiments/results/bc_study/end_to_end_latency_gpu/end_to_end_latency.md",
        ],
        [
            "| Analysis | Evidence | Conclusion |",
            "| --- | --- | --- |",
            "| Matched GPU latency, `M=2048`, batch 1 | Dense `0.7361 ms`, Natural `2.2069 ms`, Gray `5.9004 ms` | Dense remains faster in the current BC setup |",
            "| Matched GPU latency, `M=2048`, batch 256 | Dense `0.7909 ms`, Natural `2.2364 ms`, Gray `5.6997 ms` | No universal speedup claim |",
            "| Bitwise-family latency | Natural is faster than Gray | Gray is for learnability, not speed |",
            "| Head-only scaling | Dense grows with output size; bitwise avoids linear `M` logits | Efficiency potential only |",
        ],
        notes=[
            "The main paper should describe latency as a qualified secondary analysis, not as primary evidence.",
            "Matched end-to-end latency up to `M<=2048` does not outperform Dense in the locked BC artifact.",
        ],
    )


def write_manifest():
    lines = [
        "# Main Paper Assets",
        "",
        "This folder contains the main-paper figures and tables for the final submission framing.",
        "",
        "## Figures",
        "",
        "- `figures/figure1_concept_method.png`",
        "- `figures/figure2_code_geometry_diagnostic.png`",
        "- `figures/figure3_resolution_matters.png`",
        "- `figures/figure4_robustness_200roll.png`",
        "- `figures/figure5_validation_across_m.png`",
        "",
        "## Tables",
        "",
        "- `tables/table1_experimental_setup.md`",
        "- `tables/table2_main_rollout_sweep.md`",
        "- `tables/table3_reviewer_defense_controls.md`",
        "- `tables/table4_latency_boundary.md`",
        "",
        "## Regeneration",
        "",
        "Run `scripts/generate_main_paper_assets.py` with a Python environment that has `matplotlib` installed.",
        "",
    ]
    (ASSET_ROOT / "README.md").write_text("\n".join(lines))


def main():
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    make_figure5()
    make_tables()
    write_manifest()
    print("Saved assets under %s" % ASSET_ROOT)


if __name__ == "__main__":
    main()
