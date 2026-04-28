"""
plot_robosuite_results.py
=========================
Generate paper-quality figures for the PickPlaceCan quantisation study while
reusing the existing head latency benchmark from the original Robosuite study.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).parent / "results"
SUCCESS_BASE = RESULTS_DIR / "robosuite_pickplace_transition_100roll"
LAT_BASE = RESULTS_DIR / "robosuite_study"


with open(SUCCESS_BASE / "results.json") as f:
    success_raw = json.load(f)

with open(LAT_BASE / "results.json") as f:
    latency_raw = json.load(f)


action_dim = int(success_raw.get("action_dim", 7))
success_rate = success_raw["success_rate"]
M_vals = sorted(int(k) for k in success_rate.keys() if k != "inf")
sr_vals = [success_rate[str(m)] * 100 for m in M_vals]
sr_cont = success_rate.get("inf", 1.0) * 100
N_total_vals = [m ** action_dim for m in M_vals]

latency = latency_raw.get("latency", {})
N_lat_all = sorted(int(k) for k in latency.keys())
N_lat = [N for N in N_lat_all if N >= 65536]
d_ms = [latency[str(N)]["dense_ms"] for N in N_lat]
g_ms = [latency[str(N)]["gvla_ms"] for N in N_lat]

dense_mem_MB = [N * 256 * 4 / 1e6 for N in N_total_vals]
gvla_mem_MB = [int(np.ceil(np.log2(max(N, 2)))) * 256 * 4 / 1e6 for N in N_total_vals]


BLUE = "#2166ac"
RED = "#d73027"
GREEN = "#1a9641"
GRAY = "#999999"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
})


# =========================================================
# Figure 1: 3-panel paper figure
# =========================================================

fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

# ---- Panel A: Success rate vs M ----
ax = axes[0]
ax.plot(M_vals, sr_vals, "o-", color=BLUE, lw=2.5, ms=9, zorder=5, label="Quantised policy")
ax.axhline(sr_cont, color=GRAY, ls="--", lw=1.8, label=f"Continuous ({sr_cont:.0f}%)")

fail_idx = [i for i, s in enumerate(sr_vals) if s < 100]
if fail_idx:
    ax.axvspan(M_vals[0] / 1.3, M_vals[fail_idx[-1]] * 1.18, color=RED, alpha=0.07, label="Insufficient precision")

ax.set_xscale("log", base=2)
ax.set_xlabel("Action bins per dimension  $M$")
ax.set_ylabel("Task Success Rate (%)")
ax.set_title("(A)  PickPlaceCan: Precision vs Success")
ax.set_ylim(-3, 110)
ax.set_xlim(M_vals[0] * 0.85, M_vals[-1] * 1.15)
ax.legend(loc="lower right")

for m, sr in zip(M_vals, sr_vals):
    offset_y = 5 if sr < 98 else -10
    ax.annotate(f"{sr:.0f}%", xy=(m, sr), xytext=(0, offset_y), textcoords="offset points",
                ha="center", fontsize=9.5, color=BLUE, fontweight="bold")

ax.axvline(96, color=GREEN, ls=":", lw=1.5, alpha=0.8)
ax.text(96 * 1.03, 18, "M=96\n100% success", color=GREEN, fontsize=9, va="bottom")

# ---- Panel B: Latency vs N ----
ax = axes[1]
ax.plot(N_lat, d_ms, "^-", color=RED, lw=2.5, ms=9, label="Dense Head  $\\mathcal{O}(N)$")
ax.plot(N_lat, g_ms, "s-", color=GREEN, lw=2.5, ms=9, label="GVLA Head   $\\mathcal{O}(\\log N)$")
ax.axhline(10.0, color=GRAY, ls="--", lw=1.5, alpha=0.8)
ax.text(N_lat[0] * 1.45, 10.5, "100 Hz real-time budget (10 ms)", fontsize=9, color=GRAY)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Total action space  $N$")
ax.set_ylabel("Head latency (ms)")
ax.set_title("(B)  Action Head: Latency vs $N$")
ax.legend(loc="upper left")

N_big = N_lat[-1]
d_big = d_ms[-1]
g_big = g_ms[-1]
ax.annotate(f"{d_big / g_big:.0f}×\nspeedup",
            xy=(N_big, (d_big + g_big) / 2),
            xytext=(-60, 0), textcoords="offset points",
            fontsize=10, color=GREEN,
            arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))

# ---- Panel C: Memory wall ----
ax = axes[2]
ax_r = ax.twinx()
lns1 = ax.plot(M_vals, sr_vals, "o-", color=BLUE, lw=2.5, ms=9, label="Success rate (↑ better)")
ax.axhline(sr_cont, color=GRAY, ls="--", lw=1.5, alpha=0.7)
lns2 = ax_r.semilogy(M_vals, dense_mem_MB, "^--", color=RED, lw=2.2, ms=8, label="Dense head memory (↓ better)")
lns3 = ax_r.semilogy(M_vals, gvla_mem_MB, "s--", color=GREEN, lw=2.2, ms=8, label="GVLA head memory (↓ better)")
ax_r.axhline(40 * 1024, color=RED, ls=":", lw=1.4, alpha=0.6)
ax_r.text(M_vals[-1] * 0.82, 40 * 1024 * 1.6, "40 GB GPU limit", color=RED, fontsize=8.5)

ax.set_xscale("log", base=2)
ax.set_xlabel("Action bins per dimension  $M$")
ax.set_ylabel("Success Rate (%)", color=BLUE)
ax_r.set_ylabel("Action Head Memory (MB)")
ax.set_title("(C)  Precision–Memory Trade-off")
ax.set_ylim(-3, 115)

all_lines = lns1 + lns2 + lns3
ax.legend(all_lines, [l.get_label() for l in all_lines], fontsize=9.5, loc="center right")

plt.tight_layout(pad=1.6)
for suf, dpi in [("", 150), ("_paper", 300)]:
    fig.savefig(SUCCESS_BASE / f"robosuite_main{suf}.png", dpi=dpi, bbox_inches="tight")
print(f"[A] Saved 3-panel figure  → {SUCCESS_BASE}/robosuite_main.png")
plt.close()


# =========================================================
# Figure 2: 2-panel talk figure
# =========================================================

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5.2))

ax = axes2[0]
bar_colors = [RED if sr < 100 else GREEN for sr in sr_vals]
bars = ax.bar(range(len(M_vals)), sr_vals, color=bar_colors, alpha=0.85,
              edgecolor="white", linewidth=1.5, width=0.62)
ax.axhline(sr_cont, color=GRAY, ls="--", lw=2, label="Continuous baseline")
ax.set_xticks(range(len(M_vals)))
ax.set_xticklabels([f"M={m}\n$N_{{tot}}$=$10^{{{np.log10(m ** action_dim):.1f}}}$" for m in M_vals], fontsize=8.5)
ax.set_ylabel("Task Success Rate (%)")
ax.set_title("PickPlaceCan  —  Success vs Action Resolution")
ax.set_ylim(0, 115)
ax.legend()
for bar, sr in zip(bars, sr_vals):
    ax.text(bar.get_x() + bar.get_width() / 2, sr + 2, f"{sr:.0f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
            color=GREEN if sr >= 100 else RED)

ax = axes2[1]
ax.plot(N_lat, d_ms, "^-", color=RED, lw=2.5, ms=9, label="Dense Head $\\mathcal{O}(N)$")
ax.plot(N_lat, g_ms, "s-", color=GREEN, lw=2.5, ms=9, label="GVLA Head $\\mathcal{O}(\\log N)$")
ax.axhline(10, color="#fdae61", ls="--", lw=1.8, alpha=0.9, label="Real-time budget (100 Hz)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Total action space  $N$")
ax.set_ylabel("Head Latency (ms)")
ax.set_title("Action Head Latency Scaling")
ax.legend()

plt.tight_layout(pad=1.6)
for suf, dpi in [("", 150), ("_paper", 300)]:
    fig2.savefig(SUCCESS_BASE / f"robosuite_talk{suf}.png", dpi=dpi, bbox_inches="tight")
print(f"[B] Saved 2-panel talk figure → {SUCCESS_BASE}/robosuite_talk.png")
plt.close()


print("\n" + "=" * 88)
print(f"{'M':>5} | {'N_total':>24} | {'SR%':>6} | {'Dense Mem':>12} | {'GVLA Mem':>10} | {'Mem Ratio':>10}")
print("-" * 88)
print(f"{'∞':>5} | {'∞':>24} | {sr_cont:>6.1f} | {'N/A':>12} | {'N/A':>10} | {'N/A':>10}")
for m, N, sr, dm, gm in zip(M_vals, N_total_vals, sr_vals, dense_mem_MB, gvla_mem_MB):
    feasible = "✓" if dm < 40 * 1024 else "✗ OOM"
    print(f"{m:>5} | {N:>24,} | {sr:>6.1f} | {dm:>9.1f} MB {feasible:>3} | {gm:>7.4f} MB | {dm / gm:>9.0f}×")

N_ref = 96 ** action_dim
k_ref = int(np.ceil(np.log2(N_ref)))
print(f"\nLatency @ N=4M: Dense={d_ms[-1]:.1f}ms  GVLA={g_ms[-1]:.3f}ms  speedup={d_ms[-1] / g_ms[-1]:.0f}×")
print(f"M=96 requires N={N_ref:,.0f} entries in the full 7-D grid equivalent")
print(f"Dense memory @ M=96: {N_ref * 256 * 4 / 1e12:.1f} TB")
print(f"GVLA @ M=96: k={k_ref} bits → {k_ref * 256 * 4} bytes")
