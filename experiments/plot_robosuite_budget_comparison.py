"""
plot_robosuite_budget_comparison.py
===================================
Budget-constrained comparison for the Robosuite PickPlaceCan quantisation study.

Goal:
  Show that the action-resolution regime where manipulation success emerges is
  already outside the practical feasibility region of a dense action head under
  realistic memory / latency budgets.
"""

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = Path(__file__).parent / "results"
SUCCESS_BASE = RESULTS_DIR / "robosuite_pickplace_transition_100roll"
LAT_BASE = RESULTS_DIR / "robosuite_study"
OUT_BASE = SUCCESS_BASE


with open(SUCCESS_BASE / "results.json") as f:
    success_raw = json.load(f)

with open(LAT_BASE / "results.json") as f:
    latency_raw = json.load(f)


ACTION_DIM = int(success_raw.get("action_dim", 7))
HEAD_DIM = 256
BYTES_PER_PARAM = 4

success_rate = success_raw["success_rate"]
M_vals = sorted(int(k) for k in success_rate.keys() if k != "inf")
sr_vals = [success_rate[str(m)] * 100 for m in M_vals]
sr_cont = success_rate.get("inf", 1.0) * 100
N_total_vals = np.array([m ** ACTION_DIM for m in M_vals], dtype=np.float64)

latency = latency_raw["latency"]
N_lat_all = sorted(int(k) for k in latency.keys())
N_lat = np.array([N for N in N_lat_all if N >= 65536], dtype=np.float64)
dense_ms = np.array([latency[str(int(N))]["dense_ms"] for N in N_lat], dtype=np.float64)
gvla_ms = np.array([latency[str(int(N))]["gvla_ms"] for N in N_lat], dtype=np.float64)


def dense_memory_bytes(n_total: float) -> float:
    return n_total * HEAD_DIM * BYTES_PER_PARAM


def gvla_memory_bytes(n_total: float) -> float:
    k = int(math.ceil(math.log2(max(n_total, 2))))
    return k * HEAD_DIM * BYTES_PER_PARAM


def dense_latency_slope_ms_per_entry(n_vals: np.ndarray, ms_vals: np.ndarray) -> float:
    # Fit Dense latency with a line through the origin: ms ≈ alpha * N.
    # This matches the O(N) claim and keeps extrapolation easy to interpret.
    return float(np.dot(n_vals, ms_vals) / np.dot(n_vals, n_vals))


def max_m_under_memory_budget_bytes(budget_bytes: float) -> int:
    max_n = budget_bytes / (HEAD_DIM * BYTES_PER_PARAM)
    return int(math.floor(max_n ** (1.0 / ACTION_DIM)))


def max_m_under_latency_budget_ms(budget_ms: float, slope_ms_per_entry: float) -> int:
    max_n = budget_ms / slope_ms_per_entry
    return int(math.floor(max_n ** (1.0 / ACTION_DIM)))


dense_alpha = dense_latency_slope_ms_per_entry(N_lat, dense_ms)
gvla_mean_ms = float(np.mean(gvla_ms))

mem_budget_16 = 16 * (1024 ** 3)
mem_budget_40 = 40 * (1024 ** 3)
lat_budget_10 = 10.0

budget_summary = {
    "16GB memory": max_m_under_memory_budget_bytes(mem_budget_16),
    "40GB memory": max_m_under_memory_budget_bytes(mem_budget_40),
    "10ms latency": max_m_under_latency_budget_ms(lat_budget_10, dense_alpha),
}


BLUE = "#2166ac"
RED = "#d73027"
GREEN = "#1a9641"
GRAY = "#999999"
ORANGE = "#fdae61"

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.28,
})


def add_budget_region(ax, threshold_m: int, color: str, label: str, ymax: float = 112.0) -> None:
    x0 = max(M_vals[0] / 1.35, threshold_m)
    x1 = M_vals[-1] * 1.18
    if x0 < x1:
        ax.axvspan(x0, x1, color=color, alpha=0.06, label=label)


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

# ---------------------------------------------------------
# Panel A: Success rate + dense feasibility regions
# ---------------------------------------------------------
ax = axes[0]
ax.plot(M_vals, sr_vals, "o-", color=BLUE, lw=2.5, ms=9, zorder=5, label="PickPlaceCan success")
ax.axhline(sr_cont, color=GRAY, ls="--", lw=1.6, label=f"Continuous ({sr_cont:.0f}%)")

add_budget_region(ax, budget_summary["10ms latency"] + 0.5, ORANGE, "Dense infeasible under 10 ms")
add_budget_region(ax, budget_summary["16GB memory"] + 0.5, RED, "Dense infeasible under 16 GB")
add_budget_region(ax, budget_summary["40GB memory"] + 0.5, "#b2182b", "Dense infeasible under 40 GB")

ax.axvline(96, color=GREEN, ls=":", lw=1.5, alpha=0.85)
ax.text(96 * 1.03, 22, "M=96\n100% success", color=GREEN, fontsize=9, va="bottom")

for m, sr in zip(M_vals, sr_vals):
    ax.annotate(f"{sr:.0f}%", xy=(m, sr), xytext=(0, 5 if sr < 98 else -10),
                textcoords="offset points", ha="center", fontsize=9.5,
                color=BLUE, fontweight="bold")

ax.set_xscale("log", base=2)
ax.set_xlim(M_vals[0] * 0.82, M_vals[-1] * 1.2)
ax.set_ylim(-3, 110)
ax.set_xlabel("Action bins per dimension  $M$")
ax.set_ylabel("Task Success Rate (%)")
ax.set_title("(A) Success Emerges Beyond Dense Feasibility")
ax.legend(loc="upper left")

# ---------------------------------------------------------
# Panel B: Budget table as bar / line chart
# ---------------------------------------------------------
ax = axes[1]

dense_mem_gib = np.array([dense_memory_bytes(N) / (1024 ** 3) for N in N_total_vals], dtype=np.float64)
gvla_mem_kib = np.array([gvla_memory_bytes(N) / 1024.0 for N in N_total_vals], dtype=np.float64)
dense_lat_pred = dense_alpha * N_total_vals

ax2 = ax.twinx()
lns1 = ax.plot(M_vals, dense_lat_pred, "^-", color=ORANGE, lw=2.3, ms=8,
               label="Dense latency (extrapolated)")
lns2 = ax2.semilogy(M_vals, dense_mem_gib, "s--", color=RED, lw=2.3, ms=8,
                    label="Dense memory")
lns3 = ax2.semilogy(M_vals, gvla_mem_kib / (1024 ** 2), "o--", color=GREEN, lw=2.0, ms=7,
                    label="GVLA memory")

ax.axhline(lat_budget_10, color=ORANGE, ls=":", lw=1.5)
ax.text(M_vals[0] * 1.02, lat_budget_10 * 1.04, "10 ms budget", color=ORANGE, fontsize=9)
ax2.axhline(16, color=RED, ls=":", lw=1.5, alpha=0.8)
ax2.axhline(40, color="#b2182b", ls=":", lw=1.5, alpha=0.8)
ax2.text(M_vals[-1] * 0.75, 18, "16 / 40 GiB budgets", color=RED, fontsize=9)

ax.set_xscale("log", base=2)
ax.set_xlabel("Action bins per dimension  $M$")
ax.set_ylabel("Dense latency (ms)")
ax2.set_ylabel("Head memory (GiB, log scale)")
ax.set_title("(B) Dense Cost at the Success-Critical Resolution Regime")

all_lines = lns1 + lns2 + lns3
ax.legend(all_lines, [line.get_label() for line in all_lines], loc="upper left")

plt.tight_layout(pad=1.5)
for suf, dpi in [("", 150), ("_paper", 300)]:
    fig.savefig(OUT_BASE / f"robosuite_budget_comparison{suf}.png", dpi=dpi, bbox_inches="tight")
plt.close(fig)


print("=" * 92)
print("BUDGET-CONSTRAINED COMPARISON")
print("=" * 92)
print(f"Dense latency fit (through origin): ms ≈ {dense_alpha:.6e} * N")
print(f"Mean GVLA latency over measured N: {gvla_mean_ms:.4f} ms")
print()
print(f"{'Budget':<18} {'Dense max feasible M':>22}")
print("-" * 44)
for label, max_m in budget_summary.items():
    print(f"{label:<18} {max_m:>22}")

print()
print(f"{'M':>5} | {'Success %':>9} | {'N_total':>24} | {'Dense mem (GiB)':>16} | {'Dense lat pred (ms)':>19}")
print("-" * 88)
for m, sr, n_total, mem_gib, lat_ms in zip(M_vals, sr_vals, N_total_vals, dense_mem_gib, dense_lat_pred):
    print(f"{m:>5} | {sr:>8.1f}% | {int(n_total):>24,} | {mem_gib:>16,.1f} | {lat_ms:>19,.1f}")

print()
print("Interpretation:")
print("- PickPlaceCan reaches 100% success at M>=96.")
print(f"- Under a 10 ms Dense-head budget, the largest feasible M is about {budget_summary['10ms latency']}.")
print(f"- Under a 16 GiB Dense-head budget, the largest feasible M is about {budget_summary['16GB memory']}.")
print(f"- Under a 40 GiB Dense-head budget, the largest feasible M is about {budget_summary['40GB memory']}.")
print("- Therefore the success-critical high-resolution regime lies beyond realistic Dense-head budgets.")
