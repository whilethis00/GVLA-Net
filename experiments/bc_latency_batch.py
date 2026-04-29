"""
bc_latency_batch.py
===================
Dense vs GVLA head latency 측정 — batch size × M 전체 sweep.

batch=1에서는 kernel launch overhead가 지배해서 Dense가 더 빠르게 나온다.
batch가 커지면 실제 연산량(FLOPs)이 지배하기 시작하고,
Dense O(M) vs GVLA O(log M)의 이론적 차이가 드러난다.

Usage:
    python experiments/bc_latency_batch.py
    python experiments/bc_latency_batch.py --out experiments/results/bc_study/latency_batch.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.bc_train import DenseHead, GVLAHead


def measure(head, z, n_warmup=200, n_trials=2000, device=None):
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)
    with torch.no_grad():
        for _ in range(n_warmup):
            head.decode(z)
        sync()
        t0 = time.perf_counter()
        for _ in range(n_trials):
            head.decode(z)
        sync()
    return (time.perf_counter() - t0) / n_trials * 1e3   # ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str,
                        default="experiments/results/bc_study/latency_batch.json")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--action_dim", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    M_list     = [8, 32, 128, 512, 1024, 4096, 16384, 65536]
    batch_list = [1, 8, 32, 128, 512, 1024]

    results = {}   # results[head][batch][M] = ms

    for head_type in ("dense", "gvla"):
        results[head_type] = {}
        for batch in batch_list:
            results[head_type][batch] = {}
            print(f"[{head_type}]  batch={batch}")
            for M in M_list:
                k = math.ceil(math.log2(max(M, 2)))
                z = torch.randn(batch, args.latent_dim, device=device)
                if head_type == "dense":
                    head = DenseHead(args.latent_dim, args.action_dim, M).to(device).eval()
                else:
                    head = GVLAHead(args.latent_dim, args.action_dim, M).to(device).eval()
                ms = measure(head, z, device=device)
                results[head_type][batch][M] = ms
                print(f"  M={M:6d}  k={k:3d}  {ms:.4f}ms")
            print()

    # ── save ──────────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # convert int keys to str for JSON
    json_results = {
        ht: {str(b): {str(M): v for M, v in mv.items()}
             for b, mv in bv.items()}
        for ht, bv in results.items()
    }
    with open(out_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"Saved → {out_path}")

    # ── summary table ─────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print(f"{'':>12}", end="")
    for M in M_list:
        print(f"  M={M:>5}", end="")
    print()

    for head_type in ("dense", "gvla"):
        for batch in batch_list:
            label = f"{head_type} B={batch:4d}"
            print(f"{label:>14}", end="")
            for M in M_list:
                ms = results[head_type][batch][M]
                print(f"  {ms:6.2f}ms", end="")
            print()
        print()

    # ── plot ──────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=False)
        axes = axes.flatten()

        colors_dense = plt.cm.Reds(np.linspace(0.4, 0.9, len(batch_list)))
        colors_gvla  = plt.cm.Greens(np.linspace(0.4, 0.9, len(batch_list)))

        for i, batch in enumerate(batch_list):
            ax = axes[i]
            d_vals = [results["dense"][batch][M] for M in M_list]
            g_vals = [results["gvla"][batch][M]  for M in M_list]
            ax.plot(M_list, d_vals, "^-", color=colors_dense[i], lw=2, ms=7,
                    label=f"Dense")
            ax.plot(M_list, g_vals, "s-", color=colors_gvla[i],  lw=2, ms=7,
                    label=f"GVLA")
            ax.set_xscale("log", base=2)
            ax.set_title(f"batch = {batch}", fontsize=12)
            ax.set_xlabel("M (bins/dim)", fontsize=10)
            ax.set_ylabel("Latency (ms)", fontsize=10)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Dense vs GVLA Head Latency: batch size × M sweep\n"
                     "(Dense = O(M), GVLA = O(log M) — gap widens at large batch)",
                     fontsize=13)
        fig.tight_layout()

        plot_path = out_path.parent / "figures" / "latency_batch_sweep.png"
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        fig.savefig(str(plot_path).replace(".png", "_paper.png"), dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot → {plot_path}")
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
