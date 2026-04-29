"""
bc_eval.py
==========
Evaluate trained BC policies (Dense vs GVLA) on Robosuite Lift rollouts.

Measures:
  1. Task success rate  — does the robot lift the cube?
  2. Head inference latency — pure head forward pass time (excluding backbone)
  3. Sweep over n_bins to show Dense O(N) vs GVLA O(log N) scaling

Usage:
    # Evaluate a single checkpoint
    python experiments/bc_eval.py \
        --ckpt experiments/results/bc_study/checkpoints/dense_64/best.pt \
        --config experiments/results/bc_study/checkpoints/dense_64/config.json \
        --n_rollouts 50

    # Full sweep (all checkpoints in a directory)
    python experiments/bc_eval.py --sweep_dir experiments/results/bc_study/checkpoints \
        --n_rollouts 50 --out experiments/results/bc_study/eval_results.json
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.bc_train import BCPolicy  # reuse model definition


# ============================================================
# Robosuite rollout
# ============================================================

def obs_to_tensor(obs_dict: dict, device: torch.device) -> torch.Tensor:
    """Concatenate robosuite obs arrays using the same key order as training.

    Live robosuite uses "object-state" while RoboMimic HDF5 uses "object";
    we map accordingly. Keys that don't exist are skipped.
    """
    # Map: (live robosuite key) -> (robomimic training key name)
    KEY_MAP = {
        "object-state":          "object",
        "robot0_eef_pos":        "robot0_eef_pos",
        "robot0_eef_quat":       "robot0_eef_quat",
        "robot0_gripper_qpos":   "robot0_gripper_qpos",
        "robot0_gripper_qvel":   "robot0_gripper_qvel",
        "robot0_joint_pos":      "robot0_joint_pos",
        "robot0_joint_pos_cos":  "robot0_joint_pos_cos",
        "robot0_joint_pos_sin":  "robot0_joint_pos_sin",
        "robot0_joint_vel":      "robot0_joint_vel",
    }
    parts = []
    for live_key in KEY_MAP:
        if live_key in obs_dict:
            arr = np.array(obs_dict[live_key], dtype=np.float32).flatten()
            parts.append(arr)
    vec = np.concatenate(parts, axis=0)
    return torch.from_numpy(vec).unsqueeze(0).to(device)


def run_rollout(policy: BCPolicy, env, max_steps: int,
                seed: int, device: torch.device) -> tuple[bool, float]:
    """Single rollout.  Returns (success, total_reward)."""
    np.random.seed(seed)
    obs = env.reset()
    policy.eval()
    reward_sum = 0.0
    with torch.no_grad():
        for _ in range(max_steps):
            obs_t = obs_to_tensor(obs, device)
            action = policy.predict(obs_t).squeeze(0).cpu().numpy()
            action = np.clip(action, -1.0, 1.0)
            obs, reward, done, _ = env.step(action)
            reward_sum += reward
            if done or reward_sum > 0:
                return True, reward_sum
    return reward_sum > 0, reward_sum


def eval_success_rate(policy: BCPolicy, n_rollouts: int,
                      max_steps: int, device: torch.device) -> dict:
    import robosuite as suite
    env = suite.make(
        "Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, ignore_done=True,
        horizon=max_steps, reward_shaping=False,
    )
    successes = 0
    for seed in range(n_rollouts):
        ok, _ = run_rollout(policy, env, max_steps, seed, device)
        if ok:
            successes += 1
        print(f"\r  rollout {seed+1}/{n_rollouts}  success={successes}", end="", flush=True)
    print()
    env.close()
    sr = successes / n_rollouts
    return {"success_rate": sr, "n_rollouts": n_rollouts, "successes": successes}


# ============================================================
# Head latency measurement
# ============================================================

def measure_head_latency(head_type: str, n_bins_list: list,
                         latent_dim: int = 256, action_dim: int = 7,
                         n_warmup: int = 100, n_trials: int = 1000,
                         device: torch.device = torch.device("cpu")) -> dict:
    """
    Measure pure head forward-pass latency (backbone excluded).
    Returns dict[n_bins] = {head_ms, k_bits, flops_ratio}
    """
    from experiments.bc_train import DenseHead, GVLAHead
    results = {}
    sync = (lambda: torch.cuda.synchronize()) if device.type == "cuda" else (lambda: None)

    for n_bins in n_bins_list:
        k = math.ceil(math.log2(max(n_bins, 2)))
        z = torch.randn(1, latent_dim, device=device)

        if head_type == "dense":
            head = DenseHead(latent_dim, action_dim, n_bins).to(device).eval()
        else:
            head = GVLAHead(latent_dim, action_dim, n_bins).to(device).eval()

        with torch.no_grad():
            for _ in range(n_warmup):
                head.decode(z)
            sync()
            t0 = time.perf_counter()
            for _ in range(n_trials):
                head.decode(z)
            sync()
        ms = (time.perf_counter() - t0) / n_trials * 1e3

        results[n_bins] = {"head_ms": ms, "k_bits": k}
        print(f"  [{head_type}] n_bins={n_bins:6d}  k={k:3d}  latency={ms:.4f}ms")

    return results


# ============================================================
# Sweep evaluation
# ============================================================

def eval_checkpoint(ckpt_path: Path, config_path: Path,
                    n_rollouts: int, max_steps: int,
                    device: torch.device) -> dict:
    """Load a checkpoint, evaluate success rate, return result dict."""
    with open(config_path) as f:
        cfg = json.load(f)

    policy = BCPolicy(
        obs_dim=cfg["obs_dim"],
        action_dim=cfg["action_dim"],
        head_type=cfg["head"],
        n_bins=cfg["n_bins"],
        latent_dim=cfg.get("latent_dim", 256),
        gray_code=cfg.get("gray_code", False),
    ).to(device)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.eval()

    print(f"\n[{cfg['exp_name']}]  head={cfg['head']}  n_bins={cfg['n_bins']}")
    result = eval_success_rate(policy, n_rollouts, max_steps, device)
    result.update({
        "exp_name": cfg["exp_name"],
        "head": cfg["head"],
        "n_bins": cfg["n_bins"],
        "best_loss": cfg.get("best_loss"),
    })
    return result


# ============================================================
# Plotting
# ============================================================

def plot_results(results: list, latency_dense: dict,
                 latency_gvla: dict, save_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── organise results by head & n_bins ─────────────────
    dense_sr, gvla_sr = {}, {}
    for r in results:
        if r["head"] == "dense":
            dense_sr[r["n_bins"]] = r["success_rate"] * 100
        else:
            gvla_sr[r["n_bins"]] = r["success_rate"] * 100

    M_dense = sorted(dense_sr.keys())
    M_gvla  = sorted(gvla_sr.keys())
    M_all   = sorted(set(M_dense) | set(M_gvla))

    # latency lists
    N_lat   = sorted(latency_dense.keys())
    d_lat   = [latency_dense[n]["head_ms"] for n in N_lat]
    g_lat   = [latency_gvla[n]["head_ms"]  for n in N_lat]

    # ── Figure 1: success rate ────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    if M_dense:
        ax.plot(M_dense, [dense_sr[m] for m in M_dense],
                "r^-", lw=2.2, ms=9, label="Dense head  O(M)")
    if M_gvla:
        ax.plot(M_gvla, [gvla_sr[m] for m in M_gvla],
                "bs-", lw=2.2, ms=9, label="GVLA head   O(log M)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Bins per action dim (M)", fontsize=13)
    ax.set_ylabel("Task Success Rate (%)", fontsize=13)
    ax.set_title("Lift: BC Success Rate vs Action Resolution", fontsize=13)
    ax.set_ylim(-5, 110)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_dir / "bc_success_rate.png", dpi=150, bbox_inches="tight")
    fig.savefig(save_dir / "bc_success_rate_paper.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Figure 2: latency ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(N_lat, d_lat, "r^-", lw=2.2, ms=9, label="Dense head  O(M)")
    ax.plot(N_lat, g_lat, "bs-", lw=2.2, ms=9, label="GVLA head   O(log M)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Bins per action dim (M)", fontsize=13)
    ax.set_ylabel("Head latency (ms, log scale)", fontsize=13)
    ax.set_title("Action Head Inference Latency vs Discretization", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    # annotate speedup at largest M
    if N_lat and d_lat and g_lat:
        n_max = N_lat[-1]
        speedup = d_lat[-1] / g_lat[-1] if g_lat[-1] > 0 else 0
        ax.annotate(f"{speedup:.0f}× faster",
                    xy=(n_max, g_lat[-1]),
                    xytext=(-80, -25), textcoords="offset points",
                    fontsize=10, color="#1a9641",
                    arrowprops=dict(arrowstyle="->", color="#1a9641"))
    fig.tight_layout()
    fig.savefig(save_dir / "bc_latency.png", dpi=150, bbox_inches="tight")
    fig.savefig(save_dir / "bc_latency_paper.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ── Figure 3: combined (success + latency dual-axis) ──
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    c_dense, c_gvla = "#d73027", "#1a9641"
    c_sr = "#2166ac"

    if M_dense:
        ax1.plot(M_dense, [dense_sr[m] for m in M_dense],
                 "^-", color=c_dense, lw=2.5, ms=9,
                 label="Dense SR")
    if M_gvla:
        ax1.plot(M_gvla, [gvla_sr[m] for m in M_gvla],
                 "s-", color=c_gvla, lw=2.5, ms=9,
                 label="GVLA SR")

    ax2.plot(N_lat, d_lat, "^--", color=c_dense, lw=1.8, ms=7, alpha=0.7,
             label="Dense latency")
    ax2.plot(N_lat, g_lat, "s--", color=c_gvla,  lw=1.8, ms=7, alpha=0.7,
             label="GVLA latency")

    ax1.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax1.set_xlabel("Bins per action dim (M)", fontsize=13)
    ax1.set_ylabel("Task Success Rate (%)", fontsize=13)
    ax2.set_ylabel("Head Latency (ms, log)", fontsize=13)
    ax1.set_ylim(-5, 110)
    ax1.set_title("GVLA vs Dense: Precision–Efficiency Trade-off\n"
                  "(Lift, Panda, RoboMimic PH demos)", fontsize=13)
    ax1.grid(True, alpha=0.3)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_dir / "bc_combined.png", dpi=150, bbox_inches="tight")
    fig.savefig(save_dir / "bc_combined_paper.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[plot] saved to {save_dir}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BC Evaluation: Dense vs GVLA")
    # single checkpoint mode
    parser.add_argument("--ckpt",   type=str, default=None)
    parser.add_argument("--config", type=str, default=None)
    # sweep mode
    parser.add_argument("--sweep_dir", type=str, default=None,
                        help="Directory containing exp subdirs with best.pt+config.json")
    # common
    parser.add_argument("--n_rollouts", type=int, default=50)
    parser.add_argument("--max_steps",  type=int, default=500)
    parser.add_argument("--n_latency_trials", type=int, default=2000)
    parser.add_argument("--out", type=str,
                        default="experiments/results/bc_study/eval_results.json")
    parser.add_argument("--plot_dir", type=str,
                        default="experiments/results/bc_study/figures")
    parser.add_argument("--skip_rollout",  action="store_true")
    parser.add_argument("--skip_latency",  action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_results = []

    # ── rollout evaluation ─────────────────────────────────
    if not args.skip_rollout:
        if args.sweep_dir:
            sweep_dir = Path(args.sweep_dir)
            for exp_dir in sorted(sweep_dir.iterdir()):
                ckpt   = exp_dir / "best.pt"
                config = exp_dir / "config.json"
                if ckpt.exists() and config.exists():
                    r = eval_checkpoint(ckpt, config, args.n_rollouts,
                                        args.max_steps, device)
                    all_results.append(r)
                    print(f"  → success_rate={r['success_rate']*100:.1f}%")
        elif args.ckpt and args.config:
            r = eval_checkpoint(Path(args.ckpt), Path(args.config),
                                args.n_rollouts, args.max_steps, device)
            all_results.append(r)

    # ── latency sweep ──────────────────────────────────────
    lat_dense, lat_gvla = {}, {}
    if not args.skip_latency:
        M_sweep = [8, 16, 32, 64, 128, 256, 512, 1024, 4096, 16384, 65536]
        print("\n[Latency sweep — Dense head]")
        lat_dense = measure_head_latency("dense", M_sweep,
                                         n_trials=args.n_latency_trials,
                                         device=device)
        print("\n[Latency sweep — GVLA head]")
        lat_gvla = measure_head_latency("gvla", M_sweep,
                                        n_trials=args.n_latency_trials,
                                        device=device)

    # ── save JSON ──────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "rollout_results": all_results,
            "latency_dense":   {str(k): v for k, v in lat_dense.items()},
            "latency_gvla":    {str(k): v for k, v in lat_gvla.items()},
        }, f, indent=2)
    print(f"\nResults saved → {out_path}")

    # ── print summary table ────────────────────────────────
    if all_results:
        print("\n" + "=" * 55)
        print(f"{'exp_name':<18} {'head':<6} {'M':>6} {'success':>9}")
        print("-" * 55)
        for r in sorted(all_results, key=lambda x: (x["head"], x["n_bins"])):
            print(f"{r['exp_name']:<18} {r['head']:<6} {r['n_bins']:>6} "
                  f"{r['success_rate']*100:>8.1f}%")

    if lat_dense and lat_gvla:
        print("\n" + "=" * 60)
        print(f"{'M':>6}  {'Dense(ms)':>10}  {'GVLA(ms)':>10}  {'speedup':>8}")
        print("-" * 60)
        for n in sorted(lat_dense.keys()):
            if n in lat_gvla:
                d = lat_dense[n]["head_ms"]
                g = lat_gvla[n]["head_ms"]
                print(f"{n:>6}  {d:>10.4f}  {g:>10.4f}  {d/g:>7.1f}×")

    # ── plot ───────────────────────────────────────────────
    if all_results or (lat_dense and lat_gvla):
        plot_results(all_results, lat_dense, lat_gvla, Path(args.plot_dir))


if __name__ == "__main__":
    main()
