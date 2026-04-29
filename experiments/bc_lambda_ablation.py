"""
bc_lambda_ablation.py
=====================
Orthogonality regularization strength (lambda) ablation for GVLA head.

Total loss:
    L = L_bit + lambda * L_ortho
    L_bit  = sum_{d=1}^{7} sum_{j=1}^{k} BCE(p_{d,j}, c_j(b_d))
    L_ortho = sum_{d=1}^{7} ||W_d W_d^T - I||_F^2

lambda=0    : no orthogonality constraint (W can collapse)
lambda=0.01 : current default
lambda in {0, 0.001, 0.01, 0.1, 1.0} sweep

Fixed setting: head=gvla, gray_code=True, M=256 (where gray shows clear signal)

Usage:
    python experiments/bc_lambda_ablation.py
    python experiments/bc_lambda_ablation.py --fast
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.bc_train import (
    LiftDemoDataset, Backbone, GVLAHead,
    quantize_action, action_to_binary, binary_to_action,
)

CKPT_DIR = Path("experiments/results/bc_study/lambda_ablation/checkpoints")
EVAL_OUT = Path("experiments/results/bc_study/lambda_ablation/results.json")
PLOT_DIR = Path("experiments/results/bc_study/lambda_ablation/figures")

LAMBDAS = [0.0, 0.001, 0.01, 0.1, 1.0, 5.0, 10.0]
M = 256
DATA = "data/robomimic/lift/ph/low_dim_v141.hdf5"


# ── model ────────────────────────────────────────────────────────────────────

class BCPolicyLambda(nn.Module):
    """GVLA policy with configurable lambda for orthogonality loss."""

    def __init__(self, obs_dim, action_dim, n_bins, latent_dim=256,
                 gray_code=True, ortho_lambda=0.01):
        super().__init__()
        self.n_bins = n_bins
        self.action_dim = action_dim
        self.gray_code = gray_code
        self.ortho_lambda = ortho_lambda
        self.backbone = Backbone(obs_dim, latent_dim)
        self.head = GVLAHead(latent_dim, action_dim, n_bins, gray_code=gray_code)

    def forward(self, obs):
        return self.head(self.backbone(obs))

    def predict(self, obs):
        z = self.backbone(obs)
        return self.head.decode(z)

    def loss(self, obs, action):
        k = self.head.k
        bin_targets = quantize_action(action, self.n_bins)
        bin_codes = action_to_binary(bin_targets, k, gray_code=self.gray_code)
        out_list = self.forward(obs)
        bce = nn.BCEWithLogitsLoss()
        l_bit = sum(
            bce(out["projections"], bin_codes[:, d, :])
            for d, out in enumerate(out_list)
        )
        l_ortho = self.head.orthogonality_loss()
        return l_bit + self.ortho_lambda * l_ortho, l_bit.item(), l_ortho.item()


# ── train ─────────────────────────────────────────────────────────────────────

def train_one(lam, dataset, obs_dim, action_dim, epochs, batch_size, lr, device):
    exp_name = f"lambda_{str(lam).replace('.', 'p')}"
    out_dir = CKPT_DIR / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    policy = BCPolicyLambda(obs_dim, action_dim, M, ortho_lambda=lam).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        policy.train()
        epoch_loss = 0.0
        for obs_b, act_b in loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            total, _, _ = policy.loss(obs_b, act_b)
            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            epoch_loss += total.item() * len(obs_b)
        scheduler.step()
        epoch_loss /= len(dataset)
        if epoch % 20 == 0 or epoch == 1:
            print(f"  epoch {epoch:4d}/{epochs}  loss={epoch_loss:.4f}  "
                  f"elapsed={time.time()-t0:.0f}s")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(policy.state_dict(), out_dir / "best.pt")

    cfg = {"obs_dim": obs_dim, "action_dim": action_dim, "n_bins": M,
           "gray_code": True, "ortho_lambda": lam, "best_loss": best_loss,
           "exp_name": exp_name}
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"  Saved → {out_dir}  (best_loss={best_loss:.4f})\n")
    return out_dir, best_loss


# ── eval ─────────────────────────────────────────────────────────────────────

def eval_one(out_dir, n_rollouts, max_steps, device):
    cfg_path = out_dir / "config.json"
    ckpt_path = out_dir / "best.pt"
    with open(cfg_path) as f:
        cfg = json.load(f)

    policy = BCPolicyLambda(
        obs_dim=cfg["obs_dim"], action_dim=cfg["action_dim"],
        n_bins=cfg["n_bins"], gray_code=cfg["gray_code"],
        ortho_lambda=cfg["ortho_lambda"],
    ).to(device)
    policy.load_state_dict(torch.load(ckpt_path, map_location=device))
    policy.eval()

    import robosuite as suite
    env = suite.make(
        "Lift", robots="Panda",
        has_renderer=False, has_offscreen_renderer=False,
        use_camera_obs=False, ignore_done=True,
        horizon=max_steps, reward_shaping=False,
    )

    from experiments.bc_eval import obs_to_tensor, run_rollout
    successes = 0
    for seed in range(n_rollouts):
        ok, _ = run_rollout(policy, env, max_steps, seed, device)
        if ok:
            successes += 1
        print(f"\r  rollout {seed+1}/{n_rollouts}  success={successes}", end="", flush=True)
    print()
    env.close()

    sr = successes / n_rollouts
    return {"lambda": cfg["ortho_lambda"], "success_rate": sr,
            "successes": successes, "n_rollouts": n_rollouts,
            "best_loss": cfg["best_loss"]}


# ── plot ──────────────────────────────────────────────────────────────────────

def plot_results(results):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        lambdas = [r["lambda"] for r in results]
        success_rates = [r["success_rate"] * 100 for r in results]
        best_losses = [r["best_loss"] for r in results]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.plot(range(len(lambdas)), success_rates, "bo-", lw=2, ms=9)
        ax1.set_xticks(range(len(lambdas)))
        ax1.set_xticklabels([str(l) for l in lambdas])
        ax1.set_xlabel("Orthogonality lambda", fontsize=12)
        ax1.set_ylabel("Success Rate (%)", fontsize=12)
        ax1.set_title("Lambda Ablation: Success Rate\n(GVLA gray, M=256)", fontsize=12)
        ax1.axvline(x=lambdas.index(0.01), color="r", linestyle="--",
                    alpha=0.5, label="default λ=0.01")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        ax2.plot(range(len(lambdas)), best_losses, "rs-", lw=2, ms=9)
        ax2.set_xticks(range(len(lambdas)))
        ax2.set_xticklabels([str(l) for l in lambdas])
        ax2.set_xlabel("Orthogonality lambda", fontsize=12)
        ax2.set_ylabel("Best Training Loss", fontsize=12)
        ax2.set_title("Lambda Ablation: Training Loss\n(GVLA gray, M=256)", fontsize=12)
        ax2.axvline(x=lambdas.index(0.01), color="r", linestyle="--",
                    alpha=0.5, label="default λ=0.01")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(PLOT_DIR / "lambda_ablation.png", dpi=150, bbox_inches="tight")
        fig.savefig(PLOT_DIR / "lambda_ablation_paper.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Plot → {PLOT_DIR}/lambda_ablation.png")
    except Exception as e:
        print(f"Plot skipped: {e}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Quick test: epochs=50, n_rollouts=20")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--n_rollouts", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=500)
    args = parser.parse_args()

    if args.fast:
        args.epochs = 50
        args.n_rollouts = 20
        print("[fast mode]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Lambda sweep: {LAMBDAS}")
    print(f"M={M}, gray_code=True, epochs={args.epochs}, n_rollouts={args.n_rollouts}\n")

    cd_root = ROOT / "GVLA-Net" if (ROOT / "GVLA-Net").exists() else ROOT
    os.chdir(ROOT)

    dataset = LiftDemoDataset(DATA)
    obs_dim = dataset.obs.shape[1]
    action_dim = dataset.act.shape[1]

    # ── 기존 결과 로드 (있으면 재사용) ──────────────────────────
    existing = {}
    if EVAL_OUT.exists():
        with open(EVAL_OUT) as f:
            for r in json.load(f):
                existing[r["lambda"]] = r
        print(f"Loaded {len(existing)} existing results from {EVAL_OUT}")

    # ── training sweep (없는 것만) ────────────────────────────
    print("=" * 50)
    print(" Training sweep")
    print("=" * 50)
    trained_dirs = []
    for lam in LAMBDAS:
        exp_name = f"lambda_{str(lam).replace('.', 'p')}"
        out_dir = CKPT_DIR / exp_name
        if (out_dir / "best.pt").exists() and (out_dir / "config.json").exists():
            print(f"\n[SKIP] lambda={lam} — checkpoint exists")
            trained_dirs.append(out_dir)
        else:
            print(f"\n[TRAIN] lambda={lam}")
            out_dir, _ = train_one(lam, dataset, obs_dim, action_dim,
                                   args.epochs, args.batch_size, args.lr, device)
            trained_dirs.append(out_dir)

    # ── eval sweep (없는 것만) ────────────────────────────────
    print("=" * 50)
    print(" Evaluation sweep")
    print("=" * 50)
    all_results = []
    for lam, out_dir in zip(LAMBDAS, trained_dirs):
        if lam in existing:
            print(f"\n[SKIP] lambda={lam} — result exists")
            all_results.append(existing[lam])
        else:
            print(f"\n[EVAL] lambda={lam}")
            r = eval_one(out_dir, args.n_rollouts, args.max_steps, device)
            all_results.append(r)
            print(f"  success_rate={r['success_rate']*100:.1f}%")

    # λ 순서로 정렬
    all_results.sort(key=lambda r: r["lambda"])

    # ── save ─────────────────────────────────────────────────
    EVAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_OUT, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults → {EVAL_OUT}")

    # ── summary ──────────────────────────────────────────────
    print("\n" + "=" * 45)
    print(f"{'lambda':>10}  {'success':>10}  {'best_loss':>10}")
    print("-" * 45)
    for r in all_results:
        print(f"{r['lambda']:>10}  {r['success_rate']*100:>9.1f}%  {r['best_loss']:>10.4f}")

    plot_results(all_results)


if __name__ == "__main__":
    main()
