"""
bc_train.py
===========
Train BC policies with Dense head vs GVLA head on RoboMimic Lift (PH) demos.

Both heads share the same MLP backbone (obs → latent).
Only the action head differs:
  Dense head : 7 × Linear(latent_dim, n_bins)  →  CrossEntropy   O(n_bins) per dim
  GVLA head  : 7 × OrthogonalProjectionLayer   →  binary BCE     O(log n_bins) per dim

Usage:
    python experiments/bc_train.py --head dense --n_bins 64  --exp_name dense_64
    python experiments/bc_train.py --head gvla  --n_bins 64  --exp_name gvla_64

    # Sweep script:
    for HEAD in dense gvla; do
      for M in 8 16 32 64 128 256 512 1024; do
        python experiments/bc_train.py --head $HEAD --n_bins $M --exp_name ${HEAD}_${M}
      done
    done
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
from torch.utils.data import DataLoader, Dataset

# ── project root on path ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.layers import OrthogonalProjectionLayer

# ============================================================
# Dataset
# ============================================================

# Keys available in both RoboMimic HDF5 and live robosuite env.
# Excludes robot0_eef_vel_lin / robot0_eef_vel_ang which are in RoboMimic
# but NOT exposed by default in the robosuite gym interface.
ROBOSUITE_OBS_KEYS = [
    "object",            # 10-dim  (== "object-state" in live env)
    "robot0_eef_pos",    #  3-dim
    "robot0_eef_quat",   #  4-dim
    "robot0_gripper_qpos",  # 2-dim
    "robot0_gripper_qvel",  # 2-dim
    "robot0_joint_pos",     # 7-dim
    "robot0_joint_pos_cos", # 7-dim
    "robot0_joint_pos_sin", # 7-dim
    "robot0_joint_vel",     # 7-dim
]  # total = 49 dims


class LiftDemoDataset(Dataset):
    """Load low-dim obs + actions from a RoboMimic / custom HDF5 file."""

    def __init__(self, hdf5_path: str):
        self.hdf5_path = hdf5_path
        self.obs_list = []
        self.act_list = []

        with h5py.File(hdf5_path, "r") as f:
            demo_keys = sorted(f["data"].keys())
            for dk in demo_keys:
                ep = f["data"][dk]
                if "obs" in ep:
                    # RoboMimic format: use only the aligned key subset
                    obs_parts = []
                    for k in ROBOSUITE_OBS_KEYS:
                        if k in ep["obs"]:
                            obs_parts.append(ep["obs"][k][:])
                    obs = np.concatenate(obs_parts, axis=-1).astype(np.float32)
                else:
                    obs = ep["observations"][:].astype(np.float32)

                actions = ep["actions"][:].astype(np.float32)
                # clip actions to [-1, 1]
                actions = np.clip(actions, -1.0, 1.0)

                self.obs_list.append(obs)
                self.act_list.append(actions)

        self.obs = np.concatenate(self.obs_list, axis=0)
        self.act = np.concatenate(self.act_list, axis=0)
        print(f"[Dataset] {len(demo_keys)} episodes, "
              f"{len(self.obs)} steps, obs_dim={self.obs.shape[1]}, "
              f"act_dim={self.act.shape[1]}")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.obs[idx]),
            torch.from_numpy(self.act[idx]),
        )


# ============================================================
# Action quantization helpers
# ============================================================

def quantize_action(action: torch.Tensor, n_bins: int) -> torch.Tensor:
    """Continuous [-1,1] → bin index per dim.  shape: (..., action_dim)."""
    scaled = (action + 1.0) * 0.5 * n_bins       # [0, n_bins]
    idx = scaled.long().clamp(0, n_bins - 1)
    return idx


def action_to_binary(bin_idx: torch.Tensor, k: int) -> torch.Tensor:
    """Bin index → k-bit binary code (MSB first).  shape: (..., action_dim, k)."""
    # bin_idx: (..., action_dim)
    bits = []
    for i in range(k - 1, -1, -1):
        bits.append((bin_idx >> i) & 1)
    # stack along last dim → (..., action_dim, k)
    return torch.stack(bits, dim=-1).float()


def binary_to_action(binary: torch.Tensor, n_bins: int) -> torch.Tensor:
    """k-bit binary code → continuous action.  binary: (..., action_dim, k)."""
    k = binary.shape[-1]
    powers = torch.tensor(
        [2 ** (k - 1 - i) for i in range(k)],
        dtype=torch.long, device=binary.device,
    )
    # round to 0/1 first
    hard = (binary > 0.5).long()
    bin_idx = (hard * powers).sum(-1).clamp(0, n_bins - 1)   # (..., action_dim)
    return (bin_idx.float() + 0.5) / n_bins * 2.0 - 1.0


# ============================================================
# Models
# ============================================================

class Backbone(nn.Module):
    """Shared MLP: obs_dim → latent_dim."""

    def __init__(self, obs_dim: int, latent_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 512), nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, 512),     nn.LayerNorm(512), nn.ReLU(),
            nn.Linear(512, latent_dim), nn.LayerNorm(latent_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class DenseHead(nn.Module):
    """7 independent softmax heads, one per action dimension.  O(n_bins) per dim."""

    def __init__(self, latent_dim: int, action_dim: int, n_bins: int):
        super().__init__()
        self.n_bins = n_bins
        self.action_dim = action_dim
        self.heads = nn.ModuleList([
            nn.Linear(latent_dim, n_bins) for _ in range(action_dim)
        ])

    def forward(self, z: torch.Tensor):
        """Returns list of (B, n_bins) logits, one per action dim."""
        return [h(z) for h in self.heads]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z → continuous action (B, action_dim)."""
        logits_list = self.forward(z)
        parts = []
        for logits in logits_list:
            idx = logits.argmax(-1).float()             # (B,)
            cont = (idx + 0.5) / self.n_bins * 2.0 - 1.0
            parts.append(cont)
        return torch.stack(parts, dim=-1)               # (B, action_dim)


class GVLAHead(nn.Module):
    """7 OrthogonalProjectionLayers, one per action dimension.  O(log n_bins) per dim."""

    def __init__(self, latent_dim: int, action_dim: int, n_bins: int):
        super().__init__()
        self.n_bins = n_bins
        self.action_dim = action_dim
        self.k = math.ceil(math.log2(max(n_bins, 2)))
        self.heads = nn.ModuleList([
            OrthogonalProjectionLayer(latent_dim, n_bins)
            for _ in range(action_dim)
        ])

    def forward(self, z: torch.Tensor):
        """Returns list of dicts with 'binary_code', 'projections', one per dim."""
        return [h(z) for h in self.heads]

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z → continuous action (B, action_dim)."""
        out_list = self.forward(z)
        parts = []
        for out in out_list:
            cont = binary_to_action(out["binary_code"], self.n_bins)   # (B,)
            parts.append(cont)
        return torch.stack(parts, dim=-1)                              # (B, action_dim)

    def orthogonality_loss(self) -> torch.Tensor:
        return sum(h.orthogonality_loss() for h in self.heads)


class BCPolicy(nn.Module):
    """Full BC policy = backbone + head."""

    def __init__(self, obs_dim: int, action_dim: int,
                 head_type: str, n_bins: int, latent_dim: int = 256):
        super().__init__()
        self.head_type = head_type
        self.n_bins = n_bins
        self.action_dim = action_dim
        self.backbone = Backbone(obs_dim, latent_dim)
        if head_type == "dense":
            self.head = DenseHead(latent_dim, action_dim, n_bins)
        elif head_type == "gvla":
            self.head = GVLAHead(latent_dim, action_dim, n_bins)
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, obs: torch.Tensor):
        z = self.backbone(obs)
        return self.head(z)

    def predict(self, obs: torch.Tensor) -> torch.Tensor:
        """obs → continuous action (B, action_dim)."""
        z = self.backbone(obs)
        return self.head.decode(z)


# ============================================================
# Loss
# ============================================================

def compute_loss(policy: BCPolicy, obs: torch.Tensor,
                 action: torch.Tensor) -> torch.Tensor:
    n_bins = policy.n_bins
    action_dim = policy.action_dim

    if policy.head_type == "dense":
        # Target: bin index per dim
        bin_targets = quantize_action(action, n_bins)   # (B, action_dim)
        logits_list = policy(obs)                       # list of (B, n_bins)
        ce = nn.CrossEntropyLoss()
        loss = sum(
            ce(logits_list[d], bin_targets[:, d])
            for d in range(action_dim)
        )

    elif policy.head_type == "gvla":
        k = policy.head.k
        bin_targets = quantize_action(action, n_bins)           # (B, action_dim)
        bin_codes = action_to_binary(bin_targets, k)            # (B, action_dim, k)
        out_list = policy(obs)                                   # list of dicts
        bce = nn.BCEWithLogitsLoss()
        loss = 0.0
        for d, out in enumerate(out_list):
            # projections: (B, k), target: (B, k)
            loss = loss + bce(out["projections"], bin_codes[:, d, :])
        # orthogonality regularization
        loss = loss + 0.01 * policy.head.orthogonality_loss()

    return loss


# ============================================================
# Training
# ============================================================

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── dataset ────────────────────────────────────────────
    dataset = LiftDemoDataset(args.data)
    obs_dim = dataset.obs.shape[1]
    action_dim = dataset.act.shape[1]
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, num_workers=0)

    # ── model ──────────────────────────────────────────────
    policy = BCPolicy(obs_dim, action_dim,
                      head_type=args.head,
                      n_bins=args.n_bins,
                      latent_dim=args.latent_dim).to(device)
    total_params = sum(p.numel() for p in policy.parameters())
    print(f"Model: {args.head} head, n_bins={args.n_bins}, "
          f"params={total_params:,}")

    optimizer = torch.optim.AdamW(policy.parameters(),
                                  lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    # ── output dir ─────────────────────────────────────────
    out_dir = Path(args.out_dir) / args.exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── training loop ──────────────────────────────────────
    best_loss = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        policy.train()
        epoch_loss = 0.0
        for obs_batch, act_batch in loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            loss = compute_loss(policy, obs_batch, act_batch)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(obs_batch)
        scheduler.step()
        epoch_loss /= len(dataset)

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"Epoch {epoch:4d}/{args.epochs}  loss={epoch_loss:.4f}  "
                  f"elapsed={elapsed:.0f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(policy.state_dict(), out_dir / "best.pt")

    torch.save(policy.state_dict(), out_dir / "last.pt")

    # ── save config ────────────────────────────────────────
    config = vars(args)
    config.update({"obs_dim": obs_dim, "action_dim": action_dim,
                   "best_loss": best_loss, "total_params": total_params})
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nSaved to {out_dir}  (best_loss={best_loss:.4f})")
    return out_dir


# ============================================================
# Entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BC Training: Dense vs GVLA")
    parser.add_argument("--data", type=str,
                        default="data/robomimic/lift/ph/low_dim.hdf5",
                        help="Path to HDF5 demo file")
    parser.add_argument("--head", type=str, choices=["dense", "gvla"],
                        required=True)
    parser.add_argument("--n_bins", type=int, default=64,
                        help="Bins per action dimension (M)")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--out_dir", type=str,
                        default="experiments/results/bc_study/checkpoints")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
