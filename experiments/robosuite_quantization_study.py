"""
robosuite_quantization_study.py
=================================
GVLA-Net NeurIPS Experiment:
  "Fine-grained action discretization enables precise manipulation"

Design
------
1. Scripted policy for Lift task (OSC delta controller)
   → produces *continuous* expert actions at every step

2. Quantize expert actions to M uniform bins per dimension
   → simulate the effect of a discrete action codebook with M^action_dim entries

3. Sweep M in [4, 8, 16, 32, 64, 128, 256, 512, 1024, continuous]
   → measure task success rate

4. Separately measure Dense head vs GVLA head *latency* vs N
   → N = M^action_dim to make the comparison apples-to-apples

Key insight:
  • Small M → large quantization error → robot misses cube → 0% success
  • Large M → small error → robot lifts cube → high success
  • Dense head: latency O(N), infeasible for large N
  • GVLA head: latency O(log N), same cost regardless of N
  →  "Only GVLA makes fine-grained discretization computationally feasible"

Usage
-----
  python experiments/robosuite_quantization_study.py [--n_rollouts 30] [--device cuda]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

# robosuite
import robosuite as suite


# =========================================================
# Action Head Implementations
# =========================================================

class DenseHead:
    """
    Dense softmax action head.
    Stores N full d-dimensional embeddings → O(N) dot-product + argmax.
    Memory: N × d floats.
    """
    def __init__(self, d: int, N: int, device: str):
        self.codebook = torch.randn(N, d, device=device) / (d ** 0.5)

    def select(self, latent: torch.Tensor) -> torch.Tensor:
        scores = latent @ self.codebook.T          # (N,)
        return self.codebook[scores.argmax()]      # (d,)


class GVLAHead:
    """
    GVLA O(log N) action head.
    k = ceil(log2 N) orthogonal projection directions.
    Memory: k × d floats  (k ≪ N for large N).
    Routing: single matrix-vector multiply → binary code.
    """
    def __init__(self, d: int, N: int, device: str):
        k = int(np.ceil(np.log2(max(N, 2))))
        # QR-initialised orthogonal basis
        Q, _ = torch.linalg.qr(torch.randn(d, k, device=device))
        self.W = Q.T.contiguous()  # (k, d)

    def select(self, latent: torch.Tensor) -> torch.Tensor:
        y = self.W @ latent          # (k,)  — the ONLY computation
        return (y > 0).float()       # binary hash code


# =========================================================
# Scripted Policy for Lift Task (OSC Δ-pose controller)
# =========================================================

class ScriptedLiftPolicy:
    """
    Simple proportional controller for Robosuite Lift.

    Action layout (dim=7):
      [0:3]  → Δ EEF position  (x, y, z)   clipped to [-1, 1]
      [3:6]  → Δ EEF orientation (roll, pitch, yaw) — kept 0
      [6]    → gripper  (-1 = open, +1 = closed)

    Phases: hover → descend → grasp → lift → done
    """

    def __init__(self,
                 hover_height: float = 0.12,
                 approach_xy_thresh: float = 0.012,
                 approach_z_thresh: float = 0.012,
                 grasp_steps: int = 12,
                 lift_delta: float = 0.25,
                 kp: float = 10.0):
        self.hover_height = hover_height
        self.approach_xy_thresh = approach_xy_thresh
        self.approach_z_thresh = approach_z_thresh
        self.grasp_steps = grasp_steps
        self.lift_delta = lift_delta
        self.kp = kp
        self.reset()

    def reset(self):
        self.phase = "hover"
        self.grasp_counter = 0
        self.init_eef_z = None

    def act(self, obs: dict, env=None) -> np.ndarray:
        eef_pos = np.array(obs["robot0_eef_pos"])
        cube_pos = np.array(obs["cube_pos"])
        action = np.zeros(7, dtype=np.float32)

        if self.phase == "hover":
            target = cube_pos.copy()
            target[2] += self.hover_height
            delta = target - eef_pos
            action[:3] = np.clip(self.kp * delta, -1.0, 1.0)
            action[6] = -1.0  # open
            xy_err = np.linalg.norm(delta[:2])
            z_err = abs(delta[2])
            if xy_err < self.approach_xy_thresh and z_err < self.approach_z_thresh * 2:
                self.phase = "descend"

        elif self.phase == "descend":
            target = cube_pos.copy()
            target[2] += 0.005
            delta = target - eef_pos
            action[:3] = np.clip(self.kp * delta, -1.0, 1.0)
            action[6] = -1.0  # open
            if np.linalg.norm(delta) < self.approach_z_thresh:
                self.phase = "grasp"

        elif self.phase == "grasp":
            action[6] = 1.0  # close gripper
            self.grasp_counter += 1
            if self.grasp_counter >= self.grasp_steps:
                self.phase = "lift"
                self.init_eef_z = eef_pos[2]

        elif self.phase == "lift":
            action[2] = 1.0   # move up
            action[6] = 1.0   # keep closed
            if self.init_eef_z is not None and eef_pos[2] - self.init_eef_z > self.lift_delta:
                self.phase = "done"

        # phase "done": stay still, keep gripper closed
        elif self.phase == "done":
            action[6] = 1.0

        return action


class ScriptedNutAssemblySquarePolicy:
    """
    Phase-based controller for Robosuite NutAssemblySquare.

    This is intentionally simple: it keeps the same 7-D OSC delta-pose action
    interface as the Lift controller and uses environment geometry to define a
    conservative pick → align → insert routine.
    """

    def __init__(self,
                 hover_height: float = 0.14,
                 peg_hover_height: float = 0.18,
                 grasp_xy_thresh: float = 0.015,
                 grasp_z_thresh: float = 0.010,
                 align_xy_thresh: float = 0.010,
                 insert_xy_thresh: float = 0.006,
                 lift_clearance: float = 0.13,
                 retreat_height: float = 0.12,
                 grasp_steps: int = 14,
                 kp_move: float = 10.0,
                 kp_insert: float = 5.0):
        self.hover_height = hover_height
        self.peg_hover_height = peg_hover_height
        self.grasp_xy_thresh = grasp_xy_thresh
        self.grasp_z_thresh = grasp_z_thresh
        self.align_xy_thresh = align_xy_thresh
        self.insert_xy_thresh = insert_xy_thresh
        self.lift_clearance = lift_clearance
        self.retreat_height = retreat_height
        self.grasp_steps = grasp_steps
        self.kp_move = kp_move
        self.kp_insert = kp_insert
        self.reset()

    def reset(self):
        self.phase = "pregrasp"
        self.grasp_counter = 0
        self.release_counter = 0
        self.lift_start_z = None

    def _move_towards(self, eef_pos: np.ndarray, target: np.ndarray, kp: float) -> np.ndarray:
        delta = target - eef_pos
        return np.clip(kp * delta, -1.0, 1.0).astype(np.float32)

    def act(self, obs: dict, env=None) -> np.ndarray:
        if env is None:
            raise ValueError("NutAssembly policy requires env for peg geometry access.")

        eef_pos = np.array(obs["robot0_eef_pos"])
        nut_pos = np.array(obs["SquareNut_pos"])
        peg_pos = np.array(env.sim.data.body_xpos[env.peg1_body_id])
        action = np.zeros(7, dtype=np.float32)

        if self.phase == "pregrasp":
            target = nut_pos.copy()
            target[2] += self.hover_height
            action[:3] = self._move_towards(eef_pos, target, self.kp_move)
            action[6] = -1.0
            xy_err = np.linalg.norm((target - eef_pos)[:2])
            z_err = abs(target[2] - eef_pos[2])
            if xy_err < self.grasp_xy_thresh and z_err < self.grasp_z_thresh * 3:
                self.phase = "descend_to_grasp"

        elif self.phase == "descend_to_grasp":
            target = nut_pos.copy()
            target[2] += 0.010
            action[:3] = self._move_towards(eef_pos, target, self.kp_move)
            action[6] = -1.0
            delta = target - eef_pos
            if np.linalg.norm(delta[:2]) < self.grasp_xy_thresh and abs(delta[2]) < self.grasp_z_thresh:
                self.phase = "close_gripper"

        elif self.phase == "close_gripper":
            action[6] = 1.0
            self.grasp_counter += 1
            if self.grasp_counter >= self.grasp_steps:
                self.phase = "lift_clear"
                self.lift_start_z = eef_pos[2]

        elif self.phase == "lift_clear":
            action[2] = 1.0
            action[6] = 1.0
            if self.lift_start_z is not None and eef_pos[2] - self.lift_start_z > self.lift_clearance:
                self.phase = "translate_to_peg"

        elif self.phase == "translate_to_peg":
            target = peg_pos.copy()
            target[2] += self.peg_hover_height
            action[:3] = self._move_towards(eef_pos, target, self.kp_move)
            action[6] = 1.0
            xy_err = np.linalg.norm((target - eef_pos)[:2])
            z_err = abs(target[2] - eef_pos[2])
            if xy_err < self.align_xy_thresh and z_err < self.grasp_z_thresh * 3:
                self.phase = "align_for_insertion"

        elif self.phase == "align_for_insertion":
            target = peg_pos.copy()
            target[2] += 0.080
            action[:3] = self._move_towards(eef_pos, target, self.kp_insert)
            action[6] = 1.0
            xy_err = np.linalg.norm((target - eef_pos)[:2])
            if xy_err < self.insert_xy_thresh:
                self.phase = "insert"

        elif self.phase == "insert":
            target = peg_pos.copy()
            target[2] += 0.030
            action[:3] = self._move_towards(eef_pos, target, self.kp_insert)
            action[6] = 1.0
            xy_err = np.linalg.norm((target - eef_pos)[:2])
            z_err = abs(target[2] - eef_pos[2])
            if xy_err < self.insert_xy_thresh and z_err < self.grasp_z_thresh * 2:
                self.phase = "release"

        elif self.phase == "release":
            action[6] = -1.0
            self.release_counter += 1
            if self.release_counter >= 8:
                self.phase = "retreat"

        elif self.phase == "retreat":
            target = peg_pos.copy()
            target[2] += self.retreat_height
            action[:3] = self._move_towards(eef_pos, target, self.kp_move)
            action[6] = -1.0
            if abs(target[2] - eef_pos[2]) < self.grasp_z_thresh * 3:
                self.phase = "done"

        elif self.phase == "done":
            action[6] = -1.0

        return action


class ScriptedPickPlaceCanPolicy:
    """
    Phase-based controller for single-object PickPlace with a Can target.
    """

    def __init__(self,
                 hover_height: float = 0.14,
                 bin_hover_height: float = 0.20,
                 place_height: float = 0.07,
                 grasp_xy_thresh: float = 0.018,
                 grasp_z_thresh: float = 0.012,
                 place_xy_thresh: float = 0.020,
                 lift_delta: float = 0.16,
                 grasp_steps: int = 12,
                 release_steps: int = 10,
                 kp_move: float = 10.0,
                 kp_place: float = 7.0):
        self.hover_height = hover_height
        self.bin_hover_height = bin_hover_height
        self.place_height = place_height
        self.grasp_xy_thresh = grasp_xy_thresh
        self.grasp_z_thresh = grasp_z_thresh
        self.place_xy_thresh = place_xy_thresh
        self.lift_delta = lift_delta
        self.grasp_steps = grasp_steps
        self.release_steps = release_steps
        self.kp_move = kp_move
        self.kp_place = kp_place
        self.reset()

    def reset(self):
        self.phase = "pregrasp"
        self.grasp_counter = 0
        self.release_counter = 0
        self.lift_start_z = None

    def _move_towards(self, eef_pos: np.ndarray, target: np.ndarray, kp: float) -> np.ndarray:
        delta = target - eef_pos
        return np.clip(kp * delta, -1.0, 1.0).astype(np.float32)

    def act(self, obs: dict, env=None) -> np.ndarray:
        if env is None:
            raise ValueError("PickPlace policy requires env access.")

        eef_pos = np.array(obs["robot0_eef_pos"])
        obj_pos = np.array(obs["Can_pos"])
        target_bin = np.array(env.target_bin_placements[env.object_id])
        action = np.zeros(7, dtype=np.float32)

        if self.phase == "pregrasp":
            target = obj_pos.copy()
            target[2] += self.hover_height
            action[:3] = self._move_towards(eef_pos, target, self.kp_move)
            action[6] = -1.0
            xy_err = np.linalg.norm((target - eef_pos)[:2])
            z_err = abs(target[2] - eef_pos[2])
            if xy_err < self.grasp_xy_thresh and z_err < self.grasp_z_thresh * 3:
                self.phase = "descend_to_grasp"

        elif self.phase == "descend_to_grasp":
            target = obj_pos.copy()
            target[2] += 0.01
            action[:3] = self._move_towards(eef_pos, target, self.kp_move)
            action[6] = -1.0
            delta = target - eef_pos
            if np.linalg.norm(delta[:2]) < self.grasp_xy_thresh and abs(delta[2]) < self.grasp_z_thresh:
                self.phase = "close_gripper"

        elif self.phase == "close_gripper":
            action[6] = 1.0
            self.grasp_counter += 1
            if self.grasp_counter >= self.grasp_steps:
                self.phase = "lift_clear"
                self.lift_start_z = eef_pos[2]

        elif self.phase == "lift_clear":
            action[2] = 1.0
            action[6] = 1.0
            if self.lift_start_z is not None and eef_pos[2] - self.lift_start_z > self.lift_delta:
                self.phase = "translate_to_bin"

        elif self.phase == "translate_to_bin":
            target = target_bin.copy()
            target[2] += self.bin_hover_height
            action[:3] = self._move_towards(eef_pos, target, self.kp_move)
            action[6] = 1.0
            xy_err = np.linalg.norm((target - eef_pos)[:2])
            z_err = abs(target[2] - eef_pos[2])
            if xy_err < self.place_xy_thresh and z_err < self.grasp_z_thresh * 4:
                self.phase = "lower_to_place"

        elif self.phase == "lower_to_place":
            target = target_bin.copy()
            target[2] += self.place_height
            action[:3] = self._move_towards(eef_pos, target, self.kp_place)
            action[6] = 1.0
            xy_err = np.linalg.norm((target - eef_pos)[:2])
            z_err = abs(target[2] - eef_pos[2])
            if xy_err < self.place_xy_thresh and z_err < self.grasp_z_thresh * 2:
                self.phase = "release"

        elif self.phase == "release":
            action[6] = -1.0
            self.release_counter += 1
            if self.release_counter >= self.release_steps:
                self.phase = "retreat"

        elif self.phase == "retreat":
            target = target_bin.copy()
            target[2] += self.bin_hover_height
            action[:3] = self._move_towards(eef_pos, target, self.kp_move)
            action[6] = -1.0
            if abs(target[2] - eef_pos[2]) < self.grasp_z_thresh * 4:
                self.phase = "done"

        elif self.phase == "done":
            action[6] = -1.0

        return action


# =========================================================
# Quantisation Utilities
# =========================================================

def quantize_action(action: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Uniform per-dimension quantisation to n_bins bins over [-1, 1].

    Quantisation error per dim ≈ 1 / n_bins  (half the bin width).
    Equivalent total action space: N = n_bins ^ action_dim.
    """
    q = np.empty_like(action, dtype=np.float32)

    # Use symmetric nearest-centre quantisation for motion dims to avoid the
    # positive bias introduced by floor-based binning.
    motion = np.clip(action[:6], -1.0, 1.0)
    if n_bins <= 1:
        q[:6] = 0.0
    else:
        grid = np.linspace(-1.0, 1.0, n_bins, dtype=np.float32)
        step = grid[1] - grid[0]
        idx = np.rint((motion - grid[0]) / step).astype(np.int32)
        idx = np.clip(idx, 0, n_bins - 1)
        q[:6] = grid[idx]

    # The gripper is effectively binary in these scripted policies. Quantising
    # it with the same uniform grid creates artificial half-open states.
    q[6] = 1.0 if action[6] >= 0.0 else -1.0
    return q


def quantize_action_ar_tokens(action: np.ndarray, n_bins: int) -> np.ndarray:
    """
    RT-2-style per-dimension tokenization surrogate.

    Each action dimension is discretized independently to n_bins bins. Unlike
    the GVLA surrogate above, this keeps all 7 dimensions on the same tokenized
    grid to mimic token-per-dimension decoding.
    """
    motion = np.clip(action, -1.0, 1.0)
    if n_bins <= 1:
        return np.zeros_like(action, dtype=np.float32)
    grid = np.linspace(-1.0, 1.0, n_bins, dtype=np.float32)
    step = grid[1] - grid[0]
    idx = np.rint((motion - grid[0]) / step).astype(np.int32)
    idx = np.clip(idx, 0, n_bins - 1)
    return grid[idx].astype(np.float32)


def maybe_inject_token_errors(action: np.ndarray, n_bins: int,
                              token_error_prob: float, rng: np.random.RandomState) -> np.ndarray:
    """Randomly perturb tokenized dimensions by one adjacent bin."""
    if token_error_prob <= 0 or n_bins <= 1:
        return action
    out = action.copy()
    step = 2.0 / (n_bins - 1)
    for i in range(len(out)):
        if rng.rand() < token_error_prob:
            direction = -1.0 if rng.rand() < 0.5 else 1.0
            out[i] = np.clip(out[i] + direction * step, -1.0, 1.0)
    return out.astype(np.float32)


def decode_action(action: np.ndarray, n_bins,
                  decode_mode: str,
                  ar_token_bins: int,
                  token_error_prob: float,
                  rng: np.random.RandomState) -> np.ndarray:
    if n_bins is None:
        return action
    if decode_mode == "gvla":
        return quantize_action(action, n_bins)
    if decode_mode == "ar_tokens":
        bins = ar_token_bins if ar_token_bins is not None else n_bins
        decoded = quantize_action_ar_tokens(action, bins)
        return maybe_inject_token_errors(decoded, bins, token_error_prob, rng)
    raise ValueError(f"Unsupported decode mode: {decode_mode}")


# =========================================================
# Rollout & Success Rate
# =========================================================

def make_policy(task: str):
    if task == "lift":
        return ScriptedLiftPolicy()
    if task == "nut_assembly_square":
        return ScriptedNutAssemblySquarePolicy()
    if task == "pick_place_can":
        return ScriptedPickPlaceCanPolicy()
    raise ValueError(f"Unsupported task: {task}")


def run_one_rollout(env, policy,
                    n_bins, max_steps: int, seed: int,
                    decode_mode: str = "gvla",
                    ar_token_bins: int = 256,
                    token_latency_ms: float = 0.0,
                    token_error_prob: float = 0.0) -> bool:
    """
    Run one episode.  n_bins=None → continuous (no quantisation).
    Returns True if cube lifted (reward > 0).
    """
    np.random.seed(seed)
    rng = np.random.RandomState(seed)
    policy.reset()
    obs = env.reset()
    reward_accum = 0.0
    prev_action = np.zeros(7, dtype=np.float32)
    control_period_ms = 1000.0 / getattr(env, "control_freq", 20)
    total_decode_ms = token_latency_ms * 7 if decode_mode == "ar_tokens" else 0.0
    stale_steps = int(np.floor(total_decode_ms / control_period_ms))

    for _ in range(max_steps):
        action = policy.act(obs, env)
        action = decode_action(action, n_bins, decode_mode, ar_token_bins, token_error_prob, rng)

        for _ in range(stale_steps):
            obs, reward, done, _ = env.step(prev_action)
            reward_accum += reward
            if done or reward_accum > 0:
                return True

        obs, reward, done, _ = env.step(action)
        reward_accum += reward
        prev_action = action
        if done or reward_accum > 0:
            return True

    # A few extra "hold" steps in case cube is still rising
    if policy.phase in ("lift", "done"):
        hold = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(30):
            if n_bins is not None:
                hold_q = quantize_action(hold, n_bins)
            else:
                hold_q = hold
            obs, reward, done, _ = env.step(hold_q)
            if reward > 0:
                return True

    return reward_accum > 0


def measure_success_rate(env, task: str, n_bins, n_rollouts: int,
                         max_steps: int,
                         decode_mode: str = "gvla",
                         ar_token_bins: int = 256,
                         token_latency_ms: float = 0.0,
                         token_error_prob: float = 0.0) -> float:
    """Success rate over n_rollouts seeds."""
    successes = 0
    for seed in range(n_rollouts):
        policy = make_policy(task)
        if run_one_rollout(
            env, policy, n_bins, max_steps, seed,
            decode_mode=decode_mode,
            ar_token_bins=ar_token_bins,
            token_latency_ms=token_latency_ms,
            token_error_prob=token_error_prob,
        ):
            successes += 1
    return successes / n_rollouts


# =========================================================
# Head Latency Measurement
# =========================================================

def measure_head_latency(N_list: list, d: int = 256,
                          n_warmup: int = 50, n_trials: int = 500,
                          device: str = "cuda") -> dict:
    """
    Returns dict[N] = {dense_ms, gvla_ms, speedup, k}.
    Measures pure head computation (projection or dot-product),
    excluding codebook lookup (which is O(1) with a hash table in practice).
    """
    results = {}
    sync = (lambda: torch.cuda.synchronize()) if device == "cuda" else (lambda: None)

    for N in N_list:
        k = int(np.ceil(np.log2(max(N, 2))))
        print(f"  N={N:>10,}  k={k:>3} bits", end="  →  ", flush=True)

        latent = torch.randn(d, device=device)

        # --- Dense head ---
        dense = DenseHead(d, N, device)
        for _ in range(n_warmup):
            dense.select(latent)
        sync()
        t0 = time.perf_counter()
        for _ in range(n_trials):
            dense.select(latent)
        sync()
        dense_ms = (time.perf_counter() - t0) / n_trials * 1e3

        # --- GVLA head ---
        gvla = GVLAHead(d, N, device)
        for _ in range(n_warmup):
            gvla.select(latent)
        sync()
        t0 = time.perf_counter()
        for _ in range(n_trials):
            gvla.select(latent)
        sync()
        gvla_ms = (time.perf_counter() - t0) / n_trials * 1e3

        speedup = dense_ms / gvla_ms if gvla_ms > 0 else float("inf")
        results[N] = dict(dense_ms=dense_ms, gvla_ms=gvla_ms,
                          speedup=speedup, k=k)
        print(f"Dense {dense_ms:.3f}ms  GVLA {gvla_ms:.3f}ms  "
              f"speedup {speedup:.1f}×")

    return results


# =========================================================
# Plotting
# =========================================================

def plot_results(success_results: dict, latency_results: dict,
                 action_dim: int, save_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------ Data ------
    M_vals = sorted(k for k in success_results if k != "inf")
    sr_vals = [success_results[m] * 100 for m in M_vals]
    sr_cont = success_results.get("inf", None)

    N_vals = sorted(latency_results.keys())
    dense_ms = [latency_results[N]["dense_ms"] for N in N_vals]
    gvla_ms  = [latency_results[N]["gvla_ms"]  for N in N_vals]

    # N_total for x-axis alignment
    N_total_from_M = [m ** action_dim for m in M_vals]

    # ------ Figure: 2-panel ------
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A — Success rate vs bins per dim
    ax = axes[0]
    ax.plot(M_vals, sr_vals, "bo-", lw=2.2, ms=8, label="GVLA-discretised")
    if sr_cont is not None:
        ax.axhline(sr_cont * 100, color="gray", ls="--", lw=1.5,
                   label=f"Continuous (no quantisation) {sr_cont*100:.0f}%")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Bins per action dim (M)", fontsize=13)
    ax.set_ylabel("Task Success Rate (%)", fontsize=13)
    ax.set_title("Lift: Success Rate vs Action Resolution", fontsize=13)
    ax.set_ylim(-5, 110)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    for m, sr in zip(M_vals, sr_vals):
        ax.annotate(f"{sr:.0f}%", xy=(m, sr),
                    xytext=(5, 6), textcoords="offset points", fontsize=9)

    # Panel B — Head latency vs N
    ax = axes[1]
    ax.plot(N_vals, dense_ms, "r^-", lw=2.2, ms=8, label="Dense Head  O(N)")
    ax.plot(N_vals, gvla_ms,  "bs-", lw=2.2, ms=8, label="GVLA Head   O(log N)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Total action space N", fontsize=13)
    ax.set_ylabel("Head latency (ms)", fontsize=13)
    ax.set_title("Action Head Latency vs Action Space Size", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")

    plt.tight_layout()
    for suf, dpi in [("", 150), ("_paper", 300)]:
        fig.savefig(save_dir / f"robosuite_quantization_study{suf}.png",
                    dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"[plot] saved to {save_dir}")

    # ------ Figure: combined narrative ------
    fig2, ax2 = plt.subplots(figsize=(9, 6))
    color_sr  = "#2166ac"
    color_den = "#d73027"
    color_gvl = "#1a9641"

    ax2_r = ax2.twinx()

    lns1 = ax2.plot(M_vals, sr_vals, "o-", color=color_sr, lw=2.5, ms=9,
                    label="Success Rate (↑ better)")
    if sr_cont is not None:
        ax2.axhline(sr_cont * 100, color=color_sr, ls="--", lw=1.5, alpha=0.6)

    # latency: map N_total to M values for overlay
    # Find matching M for each N in latency_results
    M_for_latency = []
    dense_for_M = []
    gvla_for_M = []
    for N in N_vals:
        M_equiv = round(N ** (1 / action_dim))
        if M_equiv in M_vals:
            M_for_latency.append(M_equiv)
            dense_for_M.append(latency_results[N]["dense_ms"])
            gvla_for_M.append(latency_results[N]["gvla_ms"])

    if M_for_latency:
        lns2 = ax2_r.plot(M_for_latency, dense_for_M, "^--", color=color_den,
                           lw=2, ms=8, label="Dense latency  O(N)")
        lns3 = ax2_r.plot(M_for_latency, gvla_for_M,  "s--", color=color_gvl,
                           lw=2, ms=8, label="GVLA latency   O(log N)")
    else:
        lns2, lns3 = [], []

    ax2.set_xscale("log", base=2)
    ax2_r.set_yscale("log")
    ax2.set_xlabel("Bins per action dim (M)", fontsize=13)
    ax2.set_ylabel("Task Success Rate (%)", fontsize=13, color=color_sr)
    ax2_r.set_ylabel("Head Latency (ms, log scale)", fontsize=13)
    ax2.set_title("Precision–Efficiency Trade-off in Action Discretisation\n"
                  "(Lift, Panda, Robosuite)", fontsize=13)
    ax2.set_ylim(-5, 110)
    ax2.grid(True, alpha=0.3)

    all_lines = lns1 + (lns2 if lns2 else []) + (lns3 if lns3 else [])
    ax2.legend(all_lines, [l.get_label() for l in all_lines], fontsize=10, loc="center right")

    fig2.tight_layout()
    for suf, dpi in [("", 150), ("_paper", 300)]:
        fig2.savefig(save_dir / f"robosuite_combined{suf}.png",
                     dpi=dpi, bbox_inches="tight")
    plt.close()


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser(description="Robosuite Quantisation Study")
    parser.add_argument("--task", type=str, default="lift",
                        choices=["lift", "nut_assembly_square", "pick_place_can"],
                        help="Robosuite task to evaluate")
    parser.add_argument("--decode_mode", type=str, default="gvla",
                        choices=["gvla", "ar_tokens"],
                        help="Decode mechanism surrogate to evaluate")
    parser.add_argument("--ar_token_bins", type=int, default=256,
                        help="Bins per dimension for autoregressive token decode surrogate")
    parser.add_argument("--token_latency_ms", type=float, default=0.0,
                        help="Per-token decode latency for autoregressive token decode surrogate")
    parser.add_argument("--token_error_prob", type=float, default=0.0,
                        help="Probability of per-token adjacent-bin decode error")
    parser.add_argument("--n_rollouts",  type=int, default=30,
                        help="Rollouts per quantisation level")
    parser.add_argument("--max_steps",   type=int, default=400,
                        help="Max env steps per rollout")
    parser.add_argument("--n_trials",    type=int, default=500,
                        help="Latency measurement trials")
    parser.add_argument("--device",      type=str, default="cuda",
                        help="cuda or cpu")
    parser.add_argument("--save_dir",    type=str,
                        default="experiments/results/robosuite_study",
                        help="Output directory")
    parser.add_argument("--skip_latency", action="store_true",
                        help="Skip latency measurement (env-only run)")
    parser.add_argument("--skip_rollout", action="store_true",
                        help="Skip rollout measurement (latency-only run)")
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Task: {args.task}")
    print(f"Decode mode: {args.decode_mode}")

    # ---- Bins to sweep ----
    if args.decode_mode == "ar_tokens":
        M_bins_sweep = [args.ar_token_bins]
    elif args.task == "pick_place_can":
        M_bins_sweep = [32, 48, 64, 96, 128]
    else:
        M_bins_sweep = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

    # ---- N for latency (includes M^7 equivalents) ----
    N_latency = [1024, 4096, 16384, 65536, 262144, 1048576, 4194304]

    action_dim = 7  # Panda Lift with OSC controller

    success_results = {}
    latency_results = {}

    # =========================================================
    # PART 1: Success rate sweep
    # =========================================================
    if not args.skip_rollout:
        print("\n" + "=" * 60)
        print("PART 1: Success Rate vs Action Quantisation")
        print("=" * 60)

        env_kwargs = dict(
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            ignore_done=True,
            horizon=args.max_steps,
            reward_shaping=False,
            reward_scale=1.0,
        )
        if args.task == "lift":
            env_name = "Lift"
        elif args.task == "nut_assembly_square":
            env_name = "NutAssemblySquare"
        elif args.task == "pick_place_can":
            env_name = "PickPlace"
            env_kwargs.update(single_object_mode=2, object_type="can")
        else:
            raise ValueError(f"Unsupported task: {args.task}")

        env = suite.make(env_name, **env_kwargs)

        # Continuous baseline
        print(f"\n[Continuous — no quantisation]  ({args.n_rollouts} rollouts)")
        sr = measure_success_rate(
            env, args.task, None, args.n_rollouts, args.max_steps,
            decode_mode=args.decode_mode,
            ar_token_bins=args.ar_token_bins,
            token_latency_ms=args.token_latency_ms,
            token_error_prob=args.token_error_prob,
        )
        success_results["inf"] = sr
        print(f"  → Success rate: {sr * 100:.1f}%")

        # Quantised sweep
        for M in M_bins_sweep:
            N_total = M ** action_dim
            print(f"\n[M={M} bins/dim  |  N_total={N_total:,.0f}]"
                  f"  ({args.n_rollouts} rollouts)")
            sr = measure_success_rate(
                env, args.task, M, args.n_rollouts, args.max_steps,
                decode_mode=args.decode_mode,
                ar_token_bins=args.ar_token_bins,
                token_latency_ms=args.token_latency_ms,
                token_error_prob=args.token_error_prob,
            )
            success_results[M] = sr
            print(f"  → Success rate: {sr * 100:.1f}%")

        env.close()

    # =========================================================
    # PART 2: Head latency sweep
    # =========================================================
    if not args.skip_latency:
        print("\n" + "=" * 60)
        print("PART 2: Head Latency vs Action Space Size")
        print("=" * 60)
        latency_results = measure_head_latency(
            N_latency, d=256,
            n_warmup=50, n_trials=args.n_trials,
            device=device
        )

    # =========================================================
    # Save & plot
    # =========================================================
    json_out = {
        "success_rate": {str(k): v for k, v in success_results.items()},
        "latency":      {str(k): v for k, v in latency_results.items()},
        "config": vars(args),
        "action_dim": action_dim,
        "task": args.task,
    }
    with open(save_dir / "results.json", "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\nResults saved → {save_dir / 'results.json'}")

    if success_results and latency_results:
        plot_results(success_results, latency_results, action_dim, save_dir)

    # =========================================================
    # Summary table
    # =========================================================
    if success_results:
        print("\n" + "=" * 60)
        print("SUMMARY: Success Rate vs Quantisation")
        print(f"{'M (bins/dim)':<15}  {'N_total (M^7)':<22}  {'Success Rate':>12}")
        print("-" * 52)
        sr_cont = success_results.get("inf", None)
        if sr_cont is not None:
            print(f"{'continuous':<15}  {'∞':<22}  {sr_cont*100:>11.1f}%")
        for M in M_bins_sweep:
            if M in success_results:
                N_t = M ** action_dim
                print(f"{M:<15}  {N_t:<22,.0f}  {success_results[M]*100:>11.1f}%")

    if latency_results:
        print("\n" + "=" * 60)
        print("SUMMARY: Head Latency")
        print(f"{'N':>12}  {'k bits':>8}  {'Dense(ms)':>12}  {'GVLA(ms)':>12}  {'Speedup':>9}")
        print("-" * 60)
        for N in sorted(latency_results.keys()):
            r = latency_results[N]
            print(f"{N:>12,}  {r['k']:>8}  {r['dense_ms']:>12.3f}  "
                  f"{r['gvla_ms']:>12.3f}  {r['speedup']:>8.1f}×")


if __name__ == "__main__":
    main()
