"""
GVLA-Net vs baseline eval on ALOHA sim environment.

Loads a finetuned Octo checkpoint (action_dim=14, action_horizon=50),
wraps the AlohaCubeHandover-v0 or AlohaInsertion-v0 gym environment,
and compares:
  - Baseline: raw L1ActionHead output (continuous 14D)
  - GVLA:     ActionPrecisionAdapter(bins=2^20) quantization on 14D output

Usage:
  MUJOCO_GL=egl python experiments/octo_aloha_eval.py \
    --checkpoint_path checkpoints/octo_aloha/4999 \
    --rollouts 50
"""

import argparse
import csv
import grp
import os
import pwd
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow_datasets as tfds
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_OCTO = PROJECT_ROOT / "third_party" / "octo"
for p in [str(PROJECT_ROOT), str(THIRD_PARTY_OCTO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import jax
import jax.numpy as jnp

from octo.model.octo_model import OctoModel


_ORIG_GETPWUID = pwd.getpwuid
_ORIG_GETGRGID = grp.getgrgid


def _safe_getpwuid(uid: int):
    try:
        return _ORIG_GETPWUID(uid)
    except KeyError:
        fallback_name = os.environ.get("USER") or os.environ.get("LOGNAME") or str(uid)
        return type(
            "PwdEntry",
            (),
            {
                "pw_name": fallback_name,
                "pw_passwd": "x",
                "pw_uid": uid,
                "pw_gid": os.getgid(),
                "pw_gecos": fallback_name,
                "pw_dir": str(Path.home()),
                "pw_shell": os.environ.get("SHELL", "/bin/sh"),
            },
        )()


pwd.getpwuid = _safe_getpwuid


def _safe_getgrgid(gid: int):
    try:
        return _ORIG_GETGRGID(gid)
    except KeyError:
        fallback_name = os.environ.get("GROUP") or str(gid)
        return type(
            "GrpEntry",
            (),
            {
                "gr_name": fallback_name,
                "gr_passwd": "x",
                "gr_gid": gid,
                "gr_mem": [],
            },
        )()


grp.getgrgid = _safe_getgrgid


def resolve_checkpoint_path_and_step(checkpoint_path: str) -> Tuple[str, Optional[int]]:
    """
    OctoModel.load_pretrained expects metadata files at the checkpoint root and
    the numeric step to be passed separately. Accept either:
      - /path/to/checkpoints/octo_aloha        -> step=None (latest)
      - /path/to/checkpoints/octo_aloha/4999   -> root=/.../octo_aloha, step=4999
    """
    path = Path(checkpoint_path)
    if path.name.isdigit() and (path / "default").exists():
        return str(path.parent), int(path.name)
    return str(path), None


def task_to_instruction(task: str) -> str:
    if "Insertion" in task:
        return "insert the peg into the socket"
    if "TransferCube" in task or "CubeHandover" in task or "transfer" in task:
        # Matches language_instruction in aloha_sim_cube_scripted_dataset
        return "pick up the cube and hand it over"
    return "manipulate the object to complete the task"


def task_to_act_name(task: str) -> str:
    if "Insertion" in task:
        return "sim_insertion"
    return "sim_transfer_cube"


def sample_box_pose(rng: np.random.Generator) -> np.ndarray:
    ranges = np.array(
        [
            [0.0, 0.2],
            [0.4, 0.6],
            [0.05, 0.05],
        ],
        dtype=np.float32,
    )
    cube_position = rng.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0], dtype=np.float32)
    return np.concatenate([cube_position, cube_quat], axis=0)


def sample_insertion_pose(rng: np.random.Generator) -> np.ndarray:
    peg_ranges = np.array(
        [
            [0.1, 0.2],
            [0.4, 0.6],
            [0.05, 0.05],
        ],
        dtype=np.float32,
    )
    socket_ranges = np.array(
        [
            [-0.2, -0.1],
            [0.4, 0.6],
            [0.05, 0.05],
        ],
        dtype=np.float32,
    )
    peg_position = rng.uniform(peg_ranges[:, 0], peg_ranges[:, 1])
    socket_position = rng.uniform(socket_ranges[:, 0], socket_ranges[:, 1])
    quat = np.array([1, 0, 0, 0], dtype=np.float32)
    return np.concatenate(
        [np.concatenate([peg_position, quat]), np.concatenate([socket_position, quat])],
        axis=0,
    )


# ---------------------------------------------------------------------------
# Action precision adapter (GVLA quantization)
# ---------------------------------------------------------------------------

class ActionPrecisionAdapter:
    """Quantize continuous actions onto a fixed lattice (GVLA-Net head)."""

    def __init__(self, bins: int) -> None:
        if bins < 2:
            raise ValueError(f"bins must be >= 2, got {bins}")
        self.bins = int(bins)

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        clipped = np.clip(actions, -1.0, 1.0)
        scaled = (clipped + 1.0) * 0.5 * (self.bins - 1)
        quantized = np.round(scaled) / (self.bins - 1)
        return quantized * 2.0 - 1.0


def identity_adapter(actions: np.ndarray) -> np.ndarray:
    return np.asarray(actions)


# ---------------------------------------------------------------------------
# ALOHA env (ACT sim — original training environment)
# ---------------------------------------------------------------------------

# Ensure ACT repo is on path so sim_env.py and aloha_sim_env.py can be imported.
ACT_PATH = str(PROJECT_ROOT / "third_party" / "act")
OCTO_ENVS_PATH = str(PROJECT_ROOT / "third_party" / "octo" / "examples")
for p in [ACT_PATH, OCTO_ENVS_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)


def make_aloha_env(task: str, seed: int):
    """
    Create ALOHA sim env using ACT sim_env (same env used for dataset collection).
    This replaces gym_aloha, which had a cube-position mismatch with the dataset.

    obs returned by this env:
      obs["image_primary"]: (256, 256, 3) uint8
      obs["proprio"]:       (14,) float32  (raw qpos — normalized by caller)
    """
    from sim_env import BOX_POSE, make_sim_env
    from envs.aloha_sim_env import AlohaGymEnv

    act_task = task_to_act_name(task)
    rng = np.random.default_rng(seed)

    class TaskAwareAlohaGymEnv(AlohaGymEnv):
        def reset(self, **kwargs):
            if act_task == "sim_insertion":
                BOX_POSE[0] = sample_insertion_pose(rng)
            else:
                BOX_POSE[0] = sample_box_pose(rng)
            ts = self._env.reset(**kwargs)
            obs, images = self.get_obs(ts)
            info = {"images": images}
            self._episode_is_success = 0
            return obs, info

        def get_task(self):
            return {
                "language_instruction": [task_to_instruction(task)],
            }

    dm_env = make_sim_env(act_task)
    env = TaskAwareAlohaGymEnv(dm_env, camera_names=["top"], im_size=256, seed=seed)
    return env


def normalize_proprio(proprio: np.ndarray, proprio_stats: dict) -> np.ndarray:
    """Gaussian-normalize proprio using dataset statistics (mirrors NormalizeProprio wrapper)."""
    mean = np.array(proprio_stats["mean"], dtype=np.float32)
    std = np.array(proprio_stats["std"], dtype=np.float32)
    mask = np.array(proprio_stats.get("mask", np.ones_like(mean, dtype=bool)))
    return np.where(mask, (proprio - mean) / (std + 1e-8), proprio)


def obs_to_octo(raw_obs: dict, image_size: int = 256, proprio_stats: Optional[dict] = None) -> dict:
    """
    Convert AlohaGymEnv (ACT sim) obs to Octo observation dict.

    ACT AlohaGymEnv obs:
      raw_obs["image_primary"]: (256, 256, 3) uint8  (already resized)
      raw_obs["proprio"]:       (14,) jax/np array   (raw qpos)

    Octo expects:
      obs["image_primary"]: (H, W, 3) uint8
      obs["proprio"]:       (14,) float32  -- Gaussian-normalized with dataset statistics
    """
    pixels = np.asarray(raw_obs["image_primary"], dtype=np.uint8)
    if pixels.shape[0] != image_size or pixels.shape[1] != image_size:
        pixels = np.asarray(
            Image.fromarray(pixels).resize((image_size, image_size), Image.Resampling.BILINEAR)
        )

    proprio = np.asarray(raw_obs["proprio"], dtype=np.float32)
    if proprio_stats is not None:
        proprio = normalize_proprio(proprio, proprio_stats)

    return {
        "image_primary": pixels,
        "proprio": proprio,
    }


# ---------------------------------------------------------------------------
# Rollout utilities
# ---------------------------------------------------------------------------

@dataclass
class RolloutSummary:
    policy_name: str
    task: str
    rollouts: int
    success_rate: float
    mean_steps: float
    mean_infer_ms: float


def compute_aloha_action_bounds(dataset_dir: str, max_episodes: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    builder = tfds.builder_from_directory(dataset_dir)
    ds = tfds.as_numpy(builder.as_dataset(split="train"))
    mins = []
    maxs = []
    for i, episode in enumerate(ds):
        actions = np.stack([step["action"] for step in episode["steps"]], axis=0)
        mins.append(actions.min(axis=0))
        maxs.append(actions.max(axis=0))
        if i + 1 >= max_episodes:
            break
    if not mins:
        raise RuntimeError(f"No episodes found in dataset_dir={dataset_dir}")
    return np.min(np.stack(mins, axis=0), axis=0), np.max(np.stack(maxs, axis=0), axis=0)


def rescale_aloha_action(action: np.ndarray, action_min: np.ndarray, action_max: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(action), -1.0, 1.0)
    scaled = (clipped + 1.0) * 0.5 * (action_max - action_min) + action_min
    # Gripper channels are normalized to [0, 1] in gym_aloha.
    scaled[6] = np.clip(scaled[6], 0.0, 1.0)
    scaled[13] = np.clip(scaled[13], 0.0, 1.0)
    return scaled.astype(np.float32)


def augment_obs_for_octo(obs: dict, *, history_len: int, task_completed_dim: int) -> dict:
    """Add timestep/pad_mask fields expected by Octo."""
    # obs already has history_len prepended by HistoryWrapper — just add bookkeeping
    augmented = dict(obs)
    h = history_len
    pad_mask = np.ones(h, dtype=bool)
    augmented["timestep_pad_mask"] = pad_mask
    augmented["timestep"] = np.arange(h, dtype=np.int32)
    augmented["task_completed"] = np.zeros((h, task_completed_dim), dtype=bool)
    augmented["pad_mask_dict"] = {
        "image_primary": pad_mask.copy(),
        "timestep": pad_mask.copy(),
    }
    if "proprio" in obs:
        augmented["pad_mask_dict"]["proprio"] = pad_mask.copy()
    return augmented


def init_history(initial_obs: dict, history_len: int) -> deque:
    history = deque(maxlen=history_len)
    for _ in range(history_len):
        history.append(
            {k: np.array(v, copy=True) for k, v in initial_obs.items()}
        )
    return history


def stack_history(history: deque) -> dict:
    keys = history[0].keys()
    return {k: np.stack([step[k] for step in history], axis=0) for k in keys}


def summarize_action(name: str, step_idx: int, action_chunk: np.ndarray, env_action: np.ndarray) -> None:
    print(
        f"[debug:{name}] step={step_idx} "
        f"chunk_shape={action_chunk.shape} "
        f"raw_min={action_chunk.min():.4f} raw_max={action_chunk.max():.4f} raw_mean={action_chunk.mean():.4f} "
        f"env_min={env_action.min():.4f} env_max={env_action.max():.4f} env_mean={env_action.mean():.4f}"
    )
    print(
        f"[debug:{name}] env_action[:8]={np.array2string(env_action[:8], precision=4, separator=', ')}"
    )


def summarize_transition(
    name: str,
    step_idx: int,
    chunk_offset: int,
    qpos_before: np.ndarray,
    env_action: np.ndarray,
    qpos_after: np.ndarray,
    reward: float,
) -> None:
    qpos_delta = qpos_after - qpos_before
    action_gap = env_action - qpos_before
    print(
        f"[debug:{name}] exec_step={step_idx} chunk_offset={chunk_offset} "
        f"reward={reward:.4f} "
        f"|dq|_mean={np.abs(qpos_delta).mean():.5f} |a-q|_mean={np.abs(action_gap).mean():.5f}"
    )
    print(
        f"[debug:{name}] qpos_before[:8]={np.array2string(qpos_before[:8], precision=4, separator=', ')}"
    )
    print(
        f"[debug:{name}] action    [:8]={np.array2string(env_action[:8], precision=4, separator=', ')}"
    )
    print(
        f"[debug:{name}] qpos_after [:8]={np.array2string(qpos_after[:8], precision=4, separator=', ')}"
    )
    print(
        f"[debug:{name}] delta     [:8]={np.array2string(qpos_delta[:8], precision=4, separator=', ')}"
    )


def save_debug_frame(output_dir: Path, policy_label: str, ep_idx: int, step_idx: int, image: np.ndarray) -> None:
    frame_dir = output_dir / "debug_frames" / policy_label / f"ep_{ep_idx:03d}"
    frame_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(frame_dir / f"step_{step_idx:03d}.png")


def run_aloha_rollouts(
    model: OctoModel,
    *,
    task: str,
    rollouts: int,
    max_steps: int,
    seed: int,
    action_adapter: Callable[[np.ndarray], np.ndarray],
    policy_label: str = "policy",
    action_horizon: int = 50,
    image_size: int = 256,
    instruction: Optional[str] = None,
    debug_action_steps: int = 0,
    debug_save_frames: bool = False,
    output_dir: Optional[Path] = None,
) -> RolloutSummary:
    # Infer model metadata
    obs_spec = model.example_batch["observation"]
    history_len = int(obs_spec["timestep_pad_mask"].shape[1])
    task_completed_dim = int(obs_spec["task_completed"].shape[-1])

    dataset_stats = model.dataset_statistics
    action_stats = (
        dataset_stats["action"]
        if "action" in dataset_stats
        else next(iter(dataset_stats.values()))["action"]
    )
    # proprio normalization statistics (mirrors NormalizeProprio wrapper used during eval)
    proprio_stats = dataset_stats.get("proprio", None)

    successes = 0
    total_steps: list[int] = []
    infer_times: list[float] = []

    rng = jax.random.PRNGKey(seed)

    for ep_idx in range(rollouts):
        env = make_aloha_env(task, seed=seed + ep_idx)
        raw_obs, _ = env.reset()  # seed set at env init; dm_env.reset() takes no seed arg
        gym_task = model.create_tasks(texts=[instruction or task_to_instruction(task)])
        episode_success = False
        history = init_history(obs_to_octo(raw_obs, image_size=image_size, proprio_stats=proprio_stats), history_len)

        step_idx = 0
        while step_idx < max_steps:
            history_obs = stack_history(history)
            obs_aug = augment_obs_for_octo(
                history_obs,
                history_len=history_len,
                task_completed_dim=task_completed_dim,
            )

            rng, sample_rng = jax.random.split(rng)
            t0 = time.perf_counter()
            raw_actions = model.sample_actions(
                jax.tree_map(lambda x: x[None], obs_aug),
                gym_task,
                unnormalization_statistics=action_stats,
                rng=sample_rng,
            )
            infer_ms = (time.perf_counter() - t0) * 1000.0
            infer_times.append(infer_ms)

            action_chunk = np.asarray(raw_actions[0])  # (action_horizon, 14)
            action_chunk = action_adapter(action_chunk)
            if ep_idx == 0 and step_idx < debug_action_steps:
                summarize_action(policy_label, step_idx, action_chunk, action_chunk[0].astype(np.float32))
            if debug_save_frames and output_dir is not None and ep_idx == 0 and step_idx < max(1, debug_action_steps):
                save_debug_frame(output_dir, policy_label, ep_idx, step_idx, history_obs["image_primary"][-1])

            # Mirror the reference Octo eval: execute the predicted action chunk
            # instead of only the first action.
            remaining_steps = max_steps - step_idx
            exec_horizon = min(len(action_chunk), remaining_steps)
            reward = 0.0
            done = False
            trunc = False
            for chunk_offset in range(exec_horizon):
                env_action = action_chunk[chunk_offset].astype(np.float32)
                qpos_before = np.asarray(raw_obs["proprio"], dtype=np.float32)
                raw_obs, reward, done, trunc, info = env.step(env_action)
                qpos_after = np.asarray(raw_obs["proprio"], dtype=np.float32)
                if ep_idx == 0 and step_idx < debug_action_steps:
                    summarize_transition(
                        policy_label,
                        step_idx=step_idx,
                        chunk_offset=chunk_offset,
                        qpos_before=qpos_before,
                        env_action=env_action,
                        qpos_after=qpos_after,
                        reward=float(reward),
                    )
                history.append(
                    obs_to_octo(raw_obs, image_size=image_size, proprio_stats=proprio_stats)
                )
                step_idx += 1

                # AlohaGymEnv (ACT sim) always returns done=False; success is tracked internally.
                if reward == env._env.task.max_reward:
                    episode_success = True
                    total_steps.append(step_idx)
                    break
                if done or trunc:
                    total_steps.append(step_idx)
                    break

            if episode_success or done or trunc:
                break
        else:
            total_steps.append(step_idx)

        # fallback: also accept env's internal success tracker
        if not episode_success and hasattr(env, "get_episode_metrics"):
            episode_success = bool(env.get_episode_metrics().get("success_rate", 0))

        successes += int(episode_success)
        env.close()

        print(
            f"[{policy_label}] ep {ep_idx+1}/{rollouts}: "
            f"success={episode_success}, steps={total_steps[-1]}"
        )

    return RolloutSummary(
        policy_name=policy_label,
        task=task,
        rollouts=rollouts,
        success_rate=successes / max(rollouts, 1),
        mean_steps=float(np.mean(total_steps)) if total_steps else 0.0,
        mean_infer_ms=float(np.mean(infer_times)) if infer_times else 0.0,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GVLA-Net vs baseline eval on ALOHA sim env."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to finetuned Octo checkpoint directory.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="AlohaInsertion-v0",
        choices=[
            "AlohaInsertion-v0",
            "AlohaCubeHandover-v0",
            "AlohaTransferCube-v0",
            "gym_aloha/AlohaInsertion-v0",
            "gym_aloha/AlohaTransferCube-v0",
        ],
    )
    parser.add_argument("--rollouts", type=int, default=50)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gvla_bins", type=int, default=1 << 20)
    parser.add_argument(
        "--policy_mode",
        type=str,
        default="both",
        choices=["both", "baseline", "gvla"],
        help="Which policy branch to run.",
    )
    parser.add_argument("--instruction", type=str, default=None)
    parser.add_argument("--debug_action_steps", type=int, default=0)
    parser.add_argument("--debug_save_frames", action="store_true")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "octo_aloha_eval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint_root, checkpoint_step = resolve_checkpoint_path_and_step(
        args.checkpoint_path
    )
    print(
        f"Loading finetuned Octo checkpoint: root={checkpoint_root}, step={checkpoint_step}"
    )
    model = OctoModel.load_pretrained(checkpoint_root, step=checkpoint_step)

    # Print proprio stats for sanity check
    ds = model.dataset_statistics
    if "proprio" in ds:
        prop_mean = np.array(ds["proprio"]["mean"])
        prop_std = np.array(ds["proprio"]["std"])
        print("Proprio normalization stats (mean[:4], std[:4]):")
        print(f"  mean={np.array2string(prop_mean[:4], precision=4, separator=', ')}")
        print(f"  std ={np.array2string(prop_std[:4], precision=4, separator=', ')}")
    else:
        print("WARNING: no proprio stats in dataset_statistics — proprio will NOT be normalized")

    common_kwargs = dict(
        task=args.task,
        rollouts=args.rollouts,
        max_steps=args.max_steps,
        seed=args.seed,
        instruction=args.instruction,
        debug_action_steps=args.debug_action_steps,
        debug_save_frames=args.debug_save_frames,
        output_dir=args.output_dir,
    )

    rows = []
    if args.policy_mode in ("both", "baseline"):
        print(f"\n--- Baseline (continuous L1 head) ---")
        baseline = run_aloha_rollouts(
            model,
            action_adapter=identity_adapter,
            policy_label="octo_aloha_baseline",
            **common_kwargs,
        )
        rows.append(baseline)

    if args.policy_mode in ("both", "gvla"):
        print(f"\n--- GVLA (bins={args.gvla_bins}) ---")
        gvla = run_aloha_rollouts(
            model,
            action_adapter=ActionPrecisionAdapter(args.gvla_bins),
            policy_label=f"octo_aloha_gvla_{args.gvla_bins}",
            **common_kwargs,
        )
        rows.append(gvla)

    # Save CSV
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = args.output_dir / f"{timestamp}_aloha_eval.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(RolloutSummary.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)

    print(f"\n=== Results ===")
    for row in rows:
        print(
            f"{row.policy_name}: success_rate={row.success_rate:.3f}, "
            f"mean_steps={row.mean_steps:.1f}, mean_infer_ms={row.mean_infer_ms:.1f}"
        )
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
