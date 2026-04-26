import argparse
import csv
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Callable

import jax
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_OCTO = PROJECT_ROOT / "third_party" / "octo"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIRD_PARTY_OCTO) not in sys.path:
    sys.path.insert(0, str(THIRD_PARTY_OCTO))

from experiments.octo_lightweight_eval_env import OctoLightweightReachEnv
from octo.model.octo_model import OctoModel
from octo.utils.gym_wrappers import HistoryWrapper


@dataclass
class RolloutSummary:
    policy_name: str
    rollouts: int
    success_rate: float
    mean_return: float
    mean_steps: float
    mean_final_distance: float
    mean_infer_ms: float


class ActionPrecisionAdapter:
    """Quantize continuous actions onto a fixed lattice."""

    def __init__(self, bins: int) -> None:
        if bins < 2:
            raise ValueError(f"bins must be >= 2, got {bins}")
        self.bins = int(bins)

    def __call__(self, actions: np.ndarray) -> np.ndarray:
        clipped = np.clip(actions, -1.0, 1.0)
        scaled = (clipped + 1.0) * 0.5 * (self.bins - 1)
        quantized = np.round(scaled) / (self.bins - 1)
        return quantized * 2.0 - 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lightweight Octo rollout harness with coarse-vs-GVLA action precision adapters."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="hf://rail-berkeley/octo-base-1.5",
        help="Octo checkpoint path or hf:// identifier.",
    )
    parser.add_argument("--rollouts", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--history-horizon", type=int, default=None)
    parser.add_argument("--baseline-bins", type=int, default=256)
    parser.add_argument("--gvla-bins", type=int, default=1 << 20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--debug-action-steps", type=int, default=0)
    parser.add_argument(
        "--independent-seeds",
        action="store_true",
        default=False,
        help=(
            "Use seed+10000 for GVLA rollouts (independent episodes from baseline). "
            "Default is False: both policies run on the SAME episodes for fair comparison."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "octo_gvla_rollout",
    )
    parser.add_argument("--run-name", type=str, default="octo_lightweight_rollout")
    return parser.parse_args()


def build_run_dir(output_root: Path, run_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return output_root / f"{timestamp}_{run_name}"


def infer_model_metadata(model: OctoModel) -> tuple[int, int, int, int, int, int]:
    observation_spec = model.example_batch["observation"]
    proprio_spec = observation_spec.get("proprio")
    proprio_dim = int(proprio_spec.shape[-1]) if proprio_spec is not None else 0

    primary_spec = observation_spec.get("image_primary")
    wrist_spec = observation_spec.get("image_wrist")
    if primary_spec is None:
        raise KeyError(
            f"Could not find 'image_primary' in example_batch['observation']; "
            f"available keys: {sorted(observation_spec.keys())}"
        )
    image_size = int(primary_spec.shape[-2] if primary_spec.shape[-1] == 3 else primary_spec.shape[-3])
    wrist_image_size = (
        int(wrist_spec.shape[-2] if wrist_spec.shape[-1] == 3 else wrist_spec.shape[-3])
        if wrist_spec is not None
        else image_size
    )
    history_horizon = int(observation_spec["timestep_pad_mask"].shape[1])
    task_completed_dim = int(observation_spec["task_completed"].shape[-1])

    dataset_stats = model.dataset_statistics
    action_stats = dataset_stats["action"] if "action" in dataset_stats else next(iter(dataset_stats.values()))["action"]
    action_dim = int(len(action_stats["mean"]))
    return proprio_dim, image_size, wrist_image_size, action_dim, history_horizon, task_completed_dim


def augment_observation(obs: dict, *, task_completed_dim: int) -> dict:
    augmented = dict(obs)
    horizon = int(np.asarray(augmented["timestep_pad_mask"]).shape[0])
    timestep_mask = np.asarray(augmented["timestep_pad_mask"], dtype=bool)
    augmented["timestep"] = np.arange(horizon, dtype=np.int32)
    augmented["task_completed"] = np.zeros((horizon, task_completed_dim), dtype=bool)
    augmented["pad_mask_dict"] = {
        "image_primary": timestep_mask.copy(),
        "image_wrist": timestep_mask.copy(),
        "timestep": timestep_mask.copy(),
    }
    return augmented


def sample_action_chunk(
    model: OctoModel,
    obs: dict,
    task: dict,
    *,
    rng: jax.Array,
    action_stats: dict,
) -> tuple[np.ndarray, jax.Array, float]:
    rng, sample_rng = jax.random.split(rng)
    t0 = time.perf_counter()
    actions = model.sample_actions(
        jax.tree_map(lambda x: x[None], obs),
        task,
        unnormalization_statistics=action_stats,
        rng=sample_rng,
    )
    infer_ms = (time.perf_counter() - t0) * 1000.0
    return np.asarray(actions[0]), rng, infer_ms


def summarize_action(action: np.ndarray) -> str:
    flat = np.asarray(action, dtype=np.float32).reshape(-1)
    preview = np.array2string(flat[: min(8, len(flat))], precision=4, separator=", ")
    return (
        f"shape={tuple(action.shape)}, min={flat.min():.4f}, max={flat.max():.4f}, "
        f"mean={flat.mean():.4f}, std={flat.std():.4f}, preview={preview}"
    )


def run_policy_rollouts(
    model: OctoModel,
    *,
    rollouts: int,
    max_steps: int,
    history_horizon: int,
    seed: int,
    action_adapter: Callable[[np.ndarray], np.ndarray],
    debug_action_steps: int = 0,
    policy_label: str = "policy",
) -> RolloutSummary:
    proprio_dim, image_size, wrist_image_size, action_dim, model_horizon, task_completed_dim = infer_model_metadata(model)
    history_horizon = history_horizon if history_horizon is not None else model_horizon
    env = OctoLightweightReachEnv(
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        image_size=image_size,
        wrist_image_size=wrist_image_size,
        max_steps=max_steps,
        seed=seed,
    )
    env = HistoryWrapper(env, horizon=history_horizon)
    dataset_stats = model.dataset_statistics
    action_stats = dataset_stats["action"] if "action" in dataset_stats else next(iter(dataset_stats.values()))["action"]

    successes = 0
    returns: list[float] = []
    steps: list[int] = []
    final_distances: list[float] = []
    infer_times: list[float] = []
    rng = jax.random.PRNGKey(seed)

    for rollout_idx in range(rollouts):
        obs, _ = env.reset(seed=seed + rollout_idx)
        task = model.create_tasks(texts=[env.unwrapped.get_task()["language_instruction"]])
        episode_return = 0.0
        last_info = {"distance": np.nan, "success": False}

        for step_idx in range(max_steps):
            obs = augment_observation(obs, task_completed_dim=task_completed_dim)
            raw_action_chunk, rng, infer_ms = sample_action_chunk(
                model,
                obs,
                task,
                rng=rng,
                action_stats=action_stats,
            )
            infer_times.append(infer_ms)
            action_chunk = action_adapter(raw_action_chunk)
            if rollout_idx == 0 and step_idx < debug_action_steps:
                print(
                    f"[debug] step={step_idx} raw_action {summarize_action(raw_action_chunk)}"
                )
                print(
                    f"[debug] step={step_idx} adapted_action {summarize_action(action_chunk)}"
                )
            # Octo predicts an action chunk with shape (pred_horizon, action_dim).
            # This lightweight env executes a single control step at a time, so we
            # consume only the first action in the chunk.
            env_action = action_chunk[0] if np.asarray(action_chunk).ndim > 1 else action_chunk

            # --- DEBUG START ---
            if rollout_idx == 0 and step_idx < max(debug_action_steps, 3):
                ea = np.asarray(env_action, dtype=np.float32).reshape(-1)
                print(f"=== DEBUG ACTION ===")
                print(f"Model: {policy_label}")
                print(f"Action Type: {type(env_action)}, Shape: {ea.shape}")
                print(f"Action Values (first 8 dims): {ea[:8]}")
                print(f"Action range: min={ea.min():.4f}  max={ea.max():.4f}")
                print(f"====================")
            # --- DEBUG END ---

            obs, reward, done, trunc, info = env.step(env_action)
            episode_return += float(reward)
            last_info = info
            if done or trunc:
                steps.append(step_idx + 1)
                break
        else:
            steps.append(max_steps)

        successes += int(bool(last_info["success"]))
        returns.append(episode_return)
        final_distances.append(float(last_info["distance"]))

    return RolloutSummary(
        policy_name="",
        rollouts=rollouts,
        success_rate=successes / max(rollouts, 1),
        mean_return=float(np.mean(returns)) if returns else 0.0,
        mean_steps=float(np.mean(steps)) if steps else 0.0,
        mean_final_distance=float(np.mean(final_distances)) if final_distances else 0.0,
        mean_infer_ms=float(np.mean(infer_times)) if infer_times else 0.0,
    )


def write_summary_csv(rows: list[RolloutSummary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(RolloutSummary.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def main() -> None:
    args = parse_args()
    run_dir = build_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    model = OctoModel.load_pretrained(args.model_path)

    coarse_adapter = ActionPrecisionAdapter(args.baseline_bins)
    gvla_adapter = ActionPrecisionAdapter(args.gvla_bins)

    baseline = run_policy_rollouts(
        model,
        rollouts=args.rollouts,
        max_steps=args.max_steps,
        history_horizon=args.history_horizon,
        seed=args.seed,
        action_adapter=coarse_adapter,
        debug_action_steps=args.debug_action_steps,
        policy_label=f"octo_coarse_{args.baseline_bins}",
    )
    baseline.policy_name = f"octo_coarse_{args.baseline_bins}"

    # Use the SAME seed as baseline by default so both policies evaluate on
    # identical episodes (same env init, same model RNG).  Pass
    # --independent-seeds to revert to the old seed+10000 behaviour.
    gvla_seed = args.seed + 10_000 if args.independent_seeds else args.seed
    gvla = run_policy_rollouts(
        model,
        rollouts=args.rollouts,
        max_steps=args.max_steps,
        history_horizon=args.history_horizon,
        seed=gvla_seed,
        action_adapter=gvla_adapter,
        debug_action_steps=args.debug_action_steps,
        policy_label=f"octo_gvla_{args.gvla_bins}",
    )
    gvla.policy_name = f"octo_gvla_{args.gvla_bins}"

    rows = [baseline, gvla]
    csv_path = run_dir / "octo_gvla_rollout_summary.csv"
    write_summary_csv(rows, csv_path)

    print(f"Rollout CSV written to: {csv_path}")
    for row in rows:
        print(
            f"{row.policy_name}: success_rate={row.success_rate:.3f}, "
            f"mean_return={row.mean_return:.3f}, mean_steps={row.mean_steps:.2f}, "
            f"mean_final_distance={row.mean_final_distance:.4f}, mean_infer_ms={row.mean_infer_ms:.3f}"
        )


if __name__ == "__main__":
    main()
