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
    parser.add_argument("--history-horizon", type=int, default=1)
    parser.add_argument("--baseline-bins", type=int, default=256)
    parser.add_argument("--gvla-bins", type=int, default=1 << 20)
    parser.add_argument("--seed", type=int, default=0)
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


def infer_model_metadata(model: OctoModel) -> tuple[int, int, int]:
    observation_spec = model.example_batch["observation"]
    proprio_spec = observation_spec.get("proprio")
    proprio_dim = int(proprio_spec.shape[-1]) if proprio_spec is not None else 0

    image_key = next((key for key in observation_spec if key.startswith("image_")), None)
    if image_key is None:
        raise KeyError(
            f"Could not find an image observation key in example_batch['observation']; "
            f"available keys: {sorted(observation_spec.keys())}"
        )

    image_shape = observation_spec[image_key].shape
    image_size = int(image_shape[-2] if image_shape[-1] == 3 else image_shape[-3])

    dataset_stats = model.dataset_statistics
    action_stats = dataset_stats["action"] if "action" in dataset_stats else next(iter(dataset_stats.values()))["action"]
    action_dim = int(len(action_stats["mean"]))
    return proprio_dim, image_size, action_dim


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


def run_policy_rollouts(
    model: OctoModel,
    *,
    rollouts: int,
    max_steps: int,
    history_horizon: int,
    seed: int,
    action_adapter: Callable[[np.ndarray], np.ndarray],
) -> RolloutSummary:
    proprio_dim, image_size, action_dim = infer_model_metadata(model)
    env = OctoLightweightReachEnv(
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        image_size=image_size,
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
            action_chunk, rng, infer_ms = sample_action_chunk(
                model,
                obs,
                task,
                rng=rng,
                action_stats=action_stats,
            )
            infer_times.append(infer_ms)
            action_chunk = action_adapter(action_chunk)
            obs, reward, done, trunc, info = env.step(action_chunk)
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
    )
    baseline.policy_name = f"octo_coarse_{args.baseline_bins}"

    gvla = run_policy_rollouts(
        model,
        rollouts=args.rollouts,
        max_steps=args.max_steps,
        history_horizon=args.history_horizon,
        seed=args.seed + 10_000,
        action_adapter=gvla_adapter,
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
