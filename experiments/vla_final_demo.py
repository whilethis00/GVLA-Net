import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.layers import OrthogonalProjectionLayer
from utils.geometry import initialize_orthogonal_basis

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Continuous-action GVLA demo with tracking and FPS measurement."
    )
    parser.add_argument("--num-actions", type=int, default=65536)
    parser.add_argument("--input-dim", type=int, default=64)
    parser.add_argument("--episode-steps", type=int, default=1500)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--action-scale", type=float, default=0.075)
    parser.add_argument("--control-gain", type=float, default=1.8)
    parser.add_argument("--max-speed", type=float, default=1.0)
    parser.add_argument("--target-radius", type=float, default=1.0)
    parser.add_argument("--target-angular-speed", type=float, default=0.9)
    parser.add_argument("--noise-std", type=float, default=0.015)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=("float16", "float32", "float64"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "vla_final_demo",
    )
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return mapping[dtype_name]


def build_run_dir(output_root: Path, run_name: str | None) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = run_name if run_name is not None else "continuous_tracking"
    return output_root / f"{timestamp}_{safe_name}"


def write_args_snapshot(args: argparse.Namespace, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for key, value in sorted(vars(args).items()):
            handle.write(f"{key}: {value}\n")


def build_binary_codebook(
    num_actions: int,
    code_length: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    indices = torch.arange(num_actions, device=device, dtype=torch.long)
    bit_positions = torch.arange(code_length, device=device, dtype=torch.long)
    bits = ((indices.unsqueeze(1) >> bit_positions) & 1).to(torch.float32)
    return bits.mul(2.0).sub(1.0)


def normalize_rows(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return matrix / matrix.norm(dim=1, keepdim=True).clamp_min(eps)


def chunked_argmax_similarity(
    queries: torch.Tensor,
    candidates: torch.Tensor,
    *,
    chunk_size: int,
) -> torch.Tensor:
    best_scores = None
    best_indices = None
    start = 0

    while start < candidates.size(0):
        stop = min(start + chunk_size, candidates.size(0))
        scores = queries @ candidates[start:stop].transpose(0, 1)
        local_scores, local_indices = scores.max(dim=1)
        local_indices = local_indices + start

        if best_scores is None:
            best_scores = local_scores
            best_indices = local_indices
        else:
            better = local_scores > best_scores
            best_scores = torch.where(better, local_scores, best_scores)
            best_indices = torch.where(better, local_indices, best_indices)
        start = stop

    return best_indices


def target_trajectory(steps: int, dt: float, radius: float, angular_speed: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    time_axis = torch.arange(steps, device=device, dtype=dtype) * dt
    x = radius * torch.cos(angular_speed * time_axis)
    y = radius * torch.sin(2.0 * angular_speed * time_axis)
    return torch.stack((x, y), dim=1)


def build_action_bank(
    codebook: torch.Tensor,
    basis: torch.Tensor,
    *,
    input_dim: int,
    action_scale: float,
    max_speed: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_actions = codebook.size(0)
    relevant = codebook @ basis
    action_raw = relevant[:, :2]
    action_vectors = normalize_rows(action_raw) * action_scale

    embeddings = torch.zeros(
        num_actions,
        input_dim,
        device=codebook.device,
        dtype=basis.dtype,
    )
    embeddings[:, : codebook.size(1)] = codebook
    embeddings[:, -2:] = action_vectors / max(action_scale, 1e-6)
    return action_vectors, embeddings


def encode_control_state(
    error_vector: torch.Tensor,
    velocity: torch.Tensor,
    input_dim: int,
    max_speed: float,
) -> torch.Tensor:
    state = torch.zeros(
        error_vector.size(0),
        input_dim,
        device=error_vector.device,
        dtype=error_vector.dtype,
    )
    state[:, :2] = error_vector
    state[:, 2:4] = velocity / max(max_speed, 1e-6)
    return state


def run_controller(
    controller_name: str,
    layer: OrthogonalProjectionLayer,
    action_vectors: torch.Tensor,
    action_embeddings: torch.Tensor,
    target_positions: torch.Tensor,
    *,
    input_dim: int,
    dt: float,
    control_gain: float,
    max_speed: float,
    noise_std: float,
    chunk_size: int,
    warmup_steps: int,
) -> Dict[str, object]:
    device = target_positions.device
    dtype = target_positions.dtype
    num_steps = target_positions.size(0)

    position = torch.zeros(2, device=device, dtype=dtype)
    velocity = torch.zeros(2, device=device, dtype=dtype)

    position_trace: List[torch.Tensor] = []
    action_trace: List[torch.Tensor] = []
    error_trace: List[float] = []
    inference_times_ms: List[float] = []

    with torch.inference_mode():
        for step in range(num_steps):
            target = target_positions[step]
            observation_noise = noise_std * torch.randn_like(position)
            perceived_error = control_gain * (target - position + observation_noise)
            control_state = encode_control_state(
                perceived_error.unsqueeze(0),
                velocity.unsqueeze(0),
                input_dim=input_dim,
                max_speed=max_speed,
            )

            if device.type == "cuda":
                torch.cuda.synchronize(device=device)
            start_time = time.perf_counter()
            if controller_name == "gvla":
                outputs = layer(control_state)
                signed_code = torch.where(
                    outputs["signed_code"] >= 0,
                    torch.ones_like(outputs["signed_code"]),
                    -torch.ones_like(outputs["signed_code"]),
                )
                action_index = chunked_argmax_similarity(
                    signed_code,
                    action_embeddings[:, : layer.basis_size],
                    chunk_size=chunk_size,
                )[0]
            else:
                action_index = chunked_argmax_similarity(
                    control_state,
                    action_embeddings,
                    chunk_size=chunk_size,
                )[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device=device)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            action = action_vectors[action_index]
            velocity = action / max(dt, 1e-6)
            velocity_norm = velocity.norm()
            if velocity_norm > max_speed:
                velocity = velocity * (max_speed / velocity_norm)
            position = position + velocity * dt

            position_trace.append(position.clone())
            action_trace.append(action.clone())
            error_trace.append((target - position).norm().item())
            if step >= warmup_steps:
                inference_times_ms.append(elapsed_ms)

    mean_latency_ms = sum(inference_times_ms) / len(inference_times_ms)
    fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else float("inf")

    return {
        "positions": torch.stack(position_trace, dim=0),
        "actions": torch.stack(action_trace, dim=0),
        "errors": error_trace,
        "mean_error": sum(error_trace) / len(error_trace),
        "max_error": max(error_trace),
        "mean_latency_ms": mean_latency_ms,
        "fps": fps,
    }


def write_summary_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("controller", "mean_error", "max_error", "mean_latency_ms", "fps"),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_trace_csv(
    target_positions: torch.Tensor,
    gvla_result: Dict[str, object],
    dense_result: Dict[str, object],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gvla_positions = gvla_result["positions"]
    dense_positions = dense_result["positions"]
    gvla_errors = gvla_result["errors"]
    dense_errors = dense_result["errors"]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "step",
                "target_x",
                "target_y",
                "gvla_x",
                "gvla_y",
                "dense_x",
                "dense_y",
                "gvla_error",
                "dense_error",
            ),
        )
        writer.writeheader()
        for step in range(target_positions.size(0)):
            writer.writerow(
                {
                    "step": step,
                    "target_x": target_positions[step, 0].item(),
                    "target_y": target_positions[step, 1].item(),
                    "gvla_x": gvla_positions[step, 0].item(),
                    "gvla_y": gvla_positions[step, 1].item(),
                    "dense_x": dense_positions[step, 0].item(),
                    "dense_y": dense_positions[step, 1].item(),
                    "gvla_error": gvla_errors[step],
                    "dense_error": dense_errors[step],
                }
            )


def plot_results(
    target_positions: torch.Tensor,
    gvla_result: Dict[str, object],
    dense_result: Dict[str, object],
    output_path: Path,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install it to generate plots.")

    gvla_positions = gvla_result["positions"].cpu()
    dense_positions = dense_result["positions"].cpu()
    target_positions = target_positions.cpu()
    steps = list(range(target_positions.size(0)))

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(target_positions[:, 0], target_positions[:, 1], label="Target", linewidth=2.0)
    axes[0].plot(gvla_positions[:, 0], gvla_positions[:, 1], label="GVLA", linewidth=1.5)
    axes[0].plot(dense_positions[:, 0], dense_positions[:, 1], label="Dense baseline", linewidth=1.5)
    axes[0].set_title("Continuous Tracking Trajectory")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(steps, gvla_result["errors"], label=f"GVLA ({gvla_result['fps']:.1f} FPS)")
    axes[1].plot(steps, dense_result["errors"], label=f"Dense ({dense_result['fps']:.1f} FPS)")
    axes[1].set_title("Tracking Error Over Time")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Position error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.suptitle("GVLA-Net Continuous Action Demo")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    code_length = int(math.ceil(math.log2(args.num_actions)))
    if code_length > args.input_dim:
        raise ValueError(
            "Need ceil(log2(num_actions)) <= input_dim for the geometric observer, "
            f"got {code_length} > {args.input_dim}."
        )

    run_dir = build_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_args_snapshot(args, run_dir / "args.txt")

    basis = initialize_orthogonal_basis(
        d=args.input_dim,
        k=code_length,
        device=device,
        dtype=dtype,
    )
    codebook = build_binary_codebook(
        args.num_actions,
        code_length,
        device=device,
    ).to(dtype=dtype)
    action_vectors, action_embeddings = build_action_bank(
        codebook,
        basis,
        input_dim=args.input_dim,
        action_scale=args.action_scale,
        max_speed=args.max_speed,
    )

    gvla_layer = OrthogonalProjectionLayer(
        input_dim=args.input_dim,
        num_codes=args.num_actions,
        use_ste=False,
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        gvla_layer.weight.copy_(basis)

    target_positions = target_trajectory(
        args.episode_steps,
        args.dt,
        args.target_radius,
        args.target_angular_speed,
        device,
        dtype,
    )

    gvla_result = run_controller(
        "gvla",
        gvla_layer,
        action_vectors,
        action_embeddings,
        target_positions,
        input_dim=args.input_dim,
        dt=args.dt,
        control_gain=args.control_gain,
        max_speed=args.max_speed,
        noise_std=args.noise_std,
        chunk_size=args.chunk_size,
        warmup_steps=args.warmup_steps,
    )
    dense_result = run_controller(
        "dense",
        gvla_layer,
        action_vectors,
        action_embeddings,
        target_positions,
        input_dim=args.input_dim,
        dt=args.dt,
        control_gain=args.control_gain,
        max_speed=args.max_speed,
        noise_std=args.noise_std,
        chunk_size=args.chunk_size,
        warmup_steps=args.warmup_steps,
    )

    summary_rows = [
        {
            "controller": "gvla",
            "mean_error": gvla_result["mean_error"],
            "max_error": gvla_result["max_error"],
            "mean_latency_ms": gvla_result["mean_latency_ms"],
            "fps": gvla_result["fps"],
        },
        {
            "controller": "dense",
            "mean_error": dense_result["mean_error"],
            "max_error": dense_result["max_error"],
            "mean_latency_ms": dense_result["mean_latency_ms"],
            "fps": dense_result["fps"],
        },
    ]

    summary_path = run_dir / "summary.csv"
    trace_path = run_dir / "tracking_trace.csv"
    figure_path = run_dir / "tracking_demo.png"

    write_summary_csv(summary_rows, summary_path)
    write_trace_csv(target_positions, gvla_result, dense_result, trace_path)
    plot_results(target_positions, gvla_result, dense_result, figure_path)

    speedup = gvla_result["fps"] / dense_result["fps"] if dense_result["fps"] > 0 else float("inf")
    print(f"Summary written to: {summary_path}")
    print(f"Trace written to: {trace_path}")
    print(f"Plot written to: {figure_path}")
    print(f"GVLA FPS: {gvla_result['fps']:.2f}")
    print(f"Dense baseline FPS: {dense_result['fps']:.2f}")
    print(f"GVLA speedup over dense baseline: {speedup:.2f}x")
    print(f"GVLA mean tracking error: {gvla_result['mean_error']:.4f}")
    print(f"Dense mean tracking error: {dense_result['mean_error']:.4f}")


if __name__ == "__main__":
    main()
