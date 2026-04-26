import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

try:
    import matplotlib.pyplot as plt
    from matplotlib import animation
except ImportError:  # pragma: no cover - optional runtime dependency
    plt = None
    animation = None

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.layers import OrthogonalProjectionLayer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="2-DOF robot arm demo with GVLA geometric hashing vs dense selection."
    )
    parser.add_argument("--joint1-bins", type=int, default=512)
    parser.add_argument("--joint2-bins", type=int, default=256)
    parser.add_argument("--max-action-delta", type=float, default=0.03)
    parser.add_argument("--episode-steps", type=int, default=600)
    parser.add_argument("--dt", type=float, default=0.02)
    parser.add_argument("--control-gain", type=float, default=1.2)
    parser.add_argument("--arrival-threshold", type=float, default=0.03)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--target-x", type=float, default=1.25)
    parser.add_argument("--target-y", type=float, default=0.85)
    parser.add_argument("--init-theta1", type=float, default=0.15)
    parser.add_argument("--init-theta2", type=float, default=-1.25)
    parser.add_argument("--link1", type=float, default=1.0)
    parser.add_argument("--link2", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=0.002)
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
        default=PROJECT_ROOT / "experiments" / "results" / "robot_arm_tracking",
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
    safe_name = run_name if run_name is not None else "two_dof_demo"
    return output_root / f"{timestamp}_{safe_name}"


def write_args_snapshot(args: argparse.Namespace, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for key, value in sorted(vars(args).items()):
            handle.write(f"{key}: {value}\n")


def forward_kinematics(joint_angles: torch.Tensor, link_lengths: torch.Tensor) -> torch.Tensor:
    theta1 = joint_angles[..., 0]
    theta2 = joint_angles[..., 1]
    x = link_lengths[0] * torch.cos(theta1) + link_lengths[1] * torch.cos(theta1 + theta2)
    y = link_lengths[0] * torch.sin(theta1) + link_lengths[1] * torch.sin(theta1 + theta2)
    return torch.stack((x, y), dim=-1)


def joint_positions(joint_angles: torch.Tensor, link_lengths: torch.Tensor) -> torch.Tensor:
    theta1 = joint_angles[..., 0]
    theta2 = joint_angles[..., 1]

    elbow_x = link_lengths[0] * torch.cos(theta1)
    elbow_y = link_lengths[0] * torch.sin(theta1)
    wrist = forward_kinematics(joint_angles, link_lengths)

    origin = torch.zeros_like(wrist)
    elbow = torch.stack((elbow_x, elbow_y), dim=-1)
    return torch.stack((origin, elbow, wrist), dim=-2)


def jacobian(joint_angles: torch.Tensor, link_lengths: torch.Tensor) -> torch.Tensor:
    theta1 = joint_angles[..., 0]
    theta2 = joint_angles[..., 1]
    sin1 = torch.sin(theta1)
    cos1 = torch.cos(theta1)
    sin12 = torch.sin(theta1 + theta2)
    cos12 = torch.cos(theta1 + theta2)

    j11 = -link_lengths[0] * sin1 - link_lengths[1] * sin12
    j12 = -link_lengths[1] * sin12
    j21 = link_lengths[0] * cos1 + link_lengths[1] * cos12
    j22 = link_lengths[1] * cos12

    row1 = torch.stack((j11, j12), dim=-1)
    row2 = torch.stack((j21, j22), dim=-1)
    return torch.stack((row1, row2), dim=-2)


def desired_joint_delta(
    joint_angles: torch.Tensor,
    target: torch.Tensor,
    link_lengths: torch.Tensor,
    control_gain: float,
    max_action_delta: float,
) -> torch.Tensor:
    end_effector = forward_kinematics(joint_angles.unsqueeze(0), link_lengths)[0]
    error = target - end_effector
    jac = jacobian(joint_angles.unsqueeze(0), link_lengths)[0]
    delta = control_gain * torch.matmul(jac.transpose(0, 1), error)
    return delta.clamp(min=-max_action_delta, max=max_action_delta)


def build_action_space(
    joint1_bins: int,
    joint2_bins: int,
    max_action_delta: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    joint1_values = torch.linspace(
        -max_action_delta,
        max_action_delta,
        steps=joint1_bins,
        device=device,
        dtype=dtype,
    )
    joint2_values = torch.linspace(
        -max_action_delta,
        max_action_delta,
        steps=joint2_bins,
        device=device,
        dtype=dtype,
    )
    grid1, grid2 = torch.meshgrid(joint1_values, joint2_values, indexing="ij")
    action_bank = torch.stack((grid1.reshape(-1), grid2.reshape(-1)), dim=-1)
    return joint1_values, joint2_values, action_bank


def dense_select_action(
    desired_delta: torch.Tensor,
    action_bank: torch.Tensor,
    chunk_size: int,
) -> int:
    best_index = 0
    best_score = None
    start = 0
    query = desired_delta.unsqueeze(0)

    while start < action_bank.size(0):
        stop = min(start + chunk_size, action_bank.size(0))
        candidates = action_bank[start:stop]
        scores = -torch.sum((candidates - query) * (candidates - query), dim=-1)
        local_score, local_index = torch.max(scores, dim=0)
        local_score_value = float(local_score.item())
        if best_score is None or local_score_value > best_score:
            best_score = local_score_value
            best_index = start + int(local_index.item())
        start = stop

    return best_index


def binary_partition_features(
    normalized_value: torch.Tensor,
    num_bits: int,
) -> torch.Tensor:
    residual = normalized_value.clamp(min=0.0, max=1.0 - 1e-6)
    features: List[torch.Tensor] = []
    for _ in range(num_bits):
        signed_distance = residual - 0.5
        features.append(signed_distance)
        go_right = signed_distance >= 0
        residual = torch.where(go_right, 2.0 * residual - 1.0, 2.0 * residual)
    return torch.stack(features, dim=-1)


def encode_geometric_state(
    desired_delta: torch.Tensor,
    max_action_delta: float,
    joint1_bits: int,
    joint2_bits: int,
) -> torch.Tensor:
    normalized = (desired_delta / (2.0 * max_action_delta)) + 0.5
    features1 = binary_partition_features(normalized[0], joint1_bits)
    features2 = binary_partition_features(normalized[1], joint2_bits)
    return torch.cat((features1, features2), dim=0).unsqueeze(0)


def bits_to_index(bits: torch.Tensor) -> int:
    value = 0
    for bit in bits.tolist():
        value = (value << 1) | int(bit)
    return value


def gvla_select_action(
    desired_delta: torch.Tensor,
    layer: OrthogonalProjectionLayer,
    max_action_delta: float,
    joint1_bits: int,
    joint2_bits: int,
    joint2_bins: int,
) -> int:
    state = encode_geometric_state(
        desired_delta,
        max_action_delta,
        joint1_bits,
        joint2_bits,
    ).to(device=layer.weight.device, dtype=layer.weight.dtype)
    outputs = layer(state)
    bits = (outputs["signed_code"][0] >= 0).to(torch.long)
    index1 = bits_to_index(bits[:joint1_bits])
    index2 = bits_to_index(bits[joint1_bits:])
    return index1 * joint2_bins + index2


def initialize_hash_layer(
    num_actions: int,
    code_length: int,
    device: torch.device,
    dtype: torch.dtype,
) -> OrthogonalProjectionLayer:
    layer = OrthogonalProjectionLayer(
        input_dim=code_length,
        num_codes=num_actions,
        basis_size=code_length,
        use_ste=False,
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(code_length, device=device, dtype=dtype))
    return layer


def simulate_controller(
    mode: str,
    initial_angles: torch.Tensor,
    target: torch.Tensor,
    link_lengths: torch.Tensor,
    action_bank: torch.Tensor,
    hash_layer: OrthogonalProjectionLayer,
    *,
    joint1_bits: int,
    joint2_bits: int,
    joint2_bins: int,
    max_action_delta: float,
    control_gain: float,
    arrival_threshold: float,
    chunk_size: int,
    episode_steps: int,
    dt: float,
    noise_std: float,
) -> Dict[str, object]:
    joint_angles = initial_angles.clone()

    angle_trace: List[torch.Tensor] = []
    effector_trace: List[torch.Tensor] = []
    error_trace: List[float] = []
    inference_times_ms: List[float] = []
    arrival_step = None

    for step in range(episode_steps):
        noisy_target = target + noise_std * torch.randn_like(target)
        desired_delta = desired_joint_delta(
            joint_angles,
            noisy_target,
            link_lengths,
            control_gain,
            max_action_delta,
        )

        if joint_angles.device.type == "cuda":
            torch.cuda.synchronize(device=joint_angles.device)
        start_time = time.perf_counter()
        if mode == "gvla":
            action_index = gvla_select_action(
                desired_delta,
                hash_layer,
                max_action_delta,
                joint1_bits,
                joint2_bits,
                joint2_bins,
            )
        else:
            action_index = dense_select_action(
                desired_delta,
                action_bank,
                chunk_size,
            )
        if joint_angles.device.type == "cuda":
            torch.cuda.synchronize(device=joint_angles.device)
        inference_times_ms.append((time.perf_counter() - start_time) * 1000.0)

        joint_angles = joint_angles + action_bank[action_index]
        end_effector = forward_kinematics(joint_angles.unsqueeze(0), link_lengths)[0]
        error = torch.norm(target - end_effector).item()

        angle_trace.append(joint_angles.clone())
        effector_trace.append(end_effector.clone())
        error_trace.append(error)

        if arrival_step is None and error <= arrival_threshold:
            arrival_step = step

    mean_latency_ms = sum(inference_times_ms) / len(inference_times_ms)
    fps = 1000.0 / mean_latency_ms if mean_latency_ms > 0 else float("inf")
    return {
        "angles": torch.stack(angle_trace, dim=0),
        "end_effector": torch.stack(effector_trace, dim=0),
        "errors": error_trace,
        "arrival_step": arrival_step,
        "arrival_time_s": (arrival_step + 1) * dt if arrival_step is not None else None,
        "mean_latency_ms": mean_latency_ms,
        "fps": fps,
        "final_error": error_trace[-1],
    }


def write_summary_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "controller",
                "action_space_size",
                "arrival_step",
                "arrival_time_s",
                "mean_latency_ms",
                "fps",
                "final_error",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def write_trace_csv(
    target: torch.Tensor,
    gvla_result: Dict[str, object],
    dense_result: Dict[str, object],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gvla_angles = gvla_result["angles"]
    dense_angles = dense_result["angles"]
    gvla_ee = gvla_result["end_effector"]
    dense_ee = dense_result["end_effector"]

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "step",
                "target_x",
                "target_y",
                "gvla_theta1",
                "gvla_theta2",
                "gvla_x",
                "gvla_y",
                "dense_theta1",
                "dense_theta2",
                "dense_x",
                "dense_y",
                "gvla_error",
                "dense_error",
            ),
        )
        writer.writeheader()
        for step in range(gvla_angles.size(0)):
            writer.writerow(
                {
                    "step": step,
                    "target_x": target[0].item(),
                    "target_y": target[1].item(),
                    "gvla_theta1": gvla_angles[step, 0].item(),
                    "gvla_theta2": gvla_angles[step, 1].item(),
                    "gvla_x": gvla_ee[step, 0].item(),
                    "gvla_y": gvla_ee[step, 1].item(),
                    "dense_theta1": dense_angles[step, 0].item(),
                    "dense_theta2": dense_angles[step, 1].item(),
                    "dense_x": dense_ee[step, 0].item(),
                    "dense_y": dense_ee[step, 1].item(),
                    "gvla_error": gvla_result["errors"][step],
                    "dense_error": dense_result["errors"][step],
                }
            )


def plot_snapshot(
    target: torch.Tensor,
    link_lengths: torch.Tensor,
    gvla_result: Dict[str, object],
    dense_result: Dict[str, object],
    output_path: Path,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install it to generate plots.")

    gvla_positions = joint_positions(gvla_result["angles"][-1].unsqueeze(0), link_lengths)[0].cpu()
    dense_positions = joint_positions(dense_result["angles"][-1].unsqueeze(0), link_lengths)[0].cpu()
    gvla_path = gvla_result["end_effector"].cpu()
    dense_path = dense_result["end_effector"].cpu()
    target_cpu = target.cpu()

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, title, positions, path, result in (
        (axes[0], "GVLA Geometric Hash", gvla_positions, gvla_path, gvla_result),
        (axes[1], "Dense Softmax-style Baseline", dense_positions, dense_path, dense_result),
    ):
        axis.plot(path[:, 0], path[:, 1], linewidth=1.5, alpha=0.7)
        axis.plot(positions[:, 0], positions[:, 1], marker="o", linewidth=3.0)
        axis.scatter(target_cpu[0], target_cpu[1], marker="*", s=180)
        axis.set_title(
            f"{title}\nFPS={result['fps']:.1f}, arrival={format_arrival(result['arrival_time_s'])}"
        )
        axis.set_xlim(-1.9, 1.9)
        axis.set_ylim(-1.9, 1.9)
        axis.set_aspect("equal")
        axis.grid(True, alpha=0.3)

    figure.suptitle("2-DOF Target Tracking with 131,072 Discrete Actions")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_animation(
    target: torch.Tensor,
    link_lengths: torch.Tensor,
    gvla_result: Dict[str, object],
    dense_result: Dict[str, object],
    output_path: Path,
) -> None:
    if plt is None or animation is None:
        return

    gvla_angles = gvla_result["angles"].cpu()
    dense_angles = dense_result["angles"].cpu()
    target_cpu = target.cpu()

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    line_sets = []
    trail_sets = []
    text_sets = []

    for axis, title, result in (
        (axes[0], "GVLA", gvla_result),
        (axes[1], "Dense", dense_result),
    ):
        axis.set_xlim(-1.9, 1.9)
        axis.set_ylim(-1.9, 1.9)
        axis.set_aspect("equal")
        axis.grid(True, alpha=0.3)
        axis.scatter(target_cpu[0], target_cpu[1], marker="*", s=180)
        axis.set_title(title)
        arm_line, = axis.plot([], [], marker="o", linewidth=3.0)
        trail_line, = axis.plot([], [], linewidth=1.2, alpha=0.7)
        info_text = axis.text(-1.8, 1.55, "", fontsize=9, va="top")
        line_sets.append(arm_line)
        trail_sets.append(trail_line)
        text_sets.append((info_text, result))

    def update(frame: int):
        for idx, (angles, result) in enumerate(((gvla_angles, gvla_result), (dense_angles, dense_result))):
            positions = joint_positions(angles[frame].unsqueeze(0), link_lengths.cpu())[0]
            line_sets[idx].set_data(positions[:, 0].tolist(), positions[:, 1].tolist())
            path = result["end_effector"][: frame + 1].cpu()
            trail_sets[idx].set_data(path[:, 0].tolist(), path[:, 1].tolist())
            text_sets[idx][0].set_text(
                "step={step}\nerror={error:.3f}\nFPS={fps:.1f}\narrival={arrival}".format(
                    step=frame,
                    error=result["errors"][frame],
                    fps=result["fps"],
                    arrival=format_arrival(result["arrival_time_s"]),
                )
            )
        return line_sets + trail_sets + [item[0] for item in text_sets]

    anim = animation.FuncAnimation(
        figure,
        update,
        frames=gvla_angles.size(0),
        interval=40,
        blit=False,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        anim.save(str(output_path), writer=animation.PillowWriter(fps=25))
    finally:
        plt.close(figure)


def format_arrival(arrival_time_s: float | None) -> str:
    if arrival_time_s is None:
        return "not reached"
    return f"{arrival_time_s:.2f}s"


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    num_actions = args.joint1_bins * args.joint2_bins
    code_length = int(math.ceil(math.log2(num_actions)))
    joint1_bits = int(math.ceil(math.log2(args.joint1_bins)))
    joint2_bits = int(math.ceil(math.log2(args.joint2_bins)))

    if (1 << joint1_bits) != args.joint1_bins or (1 << joint2_bits) != args.joint2_bins:
        raise ValueError("joint bin counts must be powers of two for direct geometric decoding.")
    if code_length != joint1_bits + joint2_bits:
        raise ValueError("Expected code_length to match concatenated joint bits.")

    run_dir = build_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_args_snapshot(args, run_dir / "args.txt")

    link_lengths = torch.tensor([args.link1, args.link2], device=device, dtype=dtype)
    target = torch.tensor([args.target_x, args.target_y], device=device, dtype=dtype)
    initial_angles = torch.tensor(
        [args.init_theta1, args.init_theta2],
        device=device,
        dtype=dtype,
    )

    _, _, action_bank = build_action_space(
        args.joint1_bins,
        args.joint2_bins,
        args.max_action_delta,
        device,
        dtype,
    )
    hash_layer = initialize_hash_layer(
        num_actions,
        code_length,
        device,
        dtype,
    )

    gvla_result = simulate_controller(
        "gvla",
        initial_angles,
        target,
        link_lengths,
        action_bank,
        hash_layer,
        joint1_bits=joint1_bits,
        joint2_bits=joint2_bits,
        joint2_bins=args.joint2_bins,
        max_action_delta=args.max_action_delta,
        control_gain=args.control_gain,
        arrival_threshold=args.arrival_threshold,
        chunk_size=args.chunk_size,
        episode_steps=args.episode_steps,
        dt=args.dt,
        noise_std=args.noise_std,
    )
    dense_result = simulate_controller(
        "dense",
        initial_angles,
        target,
        link_lengths,
        action_bank,
        hash_layer,
        joint1_bits=joint1_bits,
        joint2_bits=joint2_bits,
        joint2_bins=args.joint2_bins,
        max_action_delta=args.max_action_delta,
        control_gain=args.control_gain,
        arrival_threshold=args.arrival_threshold,
        chunk_size=args.chunk_size,
        episode_steps=args.episode_steps,
        dt=args.dt,
        noise_std=args.noise_std,
    )

    summary_rows = [
        {
            "controller": "gvla",
            "action_space_size": num_actions,
            "arrival_step": gvla_result["arrival_step"],
            "arrival_time_s": gvla_result["arrival_time_s"],
            "mean_latency_ms": gvla_result["mean_latency_ms"],
            "fps": gvla_result["fps"],
            "final_error": gvla_result["final_error"],
        },
        {
            "controller": "dense",
            "action_space_size": num_actions,
            "arrival_step": dense_result["arrival_step"],
            "arrival_time_s": dense_result["arrival_time_s"],
            "mean_latency_ms": dense_result["mean_latency_ms"],
            "fps": dense_result["fps"],
            "final_error": dense_result["final_error"],
        },
    ]

    summary_path = run_dir / "summary.csv"
    trace_path = run_dir / "tracking_trace.csv"
    snapshot_path = run_dir / "robot_arm_snapshot.png"
    animation_path = run_dir / "robot_arm_tracking.gif"

    write_summary_csv(summary_rows, summary_path)
    write_trace_csv(target, gvla_result, dense_result, trace_path)
    plot_snapshot(target, link_lengths, gvla_result, dense_result, snapshot_path)
    try:
        save_animation(target, link_lengths, gvla_result, dense_result, animation_path)
    except Exception:
        pass

    speedup = (
        gvla_result["fps"] / dense_result["fps"] if dense_result["fps"] > 0 else float("inf")
    )
    print(f"Summary written to: {summary_path}")
    print(f"Trace written to: {trace_path}")
    print(f"Snapshot written to: {snapshot_path}")
    print(f"Animation path: {animation_path}")
    print(f"Action space size: {num_actions}")
    print(f"GVLA arrival time: {format_arrival(gvla_result['arrival_time_s'])}")
    print(f"Dense arrival time: {format_arrival(dense_result['arrival_time_s'])}")
    print(f"GVLA FPS: {gvla_result['fps']:.2f}")
    print(f"Dense FPS: {dense_result['fps']:.2f}")
    print(f"GVLA speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
