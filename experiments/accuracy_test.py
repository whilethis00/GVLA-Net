import argparse
import csv
import math
import sys
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
except ImportError:  # pragma: no cover - optional runtime dependency
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate GVLA action recovery accuracy and invariance under noise."
    )
    parser.add_argument("--num-actions", type=int, default=65536)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--margin-scale", type=float, default=4.0)
    parser.add_argument("--action-nuisance-scale", type=float, default=2.0)
    parser.add_argument("--hash-noise-ratio", type=float, default=0.15)
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=3.0)
    parser.add_argument("--noise-steps", type=int, default=7)
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
        default=PROJECT_ROOT / "experiments" / "results" / "accuracy_test",
    )
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return mapping[dtype_name]


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


def orthogonal_complement_noise(
    tensor: torch.Tensor,
    basis: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    projected = (tensor @ basis.transpose(0, 1)) @ basis
    residual = tensor - projected
    norms = residual.norm(dim=-1, keepdim=True).clamp_min(eps)
    return residual / norms


def build_action_bank(
    codebook: torch.Tensor,
    basis: torch.Tensor,
    *,
    margin_scale: float,
    nuisance_scale: float,
) -> torch.Tensor:
    relevant_component = margin_scale * (codebook @ basis)
    nuisance_seed = torch.randn_like(relevant_component)
    nuisance_component = nuisance_scale * orthogonal_complement_noise(
        nuisance_seed,
        basis,
    )
    return relevant_component + nuisance_component


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


def evaluate_noise_level(
    layer: OrthogonalProjectionLayer,
    action_bank: torch.Tensor,
    codebook: torch.Tensor,
    target_indices: torch.Tensor,
    *,
    noise_level: float,
    hash_noise_ratio: float,
    batch_size: int,
    chunk_size: int,
) -> Dict[str, float]:
    gvla_correct = 0
    dense_correct = 0
    invariant_hits = 0
    total_bit_flips = 0.0
    total_samples = target_indices.numel()

    basis = layer.weight.detach()

    with torch.inference_mode():
        for start in range(0, total_samples, batch_size):
            stop = min(start + batch_size, total_samples)
            batch_indices = target_indices[start:stop]
            clean_states = action_bank[batch_indices]

            nuisance_seed = torch.randn_like(clean_states)
            nuisance_noise = orthogonal_complement_noise(nuisance_seed, basis)

            hash_seed = torch.randn(
                clean_states.size(0),
                basis.size(0),
                device=clean_states.device,
                dtype=clean_states.dtype,
            )
            hash_noise = hash_seed @ basis

            noisy_states = (
                clean_states
                + noise_level * nuisance_noise
                + (noise_level * hash_noise_ratio) * hash_noise
            )

            gvla_outputs = layer(noisy_states)
            gvla_signed = torch.where(
                gvla_outputs["signed_code"] >= 0,
                torch.ones_like(gvla_outputs["signed_code"]),
                -torch.ones_like(gvla_outputs["signed_code"]),
            )

            gvla_pred = chunked_argmax_similarity(
                gvla_signed,
                codebook,
                chunk_size=chunk_size,
            )
            dense_pred = chunked_argmax_similarity(
                noisy_states,
                action_bank,
                chunk_size=chunk_size,
            )

            target_codes = codebook[batch_indices]
            bit_flip_rate = (gvla_signed != target_codes).to(torch.float32).mean(dim=1)

            gvla_correct += (gvla_pred == batch_indices).sum().item()
            dense_correct += (dense_pred == batch_indices).sum().item()
            invariant_hits += (gvla_signed == target_codes).all(dim=1).sum().item()
            total_bit_flips += bit_flip_rate.sum().item()

    return {
        "noise_level": noise_level,
        "gvla_accuracy": gvla_correct / total_samples,
        "dense_accuracy": dense_correct / total_samples,
        "invariant_rate": invariant_hits / total_samples,
        "mean_bit_flip_rate": total_bit_flips / total_samples,
    }


def write_results_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "noise_level",
                "gvla_accuracy",
                "dense_accuracy",
                "invariant_rate",
                "mean_bit_flip_rate",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: List[Dict[str, float]], output_path: Path) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is not installed. Install it to generate the invariance plot."
        )

    noise_levels = [row["noise_level"] for row in rows]
    gvla_accuracy = [row["gvla_accuracy"] for row in rows]
    dense_accuracy = [row["dense_accuracy"] for row in rows]
    invariant_rate = [row["invariant_rate"] for row in rows]
    bit_flip_rate = [row["mean_bit_flip_rate"] for row in rows]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(noise_levels, gvla_accuracy, marker="s", label="GVLA Recovery")
    axes[0].plot(noise_levels, dense_accuracy, marker="o", label="Dense Softmax-style")
    axes[0].set_title("Action Recovery Under Noise")
    axes[0].set_xlabel("Noise level")
    axes[0].set_ylabel("Top-1 accuracy")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(noise_levels, invariant_rate, marker="^", label="Hash invariant rate")
    axes[1].plot(noise_levels, bit_flip_rate, marker="d", label="Mean bit flip rate")
    axes[1].set_title("Geometric Invariance")
    axes[1].set_xlabel("Noise level")
    axes[1].set_ylabel("Rate")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.suptitle("GVLA-Net Invariance and Robustness Benchmark")
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
            "The geometric observer requires ceil(log2(num_actions)) <= input_dim, "
            f"got {code_length} > {args.input_dim}."
        )

    basis = initialize_orthogonal_basis(
        d=args.input_dim,
        k=code_length,
        device=device,
        dtype=dtype,
    )
    layer = OrthogonalProjectionLayer(
        input_dim=args.input_dim,
        num_codes=args.num_actions,
        use_ste=False,
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(basis)

    codebook = build_binary_codebook(
        args.num_actions,
        code_length,
        device=device,
    ).to(dtype=dtype)
    action_bank = build_action_bank(
        codebook,
        layer.weight.detach(),
        margin_scale=args.margin_scale,
        nuisance_scale=args.action_nuisance_scale,
    )
    target_indices = torch.randint(
        low=0,
        high=args.num_actions,
        size=(args.num_samples,),
        device=device,
        dtype=torch.long,
    )

    noise_levels = torch.linspace(
        args.noise_min,
        args.noise_max,
        steps=args.noise_steps,
        device=device,
        dtype=torch.float32,
    ).tolist()

    rows: List[Dict[str, float]] = []
    for noise_level in noise_levels:
        metrics = evaluate_noise_level(
            layer,
            action_bank,
            codebook,
            target_indices,
            noise_level=float(noise_level),
            hash_noise_ratio=args.hash_noise_ratio,
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
        rows.append(metrics)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "accuracy_results.csv"
    plot_path = args.output_dir / "accuracy_invariance.png"
    write_results_csv(rows, csv_path)
    plot_results(rows, plot_path)

    best_gap = max(row["gvla_accuracy"] - row["dense_accuracy"] for row in rows)
    final_invariant = rows[-1]["invariant_rate"]

    print(f"Results written to: {csv_path}")
    print(f"Plot written to: {plot_path}")
    print(f"Best GVLA accuracy gain over dense retrieval: {best_gap:.4f}")
    print(f"Invariant rate at max noise ({rows[-1]['noise_level']:.4f}): {final_invariant:.4f}")


if __name__ == "__main__":
    main()
