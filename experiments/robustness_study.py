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
from utils.geometry import check_orthogonality, initialize_orthogonal_basis

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional runtime dependency
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Study GVLA robustness to latent Gaussian noise and basis orthogonality."
    )
    parser.add_argument("--num-actions", type=int, default=65536)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--num-samples", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--margin-scale", type=float, default=4.0)
    parser.add_argument("--action-nuisance-scale", type=float, default=1.5)
    parser.add_argument("--sigma-min", type=float, default=0.0)
    parser.add_argument("--sigma-max", type=float, default=0.5)
    parser.add_argument("--sigma-steps", type=int, default=6)
    parser.add_argument("--perturbation-steps", type=int, default=6)
    parser.add_argument("--max-basis-perturb", type=float, default=0.5)
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
        default=PROJECT_ROOT / "experiments" / "results" / "robustness_study",
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


def normalize_rows(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return matrix / matrix.norm(dim=1, keepdim=True).clamp_min(eps)


def orthogonal_complement_noise(
    tensor: torch.Tensor,
    basis: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    projected = (tensor @ basis.transpose(0, 1)) @ basis
    residual = tensor - projected
    return residual / residual.norm(dim=-1, keepdim=True).clamp_min(eps)


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


def evaluate_retrieval(
    layer: OrthogonalProjectionLayer,
    action_bank: torch.Tensor,
    codebook: torch.Tensor,
    target_indices: torch.Tensor,
    sigma: float,
    *,
    batch_size: int,
    chunk_size: int,
) -> Tuple[float, float]:
    gvla_correct = 0
    dense_correct = 0
    total_samples = target_indices.numel()

    with torch.inference_mode():
        for start in range(0, total_samples, batch_size):
            stop = min(start + batch_size, total_samples)
            batch_indices = target_indices[start:stop]
            clean_states = action_bank[batch_indices]
            noisy_states = clean_states + sigma * torch.randn_like(clean_states)

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

            gvla_correct += (gvla_pred == batch_indices).sum().item()
            dense_correct += (dense_pred == batch_indices).sum().item()

    return gvla_correct / total_samples, dense_correct / total_samples


def perturb_basis(
    basis: torch.Tensor,
    perturbation_scale: float,
) -> torch.Tensor:
    if perturbation_scale == 0.0:
        return basis.clone()
    candidate = basis + perturbation_scale * torch.randn_like(basis)
    return normalize_rows(candidate)


def pearson_correlation(x_values: List[float], y_values: List[float]) -> float:
    x = torch.tensor(x_values, dtype=torch.float64)
    y = torch.tensor(y_values, dtype=torch.float64)
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    denominator = torch.sqrt(centered_x.square().sum() * centered_y.square().sum())
    if denominator.item() == 0.0:
        return 0.0
    return (centered_x * centered_y).sum().div(denominator).item()


def write_csv(rows: List[Dict[str, float]], output_path: Path, fieldnames: Tuple[str, ...]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(
    noise_rows: List[Dict[str, float]],
    orth_rows: List[Dict[str, float]],
    *,
    output_path: Path,
    orth_corr: float,
) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is not installed. Install it to generate robustness plots."
        )

    sigma_values = [row["sigma"] for row in noise_rows]
    gvla_accuracy = [100.0 * row["gvla_accuracy"] for row in noise_rows]
    dense_accuracy = [100.0 * row["dense_accuracy"] for row in noise_rows]

    orth_error = [row["orthogonality_error"] for row in orth_rows]
    orth_mean_accuracy = [100.0 * row["mean_gvla_accuracy"] for row in orth_rows]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(
        sigma_values,
        gvla_accuracy,
        marker="s",
        linewidth=2.0,
        label="Geometric Hash",
    )
    axes[0].plot(
        sigma_values,
        dense_accuracy,
        marker="o",
        linewidth=2.0,
        label="Softmax-style Dense Retrieval",
    )
    axes[0].set_title("Noise Ablation")
    axes[0].set_xlabel("Noise Level (sigma)")
    axes[0].set_ylabel("Retrieval Accuracy (%)")
    axes[0].set_ylim(0.0, 105.0)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].scatter(orth_error, orth_mean_accuracy, s=60, label="Perturbed basis")
    if len(orth_error) >= 2:
        x = torch.tensor(orth_error, dtype=torch.float64)
        y = torch.tensor(orth_mean_accuracy, dtype=torch.float64)
        slope = ((x - x.mean()) * (y - y.mean())).sum() / (x - x.mean()).square().sum()
        intercept = y.mean() - slope * x.mean()
        fit_x = torch.linspace(x.min(), x.max(), steps=100)
        fit_y = slope * fit_x + intercept
        axes[1].plot(
            fit_x.tolist(),
            fit_y.tolist(),
            linestyle="--",
            label="Linear trend",
        )
    axes[1].set_title("Orthogonality vs Noise Robustness")
    axes[1].set_xlabel("Orthogonality Error ||WW^T - I||_F")
    axes[1].set_ylabel("Mean GVLA Accuracy (%)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    axes[1].text(
        0.03,
        0.95,
        f"Pearson r = {orth_corr:.4f}",
        transform=axes[1].transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    figure.suptitle("GVLA-Net Robustness Study")
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

    ideal_basis = initialize_orthogonal_basis(
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
    action_bank = build_action_bank(
        codebook,
        ideal_basis,
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

    sigma_values = torch.linspace(
        args.sigma_min,
        args.sigma_max,
        steps=args.sigma_steps,
        device=device,
        dtype=torch.float32,
    ).tolist()

    layer = OrthogonalProjectionLayer(
        input_dim=args.input_dim,
        num_codes=args.num_actions,
        use_ste=False,
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(ideal_basis)

    noise_rows: List[Dict[str, float]] = []
    for sigma in sigma_values:
        gvla_accuracy, dense_accuracy = evaluate_retrieval(
            layer,
            action_bank,
            codebook,
            target_indices,
            sigma=float(sigma),
            batch_size=args.batch_size,
            chunk_size=args.chunk_size,
        )
        noise_rows.append(
            {
                "sigma": float(sigma),
                "gvla_accuracy": gvla_accuracy,
                "dense_accuracy": dense_accuracy,
                "accuracy_gap": gvla_accuracy - dense_accuracy,
            }
        )

    orth_rows: List[Dict[str, float]] = []
    perturbation_values = torch.linspace(
        0.0,
        args.max_basis_perturb,
        steps=args.perturbation_steps,
        device=device,
        dtype=torch.float32,
    ).tolist()

    for perturb_scale in perturbation_values:
        perturbed_basis = perturb_basis(ideal_basis, float(perturb_scale))
        with torch.no_grad():
            layer.weight.copy_(perturbed_basis)

        per_sigma_accuracy: List[float] = []
        for sigma in sigma_values:
            gvla_accuracy, _ = evaluate_retrieval(
                layer,
                action_bank,
                codebook,
                target_indices,
                sigma=float(sigma),
                batch_size=args.batch_size,
                chunk_size=args.chunk_size,
            )
            per_sigma_accuracy.append(gvla_accuracy)

        orth_rows.append(
            {
                "perturbation_scale": float(perturb_scale),
                "orthogonality_error": check_orthogonality(perturbed_basis),
                "mean_gvla_accuracy": sum(per_sigma_accuracy) / len(per_sigma_accuracy),
                "min_gvla_accuracy": min(per_sigma_accuracy),
                "max_gvla_accuracy": max(per_sigma_accuracy),
            }
        )

    orth_corr = pearson_correlation(
        [row["orthogonality_error"] for row in orth_rows],
        [row["mean_gvla_accuracy"] for row in orth_rows],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    noise_csv_path = args.output_dir / "robustness_noise_ablation.csv"
    orth_csv_path = args.output_dir / "robustness_orthogonality.csv"
    plot_path = args.output_dir / "robustness_study.png"

    write_csv(
        noise_rows,
        noise_csv_path,
        ("sigma", "gvla_accuracy", "dense_accuracy", "accuracy_gap"),
    )
    write_csv(
        orth_rows,
        orth_csv_path,
        (
            "perturbation_scale",
            "orthogonality_error",
            "mean_gvla_accuracy",
            "min_gvla_accuracy",
            "max_gvla_accuracy",
        ),
    )
    plot_results(
        noise_rows,
        orth_rows,
        output_path=plot_path,
        orth_corr=orth_corr,
    )

    best_gap = max(row["accuracy_gap"] for row in noise_rows)
    final_gap = noise_rows[-1]["accuracy_gap"]

    print(f"Noise ablation written to: {noise_csv_path}")
    print(f"Orthogonality study written to: {orth_csv_path}")
    print(f"Plot written to: {plot_path}")
    print(f"Best GVLA accuracy gap over dense retrieval: {best_gap:.4f}")
    print(f"GVLA accuracy gap at max sigma ({noise_rows[-1]['sigma']:.4f}): {final_gap:.4f}")
    print(f"Pearson correlation (orthogonality error vs mean GVLA accuracy): {orth_corr:.4f}")


if __name__ == "__main__":
    main()
