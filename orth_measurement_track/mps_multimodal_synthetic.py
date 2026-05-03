from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.io_utils import save_figure, write_csv_rows, write_text
from orth_measurement_track.qgvla_heads import MPSQGVLAHead, ProductQGVLAHead, code_bits_from_indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic multimodal comparison: Product-QGVLA vs MPS-QGVLA."
    )
    parser.add_argument("--input-dim", type=int, default=16)
    parser.add_argument("--num-bins", type=int, default=256)
    parser.add_argument("--train-samples", type=int, default=4096)
    parser.add_argument("--val-samples", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--bond-dims", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "orth_measurement_track" / "mps_multimodal_synthetic",
    )
    return parser.parse_args()


def make_dataset(num_samples: int, input_dim: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    latent = torch.randn(num_samples, input_dim, device=device)
    selector = latent[:, 0] > 0
    code_low = torch.zeros(num_samples, 8, device=device)
    code_high = torch.ones(num_samples, 8, device=device)
    bits = torch.where(selector.unsqueeze(1), code_high, code_low)
    return latent, bits


def entropy_from_probs(probs: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    safe = probs.clamp_min(eps)
    return -(safe * safe.log2()).sum(dim=-1)


def conditional_entropy_curve(
    probs: torch.Tensor,
    true_bits: torch.Tensor,
    code_ids: torch.Tensor,
    num_bits: int,
) -> list[float]:
    code_bits = code_bits_from_indices(code_ids, num_bits).to(probs.device)
    values = []
    for prefix_len in range(1, num_bits + 1):
        mask = (code_bits.unsqueeze(0)[:, :, :prefix_len] == true_bits.unsqueeze(1)[:, :, :prefix_len]).all(dim=-1)
        masked = probs * mask.to(probs.dtype)
        normalized = masked / masked.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        values.append(float(entropy_from_probs(normalized).mean().item()))
    return values


def train_product(model: ProductQGVLAHead, train_x: torch.Tensor, train_bits: torch.Tensor, args: argparse.Namespace) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(args.steps):
        idx = torch.randint(0, train_x.size(0), (args.batch_size,), device=train_x.device)
        batch_x = train_x[idx]
        batch_bits = train_bits[idx].unsqueeze(1)
        loss = model.born_nll(batch_x, batch_bits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def train_mps(model: MPSQGVLAHead, train_x: torch.Tensor, train_bits: torch.Tensor, args: argparse.Namespace) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    for _ in range(args.steps):
        idx = torch.randint(0, train_x.size(0), (args.batch_size,), device=train_x.device)
        batch_x = train_x[idx]
        batch_bits = train_bits[idx]
        loss = model.born_nll(batch_x, batch_bits)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def evaluate_product(
    model: ProductQGVLAHead,
    val_x: torch.Tensor,
    target_bits: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, list[float]]:
    with torch.no_grad():
        code_ids, all_probs = model.enumerate_code_probabilities(val_x)
        low_id = 0
        high_id = (1 << model.num_bits) - 1
        target_prob = float((all_probs[:, low_id] + all_probs[:, high_id]).mean().item())
        logits = model.forward(val_x).logits.squeeze(1)
        pred_bits = (logits >= 0).to(torch.float32)
        bit_acc = float((pred_bits == target_bits).to(torch.float32).mean().item())
        all_half = float(((torch.sigmoid(logits) - 0.5).abs() < 0.05).to(torch.float32).mean().item())
        pred_id = all_probs.argmax(dim=1)
        top_mode_valid_rate = float(((pred_id == low_id) | (pred_id == high_id)).to(torch.float32).mean().item())
        entropy_curve = conditional_entropy_curve(all_probs, target_bits, code_ids, model.num_bits)
    return {
        "target_mode_prob": target_prob,
        "bit_accuracy": bit_acc,
        "near_half_rate": all_half,
        "top_mode_valid_rate": top_mode_valid_rate,
    }, all_probs.mean(dim=0).cpu(), entropy_curve


def evaluate_mps(
    model: MPSQGVLAHead,
    val_x: torch.Tensor,
    target_bits: torch.Tensor,
) -> tuple[dict[str, float], torch.Tensor, list[float]]:
    with torch.no_grad():
        code_ids, probs = model.enumerate_code_probabilities(val_x)
        low_id = 0
        high_id = (1 << model.num_bits) - 1
        target_prob = float((probs[:, low_id] + probs[:, high_id]).mean().item())
        pred_id = probs.argmax(dim=1)
        low_or_high = ((pred_id == low_id) | (pred_id == high_id)).to(torch.float32).mean().item()
        entropy_curve = conditional_entropy_curve(probs, target_bits, code_ids, model.num_bits)
    return {
        "target_mode_prob": target_prob,
        "top_mode_valid_rate": float(low_or_high),
    }, probs.mean(dim=0).cpu(), entropy_curve


def plot_average_distribution(code_probs: torch.Tensor, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.plot(code_probs.numpy(), lw=1.5)
    ax.set_xlabel("Code index")
    ax.set_ylabel("Average probability")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=220)
    plt.close(fig)


def plot_entropy_curve(values: list[float], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    x = np.arange(1, len(values) + 1)
    ax.plot(x, values, marker="o", lw=2.0)
    ax.set_xlabel("Observed bit prefix length")
    ax.set_ylabel("Conditional entropy")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    train_x, train_bits = make_dataset(args.train_samples, args.input_dim, device)
    val_x, val_bits = make_dataset(args.val_samples, args.input_dim, device)
    num_bits = val_bits.size(1)
    num_bins = 1 << num_bits
    rows: list[dict[str, float | int | str]] = []

    product = ProductQGVLAHead(args.input_dim, num_bins, action_dim=1).to(device)
    train_product(product, train_x, train_bits, args)
    product_metrics, product_avg_probs, product_entropy_curve = evaluate_product(product, val_x, val_bits)
    rows.append({"model": "product", "bond_dim": 1, **product_metrics})
    plot_average_distribution(
        product_avg_probs,
        args.output_dir / "avg_distribution_product.png",
        "Product-QGVLA average code distribution",
    )
    plot_entropy_curve(
        product_entropy_curve,
        args.output_dir / "entropy_curve_product.png",
        "Product-QGVLA entropy decay",
    )

    for bond_dim in args.bond_dims:
        model = MPSQGVLAHead(args.input_dim, num_bins, bond_dim=bond_dim).to(device)
        train_mps(model, train_x, train_bits, args)
        metrics, avg_probs, entropy_curve = evaluate_mps(model, val_x, val_bits)
        rows.append({"model": "mps", "bond_dim": bond_dim, **metrics})
        plot_average_distribution(
            avg_probs,
            args.output_dir / f"avg_distribution_mps_r{bond_dim}.png",
            f"MPS-QGVLA average code distribution (r={bond_dim})",
        )
        plot_entropy_curve(
            entropy_curve,
            args.output_dir / f"entropy_curve_mps_r{bond_dim}.png",
            f"MPS-QGVLA entropy decay (r={bond_dim})",
        )

    write_csv_rows(rows, args.output_dir / "summary.csv")
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2))
    lines = [
        "# MPS Multimodal Synthetic",
        "",
        "| Model | Bond Dim | Target-Mode Prob | Top-Mode Valid Rate | Bit Accuracy | Near-Half Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['bond_dim']} | {row.get('target_mode_prob', float('nan')):.4f} | "
            f"{row.get('top_mode_valid_rate', float('nan')):.4f} | "
            f"{row.get('bit_accuracy', float('nan')):.4f} | {row.get('near_half_rate', float('nan')):.4f} |"
        )
    write_text(args.output_dir / "summary.md", "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
