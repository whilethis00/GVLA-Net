from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

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


def evaluate_product(model: ProductQGVLAHead, val_x: torch.Tensor, target_bits: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        codes = target_bits.unsqueeze(1)
        probs = model.code_probability(val_x, codes).squeeze(1)
        target_prob = float(probs.mean().item())
        logits = model.forward(val_x).logits.squeeze(1)
        pred_bits = (logits >= 0).to(torch.float32)
        bit_acc = float((pred_bits == target_bits).to(torch.float32).mean().item())
        all_half = float(((torch.sigmoid(logits) - 0.5).abs() < 0.05).to(torch.float32).mean().item())
    return {
        "target_mode_prob": target_prob,
        "bit_accuracy": bit_acc,
        "near_half_rate": all_half,
    }


def evaluate_mps(model: MPSQGVLAHead, val_x: torch.Tensor, target_bits: torch.Tensor) -> tuple[dict[str, float], torch.Tensor]:
    with torch.no_grad():
        code_ids, probs = model.enumerate_code_probabilities(val_x)
        low_id = 0
        high_id = (1 << model.num_bits) - 1
        target_prob = float((probs[:, low_id] + probs[:, high_id]).mean().item())
        pred_id = probs.argmax(dim=1)
        low_or_high = ((pred_id == low_id) | (pred_id == high_id)).to(torch.float32).mean().item()
    return {
        "target_mode_prob": target_prob,
        "top_mode_valid_rate": float(low_or_high),
    }, probs.mean(dim=0).cpu()


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
    product_metrics = evaluate_product(product, val_x, val_bits)
    rows.append({"model": "product", "bond_dim": 1, **product_metrics})

    for bond_dim in args.bond_dims:
        model = MPSQGVLAHead(args.input_dim, num_bins, bond_dim=bond_dim).to(device)
        train_mps(model, train_x, train_bits, args)
        metrics, avg_probs = evaluate_mps(model, val_x, val_bits)
        rows.append({"model": "mps", "bond_dim": bond_dim, **metrics})
        plot_average_distribution(
            avg_probs,
            args.output_dir / f"avg_distribution_mps_r{bond_dim}.png",
            f"MPS-QGVLA average code distribution (r={bond_dim})",
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
