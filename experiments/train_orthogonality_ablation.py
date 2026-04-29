"""Trainable orthogonality ablation for GVLA projection weights.

This script learns a projection matrix ``W`` to imitate a teacher hash mapping
and sweeps the orthogonality regularization coefficient. Unlike the existing
proxy-only ablation, this experiment explicitly optimizes

    BCE(hash_logits, teacher_bits) + lambda_ortho * ||WW^T - I||_F^2

so the "remove orthogonal regularization" condition is a true training-time
ablation rather than a hand-crafted correlated basis.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch import nn

from models.layers import OrthogonalProjectionLayer


PROJECT_ROOT = Path("/home/introai11/.agile/users/hsjung/projects/GVLA-Net")
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "results" / "orthogonality_training_ablation"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep training-time orthogonality regularization for GVLA."
    )
    parser.add_argument("--num-actions", type=int, default=4096)
    parser.add_argument("--input-dim", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=6000)
    parser.add_argument("--val-samples", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument(
        "--ortho-coeffs",
        type=float,
        nargs="+",
        default=[0.0, 1e-4, 1e-3, 1e-2],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, choices=("float32", "float64"), default="float32")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def make_teacher(
    input_dim: int,
    num_actions: int,
    device: torch.device,
    dtype: torch.dtype,
) -> OrthogonalProjectionLayer:
    teacher = OrthogonalProjectionLayer(
        input_dim=input_dim,
        num_codes=num_actions,
        use_ste=False,
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def mean_abs_row_cosine(weight: torch.Tensor) -> float:
    if weight.size(0) <= 1:
        return 0.0
    normalized = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
    cosine = torch.abs(normalized @ normalized.transpose(0, 1))
    mask = ~torch.eye(weight.size(0), dtype=torch.bool, device=weight.device)
    return float(cosine[mask].mean().item())


def unique_code_ratio(bits: torch.Tensor) -> float:
    bit_count = bits.size(1)
    powers = (2 ** torch.arange(bit_count, device=bits.device, dtype=torch.int64)).view(1, -1)
    ids = (bits.to(torch.int64) * powers).sum(dim=1)
    return float(ids.unique().numel() / bits.size(0))


def mean_bit_entropy(bits: torch.Tensor) -> float:
    probs = bits.to(torch.float32).mean(dim=0).clamp(1e-6, 1.0 - 1e-6)
    entropy = -(probs * torch.log2(probs) + (1.0 - probs) * torch.log2(1.0 - probs))
    return float(entropy.mean().item())


def evaluate_model(
    model: OrthogonalProjectionLayer,
    teacher_bits: torch.Tensor,
    inputs: torch.Tensor,
) -> Dict[str, float]:
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs["projections"]
        pred_bits = (logits >= 0).to(torch.int64)
        targets = teacher_bits.to(logits.dtype)
        bit_accuracy = float((pred_bits == teacher_bits).to(torch.float32).mean().item())
        bce_loss = float(F.binary_cross_entropy_with_logits(logits, targets).item())
        ortho_loss = float(model.orthogonality_loss().item())
        uniq = unique_code_ratio(pred_bits)
        return {
            "bit_accuracy": bit_accuracy,
            "bce_loss": bce_loss,
            "ortho_loss": ortho_loss,
            "mean_abs_row_cosine": mean_abs_row_cosine(model.weight.detach()),
            "unique_code_ratio": uniq,
            "collision_rate": 1.0 - uniq,
            "mean_bit_entropy": mean_bit_entropy(pred_bits),
        }


def train_single_setting(
    ortho_coeff: float,
    args: argparse.Namespace,
    train_x: torch.Tensor,
    train_bits: torch.Tensor,
    val_x: torch.Tensor,
    val_bits: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, float]:
    model = OrthogonalProjectionLayer(
        input_dim=args.input_dim,
        num_codes=args.num_actions,
        use_ste=False,
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for step in range(args.steps):
        indices = torch.randint(
            low=0,
            high=train_x.size(0),
            size=(args.batch_size,),
            device=device,
        )
        batch_x = train_x[indices]
        batch_bits = train_bits[indices].to(dtype)

        outputs = model(batch_x)
        hash_loss = F.binary_cross_entropy_with_logits(outputs["projections"], batch_bits)
        ortho_loss = model.orthogonality_loss()
        total_loss = hash_loss + ortho_coeff * ortho_loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()

        if (step + 1) % args.eval_interval == 0 or step == 0 or step + 1 == args.steps:
            train_metrics = evaluate_model(model, train_bits, train_x)
            val_metrics = evaluate_model(model, val_bits, val_x)
            print(
                "coeff={coeff:.4g} step={step:04d} "
                "train_bit_acc={train_acc:.4f} val_bit_acc={val_acc:.4f} "
                "val_collision={collision:.4f} val_mean|cos|={mean_cos:.4f}".format(
                    coeff=ortho_coeff,
                    step=step + 1,
                    train_acc=train_metrics["bit_accuracy"],
                    val_acc=val_metrics["bit_accuracy"],
                    collision=val_metrics["collision_rate"],
                    mean_cos=val_metrics["mean_abs_row_cosine"],
                )
            )

    final_train = evaluate_model(model, train_bits, train_x)
    final_val = evaluate_model(model, val_bits, val_x)
    row: Dict[str, float] = {
        "ortho_coeff": ortho_coeff,
        "steps": float(args.steps),
        "lr": args.lr,
        "seed": float(args.seed),
        "num_actions": float(args.num_actions),
        "input_dim": float(args.input_dim),
        "train_samples": float(args.train_samples),
        "val_samples": float(args.val_samples),
    }
    for key, value in final_train.items():
        row[f"train_{key}"] = value
    for key, value in final_val.items():
        row[f"val_{key}"] = value
    return row


def write_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_summary(rows: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Orthogonality Training Ablation",
        "",
        "| Ortho Coeff | Val Bit Acc | Val Collision | Val Mean |cos| | Val Unique Ratio | Val Mean Bit Entropy |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {coeff:.4g} | {acc:.4f} | {collision:.4f} | {cos:.4f} | {uniq:.4f} | {entropy:.4f} |".format(
                coeff=row["ortho_coeff"],
                acc=row["val_bit_accuracy"],
                collision=row["val_collision_rate"],
                cos=row["val_mean_abs_row_cosine"],
                uniq=row["val_unique_code_ratio"],
                entropy=row["val_mean_bit_entropy"],
            )
        )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    dtype = resolve_dtype(args.dtype)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    code_length = int(math.ceil(math.log2(args.num_actions)))
    if code_length > args.input_dim:
        raise ValueError(
            "Need ceil(log2(num_actions)) <= input_dim, got "
            f"{code_length} > {args.input_dim}."
        )

    teacher = make_teacher(args.input_dim, args.num_actions, device, dtype)
    train_x = torch.randn(args.train_samples, args.input_dim, device=device, dtype=dtype)
    val_x = torch.randn(args.val_samples, args.input_dim, device=device, dtype=dtype)
    with torch.no_grad():
        train_bits = (teacher(train_x)["projections"] >= 0).to(torch.int64)
        val_bits = (teacher(val_x)["projections"] >= 0).to(torch.int64)

    rows: List[Dict[str, float]] = []
    for ortho_coeff in args.ortho_coeffs:
        torch.manual_seed(args.seed)
        rows.append(
            train_single_setting(
                ortho_coeff=ortho_coeff,
                args=args,
                train_x=train_x,
                train_bits=train_bits,
                val_x=val_x,
                val_bits=val_bits,
                device=device,
                dtype=dtype,
            )
        )

    rows.sort(key=lambda row: row["ortho_coeff"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "orthogonality_training_ablation.csv"
    md_path = args.output_dir / "orthogonality_training_ablation.md"
    write_csv(rows, csv_path)
    write_markdown_summary(rows, md_path)
    print(f"Saved CSV to: {csv_path}")
    print(f"Saved summary to: {md_path}")


if __name__ == "__main__":
    main()
