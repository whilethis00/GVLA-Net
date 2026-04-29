"""Correlation sweep for GVLA action-code collisions.

This experiment strengthens the orthogonality ablation by sweeping the amount of
inter-row correlation in the projection matrix instead of comparing only one
"w/o orthogonality" point. The resulting trend is more defensible in the paper:
as row correlation grows, code utilization degrades and collision rate rises.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from action_codebook_collision_test import (
    PROJECT_ROOT,
    ProxyRow,
    correlated_rows,
    evaluate_projection_method,
    orthogonal_rows,
    sample_actions,
    train_test_split,
)


@dataclass
class SweepRow:
    mix: float
    mean_abs_row_cosine: float
    collision_rate: float
    singleton_rate: float
    unique_code_ratio: float
    occupancy_efficiency: float
    reconstruction_mse: float
    total_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep projection-row correlation and measure GVLA proxy collapse."
    )
    parser.add_argument("--num-actions", type=int, default=1 << 20)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--code-bits", type=int, default=24)
    parser.add_argument("--train-samples", type=int, default=65536)
    parser.add_argument("--eval-samples", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--mix-values",
        type=float,
        nargs="+",
        default=[0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 0.9],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "orthogonality_correlation_sweep",
    )
    parser.add_argument("--run-name", type=str, default="proxy_corr_sweep")
    return parser.parse_args()


def build_run_dir(output_root: Path, run_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return output_root / f"{timestamp}_{run_name}"


def to_sweep_row(proxy_row: ProxyRow, mix: float) -> SweepRow:
    return SweepRow(
        mix=mix,
        mean_abs_row_cosine=proxy_row.mean_abs_row_cosine,
        collision_rate=proxy_row.collision_rate,
        singleton_rate=proxy_row.singleton_rate,
        unique_code_ratio=proxy_row.unique_code_ratio,
        occupancy_efficiency=proxy_row.occupancy_efficiency,
        reconstruction_mse=proxy_row.reconstruction_mse,
        total_ms=proxy_row.total_ms,
    )


def write_csv(rows: list[SweepRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(SweepRow.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def write_markdown(rows: list[SweepRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Orthogonality Correlation Sweep",
        "",
        (
            "Higher `mix` means stronger inter-row coupling in the projection matrix. "
            "As `mix` rises, `mean |cos|` should increase, `unique_code_ratio` should fall, "
            "and `collision_rate` should rise."
        ),
        "",
        "| Mix | Mean |cos| | Collision Rate | Occupancy Efficiency | Unique Code Ratio | Recon. MSE |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {mix:.2f} | {cos:.4f} | {collision:.4f} | {eff:.4f} | {uniq:.4f} | {mse:.6f} |".format(
                mix=row.mix,
                cos=row.mean_abs_row_cosine,
                collision=row.collision_rate,
                eff=row.occupancy_efficiency,
                uniq=row.unique_code_ratio,
                mse=row.reconstruction_mse,
            )
        )
    output_path.write_text("\n".join(lines) + "\n")


def write_tex(rows: list[SweepRow], output_path: Path, code_bits: int) -> None:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Correlation sweep within the GVLA projection family at "
            f"$N=2^{{20}}$ and $k={code_bits}$ bits. We progressively increase "
            "inter-row coupling in the projection matrix; as orthogonality degrades, "
            "collision rate rises and occupancy efficiency collapses.}"
        ),
        "\\label{tab:gvla_corr_sweep}",
        "\\scriptsize",
        "\\begin{tabular}{rrrrrr}",
        "\\toprule",
        "Mix & Mean $|\\cos|$ & Collision Rate & Occ. Eff. & Unique Code Ratio & Recon. MSE \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.mix:.2f} & {row.mean_abs_row_cosine:.4f} & {row.collision_rate:.4f} & "
            f"{row.occupancy_efficiency:.4f} & {row.unique_code_ratio:.4f} & {row.reconstruction_mse:.6f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    output_path.write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    if args.code_bits > args.action_dim:
        raise ValueError(
            f"Need code_bits <= action_dim for orthogonal reference, got {args.code_bits} > {args.action_dim}."
        )

    run_dir = build_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    actions = sample_actions(args.num_actions, args.action_dim, rng)
    train_actions, test_actions = train_test_split(actions, args.train_samples, args.eval_samples)

    rows: list[SweepRow] = []
    mix_values = sorted(set(float(v) for v in args.mix_values))
    for mix in mix_values:
        if mix == 0.0:
            weight = orthogonal_rows(
                args.code_bits,
                args.action_dim,
                np.random.default_rng(args.seed + 37 + args.code_bits),
            )
            method_name = "GVLA-Net (Orthogonal)"
        else:
            weight = correlated_rows(
                args.code_bits,
                args.action_dim,
                np.random.default_rng(args.seed + 31 + args.code_bits + int(mix * 1000)),
                mix=mix,
            )
            method_name = f"GVLA CorrMix={mix:.2f}"
        proxy_row = evaluate_projection_method(
            method_name,
            train_actions,
            test_actions,
            actions,
            weight,
        )
        row = to_sweep_row(proxy_row, mix)
        rows.append(row)
        print(
            f"mix={row.mix:.2f}: mean_abs_row_cosine={row.mean_abs_row_cosine:.4f}, "
            f"collision_rate={row.collision_rate:.4f}, occupancy_efficiency={row.occupancy_efficiency:.4f}, "
            f"unique_code_ratio={row.unique_code_ratio:.4f}, reconstruction_mse={row.reconstruction_mse:.6f}"
        )

    csv_path = run_dir / "orthogonality_correlation_sweep.csv"
    md_path = run_dir / "orthogonality_correlation_sweep.md"
    tex_path = run_dir / "orthogonality_correlation_sweep.tex"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    write_tex(rows, tex_path, code_bits=args.code_bits)
    print(f"CSV written to: {csv_path}")
    print(f"Markdown written to: {md_path}")
    print(f"LaTeX written to: {tex_path}")


if __name__ == "__main__":
    main()
