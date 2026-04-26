import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.layers import OrthogonalProjectionLayer

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional runtime dependency
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare a softmax VLA action head against GVLA geometric hashing."
    )
    parser.add_argument("--embedding-dims", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--min-exp", type=int, default=15)
    parser.add_argument("--max-exp", type=int, default=19)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--measure-iters", type=int, default=100)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16" if torch.cuda.is_available() else "float32",
        choices=("float16", "float32", "float64"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "vla_backbone_comparison",
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
    safe_name = run_name if run_name is not None else "sota_vla_head_comparison"
    return output_root / f"{timestamp}_{safe_name}"


def write_args_snapshot(args: argparse.Namespace, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for key, value in sorted(vars(args).items()):
            handle.write(f"{key}: {value}\n")


def benchmark_module(
    module: nn.Module,
    inputs: torch.Tensor,
    *,
    warmup_iters: int,
    measure_iters: int,
) -> float:
    module.eval()

    def run_step() -> None:
        output = module(inputs)
        if isinstance(output, dict):
            _ = output["binary_code"]

    with torch.inference_mode():
        for _ in range(warmup_iters):
            run_step()

        if inputs.device.type == "cuda":
            torch.cuda.synchronize(device=inputs.device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(measure_iters):
                run_step()
            end_event.record()
            torch.cuda.synchronize(device=inputs.device)
            return start_event.elapsed_time(end_event) / measure_iters

        start_time = time.perf_counter()
        for _ in range(measure_iters):
            run_step()
        end_time = time.perf_counter()
        return ((end_time - start_time) * 1000.0) / measure_iters


def estimate_softmax_head_flops(batch_size: int, embedding_dim: int, num_actions: int) -> float:
    projection_flops = 2.0 * batch_size * embedding_dim * num_actions
    softmax_flops = 5.0 * batch_size * num_actions
    return projection_flops + softmax_flops


def estimate_gvla_head_flops(batch_size: int, embedding_dim: int, num_actions: int) -> float:
    code_length = int(math.ceil(math.log2(num_actions)))
    projection_flops = 2.0 * batch_size * embedding_dim * code_length
    hashing_flops = 2.0 * batch_size * code_length
    return projection_flops + hashing_flops


def efficiency_to_precision_ratio(
    num_actions: int,
    latency_ms: float,
    flops: float,
) -> float:
    # Higher is better: more discrete control precision delivered per unit runtime and compute.
    return num_actions / max(latency_ms * flops, 1e-12)


def write_results_csv(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "backbone_dim",
                "num_actions",
                "precision_bits",
                "method",
                "latency_ms",
                "flops",
                "gflops",
                "efficiency_to_precision_ratio",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def plot_results(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install it to generate plots.")

    dims = sorted({int(row["backbone_dim"]) for row in rows})
    methods = ("softmax", "gvla")
    colors = {"softmax": "#d94841", "gvla": "#1f78b4"}
    linestyles = {2048: "-", 4096: "--"}

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    for dim in dims:
        for method in methods:
            subset = [
                row
                for row in rows
                if int(row["backbone_dim"]) == dim and row["method"] == method
            ]
            subset.sort(key=lambda row: int(row["num_actions"]))
            x_values = [int(row["num_actions"]) for row in subset]
            latency = [float(row["latency_ms"]) for row in subset]
            gflops = [float(row["gflops"]) for row in subset]
            epr = [float(row["efficiency_to_precision_ratio"]) for row in subset]
            label = f"{method.upper()} d={dim}"

            axes[0].plot(
                x_values,
                latency,
                marker="o",
                color=colors[method],
                linestyle=linestyles.get(dim, "-."),
                label=label,
            )
            axes[1].plot(
                x_values,
                gflops,
                marker="o",
                color=colors[method],
                linestyle=linestyles.get(dim, "-."),
                label=label,
            )
            axes[2].plot(
                x_values,
                epr,
                marker="o",
                color=colors[method],
                linestyle=linestyles.get(dim, "-."),
                label=label,
            )

    axes[0].set_title("Latency vs Action Precision")
    axes[0].set_xlabel("Number of discrete actions (N)")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_xscale("log", base=2)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title("Head FLOPs vs Action Precision")
    axes[1].set_xlabel("Number of discrete actions (N)")
    axes[1].set_ylabel("GFLOPs")
    axes[1].set_xscale("log", base=2)
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title("Efficiency-to-Precision Ratio")
    axes[2].set_xlabel("Number of discrete actions (N)")
    axes[2].set_ylabel("N / (latency * FLOPs)")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    handles, labels = axes[2].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("SOTA VLA Backbone Head Comparison")
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    run_dir = build_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_args_snapshot(args, run_dir / "args.txt")

    num_actions_list = [1 << exponent for exponent in range(args.min_exp, args.max_exp + 1)]
    rows: List[Dict[str, float | str]] = []

    for backbone_dim in args.embedding_dims:
        inputs = torch.randn(
            args.batch_size,
            backbone_dim,
            device=device,
            dtype=dtype,
        )

        for num_actions in num_actions_list:
            precision_bits = int(math.ceil(math.log2(num_actions)))

            softmax_head = nn.Sequential(
                nn.Linear(
                    backbone_dim,
                    num_actions,
                    bias=False,
                    device=device,
                    dtype=dtype,
                ),
                nn.Softmax(dim=-1),
            )
            gvla_head = OrthogonalProjectionLayer(
                input_dim=backbone_dim,
                num_codes=num_actions,
                use_ste=False,
                device=device,
                dtype=dtype,
            ).to(device=device, dtype=dtype)

            softmax_latency_ms = benchmark_module(
                softmax_head,
                inputs,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
            )
            gvla_latency_ms = benchmark_module(
                gvla_head,
                inputs,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
            )

            softmax_flops = estimate_softmax_head_flops(
                args.batch_size,
                backbone_dim,
                num_actions,
            )
            gvla_flops = estimate_gvla_head_flops(
                args.batch_size,
                backbone_dim,
                num_actions,
            )

            rows.append(
                {
                    "backbone_dim": backbone_dim,
                    "num_actions": num_actions,
                    "precision_bits": precision_bits,
                    "method": "softmax",
                    "latency_ms": softmax_latency_ms,
                    "flops": softmax_flops,
                    "gflops": softmax_flops / 1e9,
                    "efficiency_to_precision_ratio": efficiency_to_precision_ratio(
                        num_actions,
                        softmax_latency_ms,
                        softmax_flops,
                    ),
                }
            )
            rows.append(
                {
                    "backbone_dim": backbone_dim,
                    "num_actions": num_actions,
                    "precision_bits": precision_bits,
                    "method": "gvla",
                    "latency_ms": gvla_latency_ms,
                    "flops": gvla_flops,
                    "gflops": gvla_flops / 1e9,
                    "efficiency_to_precision_ratio": efficiency_to_precision_ratio(
                        num_actions,
                        gvla_latency_ms,
                        gvla_flops,
                    ),
                }
            )

            del softmax_head
            del gvla_head
            if device.type == "cuda":
                torch.cuda.empty_cache()

    csv_path = run_dir / "vla_backbone_comparison.csv"
    figure_path = run_dir / "vla_backbone_comparison.png"
    write_results_csv(rows, csv_path)
    plot_results(rows, figure_path)

    print(f"Results written to: {csv_path}")
    print(f"Plot written to: {figure_path}")


if __name__ == "__main__":
    main()
