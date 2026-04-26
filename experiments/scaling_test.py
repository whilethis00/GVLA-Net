import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
import torch.distributed as dist
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.layers import OrthogonalProjectionLayer

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional runtime dependency
    plt = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark GVLA-Net logarithmic hashing against a softmax head."
    )
    parser.add_argument("--start-n", type=int, default=10_000)
    parser.add_argument("--end-n", type=int, default=100_000)
    parser.add_argument("--step-n", type=int, default=10_000)
    parser.add_argument("--input-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
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
        default=PROJECT_ROOT / "experiments" / "results" / "scaling",
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


def build_search_space(start_n: int, end_n: int, step_n: int) -> List[int]:
    if start_n <= 1:
        raise ValueError(f"'start_n' must be greater than 1, got {start_n}.")
    if end_n < start_n:
        raise ValueError(f"'end_n' must be >= start_n, got {end_n} < {start_n}.")
    if step_n <= 0:
        raise ValueError(f"'step_n' must be positive, got {step_n}.")
    return list(range(start_n, end_n + 1, step_n))


def init_distributed(requested_device: str) -> Tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1

    if is_distributed and not dist.is_initialized():
        backend = "nccl" if requested_device.startswith("cuda") else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for DDP, but no GPU is available.")
        device = torch.device("cuda", local_rank if is_distributed else 0)
        torch.cuda.set_device(device)
    else:
        device = torch.device(requested_device)

    return device, rank, world_size, is_distributed


def shard_search_space(search_space: List[int], rank: int, world_size: int) -> List[int]:
    return search_space[rank::world_size]


def gather_rows(rows: List[Dict[str, float]], is_distributed: bool) -> List[Dict[str, float]]:
    if not is_distributed:
        return rows

    gathered_rows: List[List[Dict[str, float]]] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered_rows, rows)

    merged_rows: List[Dict[str, float]] = []
    for rank_rows in gathered_rows:
        merged_rows.extend(rank_rows)
    merged_rows.sort(key=lambda row: int(row["N"]))
    return merged_rows


def benchmark_module(
    module: nn.Module,
    inputs: torch.Tensor,
    *,
    warmup_iters: int,
    measure_iters: int,
    use_cuda_timing: bool,
) -> float:
    def run_step() -> None:
        output = module(inputs)
        if isinstance(output, dict):
            _ = output["binary_code"]

    module.eval()
    with torch.inference_mode():
        for _ in range(warmup_iters):
            run_step()
        if use_cuda_timing:
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


def linear_log_fit(x_values: Iterable[int], y_values: Iterable[float]) -> Dict[str, float]:
    log_x = torch.tensor([math.log2(value) for value in x_values], dtype=torch.float64)
    y = torch.tensor(list(y_values), dtype=torch.float64)

    centered_x = log_x - log_x.mean()
    centered_y = y - y.mean()
    slope = (centered_x * centered_y).sum() / centered_x.square().sum()
    intercept = y.mean() - slope * log_x.mean()
    predicted = slope * log_x + intercept

    residual = y - predicted
    ss_res = residual.square().sum()
    ss_tot = centered_y.square().sum()
    r2 = 1.0 - (ss_res / ss_tot).item() if ss_tot.item() > 0 else 1.0
    return {
        "slope": slope.item(),
        "intercept": intercept.item(),
        "r2": r2,
    }


def write_results_csv(rows: List[Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "N",
                "log2_N",
                "softmax_latency_ms",
                "orthogonal_latency_ms",
                "speedup_x",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def build_run_dir(output_root: Path, run_name: str | None, rank: int) -> Path:
    if rank != 0:
        return output_root

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = run_name if run_name is not None else "scaling_benchmark"
    return output_root / f"{timestamp}_{safe_name}"


def write_args_snapshot(args: argparse.Namespace, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for key, value in sorted(vars(args).items()):
            handle.write(f"{key}: {value}\n")


def plot_results(
    rows: List[Dict[str, float]],
    *,
    output_path: Path,
    orth_fit: Dict[str, float],
) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is not installed. Install it to generate the latency plot."
        )

    n_values = [row["N"] for row in rows]
    log_n_values = [row["log2_N"] for row in rows]
    softmax_latencies = [row["softmax_latency_ms"] for row in rows]
    orthogonal_latencies = [row["orthogonal_latency_ms"] for row in rows]

    fit_curve = [
        orth_fit["slope"] * math.log2(n_value) + orth_fit["intercept"]
        for n_value in n_values
    ]

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(n_values, softmax_latencies, marker="o", label="Softmax Head")
    axes[0].plot(n_values, orthogonal_latencies, marker="s", label="GVLA Orthogonal Hash")
    axes[0].set_title("Latency vs Search Space Size")
    axes[0].set_xlabel("N")
    axes[0].set_ylabel("Average latency (ms)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(log_n_values, orthogonal_latencies, marker="s", label="Measured GVLA")
    axes[1].plot(log_n_values, fit_curve, linestyle="--", label="Linear fit on log2(N)")
    axes[1].set_title("GVLA Latency vs log2(N)")
    axes[1].set_xlabel("log2(N)")
    axes[1].set_ylabel("Average latency (ms)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(title="Log fit")
    axes[1].text(
        0.03,
        0.95,
        f"R^2 = {orth_fit['r2']:.4f}",
        transform=axes[1].transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
    )

    figure.suptitle("GVLA-Net Scaling Benchmark")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    device, rank, world_size, is_distributed = init_distributed(args.device)
    try:
        run_dir = build_run_dir(args.output_dir, args.run_name, rank)
        dtype = resolve_dtype(args.dtype)
        search_space = build_search_space(args.start_n, args.end_n, args.step_n)
        local_search_space = shard_search_space(search_space, rank, world_size)
        use_cuda_timing = device.type == "cuda"

        if rank == 0:
            run_dir.mkdir(parents=True, exist_ok=True)
            write_args_snapshot(args, run_dir / "args.txt")

        rows: List[Dict[str, float]] = []
        for n_value in local_search_space:
            inputs = torch.randn(
                args.batch_size,
                args.input_dim,
                device=device,
                dtype=dtype,
            )

            softmax_head = nn.Sequential(
                nn.Linear(args.input_dim, n_value, bias=False, device=device, dtype=dtype),
                nn.Softmax(dim=-1),
            )
            orthogonal_head = OrthogonalProjectionLayer(
                input_dim=args.input_dim,
                num_codes=n_value,
                use_ste=True,
                device=device,
                dtype=dtype,
            )
            orthogonal_head = orthogonal_head.to(device=device, dtype=dtype)

            softmax_latency = benchmark_module(
                softmax_head,
                inputs,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                use_cuda_timing=use_cuda_timing,
            )
            orthogonal_latency = benchmark_module(
                orthogonal_head,
                inputs,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
                use_cuda_timing=use_cuda_timing,
            )

            rows.append(
                {
                    "N": n_value,
                    "log2_N": math.log2(n_value),
                    "softmax_latency_ms": softmax_latency,
                    "orthogonal_latency_ms": orthogonal_latency,
                    "speedup_x": softmax_latency / orthogonal_latency,
                }
            )

            del softmax_head
            del orthogonal_head
            del inputs
            if device.type == "cuda":
                torch.cuda.empty_cache()

        rows = gather_rows(rows, is_distributed=is_distributed)

        if rank == 0:
            orth_fit = linear_log_fit(
                x_values=[int(row["N"]) for row in rows],
                y_values=[row["orthogonal_latency_ms"] for row in rows],
            )

            csv_path = run_dir / "scaling_results.csv"
            figure_path = run_dir / "scaling_latency.png"
            write_results_csv(rows, csv_path)
            plot_results(rows, output_path=figure_path, orth_fit=orth_fit)

            print(f"Results written to: {csv_path}")
            print(f"Plot written to: {figure_path}")
            print(f"Orthogonal latency vs log2(N) fit R^2: {orth_fit['r2']:.4f}")
    finally:
        if is_distributed and dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
