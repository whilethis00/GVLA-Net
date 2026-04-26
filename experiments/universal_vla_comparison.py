import argparse
import csv
import math
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.vla_universal_adapter import (
    SyntheticUniversalAdapter,
    list_universal_specs,
    resolve_octo_backbone_dim,
    resolve_openpi05_backbone_dim,
)
from models.layers import OrthogonalProjectionLayer


@dataclass
class UniversalBenchmarkRow:
    model_name: str
    backbone_dim: int
    dim_status: str
    native_head_type: str
    native_action_interface: str
    attack_point: str
    public_source: str
    num_actions: int
    precision_bits: int
    dense_head_latency_ms: float
    dense_latency_mode: str
    gvla_head_latency_ms: float
    latency_delta_ms: float
    speedup_x: float
    dense_head_memory_mb: float
    gvla_head_memory_mb: float
    memory_reduction_x: float
    dense_head_flops_g: float
    gvla_head_flops_g: float
    flops_reduction_x: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Universal head-level comparison across representative VLA families."
    )
    parser.add_argument(
        "--num-actions",
        type=int,
        nargs="+",
        default=[1 << 10, 1 << 15, 1 << 20],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--measure-iters", type=int, default=50)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16" if torch.cuda.is_available() else "float32",
        choices=("float16", "bfloat16", "float32", "float64"),
    )
    parser.add_argument(
        "--max-dense-weight-mb",
        type=float,
        default=1024.0,
        help="Above this threshold the dense head is projected instead of instantiated.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "universal_vla_comparison",
    )
    parser.add_argument(
        "--stable-tex-path",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "neurips_tables" / "SOTA_VLA_Comparison.tex",
    )
    parser.add_argument(
        "--stable-csv-path",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "neurips_tables" / "SOTA_VLA_Comparison.csv",
    )
    parser.add_argument(
        "--third-party-root",
        type=Path,
        default=PROJECT_ROOT / "third_party",
    )
    parser.add_argument(
        "--octo-python",
        type=Path,
        default=Path.home() / ".conda" / "envs" / "octo_env" / "bin" / "python",
    )
    parser.add_argument(
        "--openpi-python",
        type=Path,
        default=Path.home() / ".conda" / "envs" / "openpi_env" / "bin" / "python",
    )
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
        "float64": torch.float64,
    }
    return mapping[dtype_name]


def build_run_dir(output_root: Path, run_name: str | None) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = run_name if run_name is not None else "universal_vla_comparison"
    return output_root / f"{timestamp}_{safe_name}"


def write_args_snapshot(args: argparse.Namespace, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for key, value in sorted(vars(args).items()):
            handle.write(f"{key}: {value}\n")


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


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
        synchronize_if_needed(inputs.device)

        if inputs.device.type == "cuda":
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


def dense_head_parameter_bytes(
    backbone_dim: int,
    num_actions: int,
    dtype: torch.dtype,
) -> int:
    dtype_bytes = max(torch.tensor([], dtype=dtype).element_size(), 1)
    return backbone_dim * num_actions * dtype_bytes


def gvla_head_parameter_bytes(
    backbone_dim: int,
    num_actions: int,
    dtype: torch.dtype,
) -> int:
    dtype_bytes = max(torch.tensor([], dtype=dtype).element_size(), 1)
    return backbone_dim * int(math.ceil(math.log2(num_actions))) * dtype_bytes


def estimate_dense_head_flops(
    batch_size: int,
    backbone_dim: int,
    num_actions: int,
) -> float:
    projection_flops = 2.0 * batch_size * backbone_dim * num_actions
    softmax_flops = 5.0 * batch_size * num_actions
    return projection_flops + softmax_flops


def estimate_gvla_head_flops(
    batch_size: int,
    backbone_dim: int,
    num_actions: int,
) -> float:
    code_length = int(math.ceil(math.log2(num_actions)))
    projection_flops = 2.0 * batch_size * backbone_dim * code_length
    hashing_flops = 2.0 * batch_size * code_length
    return projection_flops + hashing_flops


def fit_linear_latency_projection(samples: List[tuple[int, float]]) -> tuple[float, float]:
    if not samples:
        raise ValueError("Need at least one latency sample for projection.")
    if len(samples) == 1:
        n_value, latency_ms = samples[0]
        return latency_ms / max(n_value, 1), 0.0

    x = torch.tensor([sample[0] for sample in samples], dtype=torch.float64)
    y = torch.tensor([sample[1] for sample in samples], dtype=torch.float64)
    centered_x = x - x.mean()
    centered_y = y - y.mean()
    slope = (centered_x * centered_y).sum() / centered_x.square().sum()
    intercept = y.mean() - slope * x.mean()
    return slope.item(), intercept.item()


def should_measure_dense_head(
    backbone_dim: int,
    num_actions: int,
    *,
    dtype: torch.dtype,
    max_dense_weight_mb: float,
) -> bool:
    weight_mb = dense_head_parameter_bytes(backbone_dim, num_actions, dtype) / (1024.0 * 1024.0)
    return weight_mb <= max_dense_weight_mb


def write_results_csv(rows: List[UniversalBenchmarkRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(UniversalBenchmarkRow.__annotations__.keys()),
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def build_table_rows(rows: List[UniversalBenchmarkRow]) -> List[UniversalBenchmarkRow]:
    by_model = {(row.model_name, row.num_actions): row for row in rows}
    target_actions = sorted({row.num_actions for row in rows})
    selected_rows: List[UniversalBenchmarkRow] = []
    for model_name in ("OpenVLA-7B", "RT-2-X", "Octo-Base", "pi0.5"):
        for num_actions in target_actions:
            row = by_model.get((model_name, num_actions))
            if row is not None:
                selected_rows.append(row)
    return selected_rows


def render_latex_table(rows: List[UniversalBenchmarkRow]) -> str:
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        (
            "\\caption{Universal head-swap comparison across representative VLA families. "
            "For each backbone, $d$ is resolved from model config/output when available and "
            "otherwise falls back to a documented proxy. Dense head latency is measured when "
            "the action head fits within the configured VRAM budget and linearly projected "
            "beyond that point; GVLA latency is measured directly because it scales with "
            "$\\lceil \\log_2 N \\rceil$. Memory reports parameter-state footprint for the "
            "terminal action-routing head only, isolating the savings unlocked by replacing "
            "dense routing with orthogonal geometric hashing.}"
        ),
        "\\label{tab:sota_vla_comparison}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{llrrrcccr}",
        "\\toprule",
        "Model & Attack Point & $d$ & $N$ & Dense$\\to$GVLA $\\Delta$ms & Speedup & Dense Mem. MB & GVLA Mem. MB & Mem. Red. \\\\",
        "\\midrule",
    ]

    for row in rows:
        exponent = int(round(math.log2(row.num_actions)))
        lines.append(
            (
                f"{row.model_name} & "
                f"{row.attack_point} & "
                f"{row.backbone_dim} & "
                f"$2^{{{exponent}}}$ & "
                f"{row.latency_delta_ms:.3f} & "
                f"{row.speedup_x:.2f}$\\times$ & "
                f"{row.dense_head_memory_mb:.1f} & "
                f"{row.gvla_head_memory_mb:.4f} & "
                f"{row.memory_reduction_x:.0f}$\\times$ \\\\"
            )
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)

    run_dir = build_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_args_snapshot(args, run_dir / "args.txt")

    rows: List[UniversalBenchmarkRow] = []
    for spec in list_universal_specs():
        adapter = SyntheticUniversalAdapter(
            spec,
            device=device,
            dtype=dtype,
        )
        if spec.model_name == "Octo-Base":
            adapter.backbone_dim, adapter.dim_status = resolve_octo_backbone_dim(
                args.third_party_root,
                checkpoint_name="octo-base-1.5",
                env_python=args.octo_python,
            )
        elif spec.model_name == "pi0.5":
            adapter.backbone_dim, adapter.dim_status = resolve_openpi05_backbone_dim(
                args.third_party_root,
                env_python=args.openpi_python,
            )
        inputs = adapter.synthetic_embeddings(args.batch_size)

        dense_latency_samples: List[tuple[int, float]] = []
        gvla_latencies: Dict[int, float] = {}

        for num_actions in sorted(args.num_actions):
            if should_measure_dense_head(
                adapter.backbone_dim,
                num_actions,
                dtype=dtype,
                max_dense_weight_mb=args.max_dense_weight_mb,
            ):
                dense_head = nn.Sequential(
                    nn.Linear(
                        adapter.backbone_dim,
                        num_actions,
                        bias=False,
                        device=device,
                        dtype=dtype,
                    ),
                    nn.Softmax(dim=-1),
                )
                dense_latency_ms = benchmark_module(
                    dense_head,
                    inputs,
                    warmup_iters=args.warmup_iters,
                    measure_iters=args.measure_iters,
                )
                dense_latency_samples.append((num_actions, dense_latency_ms))
                del dense_head
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            gvla_head = OrthogonalProjectionLayer(
                input_dim=adapter.backbone_dim,
                num_codes=num_actions,
                use_ste=True,
                device=device,
                dtype=dtype,
            ).to(device=device, dtype=dtype)
            gvla_latencies[num_actions] = benchmark_module(
                gvla_head,
                inputs,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
            )
            del gvla_head

        slope, intercept = fit_linear_latency_projection(dense_latency_samples)

        dense_latency_lookup = {n_value: latency_ms for n_value, latency_ms in dense_latency_samples}
        for num_actions in sorted(args.num_actions):
            dense_latency_ms = dense_latency_lookup.get(num_actions)
            dense_latency_mode = "measured"
            if dense_latency_ms is None:
                dense_latency_ms = max(slope * num_actions + intercept, 0.0)
                dense_latency_mode = "projected"

            gvla_latency_ms = gvla_latencies[num_actions]
            dense_memory_mb = dense_head_parameter_bytes(
                adapter.backbone_dim,
                num_actions,
                dtype,
            ) / (1024.0 * 1024.0)
            gvla_memory_mb = gvla_head_parameter_bytes(
                adapter.backbone_dim,
                num_actions,
                dtype,
            ) / (1024.0 * 1024.0)
            dense_flops_g = estimate_dense_head_flops(
                args.batch_size,
                adapter.backbone_dim,
                num_actions,
            ) / 1e9
            gvla_flops_g = estimate_gvla_head_flops(
                args.batch_size,
                adapter.backbone_dim,
                num_actions,
            ) / 1e9

            rows.append(
                UniversalBenchmarkRow(
                    model_name=spec.model_name,
                    backbone_dim=adapter.backbone_dim,
                    dim_status=adapter.dim_status,
                    native_head_type=spec.native_head_type,
                    native_action_interface=spec.native_action_interface,
                    attack_point=spec.attack_point,
                    public_source=spec.public_source,
                    num_actions=num_actions,
                    precision_bits=int(math.ceil(math.log2(num_actions))),
                    dense_head_latency_ms=dense_latency_ms,
                    dense_latency_mode=dense_latency_mode,
                    gvla_head_latency_ms=gvla_latency_ms,
                    latency_delta_ms=dense_latency_ms - gvla_latency_ms,
                    speedup_x=dense_latency_ms / max(gvla_latency_ms, 1e-12),
                    dense_head_memory_mb=dense_memory_mb,
                    gvla_head_memory_mb=gvla_memory_mb,
                    memory_reduction_x=dense_memory_mb / max(gvla_memory_mb, 1e-12),
                    dense_head_flops_g=dense_flops_g,
                    gvla_head_flops_g=gvla_flops_g,
                    flops_reduction_x=dense_flops_g / max(gvla_flops_g, 1e-12),
                )
            )

    rows.sort(key=lambda row: (row.model_name, row.num_actions))
    csv_path = run_dir / "universal_vla_comparison.csv"
    tex_path = run_dir / "SOTA_VLA_Comparison.tex"
    write_results_csv(rows, csv_path)
    tex_path.write_text(render_latex_table(build_table_rows(rows)))

    args.stable_csv_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(csv_path, args.stable_csv_path)
    shutil.copyfile(tex_path, args.stable_tex_path)

    print(f"CSV written to: {csv_path}")
    print(f"LaTeX written to: {tex_path}")
    print(f"Stable CSV updated: {args.stable_csv_path}")
    print(f"Stable LaTeX updated: {args.stable_tex_path}")


if __name__ == "__main__":
    main()
