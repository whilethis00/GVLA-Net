import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import tensorstore as ts
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.universal_vla_comparison import render_latex_table
from models.layers import OrthogonalProjectionLayer


STABLE_FIELDS = (
    "model_name",
    "backbone_dim",
    "dim_status",
    "native_head_type",
    "native_action_interface",
    "attack_point",
    "public_source",
    "num_actions",
    "precision_bits",
    "dense_head_latency_ms",
    "dense_latency_mode",
    "gvla_head_latency_ms",
    "latency_delta_ms",
    "speedup_x",
    "dense_head_memory_mb",
    "gvla_head_memory_mb",
    "memory_reduction_x",
    "dense_head_flops_g",
    "gvla_head_flops_g",
    "flops_reduction_x",
)


@dataclass
class Pi05VerifiedRow:
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


class DenseTailModule(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        out_features, in_features = weight.shape
        self.proj = nn.Linear(in_features, out_features, bias=True, device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            self.proj.weight.copy_(weight)
            self.proj.bias.copy_(bias)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.proj(hidden))


class GVLATailModule(nn.Module):
    def __init__(self, input_dim: int, num_actions: int, *, device: torch.device, dtype: torch.dtype) -> None:
        super().__init__()
        self.head = OrthogonalProjectionLayer(
            input_dim=input_dim,
            num_codes=num_actions,
            use_ste=True,
            device=device,
            dtype=dtype,
        ).to(device=device, dtype=dtype)

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.head(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verified pi0.5 tail benchmark with direct tensorstore checkpoint weights."
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path.home() / ".cache" / "openpi" / "openpi-assets" / "checkpoints" / "pi05_droid",
    )
    parser.add_argument(
        "--num-actions",
        type=int,
        nargs="+",
        default=[1 << 10, 1 << 15, 1 << 20],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=15)
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
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16" if torch.cuda.is_available() else "float32",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "pi05_verified_gvla_benchmark",
    )
    parser.add_argument("--run-name", type=str, default="pi05_verified")
    parser.add_argument(
        "--stable-csv-path",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "neurips_tables" / "SOTA_VLA_Comparison.csv",
    )
    parser.add_argument(
        "--stable-tex-path",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "neurips_tables" / "SOTA_VLA_Comparison.tex",
    )
    parser.add_argument(
        "--skip-stable-update",
        action="store_true",
        help="Do not overwrite the stable comparison CSV/TEX.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def build_run_dir(output_root: Path, run_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return output_root / f"{timestamp}_{run_name}"


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
        return (time.perf_counter() - start_time) * 1000.0 / measure_iters


def read_tensorstore_array(base_kv: dict, path: str) -> torch.Tensor:
    arr = ts.open({"driver": "zarr", "kvstore": base_kv, "path": path}).result()
    return torch.from_numpy(arr.read().result())


def load_pi05_action_head_weights(checkpoint_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    params_dir = checkpoint_dir / "params" / "ocdbt.process_0"
    base_kv = {
        "driver": "ocdbt",
        "base": {
            "driver": "file",
            "path": str(params_dir),
        },
    }
    kernel = read_tensorstore_array(base_kv, "params.action_out_proj.kernel")
    bias = read_tensorstore_array(base_kv, "params.action_out_proj.bias")
    return kernel, bias


def tile_dense_head(
    kernel: torch.Tensor,
    bias: torch.Tensor,
    num_actions: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if kernel.ndim != 2:
        raise ValueError(f"Expected 2D kernel, got shape={tuple(kernel.shape)}")
    input_dim, native_actions = kernel.shape
    repeats = math.ceil(num_actions / native_actions)
    weight = kernel.transpose(0, 1).repeat(repeats, 1)[:num_actions, :]
    tiled_bias = bias.repeat(repeats)[:num_actions]
    return weight.to(device=device, dtype=dtype), tiled_bias.to(device=device, dtype=dtype)


def dense_head_parameter_bytes(backbone_dim: int, num_actions: int, dtype: torch.dtype) -> int:
    dtype_bytes = max(torch.tensor([], dtype=dtype).element_size(), 1)
    return backbone_dim * num_actions * dtype_bytes


def gvla_head_parameter_bytes(backbone_dim: int, num_actions: int, dtype: torch.dtype) -> int:
    dtype_bytes = max(torch.tensor([], dtype=dtype).element_size(), 1)
    return backbone_dim * int(math.ceil(math.log2(num_actions))) * dtype_bytes


def estimate_dense_head_flops(batch_size: int, action_horizon: int, backbone_dim: int, num_actions: int) -> float:
    projection_flops = 2.0 * batch_size * action_horizon * backbone_dim * num_actions
    softmax_flops = 5.0 * batch_size * action_horizon * num_actions
    return projection_flops + softmax_flops


def estimate_gvla_head_flops(batch_size: int, action_horizon: int, backbone_dim: int, num_actions: int) -> float:
    code_length = int(math.ceil(math.log2(num_actions)))
    projection_flops = 2.0 * batch_size * action_horizon * backbone_dim * code_length
    hashing_flops = 2.0 * batch_size * action_horizon * code_length
    return projection_flops + hashing_flops


def read_stable_rows(csv_path: Path) -> list[OrderedDict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open() as handle:
        return [OrderedDict(row) for row in csv.DictReader(handle)]


def overwrite_pi05_rows(
    existing_rows: list[OrderedDict[str, str]],
    verified_rows: Iterable[Pi05VerifiedRow],
) -> list[OrderedDict[str, str]]:
    verified_map = {(row.model_name, str(row.num_actions)): row for row in verified_rows}
    retained = [
        row
        for row in existing_rows
        if (row.get("model_name"), row.get("num_actions")) not in verified_map
    ]

    for row in verified_rows:
        retained.append(
            OrderedDict(
                (field, str(getattr(row, field)))
                for field in STABLE_FIELDS
            )
        )

    def sort_key(row: OrderedDict[str, str]) -> tuple[str, int]:
        return row["model_name"], int(row["num_actions"])

    retained.sort(key=sort_key)
    return retained


def write_stable_csv(rows: list[OrderedDict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STABLE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def latex_rows_from_csv(rows: list[OrderedDict[str, str]]) -> list[Pi05VerifiedRow]:
    parsed_rows: list[Pi05VerifiedRow] = []
    for row in rows:
        parsed_rows.append(
            Pi05VerifiedRow(
                model_name=row["model_name"],
                backbone_dim=int(row["backbone_dim"]),
                dim_status=row["dim_status"],
                native_head_type=row["native_head_type"],
                native_action_interface=row["native_action_interface"],
                attack_point=row["attack_point"],
                public_source=row["public_source"],
                num_actions=int(row["num_actions"]),
                precision_bits=int(row["precision_bits"]),
                dense_head_latency_ms=float(row["dense_head_latency_ms"]),
                dense_latency_mode=row["dense_latency_mode"],
                gvla_head_latency_ms=float(row["gvla_head_latency_ms"]),
                latency_delta_ms=float(row["latency_delta_ms"]),
                speedup_x=float(row["speedup_x"]),
                dense_head_memory_mb=float(row["dense_head_memory_mb"]),
                gvla_head_memory_mb=float(row["gvla_head_memory_mb"]),
                memory_reduction_x=float(row["memory_reduction_x"]),
                dense_head_flops_g=float(row["dense_head_flops_g"]),
                gvla_head_flops_g=float(row["gvla_head_flops_g"]),
                flops_reduction_x=float(row["flops_reduction_x"]),
            )
        )
    return parsed_rows


def write_verified_rows_csv(rows: list[Pi05VerifiedRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STABLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: getattr(row, field) for field in STABLE_FIELDS})


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    torch.manual_seed(args.seed)

    run_dir = build_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_args_snapshot(args, run_dir / "args.txt")

    kernel, bias = load_pi05_action_head_weights(args.checkpoint_dir)
    backbone_dim = int(kernel.shape[0])
    native_action_dim = int(kernel.shape[1])
    tail_inputs = torch.randn(
        args.batch_size,
        args.action_horizon,
        backbone_dim,
        device=device,
        dtype=dtype,
    )

    rows: list[Pi05VerifiedRow] = []
    for num_actions in sorted(args.num_actions):
        dense_weight, dense_bias = tile_dense_head(
            kernel,
            bias,
            num_actions,
            device=device,
            dtype=dtype,
        )
        dense_module = DenseTailModule(dense_weight, dense_bias)
        dense_latency_ms = benchmark_module(
            dense_module,
            tail_inputs,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )
        del dense_module, dense_weight, dense_bias

        gvla_module = GVLATailModule(
            input_dim=backbone_dim,
            num_actions=num_actions,
            device=device,
            dtype=dtype,
        )
        gvla_latency_ms = benchmark_module(
            gvla_module,
            tail_inputs,
            warmup_iters=args.warmup_iters,
            measure_iters=args.measure_iters,
        )
        del gvla_module
        if device.type == "cuda":
            torch.cuda.empty_cache()

        dense_memory_mb = dense_head_parameter_bytes(backbone_dim, num_actions, dtype) / (1024.0 * 1024.0)
        gvla_memory_mb = gvla_head_parameter_bytes(backbone_dim, num_actions, dtype) / (1024.0 * 1024.0)
        dense_flops_g = estimate_dense_head_flops(
            args.batch_size,
            args.action_horizon,
            backbone_dim,
            num_actions,
        ) / 1e9
        gvla_flops_g = estimate_gvla_head_flops(
            args.batch_size,
            args.action_horizon,
            backbone_dim,
            num_actions,
        ) / 1e9

        rows.append(
            Pi05VerifiedRow(
                model_name="pi0.5",
                backbone_dim=backbone_dim,
                dim_status=(
                    "verified from local pi05_droid tensorstore checkpoint "
                    f"(action_out_proj={backbone_dim}x{native_action_dim}, action_horizon={args.action_horizon})"
                ),
                native_head_type="flow-matching action head",
                native_action_interface="continuous action chunks via flow matching",
                attack_point="Verified checkpoint-tail swap: replace dense action routing with geometric hashing.",
                public_source="https://github.com/Physical-Intelligence/openpi",
                num_actions=num_actions,
                precision_bits=int(math.ceil(math.log2(num_actions))),
                dense_head_latency_ms=dense_latency_ms,
                dense_latency_mode="verified_tiled_actual_weights",
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

    verified_csv = run_dir / "pi05_verified_gvla_benchmark.csv"
    write_verified_rows_csv(rows, verified_csv)

    manifest = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "loaded_weight_paths": [
            "params.action_out_proj.kernel",
            "params.action_out_proj.bias",
        ],
        "kernel_shape": list(kernel.shape),
        "bias_shape": list(bias.shape),
        "device": str(device),
        "dtype": args.dtype,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    stable_rows = overwrite_pi05_rows(read_stable_rows(args.stable_csv_path), rows)
    merged_csv = run_dir / "SOTA_VLA_Comparison.csv"
    merged_tex = run_dir / "SOTA_VLA_Comparison.tex"
    write_stable_csv(stable_rows, merged_csv)
    merged_tex.write_text(render_latex_table(latex_rows_from_csv(stable_rows)))

    if not args.skip_stable_update:
        args.stable_csv_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(merged_csv, args.stable_csv_path)
        shutil.copyfile(merged_tex, args.stable_tex_path)

    print(f"Verified CSV written to: {verified_csv}")
    print(f"Merged comparison CSV written to: {merged_csv}")
    print(f"Merged comparison TeX written to: {merged_tex}")
    if not args.skip_stable_update:
        print(f"Stable CSV updated: {args.stable_csv_path}")
        print(f"Stable TeX updated: {args.stable_tex_path}")


if __name__ == "__main__":
    main()
