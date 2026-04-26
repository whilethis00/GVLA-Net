import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a NeurIPS-style LaTeX table from GVLA experiment results."
    )
    parser.add_argument(
        "--backbone-csv",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "results"
        / "vla_backbone_comparison"
        / "20260425_232225_sota_vla_head_comparison"
        / "vla_backbone_comparison.csv",
    )
    parser.add_argument(
        "--robustness-csv",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "results"
        / "legacy_artifacts"
        / "20260425_migrated_artifacts"
        / "robustness_noise_ablation.csv",
    )
    parser.add_argument("--backbone-dim", type=int, default=4096)
    parser.add_argument("--min-exp", type=int, default=15)
    parser.add_argument("--max-exp", type=int, default=19)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "neurips_tables",
    )
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def build_run_dir(output_root: Path, run_name: str | None) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = run_name if run_name is not None else "neurips_table_export"
    return output_root / f"{timestamp}_{safe_name}"


def write_args_snapshot(args: argparse.Namespace, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for key, value in sorted(vars(args).items()):
            handle.write(f"{key}: {value}\n")


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def load_accuracy_gap(path: Path) -> float:
    rows = read_csv(path)
    if not rows:
        return 0.0
    return max(float(row["accuracy_gap"]) for row in rows)


def filter_backbone_rows(
    rows: List[Dict[str, str]],
    backbone_dim: int,
    min_exp: int,
    max_exp: int,
) -> List[Dict[str, str]]:
    valid_actions = {1 << exponent for exponent in range(min_exp, max_exp + 1)}
    return [
        row
        for row in rows
        if int(row["backbone_dim"]) == backbone_dim
        and int(row["num_actions"]) in valid_actions
    ]


def build_table_rows(rows: List[Dict[str, str]]) -> List[Dict[str, float]]:
    by_key = {
        (int(row["num_actions"]), row["method"]): row
        for row in rows
    }
    action_sizes = sorted({int(row["num_actions"]) for row in rows})

    table_rows: List[Dict[str, float]] = []
    for num_actions in action_sizes:
        softmax_row = by_key[(num_actions, "softmax")]
        gvla_row = by_key[(num_actions, "gvla")]

        softmax_latency = float(softmax_row["latency_ms"])
        gvla_latency = float(gvla_row["latency_ms"])
        softmax_gflops = float(softmax_row["gflops"])
        gvla_gflops = float(gvla_row["gflops"])
        softmax_epr = float(softmax_row["efficiency_to_precision_ratio"])
        gvla_epr = float(gvla_row["efficiency_to_precision_ratio"])

        table_rows.append(
            {
                "num_actions": num_actions,
                "softmax_latency": softmax_latency,
                "gvla_latency": gvla_latency,
                "speedup": softmax_latency / gvla_latency,
                "softmax_gflops": softmax_gflops,
                "gvla_gflops": gvla_gflops,
                "flops_reduction": softmax_gflops / gvla_gflops,
                "epr_gain": gvla_epr / softmax_epr,
            }
        )

    return table_rows


def format_table(
    table_rows: List[Dict[str, float]],
    *,
    backbone_dim: int,
    accuracy_gap: float,
) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Head-level scaling comparison at OpenVLA-scale backbone width "
            f"($d={backbone_dim}$). Softmax denotes a linear projection followed by Softmax, "
            "while GVLA uses the proposed OrthogonalProjectionLayer with $O(\\log N)$ inference. "
            "Accuracy Gap$^\\dagger$ is taken from the robustness study and is reported "
            "separately because it was not measured per-$N$ in the backbone benchmark.}"
        ),
        "\\label{tab:vla_backbone_comparison}",
        "\\small",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{rcccccccc}",
        "\\toprule",
        "$N$ & Softmax Lat. & GVLA Lat. & Speedup & Softmax GFLOPs & GVLA GFLOPs & FLOPs Red. & EPR Gain & Acc. Gap$^\\dagger$ \\\\",
        "\\midrule",
    ]

    for row in table_rows:
        exponent = int(round(row["num_actions"]).bit_length() - 1)
        lines.append(
            (
                f"$2^{{{exponent}}}$ & "
                f"{row['softmax_latency']:.3f} & "
                f"{row['gvla_latency']:.3f} & "
                f"{row['speedup']:.2f}$\\times$ & "
                f"{row['softmax_gflops']:.3f} & "
                f"{row['gvla_gflops']:.5f} & "
                f"{row['flops_reduction']:.0f}$\\times$ & "
                f"{row['epr_gain']:.0f}$\\times$ & "
                f"{accuracy_gap:.2f} \\\\"
            )
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "",
            "\\vspace{2mm}",
            "\\begin{minipage}{0.97\\linewidth}",
            "\\footnotesize",
            (
                f"$^\\dagger$Robustness study (`\\sigma \\in [0, 0.5]`) yielded a maximum "
                f"Top-1 Accuracy Gap of {accuracy_gap:.2f} in the saved run. "
                "Orthogonality ablation further showed that GVLA accuracy drops sharply "
                "once $\\|WW^\\top - I\\|_F$ departs from near-zero, confirming that "
                "noise robustness is tightly coupled to preserving orthogonality."
            ),
            "\\end{minipage}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = build_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_args_snapshot(args, run_dir / "args.txt")

    backbone_rows = read_csv(args.backbone_csv)
    filtered_rows = filter_backbone_rows(
        backbone_rows,
        args.backbone_dim,
        args.min_exp,
        args.max_exp,
    )
    table_rows = build_table_rows(filtered_rows)
    accuracy_gap = load_accuracy_gap(args.robustness_csv)

    latex = format_table(
        table_rows,
        backbone_dim=args.backbone_dim,
        accuracy_gap=accuracy_gap,
    )

    output_path = run_dir / "neurips_vla_backbone_table.tex"
    output_path.write_text(latex)
    print(f"LaTeX table written to: {output_path}")


if __name__ == "__main__":
    main()
