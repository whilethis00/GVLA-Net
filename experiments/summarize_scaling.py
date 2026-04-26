import argparse
import csv
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize scaling benchmark results into a paper-ready table."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT
        / "experiments"
        / "results"
        / "legacy_artifacts"
        / "20260425_migrated_artifacts"
        / "scaling_results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "scaling_summary",
    )
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=[10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000],
    )
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def build_run_dir(output_root: Path, run_name: str | None) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = run_name if run_name is not None else "scaling_summary"
    return output_root / f"{timestamp}_{safe_name}"


def read_rows(input_csv: Path) -> List[Dict[str, float]]:
    with input_csv.open() as handle:
        reader = csv.DictReader(handle)
        return [{key: float(value) for key, value in row.items()} for row in reader]


def linear_fit(x_values: List[float], y_values: List[float]) -> Tuple[float, float]:
    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    denominator = sum((x - mean_x) ** 2 for x in x_values)
    slope = numerator / denominator
    intercept = mean_y - slope * mean_x
    return slope, intercept


def build_summary_rows(
    rows: List[Dict[str, float]],
    targets: List[int],
) -> List[Dict[str, float | str]]:
    by_n = {int(row["N"]): row for row in rows}

    soft_slope, soft_intercept = linear_fit(
        [row["N"] for row in rows],
        [row["softmax_latency_ms"] for row in rows],
    )
    orth_slope, orth_intercept = linear_fit(
        [math.log2(row["N"]) for row in rows],
        [row["orthogonal_latency_ms"] for row in rows],
    )

    summary_rows: List[Dict[str, float | str]] = []
    for target in targets:
        observed = by_n.get(target)
        if observed is not None:
            softmax_latency = observed["softmax_latency_ms"]
            orthogonal_latency = observed["orthogonal_latency_ms"]
            speedup = observed["speedup_x"]
            source = "measured"
        else:
            softmax_latency = soft_slope * target + soft_intercept
            orthogonal_latency = orth_slope * math.log2(target) + orth_intercept
            speedup = softmax_latency / orthogonal_latency
            source = "extrapolated"

        summary_rows.append(
            {
                "N": target,
                "softmax_latency_ms": softmax_latency,
                "orthogonal_latency_ms": orthogonal_latency,
                "speedup_x": speedup,
                "source": source,
            }
        )

    return summary_rows


def write_csv(rows: List[Dict[str, float | str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "N",
                "softmax_latency_ms",
                "orthogonal_latency_ms",
                "speedup_x",
                "source",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def format_markdown_table(rows: List[Dict[str, float | str]]) -> str:
    lines = [
        "| N | Softmax Latency (ms) | GVLA Latency (ms) | Speedup | Source |",
        "|---:|---:|---:|---:|:---|",
    ]
    for row in rows:
        lines.append(
            "| {N:,} | {soft:.4f} | {orth:.4f} | {speed:.2f}x | {source} |".format(
                N=int(row["N"]),
                soft=row["softmax_latency_ms"],
                orth=row["orthogonal_latency_ms"],
                speed=row["speedup_x"],
                source=row["source"],
            )
        )
    return "\n".join(lines) + "\n"


def write_markdown(
    rows: List[Dict[str, float | str]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(format_markdown_table(rows))


def write_args_snapshot(args: argparse.Namespace, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for key, value in sorted(vars(args).items()):
            handle.write(f"{key}: {value}\n")


def main() -> None:
    args = parse_args()
    run_dir = build_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_args_snapshot(args, run_dir / "args.txt")

    rows = read_rows(args.input_csv)
    summary_rows = build_summary_rows(rows, args.targets)

    csv_path = run_dir / "scaling_summary.csv"
    markdown_path = run_dir / "scaling_summary.md"
    write_csv(summary_rows, csv_path)
    write_markdown(summary_rows, markdown_path)

    print(f"Summary CSV written to: {csv_path}")
    print(f"Summary markdown written to: {markdown_path}")


if __name__ == "__main__":
    main()
