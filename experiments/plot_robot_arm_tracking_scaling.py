import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "experiments" / "results" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot robot-arm tracking scaling and trajectory quality for GVLA vs dense baselines."
    )
    parser.add_argument(
        "--run-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="Run directories produced by robot_arm_tracking_demo.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default="fig_tracking_scaling",
    )
    return parser.parse_args()


def read_summary(summary_path: Path) -> Dict[str, Dict[str, float]]:
    rows: Dict[str, Dict[str, float]] = {}
    with summary_path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            controller = row["controller"]
            rows[controller] = {
                "action_space_size": float(row["action_space_size"]),
                "arrival_step": float(row["arrival_step"]),
                "arrival_time_s": float(row["arrival_time_s"]),
                "mean_latency_ms": float(row["mean_latency_ms"]),
                "fps": float(row["fps"]),
                "final_error": float(row["final_error"]),
            }
    if {"gvla", "dense"} - rows.keys():
        raise ValueError(f"Expected gvla and dense rows in {summary_path}")
    return rows


def read_trace(trace_path: Path) -> Dict[str, np.ndarray]:
    with trace_path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    columns = reader.fieldnames or []
    trace: Dict[str, np.ndarray] = {}
    for column in columns:
        trace[column] = np.asarray([float(row[column]) for row in rows], dtype=np.float64)
    return trace


def summarize_trace(trace: Dict[str, np.ndarray], window: int = 50) -> Dict[str, float]:
    gvla_error = trace["gvla_error"]
    dense_error = trace["dense_error"]
    win = min(window, gvla_error.shape[0])
    return {
        "gvla_mean_last_window_error": float(np.mean(gvla_error[-win:])),
        "dense_mean_last_window_error": float(np.mean(dense_error[-win:])),
        "gvla_median_last_window_error": float(np.median(gvla_error[-win:])),
        "dense_median_last_window_error": float(np.median(dense_error[-win:])),
        "window_size": float(win),
    }


def format_action_space(value: float) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"{int(round(value / 1_000))}k"
    return str(int(value))


def load_runs(run_dirs: List[Path]) -> Tuple[List[Dict[str, object]], Dict[str, np.ndarray], Path]:
    loaded_runs: List[Dict[str, object]] = []
    largest_run: Dict[str, object] | None = None
    largest_trace: Dict[str, np.ndarray] | None = None
    largest_run_dir: Path | None = None

    for run_dir in run_dirs:
        summary = read_summary(run_dir / "summary.csv")
        trace = read_trace(run_dir / "tracking_trace.csv")
        trace_stats = summarize_trace(trace)
        action_space_size = summary["gvla"]["action_space_size"]
        loaded_runs.append(
            {
                "run_dir": run_dir,
                "action_space_size": action_space_size,
                "gvla": summary["gvla"],
                "dense": summary["dense"],
                "trace_stats": trace_stats,
            }
        )
        if largest_run is None or action_space_size > float(largest_run["action_space_size"]):
            largest_run = loaded_runs[-1]
            largest_trace = trace
            largest_run_dir = run_dir

    loaded_runs.sort(key=lambda item: float(item["action_space_size"]))
    if largest_trace is None or largest_run_dir is None:
        raise ValueError("No run data loaded.")
    return loaded_runs, largest_trace, largest_run_dir


def build_aggregate_rows(loaded_runs: List[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for run in loaded_runs:
        for controller in ("gvla", "dense"):
            row = {
                "run_name": Path(run["run_dir"]).name,
                "controller": controller,
                "action_space_size": int(run["action_space_size"]),
                "mean_last_window_error": run["trace_stats"][f"{controller}_mean_last_window_error"],
                "median_last_window_error": run["trace_stats"][f"{controller}_median_last_window_error"],
                "window_size": int(run["trace_stats"]["window_size"]),
            }
            row.update(run[controller])  # type: ignore[arg-type]
            rows.append(row)
    return rows


def write_aggregate_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "run_name",
        "controller",
        "action_space_size",
        "arrival_step",
        "arrival_time_s",
        "mean_latency_ms",
        "fps",
        "final_error",
        "mean_last_window_error",
        "median_last_window_error",
        "window_size",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def add_gradient_path(ax, x: np.ndarray, y: np.ndarray, cmap_name: str, linewidth: float, label: str) -> None:
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    t = np.linspace(0.0, 1.0, len(segments))
    collection = LineCollection(segments, cmap=plt.get_cmap(cmap_name), norm=plt.Normalize(0.0, 1.0))
    collection.set_array(t)
    collection.set_linewidth(linewidth)
    ax.add_collection(collection)
    ax.plot([], [], color=plt.get_cmap(cmap_name)(0.75), linewidth=linewidth, label=label)


def plot_scaling(loaded_runs: List[Dict[str, object]], largest_trace: Dict[str, np.ndarray], largest_run_dir: Path, output_png: Path, output_pdf: Path) -> None:
    gvla_color = "#b73a52"
    dense_color = "#69707a"
    target_color = "#111111"
    accent_color = "#d68c1f"

    action_sizes = np.asarray([float(run["action_space_size"]) for run in loaded_runs], dtype=np.float64)
    gvla_fps = np.asarray([float(run["gvla"]["fps"]) for run in loaded_runs], dtype=np.float64)  # type: ignore[index]
    dense_fps = np.asarray([float(run["dense"]["fps"]) for run in loaded_runs], dtype=np.float64)  # type: ignore[index]
    gvla_error = np.asarray([float(run["trace_stats"]["gvla_mean_last_window_error"]) for run in loaded_runs], dtype=np.float64)  # type: ignore[index]
    dense_error = np.asarray([float(run["trace_stats"]["dense_mean_last_window_error"]) for run in loaded_runs], dtype=np.float64)  # type: ignore[index]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5), constrained_layout=True)
    ax_fps, ax_error, ax_xy, ax_time = axes.flatten()

    for ax in (ax_fps, ax_error):
        ax.grid(True, alpha=0.22, linewidth=0.8)
        ax.set_xscale("log", base=2)
        ax.set_xticks(action_sizes)
        ax.set_xticklabels([format_action_space(value) for value in action_sizes])

    ax_fps.plot(action_sizes, gvla_fps, marker="o", markersize=8, linewidth=2.8, color=gvla_color, label="GVLA-Net")
    ax_fps.plot(action_sizes, dense_fps, marker="o", markersize=8, linewidth=2.8, color=dense_color, label="Dense")
    ax_fps.set_title("Scaling Crossover")
    ax_fps.set_xlabel("Action Space Size")
    ax_fps.set_ylabel("Controller Throughput (FPS)")
    ax_fps.legend(frameon=False, loc="upper right")
    crossover_idx = int(np.argmax(gvla_fps > dense_fps))
    if np.any(gvla_fps > dense_fps):
        ax_fps.scatter(
            [action_sizes[crossover_idx]],
            [gvla_fps[crossover_idx]],
            s=120,
            facecolors="none",
            edgecolors=accent_color,
            linewidths=2.0,
            zorder=5,
        )
        ax_fps.annotate(
            "GVLA overtakes dense",
            xy=(action_sizes[crossover_idx], gvla_fps[crossover_idx]),
            xytext=(18, 20),
            textcoords="offset points",
            fontsize=10,
            color=accent_color,
            arrowprops={"arrowstyle": "->", "color": accent_color, "lw": 1.4},
        )

    ax_error.plot(action_sizes, gvla_error, marker="o", markersize=8, linewidth=2.8, color=gvla_color, label="GVLA-Net")
    ax_error.plot(action_sizes, dense_error, marker="o", markersize=8, linewidth=2.8, color=dense_color, label="Dense")
    ax_error.set_title("Steady-State Tracking Error")
    ax_error.set_xlabel("Action Space Size")
    ax_error.set_ylabel("Mean Error Over Last 50 Steps")

    target_x = largest_trace["target_x"][0]
    target_y = largest_trace["target_y"][0]
    add_gradient_path(ax_xy, largest_trace["dense_x"], largest_trace["dense_y"], "Greys", 2.6, "Dense path")
    add_gradient_path(ax_xy, largest_trace["gvla_x"], largest_trace["gvla_y"], "YlOrRd", 2.6, "GVLA path")
    ax_xy.scatter([target_x], [target_y], color=target_color, s=55, marker="x", linewidths=2.0, label="Target")
    ax_xy.scatter([largest_trace["dense_x"][0]], [largest_trace["dense_y"][0]], color="#5f6368", s=28, alpha=0.9, marker="o")
    ax_xy.scatter([largest_trace["gvla_x"][0]], [largest_trace["gvla_y"][0]], color="#8f1731", s=28, alpha=0.9, marker="o")
    ax_xy.set_title(f"Trajectory Overlay ({largest_run_dir.name.split('_', 2)[-1]})")
    ax_xy.set_xlabel("X Position")
    ax_xy.set_ylabel("Y Position")
    ax_xy.grid(True, alpha=0.22, linewidth=0.8)
    all_x = np.concatenate((largest_trace["dense_x"], largest_trace["gvla_x"], np.asarray([target_x])))
    all_y = np.concatenate((largest_trace["dense_y"], largest_trace["gvla_y"], np.asarray([target_y])))
    x_mid = 0.5 * (float(np.min(all_x)) + float(np.max(all_x)))
    y_mid = 0.5 * (float(np.min(all_y)) + float(np.max(all_y)))
    span = max(float(np.max(all_x) - np.min(all_x)), float(np.max(all_y) - np.min(all_y)))
    margin = 0.08 * span
    half_width = 0.5 * span + margin
    ax_xy.set_xlim(x_mid - half_width, x_mid + half_width)
    ax_xy.set_ylim(y_mid - half_width, y_mid + half_width)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.legend(frameon=False, loc="lower right")

    steps = largest_trace["step"]
    focus = steps <= 120
    ax_time.plot(steps[focus], largest_trace["dense_error"][focus], color=dense_color, linewidth=2.2, label="Dense error")
    ax_time.plot(steps[focus], largest_trace["gvla_error"][focus], color=gvla_color, linewidth=2.2, label="GVLA error")
    ax_time.axhline(y=0.03, color=accent_color, linewidth=1.5, linestyle="--", alpha=0.8)
    ax_time.text(0.98, 0.04, "arrival threshold", transform=ax_time.transAxes, ha="right", va="bottom", color=accent_color, fontsize=9)
    ax_time.set_title("Convergence Window")
    ax_time.set_xlabel("Control Step (0-120)")
    ax_time.set_ylabel("Euclidean Tracking Error")
    ax_time.grid(True, alpha=0.22, linewidth=0.8)

    fig.suptitle("GVLA Tracking Crossover Under Expanding Action Resolution", fontsize=18, y=1.02)
    fig.savefig(output_png, dpi=220, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded_runs, largest_trace, largest_run_dir = load_runs(args.run_dirs)
    aggregate_rows = build_aggregate_rows(loaded_runs)

    output_png = args.output_dir / f"{args.figure_name}.png"
    output_pdf = args.output_dir / f"{args.figure_name}.pdf"
    output_csv = args.output_dir / f"{args.figure_name}.csv"

    write_aggregate_csv(aggregate_rows, output_csv)
    plot_scaling(loaded_runs, largest_trace, largest_run_dir, output_png, output_pdf)

    print(f"Aggregate summary written to: {output_csv}")
    print(f"Figure written to: {output_png}")
    print(f"Figure written to: {output_pdf}")


if __name__ == "__main__":
    main()
