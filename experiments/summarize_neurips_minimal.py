"""
summarize_neurips_minimal.py
============================

Collect the minimal non-GPU artifacts for the NeurIPS framing:
  - precision custom2.5 success rates with Wilson CIs
  - Gray code ablation table
  - latency summaries from head-only and batch results
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Union


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "experiments" / "results"
OUT_DIR = RESULTS_DIR / "neurips_minimal_summary"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    if total <= 0:
        return 0.0, 0.0
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    radius = (
        z
        * math.sqrt((phat * (1 - phat) + z * z / (4 * total)) / total)
        / denom
    )
    return center - radius, center + radius


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def precision_table(path: Path) -> List[Dict[str, Union[float, int, str]]]:
    data = load_json(path)
    results = data["results"]
    # Newer files already store structured rows with CI values.
    if isinstance(results, list):
        rows = []
        n_rollouts = int(data.get("n_rollouts", 200))
        for row in results:
            rows.append(
                {
                    "setting": row["setting"],
                    "label": row["label"],
                    "success_rate": float(row["success_rate"]),
                    "successes": int(row["successes"]),
                    "n_rollouts": n_rollouts,
                    "ci95_low": float(row["ci95_low"]),
                    "ci95_high": float(row["ci95_high"]),
                }
            )
        return rows

    # Backward compatibility for older dict-style result files.
    n_rollouts = int(data.get("n_rollouts", 200))
    rows = []
    for key in ["inf", "256", "1024", "2048"]:
        success_rate = float(results[key])
        successes = int(round(success_rate * n_rollouts))
        ci_low, ci_high = wilson_interval(successes, n_rollouts)
        rows.append(
            {
                "setting": "continuous" if key == "inf" else f"gvla_{key}",
                "label": "continuous" if key == "inf" else key,
                "success_rate": success_rate,
                "successes": successes,
                "n_rollouts": n_rollouts,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    return rows


def gray_table(path: Path) -> List[Dict[str, Union[float, int, str]]]:
    data = load_json(path)
    rows = data["rollout_results"]
    target = {(128, "dense"), (128, "gvla"), (128, "gvla_gray"), (1024, "dense"), (1024, "gvla"), (1024, "gvla_gray")}
    selected = []
    for row in rows:
        head_key = row["exp_name"].rsplit("_", 1)[0]
        key = (int(row["n_bins"]), head_key)
        if key not in target:
            continue
        successes = int(row["successes"])
        n_rollouts = int(row["n_rollouts"])
        ci_low, ci_high = wilson_interval(successes, n_rollouts)
        selected.append(
            {
                "M": int(row["n_bins"]),
                "variant": head_key,
                "success_rate": float(row["success_rate"]),
                "successes": successes,
                "n_rollouts": n_rollouts,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
            }
        )
    selected.sort(key=lambda x: (x["M"], x["variant"]))
    return selected


def latency_summary(head_only_path: Path, batch_path: Path) -> dict:
    head_only = load_json(head_only_path)["latency"]
    batch = load_json(batch_path)
    return {
        "head_only": {
            "smallest_N": {"N": 1024, **head_only["1024"]},
            "largest_N": {"N": 4194304, **head_only["4194304"]},
        },
        "batch_latency": {
            "batch_1_M_65536": {
                "dense_ms": batch["dense"]["1"]["65536"],
                "gvla_ms": batch["gvla"]["1"]["65536"],
            },
            "batch_1024_M_65536": {
                "dense_ms": batch["dense"]["1024"]["65536"],
                "gvla_ms": batch["gvla"]["1024"]["65536"],
            },
        },
    }


def write_markdown(
    precision_rows: List[Dict[str, Union[float, int, str]]],
    gray_rows: List[Dict[str, Union[float, int, str]]],
    latency: Dict,
    out_path: Path,
) -> None:
    lines = [
        "# NeurIPS Minimal CPU Summary",
        "",
        "## Precision custom2.5 (200 rollouts)",
        "",
        "| Setting | Successes | Rate | 95% CI |",
        "|---|---:|---:|---:|",
    ]
    for row in precision_rows:
        lines.append(
            f"| {row['label']} | {row['successes']}/{row['n_rollouts']} | "
            f"{row['success_rate']*100:.1f}% | "
            f"[{row['ci95_low']*100:.1f}, {row['ci95_high']*100:.1f}] |"
        )

    lines.extend(
        [
            "",
            "## Gray code main ablation",
            "",
            "| M | Variant | Successes | Rate | 95% CI |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in gray_rows:
        lines.append(
            f"| {row['M']} | {row['variant']} | {row['successes']}/{row['n_rollouts']} | "
            f"{row['success_rate']*100:.1f}% | "
            f"[{row['ci95_low']*100:.1f}, {row['ci95_high']*100:.1f}] |"
        )

    lines.extend(
        [
            "",
            "## Latency anchors",
            "",
            f"- Head-only N=1024: dense {latency['head_only']['smallest_N']['dense_ms']:.4f} ms, "
            f"GVLA {latency['head_only']['smallest_N']['gvla_ms']:.4f} ms, "
            f"speedup {latency['head_only']['smallest_N']['speedup']:.1f}x.",
            f"- Head-only N=4,194,304: dense {latency['head_only']['largest_N']['dense_ms']:.4f} ms, "
            f"GVLA {latency['head_only']['largest_N']['gvla_ms']:.4f} ms, "
            f"speedup {latency['head_only']['largest_N']['speedup']:.1f}x.",
            f"- Batch=1, M=65536: dense {latency['batch_latency']['batch_1_M_65536']['dense_ms']:.4f} ms, "
            f"GVLA {latency['batch_latency']['batch_1_M_65536']['gvla_ms']:.4f} ms.",
            f"- Batch=1024, M=65536: dense {latency['batch_latency']['batch_1024_M_65536']['dense_ms']:.4f} ms, "
            f"GVLA {latency['batch_latency']['batch_1024_M_65536']['gvla_ms']:.4f} ms.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    precision_rows = precision_table(RESULTS_DIR / "precision_custom25_200roll_core.json")
    gray_rows = gray_table(RESULTS_DIR / "bc_study" / "eval_results.json")
    latency = latency_summary(
        RESULTS_DIR / "robosuite_study" / "results.json",
        RESULTS_DIR / "bc_study" / "latency_batch.json",
    )

    summary = {
        "precision_custom25_200roll_core": precision_rows,
        "gray_code_main_ablation": gray_rows,
        "latency_summary": latency,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    write_markdown(precision_rows, gray_rows, latency, OUT_DIR / "summary.md")
    print(f"Saved summary → {OUT_DIR / 'summary.json'}")
    print(f"Saved markdown → {OUT_DIR / 'summary.md'}")


if __name__ == "__main__":
    main()
