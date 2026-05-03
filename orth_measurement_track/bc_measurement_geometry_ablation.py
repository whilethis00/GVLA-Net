from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.bc_train import BCPolicy, LiftDemoDataset, compute_loss, quantize_action
from experiments.io_utils import save_figure, write_csv_rows, write_text
from models.layers import OrthogonalProjectionLayer


@dataclass(frozen=True)
class VariantSpec:
    name: str
    encoding: str
    ortho_lambda: float
    data_orth_lambda: float
    freeze_measurement: bool = False


CORE_VARIANTS = (
    VariantSpec("gray_no_orth", "gray", 0.0, 0.0),
    VariantSpec("gray_param_orth", "gray", 1e-2, 0.0),
    VariantSpec("gray_data_orth", "gray", 0.0, 1e-2),
    VariantSpec("natural_no_orth", "natural", 0.0, 0.0),
    VariantSpec("natural_param_orth", "natural", 1e-2, 0.0),
    VariantSpec("random_fixed_orth", "random", 0.0, 0.0, True),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BC-specific measurement geometry ablation, isolated from NeurIPS artifacts."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "robomimic" / "lift" / "ph" / "low_dim_v141.hdf5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "orth_measurement_track" / "bc_measurement_geometry_ablation",
    )
    parser.add_argument("--n-bins", type=int, default=256)
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=20260504)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--rollouts", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=[variant.name for variant in CORE_VARIANTS],
        help="Subset of predefined variants to run.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_split(dataset: LiftDemoDataset, val_fraction: float, split_seed: int) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("--val-fraction must be between 0 and 1.")
    rng = np.random.default_rng(split_seed)
    indices = np.arange(len(dataset))
    rng.shuffle(indices)
    val_count = max(1, int(round(len(indices) * val_fraction)))
    val_indices = np.sort(indices[-val_count:])
    train_indices = np.sort(indices[:-val_count])
    return train_indices, val_indices


def get_variant_specs(names: list[str]) -> list[VariantSpec]:
    lookup = {variant.name: variant for variant in CORE_VARIANTS}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise ValueError(f"Unknown variant(s): {missing}")
    return [lookup[name] for name in names]


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_policy(
    obs_dim: int,
    action_dim: int,
    args: argparse.Namespace,
    variant: VariantSpec,
    device: torch.device,
) -> BCPolicy:
    policy = BCPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        head_type="gvla",
        n_bins=args.n_bins,
        latent_dim=args.latent_dim,
        gray_code=(variant.encoding == "gray"),
        encoding=variant.encoding,
        random_code_seed=args.seed,
        ortho_lambda=variant.ortho_lambda,
    ).to(device)
    if variant.freeze_measurement:
        for head in policy.head.heads:
            if not isinstance(head, OrthogonalProjectionLayer):
                continue
            head.weight.requires_grad_(False)
    return policy


def correlation_penalty(matrix: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(eps)
    normalized = centered / std
    corr = normalized.transpose(0, 1) @ normalized / normalized.size(0)
    identity = torch.eye(corr.size(0), device=corr.device, dtype=corr.dtype)
    diff = corr - identity
    return torch.sum(diff * diff)


def data_orth_loss(policy: BCPolicy, obs: torch.Tensor) -> torch.Tensor:
    z = policy.backbone(obs)
    out_list = policy.head(z)
    penalty = 0.0
    for out in out_list:
        penalty = penalty + correlation_penalty(out["projections"])
    return penalty / len(out_list)


def predict_bins_and_bits_local(policy: BCPolicy, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = policy.backbone(obs)
    out_list = policy.head(z)
    hard_bits = []
    pred_bins = []
    for out in out_list:
        bits = (out["binary_code"] > 0.5).long()
        bins = policy.head.decode_binary_to_bins(out["binary_code"])
        hard_bits.append(bits)
        pred_bins.append(bins)
    return torch.stack(pred_bins, dim=-1), torch.stack(hard_bits, dim=1).float()


def train_variant(
    policy: BCPolicy,
    train_loader: DataLoader,
    args: argparse.Namespace,
    variant: VariantSpec,
    device: torch.device,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(
        [param for param in policy.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        policy.train()
        total_loss = 0.0
        total_items = 0
        for obs_batch, act_batch in train_loader:
            obs_batch = obs_batch.to(device)
            act_batch = act_batch.to(device)
            loss = compute_loss(policy, obs_batch, act_batch)
            if variant.data_orth_lambda > 0:
                loss = loss + variant.data_orth_lambda * data_orth_loss(policy, obs_batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()
            batch_size = obs_batch.size(0)
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size
        scheduler.step()
        mean_loss = total_loss / max(total_items, 1)
        history.append({"epoch": epoch, "train_loss": mean_loss})
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f"[{variant.name}] epoch={epoch:03d}/{args.epochs} loss={mean_loss:.4f}", flush=True)
    return history


def offdiag_abs_mean(matrix: np.ndarray) -> float:
    if matrix.shape[0] <= 1:
        return 0.0
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.abs(matrix[mask]).mean())


def compute_sign_corr(bits: np.ndarray) -> np.ndarray:
    x = bits.astype(np.float64)
    if x.shape[1] <= 1:
        return np.eye(x.shape[1], dtype=np.float64)
    if np.allclose(x.std(axis=0), 0.0):
        return np.eye(x.shape[1], dtype=np.float64)
    return np.corrcoef(x, rowvar=False)


def compute_conditional_entropy(gt_bins: np.ndarray, pred_bits: np.ndarray) -> list[float]:
    entropies: list[float] = []
    n_samples = gt_bins.shape[0]
    for prefix_len in range(1, pred_bits.shape[1] + 1):
        groups: dict[tuple[int, ...], list[int]] = {}
        for sample_idx in range(n_samples):
            key = tuple(pred_bits[sample_idx, :prefix_len].astype(int).tolist())
            groups.setdefault(key, []).append(int(gt_bins[sample_idx]))
        conditional_entropy = 0.0
        for values in groups.values():
            counts = np.bincount(values)
            probs = counts[counts > 0] / len(values)
            entropy = -float(np.sum(probs * np.log2(probs)))
            conditional_entropy += (len(values) / n_samples) * entropy
        entropies.append(conditional_entropy)
    return entropies


def collect_validation_outputs(
    policy: BCPolicy,
    val_loader: DataLoader,
    val_indices: np.ndarray,
    device: torch.device,
) -> dict[str, np.ndarray]:
    sample_ids_chunks = []
    gt_action_chunks = []
    pred_action_chunks = []
    gt_bin_chunks = []
    pred_bin_chunks = []
    gt_bit_chunks = []
    pred_bit_chunks = []
    proj_chunks: list[list[np.ndarray]] | None = None
    row_cos_chunks = []

    offset = 0
    policy.eval()
    with torch.inference_mode():
        for obs_batch, action_batch in val_loader:
            obs_batch = obs_batch.to(device)
            action_batch = action_batch.to(device)
            z = policy.backbone(obs_batch)
            out_list = policy.head(z)
            pred_action_batch = policy.head.decode(z)
            pred_bin_batch, pred_bit_batch = predict_bins_and_bits_local(policy, obs_batch)
            gt_bin_batch = quantize_action(action_batch, policy.n_bins)
            gt_bit_batch = policy.head.encode_bins(gt_bin_batch)

            if proj_chunks is None:
                proj_chunks = [[] for _ in range(len(out_list))]
            for dim_idx, out in enumerate(out_list):
                proj_chunks[dim_idx].append(out["projections"].cpu().numpy())
            if not row_cos_chunks:
                for head in policy.head.heads:
                    weight = head.weight.detach()
                    normalized = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
                    row_cos_chunks.append((normalized @ normalized.transpose(0, 1)).cpu().numpy())

            batch_size = obs_batch.shape[0]
            batch_indices = val_indices[offset:offset + batch_size]
            offset += batch_size

            sample_ids_chunks.append(batch_indices)
            gt_action_chunks.append(action_batch.cpu().numpy())
            pred_action_chunks.append(pred_action_batch.cpu().numpy())
            gt_bin_chunks.append(gt_bin_batch.cpu().numpy())
            pred_bin_chunks.append(pred_bin_batch.cpu().numpy())
            gt_bit_chunks.append(gt_bit_batch.cpu().numpy())
            pred_bit_chunks.append(pred_bit_batch.cpu().numpy())

    projections = [np.concatenate(chunks, axis=0) for chunks in (proj_chunks or [])]
    return {
        "sample_ids": np.concatenate(sample_ids_chunks, axis=0),
        "gt_actions": np.concatenate(gt_action_chunks, axis=0),
        "pred_actions": np.concatenate(pred_action_chunks, axis=0),
        "gt_bins": np.concatenate(gt_bin_chunks, axis=0),
        "pred_bins": np.concatenate(pred_bin_chunks, axis=0),
        "gt_bits": np.concatenate(gt_bit_chunks, axis=0),
        "pred_bits": np.concatenate(pred_bit_chunks, axis=0),
        "projections": projections,
        "row_cos_matrices": row_cos_chunks,
    }


def summarize_validation_metrics(outputs: dict[str, np.ndarray]) -> dict[str, object]:
    gt_actions = outputs["gt_actions"]
    pred_actions = outputs["pred_actions"]
    gt_bins = outputs["gt_bins"]
    pred_bins = outputs["pred_bins"]
    gt_bits = outputs["gt_bits"]
    pred_bits = outputs["pred_bits"]
    projections = outputs["projections"]

    action_abs = np.abs(pred_actions - gt_actions)
    action_sq = np.square(pred_actions - gt_actions)
    bin_diff = np.abs(pred_bins - gt_bins)
    bit_diff = np.not_equal(pred_bits.astype(int), gt_bits.astype(int))

    row_cos_means = []
    logit_corr_means = []
    sign_corr_means = []
    entropy_curves = []
    row_cos_first = None
    logit_corr_first = None
    sign_corr_first = None
    row_cos_matrices = outputs["row_cos_matrices"]
    for dim_idx, proj in enumerate(projections):
        corr = np.corrcoef(proj, rowvar=False)
        sign_corr = compute_sign_corr(pred_bits[:, dim_idx, :])
        row_cos = row_cos_matrices[dim_idx]
        row_cos_means.append(offdiag_abs_mean(row_cos))
        if row_cos_first is None:
            row_cos_first = row_cos
        if logit_corr_first is None:
            logit_corr_first = corr
            sign_corr_first = sign_corr
        logit_corr_means.append(offdiag_abs_mean(corr))
        sign_corr_means.append(offdiag_abs_mean(sign_corr))
        entropy_curves.append(compute_conditional_entropy(gt_bins[:, dim_idx], pred_bits[:, dim_idx, :]))

    entropy_curve_mean = np.mean(np.array(entropy_curves, dtype=np.float64), axis=0).tolist()

    return {
        "mean_action_l1": float(action_abs.mean()),
        "mean_action_l2": float(np.sqrt(action_sq.sum(axis=-1)).mean()),
        "mean_bin_error": float(bin_diff.mean()),
        "mean_hamming_error": float(bit_diff.mean()),
        "adjacent_bin_error_rate": float((bin_diff == 1).mean()),
        "exact_bin_match_rate": float((bin_diff == 0).mean()),
        "mean_abs_row_cosine": float(np.mean(row_cos_means)),
        "mean_abs_logit_corr": float(np.mean(logit_corr_means)),
        "mean_abs_sign_corr": float(np.mean(sign_corr_means)),
        "conditional_entropy_curve": entropy_curve_mean,
        "row_cos_first_dim": row_cos_first.tolist() if row_cos_first is not None else [],
        "logit_corr_first_dim": logit_corr_first.tolist() if logit_corr_first is not None else [],
        "sign_corr_first_dim": sign_corr_first.tolist() if sign_corr_first is not None else [],
    }


def mean_abs_row_cosine(policy: BCPolicy) -> float:
    values = []
    for head in policy.head.heads:
        weight = head.weight.detach()
        normalized = weight / weight.norm(dim=1, keepdim=True).clamp_min(1e-6)
        cosine = torch.abs(normalized @ normalized.transpose(0, 1))
        mask = ~torch.eye(cosine.size(0), dtype=torch.bool, device=cosine.device)
        values.append(float(cosine[mask].mean().item()))
    return float(np.mean(values))


def save_heatmap(matrix: np.ndarray, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    im = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=220)
    plt.close(fig)


def save_entropy_curve(values: list[float], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    x = np.arange(1, len(values) + 1)
    ax.plot(x, values, marker="o", lw=2.0)
    ax.set_xlabel("Observed bit prefix length")
    ax.set_ylabel("Conditional entropy of target bin")
    ax.set_title("Entropy decay under predicted bits")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output_path, dpi=220)
    plt.close(fig)


def maybe_rollout_success(policy: BCPolicy, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    if args.rollouts <= 0:
        return {"rollout_success_rate": None, "rollout_successes": None}
    import robosuite as suite
    from experiments.bc_eval import run_rollout

    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        horizon=args.max_steps,
        reward_shaping=False,
    )
    successes = 0
    for seed in range(args.rollouts):
        ok, _ = run_rollout(policy, env, args.max_steps, seed, device)
        successes += int(ok)
    env.close()
    return {
        "rollout_success_rate": successes / args.rollouts,
        "rollout_successes": successes,
    }


def render_summary_markdown(rows: list[dict[str, object]]) -> str:
    lines = [
        "# BC Measurement Geometry Ablation",
        "",
        "| Variant | Encoding | Ortho | Data-Orth | Action L1 | Action L2 | Bin Error | Hamming | Mean |logit corr| | Mean |sign corr| | Exact Match | Rollout SR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        rollout = "N/A" if row["rollout_success_rate"] is None else f"{100.0 * row['rollout_success_rate']:.1f}%"
        lines.append(
            f"| {row['variant']} | {row['encoding']} | {row['ortho_lambda']:.4g} | {row['data_orth_lambda']:.4g} | "
            f"{row['mean_action_l1']:.4f} | {row['mean_action_l2']:.4f} | {row['mean_bin_error']:.4f} | "
            f"{row['mean_hamming_error']:.4f} | {row['mean_abs_logit_corr']:.4f} | {row['mean_abs_sign_corr']:.4f} | "
            f"{row['exact_bin_match_rate']:.4f} | {rollout} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    set_seed(args.seed)

    dataset = LiftDemoDataset(str(args.data))
    train_indices, val_indices = build_split(dataset, args.val_fraction, args.split_seed)
    train_subset = Subset(dataset, train_indices.tolist())
    val_subset = Subset(dataset, val_indices.tolist())

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    split_payload = {
        "dataset_path": str(args.data),
        "dataset_size": len(dataset),
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "val_fraction": args.val_fraction,
        "split_seed": args.split_seed,
        "train_indices": train_indices.tolist(),
        "val_indices": val_indices.tolist(),
    }
    (args.output_dir / "split_lock.json").write_text(json.dumps(split_payload, indent=2))

    obs_dim = dataset.obs.shape[1]
    action_dim = dataset.act.shape[1]
    summary_rows: list[dict[str, object]] = []

    for variant in get_variant_specs(args.variants):
        print(f"\n=== {variant.name} ===", flush=True)
        set_seed(args.seed)
        policy = build_policy(obs_dim, action_dim, args, variant, device)
        variant_dir = args.output_dir / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)

        history = train_variant(policy, train_loader, args, variant, device)
        write_csv_rows(history, variant_dir / "train_history.csv")

        outputs = collect_validation_outputs(policy, val_loader, val_indices, device)
        metrics = summarize_validation_metrics(outputs)
        rollout = maybe_rollout_success(policy, args, device)
        row = {
            "variant": variant.name,
            "encoding": variant.encoding,
            "ortho_lambda": variant.ortho_lambda,
            "data_orth_lambda": variant.data_orth_lambda,
            "freeze_measurement": variant.freeze_measurement,
            "n_bins": args.n_bins,
            "val_size": len(val_indices),
            "mean_abs_row_cosine": mean_abs_row_cosine(policy),
            **metrics,
            **rollout,
        }
        summary_rows.append(row)
        (variant_dir / "summary.json").write_text(json.dumps(row, indent=2))

        save_heatmap(
            np.array(metrics["row_cos_first_dim"], dtype=np.float64),
            f"{variant.name} row cosine (dim 0)",
            variant_dir / "row_cos_first_dim.png",
        )
        save_heatmap(
            np.array(metrics["logit_corr_first_dim"], dtype=np.float64),
            f"{variant.name} logit corr (dim 0)",
            variant_dir / "logit_corr_first_dim.png",
        )
        save_heatmap(
            np.array(metrics["sign_corr_first_dim"], dtype=np.float64),
            f"{variant.name} sign corr (dim 0)",
            variant_dir / "sign_corr_first_dim.png",
        )
        save_entropy_curve(
            list(metrics["conditional_entropy_curve"]),
            variant_dir / "conditional_entropy_curve.png",
        )

    write_csv_rows(summary_rows, args.output_dir / "summary.csv")
    (args.output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    write_text(args.output_dir / "summary.md", render_summary_markdown(summary_rows))


if __name__ == "__main__":
    main()
