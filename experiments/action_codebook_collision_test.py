import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ProxyRow:
    method: str
    num_actions: int
    action_dim: int
    code_bits: int
    load_factor: float
    ideal_unique_code_ratio: float
    mean_abs_row_cosine: float
    collision_rate: float
    singleton_rate: float
    unique_code_ratio: float
    occupancy_efficiency: float
    reconstruction_mse: float
    encode_ms: float
    decode_ms: float
    total_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Proxy comparison of PQ, random LSH, and GVLA on action code collisions and reconstruction error."
    )
    parser.add_argument("--num-actions", type=int, default=1 << 20)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--code-bits", type=int, nargs="+", default=[20])
    parser.add_argument("--train-samples", type=int, default=65536)
    parser.add_argument("--eval-samples", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--include-no-ortho-ablation", action="store_true")
    parser.add_argument("--no-ortho-mix", type=float, default=0.55)
    parser.add_argument("--pq-subspaces", type=int, default=4)
    parser.add_argument("--pq-centroids", type=int, default=32)
    parser.add_argument("--pq-iters", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "action_codebook_collision_test",
    )
    parser.add_argument("--run-name", type=str, default="proxy_collision")
    return parser.parse_args()


def build_run_dir(output_root: Path, run_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return output_root / f"{timestamp}_{run_name}"


def sample_actions(num_actions: int, action_dim: int, rng: np.random.Generator) -> np.ndarray:
    actions = rng.standard_normal((num_actions, action_dim), dtype=np.float32)
    actions /= np.linalg.norm(actions, axis=1, keepdims=True) + 1e-6
    return actions.astype(np.float32)


def train_test_split(actions: np.ndarray, train_samples: int, eval_samples: int) -> tuple[np.ndarray, np.ndarray]:
    train = actions[:train_samples]
    test = actions[train_samples : train_samples + eval_samples]
    return train, test


def orthogonal_rows(num_rows: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    matrix = rng.standard_normal((dim, dim), dtype=np.float32)
    q, _ = np.linalg.qr(matrix)
    return q[:, :num_rows].T.astype(np.float32)


def random_rows(num_rows: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    rows = rng.standard_normal((num_rows, dim), dtype=np.float32)
    rows /= np.linalg.norm(rows, axis=1, keepdims=True) + 1e-6
    return rows.astype(np.float32)


def correlated_rows(
    num_rows: int,
    dim: int,
    rng: np.random.Generator,
    *,
    mix: float,
) -> np.ndarray:
    """Proxy for learned projections without orthogonal regularization.

    We start from a random normalized basis and then repeatedly mix neighboring
    rows together. This preserves the projection-based decoder shape while
    intentionally introducing inter-row correlation, which is the failure mode
    orthogonal regularization is meant to suppress.
    """
    rows = random_rows(num_rows, dim, rng)
    mixed = rows.copy()
    for i in range(1, num_rows):
        mixed[i] = (1.0 - mix) * mixed[i] + mix * mixed[i - 1]
    mixed /= np.linalg.norm(mixed, axis=1, keepdims=True) + 1e-6
    return mixed.astype(np.float32)


def mean_abs_row_cosine(weight: np.ndarray) -> float:
    if weight.shape[0] <= 1:
        return 0.0
    normalized = weight / (np.linalg.norm(weight, axis=1, keepdims=True) + 1e-6)
    cosine = np.abs(normalized @ normalized.T)
    mask = ~np.eye(weight.shape[0], dtype=bool)
    return float(cosine[mask].mean())


def bits_to_ids(bits: np.ndarray) -> np.ndarray:
    code_bits = bits.shape[1]
    packed = np.zeros(bits.shape[0], dtype=np.uint32)
    for i in range(code_bits):
        packed |= (bits[:, i].astype(np.uint32) << i)
    return packed


def projection_codes(actions: np.ndarray, weight: np.ndarray) -> np.ndarray:
    projections = actions @ weight.T
    bits = projections >= 0
    return bits_to_ids(bits)


def compute_collision_stats(code_ids: np.ndarray, num_codes: int) -> tuple[float, float, float]:
    counts = np.bincount(code_ids.astype(np.int64), minlength=num_codes)
    collision_actions = counts[counts > 1].sum()
    singletons = counts[counts == 1].sum()
    unique_codes = np.count_nonzero(counts)
    total = max(code_ids.size, 1)
    return (
        float(collision_actions) / total,
        float(singletons) / total,
        float(unique_codes) / total,
    )


def occupancy_terms(num_actions: int, code_bits: int) -> tuple[float, float]:
    num_codes = float(1 << code_bits)
    load_factor = float(num_actions) / num_codes
    ideal_occupied_bins = num_codes * (1.0 - math.exp(-load_factor))
    ideal_unique_code_ratio = ideal_occupied_bins / max(float(num_actions), 1.0)
    return load_factor, ideal_unique_code_ratio


def build_prototypes(train_actions: np.ndarray, code_ids: np.ndarray, num_codes: int) -> tuple[np.ndarray, np.ndarray]:
    dim = train_actions.shape[1]
    sums = np.zeros((num_codes, dim), dtype=np.float32)
    counts = np.bincount(code_ids.astype(np.int64), minlength=num_codes).astype(np.int32)
    np.add.at(sums, code_ids.astype(np.int64), train_actions)
    prototypes = np.zeros_like(sums)
    valid = counts > 0
    prototypes[valid] = sums[valid] / counts[valid, None]
    return prototypes, counts


def decode_prototypes(code_ids: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    return prototypes[code_ids.astype(np.int64)]


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def benchmark_encode_decode(
    encoder: Callable[[np.ndarray], np.ndarray],
    decoder: Callable[[np.ndarray], np.ndarray],
    actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    t0 = time.perf_counter()
    code_ids = encoder(actions)
    t1 = time.perf_counter()
    recon = decoder(code_ids)
    t2 = time.perf_counter()
    return code_ids, recon, (t1 - t0) * 1000.0, (t2 - t1) * 1000.0


def kmeans_subspace(data: np.ndarray, num_centroids: int, num_iters: int, rng: np.random.Generator) -> np.ndarray:
    indices = rng.choice(data.shape[0], size=num_centroids, replace=False)
    centroids = data[indices].copy()
    for _ in range(num_iters):
        distances = (
            np.sum(data**2, axis=1, keepdims=True)
            - 2.0 * data @ centroids.T
            + np.sum(centroids**2, axis=1)[None, :]
        )
        assignments = np.argmin(distances, axis=1)
        for cid in range(num_centroids):
            mask = assignments == cid
            if np.any(mask):
                centroids[cid] = data[mask].mean(axis=0)
    return centroids.astype(np.float32)


class ProductQuantizer:
    def __init__(
        self,
        *,
        action_dim: int,
        num_subspaces: int,
        num_centroids: int,
        num_iters: int,
        rng: np.random.Generator,
    ) -> None:
        if action_dim % num_subspaces != 0:
            raise ValueError(f"action_dim={action_dim} must be divisible by num_subspaces={num_subspaces}")
        self.action_dim = action_dim
        self.num_subspaces = num_subspaces
        self.num_centroids = num_centroids
        self.num_iters = num_iters
        self.sub_dim = action_dim // num_subspaces
        self.rng = rng
        self.codebooks: list[np.ndarray] = []

    def fit(self, actions: np.ndarray) -> None:
        self.codebooks = []
        for i in range(self.num_subspaces):
            start = i * self.sub_dim
            stop = start + self.sub_dim
            centroids = kmeans_subspace(
                actions[:, start:stop],
                num_centroids=self.num_centroids,
                num_iters=self.num_iters,
                rng=self.rng,
            )
            self.codebooks.append(centroids)

    def encode(self, actions: np.ndarray) -> np.ndarray:
        code_ids = np.zeros(actions.shape[0], dtype=np.uint32)
        base = 1
        for i, centroids in enumerate(self.codebooks):
            start = i * self.sub_dim
            stop = start + self.sub_dim
            chunk = actions[:, start:stop]
            distances = (
                np.sum(chunk**2, axis=1, keepdims=True)
                - 2.0 * chunk @ centroids.T
                + np.sum(centroids**2, axis=1)[None, :]
            )
            assignments = np.argmin(distances, axis=1).astype(np.uint32)
            code_ids += base * assignments
            base *= self.num_centroids
        return code_ids

    def decode(self, code_ids: np.ndarray) -> np.ndarray:
        recon = np.zeros((code_ids.shape[0], self.action_dim), dtype=np.float32)
        residual = code_ids.astype(np.uint32).copy()
        for i, centroids in enumerate(self.codebooks):
            start = i * self.sub_dim
            stop = start + self.sub_dim
            assignments = residual % self.num_centroids
            residual //= self.num_centroids
            recon[:, start:stop] = centroids[assignments]
        return recon


def evaluate_pq(
    train_actions: np.ndarray,
    test_actions: np.ndarray,
    all_actions: np.ndarray,
    *,
    action_dim: int,
    num_subspaces: int,
    num_centroids: int,
    num_iters: int,
    rng: np.random.Generator,
) -> ProxyRow:
    pq = ProductQuantizer(
        action_dim=action_dim,
        num_subspaces=num_subspaces,
        num_centroids=num_centroids,
        num_iters=num_iters,
        rng=rng,
    )
    pq.fit(train_actions)
    all_codes = pq.encode(all_actions)
    collision_rate, singleton_rate, unique_code_ratio = compute_collision_stats(
        all_codes,
        num_codes=num_centroids**num_subspaces,
    )
    _, recon, encode_ms, decode_ms = benchmark_encode_decode(pq.encode, pq.decode, test_actions)
    load_factor, ideal_unique_code_ratio = occupancy_terms(all_actions.shape[0], int(num_subspaces * math.log2(num_centroids)))
    return ProxyRow(
        method="PQ",
        num_actions=all_actions.shape[0],
        action_dim=action_dim,
        code_bits=int(num_subspaces * math.log2(num_centroids)),
        load_factor=load_factor,
        ideal_unique_code_ratio=ideal_unique_code_ratio,
        mean_abs_row_cosine=float("nan"),
        collision_rate=collision_rate,
        singleton_rate=singleton_rate,
        unique_code_ratio=unique_code_ratio,
        occupancy_efficiency=unique_code_ratio / max(ideal_unique_code_ratio, 1e-12),
        reconstruction_mse=mse(test_actions, recon),
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        total_ms=encode_ms + decode_ms,
    )


def evaluate_projection_method(
    method_name: str,
    train_actions: np.ndarray,
    test_actions: np.ndarray,
    all_actions: np.ndarray,
    weight: np.ndarray,
) -> ProxyRow:
    code_bits = weight.shape[0]
    num_codes = 1 << code_bits
    train_codes = projection_codes(train_actions, weight)
    prototypes, _ = build_prototypes(train_actions, train_codes, num_codes=num_codes)

    encoder = lambda x: projection_codes(x, weight)
    decoder = lambda code_ids: decode_prototypes(code_ids, prototypes)

    all_codes = encoder(all_actions)
    collision_rate, singleton_rate, unique_code_ratio = compute_collision_stats(all_codes, num_codes=num_codes)
    _, recon, encode_ms, decode_ms = benchmark_encode_decode(encoder, decoder, test_actions)

    load_factor, ideal_unique_code_ratio = occupancy_terms(all_actions.shape[0], code_bits)
    return ProxyRow(
        method=method_name,
        num_actions=all_actions.shape[0],
        action_dim=all_actions.shape[1],
        code_bits=code_bits,
        load_factor=load_factor,
        ideal_unique_code_ratio=ideal_unique_code_ratio,
        mean_abs_row_cosine=mean_abs_row_cosine(weight),
        collision_rate=collision_rate,
        singleton_rate=singleton_rate,
        unique_code_ratio=unique_code_ratio,
        occupancy_efficiency=unique_code_ratio / max(ideal_unique_code_ratio, 1e-12),
        reconstruction_mse=mse(test_actions, recon),
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        total_ms=encode_ms + decode_ms,
    )


def write_csv(rows: list[ProxyRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(ProxyRow.__annotations__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def render_table(rows: list[ProxyRow]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Action codebook proxy test across fixed and overcomplete code budgets. "
            "We compare a product-quantization baseline (PQ), random-projection LSH, "
            "and GVLA's orthogonal geometric hashing on $N=2^{20}$ synthetic continuous actions. "
            "Collision rate measures the fraction of actions that share a code with at least one other action; "
            "occupancy efficiency is the realized unique-code ratio divided by the ideal Poisson occupancy limit "
            "$1-e^{-\\lambda}$ at load factor $\\lambda=N/2^k$; reconstruction MSE is computed after decoding from "
            "the learned or induced codebook prototype.}"
        ),
        "\\label{tab:proxy_collision_test}",
        "\\scriptsize",
        "\\begin{tabular}{lrrrrrrr}",
        "\\toprule",
        "Method & Bits & Mean $|\\cos|$ & Collision Rate & Occ. Eff. & Unique Code Ratio & Recon. MSE & Total ms \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row.method} & {row.code_bits} & {row.mean_abs_row_cosine:.4f} & {row.collision_rate:.4f} & {row.occupancy_efficiency:.4f} & "
            f"{row.unique_code_ratio:.4f} & {row.reconstruction_mse:.6f} & {row.total_ms:.2f} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    pq_bits = int(args.pq_subspaces * math.log2(args.pq_centroids))

    run_dir = build_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    actions = sample_actions(args.num_actions, args.action_dim, rng)
    train_actions, test_actions = train_test_split(actions, args.train_samples, args.eval_samples)

    rows: list[ProxyRow] = []
    requested_bits = sorted(set(int(bits) for bits in args.code_bits))

    if pq_bits in requested_bits:
        rows.append(
            evaluate_pq(
                train_actions,
                test_actions,
                actions,
                action_dim=args.action_dim,
                num_subspaces=args.pq_subspaces,
                num_centroids=args.pq_centroids,
                num_iters=args.pq_iters,
                rng=np.random.default_rng(args.seed + 11),
            )
        )
    else:
        print(
            f"Skipping PQ: requested code bits {requested_bits} do not include PQ budget {pq_bits}. "
            f"Use --pq-centroids/--pq-subspaces to match an overcomplete target if needed."
        )

    for code_bits in requested_bits:
        random_weight = random_rows(code_bits, args.action_dim, np.random.default_rng(args.seed + 23 + code_bits))
        rows.append(
            evaluate_projection_method(
                "LSH (Random Projection)",
                train_actions,
                test_actions,
                actions,
                random_weight,
            )
        )

        if args.include_no_ortho_ablation:
            no_ortho_weight = correlated_rows(
                code_bits,
                args.action_dim,
                np.random.default_rng(args.seed + 31 + code_bits),
                mix=args.no_ortho_mix,
            )
            rows.append(
                evaluate_projection_method(
                    "GVLA w/o Orthogonal Reg.",
                    train_actions,
                    test_actions,
                    actions,
                    no_ortho_weight,
                )
            )

        orth_weight = orthogonal_rows(code_bits, args.action_dim, np.random.default_rng(args.seed + 37 + code_bits))
        rows.append(
            evaluate_projection_method(
                "GVLA-Net (Ours)",
                train_actions,
                test_actions,
                actions,
                orth_weight,
            )
        )

    method_order = {
        "GVLA-Net (Ours)": 0,
        "GVLA w/o Orthogonal Reg.": 1,
        "LSH (Random Projection)": 2,
        "PQ": 3,
    }
    rows.sort(key=lambda row: (row.code_bits, method_order.get(row.method, 99), row.method))

    csv_path = run_dir / "action_codebook_collision_test.csv"
    tex_path = run_dir / "action_codebook_collision_test.tex"
    write_csv(rows, csv_path)
    tex_path.write_text(render_table(rows))

    print(f"CSV written to: {csv_path}")
    print(f"LaTeX written to: {tex_path}")
    for row in rows:
        print(
            f"{row.method} @ {row.code_bits} bits: collision_rate={row.collision_rate:.4f}, "
            f"mean_abs_row_cosine={row.mean_abs_row_cosine:.4f}, "
            f"occupancy_efficiency={row.occupancy_efficiency:.4f}, "
            f"unique_code_ratio={row.unique_code_ratio:.4f}, "
            f"reconstruction_mse={row.reconstruction_mse:.6f}, "
            f"encode_ms={row.encode_ms:.2f}, decode_ms={row.decode_ms:.2f}"
        )


if __name__ == "__main__":
    main()
