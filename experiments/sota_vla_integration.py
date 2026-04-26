import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

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

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional runtime dependency
    Image = None


@dataclass
class BenchmarkMeasurement:
    case_name: str
    num_actions: int
    backbone_dim: int
    backbone_latency_ms: float
    softmax_head_latency_ms: float
    gvla_head_latency_ms: float
    softmax_total_latency_ms: float
    gvla_total_latency_ms: float
    softmax_head_flops: float
    gvla_head_flops: float
    softmax_speedup: float
    head_speedup: float
    head_flops_reduction: float
    total_epr_softmax: float
    total_epr_gvla: float
    total_epr_gain: float


@dataclass
class PublicVLASpec:
    model_name: str
    backbone_dim: int
    native_action_interface: str
    public_source: str
    dim_status: str


# Public-comparison rows are intentionally explicit about inferred dimensions.
# OpenVLA hidden size is specified by this experiment. For the others, public pages
# clearly expose the model family and action interface, but not always the exact
# transformer width in an easily machine-readable way. We therefore use conservative
# proxy widths and mark them as inferred in the final table.
PUBLIC_VLA_SPECS: Tuple[PublicVLASpec, ...] = (
    PublicVLASpec(
        model_name="OpenVLA-7B",
        backbone_dim=4096,
        native_action_interface="7-DoF continuous deltas",
        public_source="https://huggingface.co/openvla/openvla-7b",
        dim_status="measured/integration target",
    ),
    PublicVLASpec(
        model_name="pi0",
        backbone_dim=4096,
        native_action_interface="continuous 8-D actions",
        public_source="https://github.com/Physical-Intelligence/openpi",
        dim_status="proxy dim (public interface, dim not explicit)",
    ),
    PublicVLASpec(
        model_name="RT-2",
        backbone_dim=4096,
        native_action_interface="7-DoF, 256 bins per dimension",
        public_source="https://arxiv.org/abs/2307.15818",
        dim_status="proxy dim (PaLI-X/PaLM-E family)",
    ),
    PublicVLASpec(
        model_name="Octo-Base",
        backbone_dim=1024,
        native_action_interface="continuous action chunks",
        public_source="https://github.com/octo-models/octo",
        dim_status="proxy dim (93M public checkpoint family)",
    ),
    PublicVLASpec(
        model_name="VIMA-200M",
        backbone_dim=1024,
        native_action_interface="autoregressive multimodal actions",
        public_source="https://github.com/vimalabs/VIMA",
        dim_status="proxy dim (200M public checkpoint family)",
    ),
)


class SoftmaxActionHead(nn.Module):
    """Dense VLA-style action head used as the controlled softmax baseline.

    The purpose of this head is to isolate the action-selection bottleneck on top
    of a real OpenVLA backbone embedding. It mirrors the dense ``d -> N`` pattern
    that scales linearly with action precision and therefore acts as the direct
    cost-comparison target for GVLA's logarithmic hash head.
    """

    def __init__(
        self,
        input_dim: int,
        num_actions: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.proj = nn.Linear(
            input_dim,
            num_actions,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.softmax(self.proj(embeddings))


class OpenVLABackboneAdapter:
    """Load OpenVLA and expose pooled backbone embeddings for head swapping.

    The adapter keeps the original vision-language backbone intact while making the
    final action selection explicit and replaceable. This lets the experiment split
    latency into: backbone cost (vision + language reasoning) and head cost (dense
    softmax vs orthogonal geometric hashing).
    """

    def __init__(
        self,
        model_id: str,
        *,
        device: torch.device,
        dtype: torch.dtype,
        quantization: str,
        attn_implementation: Optional[str],
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.quantization = quantization
        self.attn_implementation = attn_implementation
        self.processor = None
        self.model = None
        self.embedding_dim = None
        self.error_message = None

    @staticmethod
    def _validate_transformers_version(transformers_module: Any) -> None:
        version = getattr(transformers_module, "__version__", "unknown")
        major_str = str(version).split(".", 1)[0]
        try:
            major = int(major_str)
        except ValueError:
            return

        if major >= 5:
            raise RuntimeError(
                "OpenVLA remote processor code is not compatible with "
                f"transformers {version}. Install a 4.x release such as "
                "transformers==4.53.2 with tokenizers<0.22."
            )

    @staticmethod
    def _has_flash_attention() -> bool:
        import importlib.util

        return importlib.util.find_spec("flash_attn") is not None

    def _resolve_attn_implementation(self) -> Optional[str]:
        if self.attn_implementation in (None, "auto"):
            return "flash_attention_2" if self._has_flash_attention() else "sdpa"
        if self.attn_implementation == "flash_attention_2" and not self._has_flash_attention():
            return "sdpa"
        return self.attn_implementation

    def load(self) -> None:
        try:
            import transformers
            from transformers import AutoProcessor
        except ImportError as exc:  # pragma: no cover - dependency error
            raise RuntimeError(
                "transformers is required for OpenVLA integration. "
                "Install it in the active environment."
            ) from exc

        self._validate_transformers_version(transformers)
        resolved_attn_implementation = self._resolve_attn_implementation()

        try:
            from transformers import AutoModelForVision2Seq as AutoModelClass
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText as AutoModelClass
            except ImportError:
                try:
                    from transformers import AutoModel as AutoModelClass
                except ImportError as exc:  # pragma: no cover - dependency error
                    raise RuntimeError(
                        "The installed transformers build does not expose "
                        "AutoModelForVision2Seq, AutoModelForImageTextToText, or AutoModel. "
                        "Upgrade transformers to a newer release."
                    ) from exc

        model_kwargs: Dict[str, Any] = {
            "torch_dtype": self.dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        if resolved_attn_implementation is not None:
            model_kwargs["attn_implementation"] = resolved_attn_implementation

        if self.quantization != "none":
            try:
                from transformers import BitsAndBytesConfig
            except ImportError as exc:  # pragma: no cover - dependency error
                raise RuntimeError(
                    "bitsandbytes quantization was requested, but BitsAndBytesConfig "
                    "is unavailable in the installed transformers package."
                ) from exc

            if self.quantization == "4bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=self.dtype,
                )
            elif self.quantization == "8bit":
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                raise ValueError(f"Unsupported quantization mode: {self.quantization}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )
        try:
            self.model = AutoModelClass.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        except ImportError as exc:
            if (
                model_kwargs.get("attn_implementation") == "flash_attention_2"
                and "flash_attn" in str(exc)
            ):
                model_kwargs["attn_implementation"] = "sdpa"
                self.model = AutoModelClass.from_pretrained(
                    self.model_id,
                    **model_kwargs,
                )
            else:
                raise
        if self.quantization == "none":
            self.model = self.model.to(self.device)

        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def _random_image(self, size: Tuple[int, int] = (224, 224)) -> "Image.Image":
        if Image is None:  # pragma: no cover - dependency error
            raise RuntimeError("Pillow is required to generate synthetic image inputs.")
        tensor = torch.randint(0, 256, (size[1], size[0], 3), dtype=torch.uint8)
        return Image.fromarray(tensor.cpu().numpy(), mode="RGB")

    def make_synthetic_batch(
        self,
        batch_size: int,
        instruction: str,
    ) -> Dict[str, torch.Tensor]:
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        images = [self._random_image() for _ in range(batch_size)]
        prompts = [prompt for _ in range(batch_size)]
        inputs = self.processor(prompts, images, return_tensors="pt", padding=True)

        prepared_inputs: Dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            if torch.is_tensor(value):
                if value.dtype.is_floating_point:
                    prepared_inputs[key] = value.to(self.device, dtype=self.dtype)
                else:
                    prepared_inputs[key] = value.to(self.device)
        return prepared_inputs

    def extract_backbone_embeddings(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("OpenVLA model has not been loaded.")

        with torch.inference_mode():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        candidate_tensors: List[torch.Tensor] = []
        if hasattr(outputs, "hidden_states") and outputs.hidden_states:
            candidate_tensors.append(outputs.hidden_states[-1])
        if hasattr(outputs, "decoder_hidden_states") and outputs.decoder_hidden_states:
            candidate_tensors.append(outputs.decoder_hidden_states[-1])
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            candidate_tensors.append(outputs.last_hidden_state)
        if hasattr(outputs, "encoder_last_hidden_state") and outputs.encoder_last_hidden_state is not None:
            candidate_tensors.append(outputs.encoder_last_hidden_state)

        if not candidate_tensors:
            raise RuntimeError(
                "Could not locate hidden states in OpenVLA forward output. "
                "Inspect the remote model output structure and update the adapter."
            )

        last_hidden = candidate_tensors[0]
        if last_hidden.ndim == 2:
            pooled = last_hidden
        elif last_hidden.ndim == 3:
            pooled = last_hidden.mean(dim=1)
        else:
            raise RuntimeError(
                f"Unsupported hidden-state rank for pooling: {tuple(last_hidden.shape)}"
            )

        self.embedding_dim = int(pooled.shape[-1])
        return pooled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Real-world OpenVLA integration benchmark with GVLA head swapping."
    )
    parser.add_argument("--model-id", type=str, default="openvla/openvla-7b")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=10)
    parser.add_argument("--instruction", type=str, default="pick up the red block")
    parser.add_argument(
        "--num-actions",
        type=int,
        nargs="+",
        default=[256, 65536],
    )
    parser.add_argument(
        "--quantization",
        type=str,
        choices=("none", "4bit", "8bit"),
        default="none",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="auto",
        choices=("auto", "flash_attention_2", "sdpa", "eager"),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=("float16", "bfloat16", "float32"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "experiments" / "results" / "sota_vla_integration",
    )
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def resolve_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[dtype_name]


def build_run_dir(output_root: Path, run_name: Optional[str]) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_name = run_name if run_name is not None else "openvla_real_world_integration"
    return output_root / f"{timestamp}_{safe_name}"


def write_args_snapshot(args: argparse.Namespace, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        for key, value in sorted(vars(args).items()):
            handle.write(f"{key}: {value}\n")


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


def measure_callable(
    fn,
    *,
    warmup_iters: int,
    measure_iters: int,
    device: torch.device,
) -> Tuple[float, Any]:
    last_output = None
    for _ in range(warmup_iters):
        last_output = fn()

    synchronize_if_needed(device)
    start_time = time.perf_counter()
    for _ in range(measure_iters):
        last_output = fn()
    synchronize_if_needed(device)
    end_time = time.perf_counter()
    mean_latency_ms = ((end_time - start_time) * 1000.0) / measure_iters
    return mean_latency_ms, last_output


def estimate_softmax_head_flops(
    batch_size: int,
    embedding_dim: int,
    num_actions: int,
) -> float:
    projection_flops = 2.0 * batch_size * embedding_dim * num_actions
    softmax_flops = 5.0 * batch_size * num_actions
    return projection_flops + softmax_flops


def estimate_gvla_head_flops(
    batch_size: int,
    embedding_dim: int,
    num_actions: int,
) -> float:
    code_length = int(math.ceil(math.log2(num_actions)))
    projection_flops = 2.0 * batch_size * embedding_dim * code_length
    hashing_flops = 2.0 * batch_size * code_length
    return projection_flops + hashing_flops


def efficiency_to_precision_ratio(
    num_actions: int,
    latency_ms: float,
    flops: float,
) -> float:
    return num_actions / max(latency_ms * flops, 1e-12)


def benchmark_case(
    adapter: OpenVLABackboneAdapter,
    inputs: Dict[str, torch.Tensor],
    num_actions: int,
    *,
    warmup_iters: int,
    measure_iters: int,
) -> BenchmarkMeasurement:
    device = adapter.device
    dtype = adapter.dtype

    backbone_latency_ms, embeddings = measure_callable(
        lambda: adapter.extract_backbone_embeddings(inputs),
        warmup_iters=warmup_iters,
        measure_iters=measure_iters,
        device=device,
    )
    embedding_dim = int(embeddings.shape[-1])

    softmax_head = SoftmaxActionHead(
        embedding_dim,
        num_actions,
        device=device,
        dtype=dtype,
    ).eval()
    gvla_head = OrthogonalProjectionLayer(
        input_dim=embedding_dim,
        num_codes=num_actions,
        use_ste=False,
        device=device,
        dtype=dtype,
    ).to(device=device, dtype=dtype).eval()

    with torch.inference_mode():
        softmax_head_latency_ms, _ = measure_callable(
            lambda: softmax_head(embeddings),
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            device=device,
        )
        gvla_head_latency_ms, _ = measure_callable(
            lambda: gvla_head(embeddings),
            warmup_iters=warmup_iters,
            measure_iters=measure_iters,
            device=device,
        )

    softmax_total_latency_ms = backbone_latency_ms + softmax_head_latency_ms
    gvla_total_latency_ms = backbone_latency_ms + gvla_head_latency_ms

    softmax_head_flops = estimate_softmax_head_flops(
        batch_size=embeddings.shape[0],
        embedding_dim=embedding_dim,
        num_actions=num_actions,
    )
    gvla_head_flops = estimate_gvla_head_flops(
        batch_size=embeddings.shape[0],
        embedding_dim=embedding_dim,
        num_actions=num_actions,
    )

    return BenchmarkMeasurement(
        case_name=f"openvla_d{embedding_dim}_N{num_actions}",
        num_actions=num_actions,
        backbone_dim=embedding_dim,
        backbone_latency_ms=backbone_latency_ms,
        softmax_head_latency_ms=softmax_head_latency_ms,
        gvla_head_latency_ms=gvla_head_latency_ms,
        softmax_total_latency_ms=softmax_total_latency_ms,
        gvla_total_latency_ms=gvla_total_latency_ms,
        softmax_head_flops=softmax_head_flops,
        gvla_head_flops=gvla_head_flops,
        softmax_speedup=softmax_total_latency_ms / gvla_total_latency_ms,
        head_speedup=softmax_head_latency_ms / gvla_head_latency_ms,
        head_flops_reduction=softmax_head_flops / gvla_head_flops,
        total_epr_softmax=efficiency_to_precision_ratio(
            num_actions,
            softmax_total_latency_ms,
            softmax_head_flops,
        ),
        total_epr_gvla=efficiency_to_precision_ratio(
            num_actions,
            gvla_total_latency_ms,
            gvla_head_flops,
        ),
        total_epr_gain=0.0,  # filled after construction
    )


def write_measurements_csv(
    measurements: Iterable[BenchmarkMeasurement],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_name",
                "num_actions",
                "backbone_dim",
                "backbone_latency_ms",
                "softmax_head_latency_ms",
                "gvla_head_latency_ms",
                "softmax_total_latency_ms",
                "gvla_total_latency_ms",
                "softmax_head_flops",
                "gvla_head_flops",
                "softmax_speedup",
                "head_speedup",
                "head_flops_reduction",
                "total_epr_softmax",
                "total_epr_gvla",
                "total_epr_gain",
            ),
        )
        writer.writeheader()
        for measurement in measurements:
            row = measurement.__dict__.copy()
            writer.writerow(row)


def plot_measurements(
    measurements: List[BenchmarkMeasurement],
    output_path: Path,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed. Install it to generate plots.")

    measurements = sorted(measurements, key=lambda item: item.num_actions)
    action_sizes = [item.num_actions for item in measurements]
    backbone_latency = [item.backbone_latency_ms for item in measurements]
    softmax_head_latency = [item.softmax_head_latency_ms for item in measurements]
    gvla_head_latency = [item.gvla_head_latency_ms for item in measurements]
    softmax_total = [item.softmax_total_latency_ms for item in measurements]
    gvla_total = [item.gvla_total_latency_ms for item in measurements]
    epr_gain = [item.total_epr_gain for item in measurements]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    axes[0].plot(action_sizes, backbone_latency, marker="o", label="Backbone only")
    axes[0].plot(action_sizes, softmax_head_latency, marker="o", label="Softmax head")
    axes[0].plot(action_sizes, gvla_head_latency, marker="o", label="GVLA head")
    axes[0].set_title("Latency Decomposition")
    axes[0].set_xlabel("Action precision N")
    axes[0].set_ylabel("Latency (ms)")
    axes[0].set_xscale("log", base=2)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(action_sizes, softmax_total, marker="o", label="OpenVLA + Softmax")
    axes[1].plot(action_sizes, gvla_total, marker="o", label="OpenVLA + GVLA")
    axes[1].set_title("End-to-End Latency")
    axes[1].set_xlabel("Action precision N")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_xscale("log", base=2)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(action_sizes, epr_gain, marker="o")
    axes[2].set_title("Efficiency-to-Precision Gain")
    axes[2].set_xlabel("Action precision N")
    axes[2].set_ylabel("GVLA / Softmax EPR")
    axes[2].set_xscale("log", base=2)
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    figure.suptitle("OpenVLA Real-world Integration Benchmark")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def project_sota_rows(
    specs: Iterable[PublicVLASpec],
    reference_measurement: BenchmarkMeasurement,
    *,
    target_num_actions: int,
) -> List[Dict[str, Any]]:
    reference_head_latency_per_dim = (
        reference_measurement.softmax_head_latency_ms / reference_measurement.backbone_dim
    )
    reference_gvla_latency_per_dim_per_bit = (
        reference_measurement.gvla_head_latency_ms
        / (reference_measurement.backbone_dim * math.ceil(math.log2(reference_measurement.num_actions)))
    )
    reference_softmax_flops_per_dim_per_action = (
        reference_measurement.softmax_head_flops
        / (reference_measurement.backbone_dim * reference_measurement.num_actions)
    )
    reference_gvla_flops_per_dim_per_bit = (
        reference_measurement.gvla_head_flops
        / (
            reference_measurement.backbone_dim
            * math.ceil(math.log2(reference_measurement.num_actions))
        )
    )

    target_bits = int(math.ceil(math.log2(target_num_actions)))
    dream_rows: List[Dict[str, Any]] = []
    for spec in specs:
        projected_softmax_head_latency_ms = (
            reference_head_latency_per_dim * spec.backbone_dim * target_num_actions
            / reference_measurement.num_actions
        )
        projected_gvla_head_latency_ms = (
            reference_gvla_latency_per_dim_per_bit * spec.backbone_dim * target_bits
        )
        projected_softmax_flops = (
            reference_softmax_flops_per_dim_per_action
            * spec.backbone_dim
            * target_num_actions
        )
        projected_gvla_flops = (
            reference_gvla_flops_per_dim_per_bit * spec.backbone_dim * target_bits
        )

        dream_rows.append(
            {
                "model_name": spec.model_name,
                "backbone_dim": spec.backbone_dim,
                "native_action_interface": spec.native_action_interface,
                "projected_gvla_actions": target_num_actions,
                "projected_softmax_head_latency_ms": projected_softmax_head_latency_ms,
                "projected_gvla_head_latency_ms": projected_gvla_head_latency_ms,
                "projected_head_speedup": projected_softmax_head_latency_ms
                / projected_gvla_head_latency_ms,
                "projected_softmax_head_gflops": projected_softmax_flops / 1e9,
                "projected_gvla_head_gflops": projected_gvla_flops / 1e9,
                "projected_flops_reduction": projected_softmax_flops / projected_gvla_flops,
                "dim_status": spec.dim_status,
                "public_source": spec.public_source,
            }
        )
    return dream_rows


def write_dream_table_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "model_name",
                "backbone_dim",
                "native_action_interface",
                "projected_gvla_actions",
                "projected_softmax_head_latency_ms",
                "projected_gvla_head_latency_ms",
                "projected_head_speedup",
                "projected_softmax_head_gflops",
                "projected_gvla_head_gflops",
                "projected_flops_reduction",
                "dim_status",
                "public_source",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)


def render_latex_table(rows: List[Dict[str, Any]], target_num_actions: int) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        (
            "\\caption{Projected impact of swapping dense action heads for GVLA at "
            f"$N={target_num_actions:,}$ on representative VLA backbones. OpenVLA rows "
            "use measured integration data from this experiment; the remaining rows use "
            "public model/interface specifications with conservative dimension proxies "
            "when the exact hidden width is not explicitly published.}"
        ),
        "\\label{tab:gvla_sota_integration}",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lrrrccc}",
        "\\toprule",
        "Model & $d$ & $N$ & Softmax Head (ms) & GVLA Head (ms) & Speedup & FLOPs Red. \\\\",
        "\\midrule",
    ]

    for row in rows:
        lines.append(
            (
                f"{row['model_name']} & "
                f"{int(row['backbone_dim'])} & "
                f"{int(row['projected_gvla_actions']):,} & "
                f"{row['projected_softmax_head_latency_ms']:.3f} & "
                f"{row['projected_gvla_head_latency_ms']:.3f} & "
                f"{row['projected_head_speedup']:.2f}$\\times$ & "
                f"{row['projected_flops_reduction']:.0f}$\\times$ \\\\"
            )
        )

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_failure_report(output_path: Path, message: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        handle.write(message + "\n")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = resolve_dtype(args.dtype)
    run_dir = build_run_dir(args.output_dir, args.run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_args_snapshot(args, run_dir / "args.txt")

    adapter = OpenVLABackboneAdapter(
        args.model_id,
        device=device,
        dtype=dtype,
        quantization=args.quantization,
        attn_implementation=args.attn_implementation,
    )

    try:
        adapter.load()
        inputs = adapter.make_synthetic_batch(args.batch_size, args.instruction)

        measurements: List[BenchmarkMeasurement] = []
        for num_actions in args.num_actions:
            measurement = benchmark_case(
                adapter,
                inputs,
                num_actions,
                warmup_iters=args.warmup_iters,
                measure_iters=args.measure_iters,
            )
            measurement.total_epr_gain = (
                measurement.total_epr_gvla / measurement.total_epr_softmax
            )
            measurements.append(measurement)

        measurement_csv = run_dir / "openvla_head_swap_benchmark.csv"
        figure_path = run_dir / "openvla_head_swap_benchmark.png"
        write_measurements_csv(measurements, measurement_csv)
        plot_measurements(measurements, figure_path)

        reference_measurement = max(measurements, key=lambda item: item.num_actions)
        dream_rows = project_sota_rows(
            PUBLIC_VLA_SPECS,
            reference_measurement,
            target_num_actions=reference_measurement.num_actions,
        )
        dream_csv_path = run_dir / "sota_dream_table.csv"
        latex_path = run_dir / "sota_dream_table.tex"
        write_dream_table_csv(dream_rows, dream_csv_path)
        latex_path.write_text(
            render_latex_table(dream_rows, reference_measurement.num_actions)
        )

        print(f"Benchmark CSV written to: {measurement_csv}")
        print(f"Benchmark plot written to: {figure_path}")
        print(f"Dream-table CSV written to: {dream_csv_path}")
        print(f"Dream-table LaTeX written to: {latex_path}")
    except Exception as exc:
        failure_path = run_dir / "integration_failure.txt"
        write_failure_report(
            failure_path,
            f"OpenVLA integration failed: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
