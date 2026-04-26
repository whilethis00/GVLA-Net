import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import torch


COMMON_HIDDEN_DIM_KEYS: Sequence[str] = (
    "hidden_size",
    "d_model",
    "dim",
    "embed_dim",
    "embedding_dim",
    "n_embd",
    "width",
    "model_dim",
)


@dataclass(frozen=True)
class UniversalVLASpec:
    model_name: str
    public_source: str
    architecture_summary: str
    native_action_interface: str
    attack_point: str
    native_head_type: str
    model_id: Optional[str]
    proxy_backbone_dim: int
    dim_status: str


UNIVERSAL_VLA_SPECS: tuple[UniversalVLASpec, ...] = (
    UniversalVLASpec(
        model_name="OpenVLA-7B",
        public_source="https://huggingface.co/openvla/openvla-7b",
        architecture_summary="Prismatic VLM with pooled vision-language hidden states.",
        native_action_interface="7-DoF continuous deltas",
        attack_point="Measured anchor: replace dense action routing without touching the backbone.",
        native_head_type="dense autoregressive action head",
        model_id="openvla/openvla-7b",
        proxy_backbone_dim=4096,
        dim_status="measured in local integration run",
    ),
    UniversalVLASpec(
        model_name="RT-2-X",
        public_source="https://arxiv.org/abs/2307.15818",
        architecture_summary="PaLI-X style VLA that emits action tokens through a large VLM stack.",
        native_action_interface="tokenized 7-DoF robot actions",
        attack_point="Real-time retrofit: keep the giant backbone, remove O(N) action routing pressure.",
        native_head_type="token logits over discretized actions",
        model_id=None,
        proxy_backbone_dim=4096,
        dim_status="proxy from PaLI-X family; exact width should be probed from the target checkpoint",
    ),
    UniversalVLASpec(
        model_name="Octo-Base",
        public_source="https://github.com/octo-models/octo",
        architecture_summary="Transformer-based robot policy with diffusion-style action prediction.",
        native_action_interface="continuous action chunks",
        attack_point="Precision transplant: push Octo to much finer action resolution with logarithmic routing.",
        native_head_type="diffusion readout head",
        model_id="hf://rail-berkeley/octo-base-1.5",
        proxy_backbone_dim=768,
        dim_status="resolved from Octo base config token_embedding_size",
    ),
    UniversalVLASpec(
        model_name="pi0.5",
        public_source="https://github.com/Physical-Intelligence/openpi",
        architecture_summary="Flow-based VLA with a transformer backbone and iterative action generation.",
        native_action_interface="continuous action chunks via flow matching",
        attack_point="Compute compression: replace expensive high-precision action routing with geometric hashing.",
        native_head_type="flow-matching action head",
        model_id=None,
        proxy_backbone_dim=1024,
        dim_status="resolved from openpi action expert config width for pi0.5",
    ),
)


def infer_hidden_dim_from_config(config_like: Any) -> Optional[int]:
    if config_like is None:
        return None

    if isinstance(config_like, dict):
        config_dict = config_like
    else:
        try:
            config_dict = vars(config_like)
        except TypeError:
            config_dict = {
                key: getattr(config_like, key)
                for key in dir(config_like)
                if not key.startswith("_")
            }

    for key in COMMON_HIDDEN_DIM_KEYS:
        value = config_dict.get(key)
        if isinstance(value, int) and value > 0:
            return value

    for nested_key in ("text_config", "vision_config", "model_config", "backbone_config"):
        nested = config_dict.get(nested_key)
        nested_value = infer_hidden_dim_from_config(nested)
        if nested_value is not None:
            return nested_value

    return None


def infer_hidden_dim_from_outputs(outputs: Any) -> Optional[int]:
    if outputs is None:
        return None

    candidate_names = (
        "hidden_states",
        "decoder_hidden_states",
        "last_hidden_state",
        "encoder_last_hidden_state",
        "logits",
    )
    for name in candidate_names:
        value = getattr(outputs, name, None)
        if value is None:
            continue
        if isinstance(value, (tuple, list)) and value:
            value = value[-1]
        if torch.is_tensor(value) and value.ndim >= 2:
            return int(value.shape[-1])
    return None


def infer_hidden_dim(
    *,
    spec: UniversalVLASpec,
    config_like: Any = None,
    outputs: Any = None,
) -> tuple[int, str]:
    config_dim = infer_hidden_dim_from_config(config_like)
    if config_dim is not None:
        return config_dim, "resolved from model config"

    output_dim = infer_hidden_dim_from_outputs(outputs)
    if output_dim is not None:
        return output_dim, "resolved from model outputs"

    return spec.proxy_backbone_dim, spec.dim_status


class SyntheticUniversalAdapter:
    """Common benchmarking shim for VLA backbones with optional metadata probing.

    The adapter is intentionally lightweight: when the real model implementation is not
    available in the current environment, it still provides a consistent latent-space
    interface for head-level benchmarking by falling back to a documented proxy width.
    """

    def __init__(
        self,
        spec: UniversalVLASpec,
        *,
        device: torch.device,
        dtype: torch.dtype,
        config_like: Any = None,
        outputs: Any = None,
    ) -> None:
        self.spec = spec
        self.device = device
        self.dtype = dtype
        self.backbone_dim, self.dim_status = infer_hidden_dim(
            spec=spec,
            config_like=config_like,
            outputs=outputs,
        )

    def synthetic_embeddings(self, batch_size: int) -> torch.Tensor:
        return torch.randn(
            batch_size,
            self.backbone_dim,
            device=self.device,
            dtype=self.dtype,
        )


def _prepend_sys_path(path: Path) -> None:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def _default_env_python(env_name: str) -> Optional[Path]:
    candidate = Path.home() / ".conda" / "envs" / env_name / "bin" / "python"
    if candidate.exists():
        return candidate
    return None


def _probe_with_external_env(
    env_python: Optional[Path],
    *,
    model_family: str,
    third_party_root: Path,
    checkpoint_name: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    if env_python is None or not env_python.exists():
        return None

    script_path = Path(__file__).resolve().parent / "probe_third_party_vla.py"
    command = [str(env_python), str(script_path), "--model-family", model_family, "--third-party-root", str(third_party_root)]
    if checkpoint_name is not None:
        command.extend(["--checkpoint-name", checkpoint_name])

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return None


def resolve_octo_backbone_dim(
    third_party_root: Path,
    *,
    checkpoint_name: str = "octo-base-1.5",
    env_python: Optional[Path] = None,
) -> tuple[int, str]:
    probe = _probe_with_external_env(
        env_python or _default_env_python("octo_env"),
        model_family="octo",
        third_party_root=third_party_root,
        checkpoint_name=checkpoint_name,
    )
    if probe is not None:
        return int(probe["backbone_dim"]), str(probe["status"])

    try:
        _prepend_sys_path(third_party_root / "octo")
        from octo.model.components.transformer import common_transformer_sizes

        if "small" in checkpoint_name:
            transformer_size = "vit_s"
        else:
            transformer_size = "vit_b"
        token_embedding_size, _ = common_transformer_sizes(transformer_size)
        return int(token_embedding_size), f"resolved from Octo {checkpoint_name} transformer config"
    except Exception:
        if "small" in checkpoint_name:
            return 384, "resolved from public Octo small HF config token_embedding_size"
        return 768, "resolved from public Octo base HF config token_embedding_size"


def resolve_openpi05_backbone_dim(
    third_party_root: Path,
    *,
    env_python: Optional[Path] = None,
) -> tuple[int, str]:
    probe = _probe_with_external_env(
        env_python or _default_env_python("openpi_env"),
        model_family="openpi",
        third_party_root=third_party_root,
    )
    if probe is not None:
        return int(probe["backbone_dim"]), str(probe["status"])

    try:
        _prepend_sys_path(third_party_root / "openpi" / "src")

        import openpi.models.gemma as gemma
        import openpi.models.pi0_config as pi0_config

        config = pi0_config.Pi0Config(pi05=True)
        action_expert = gemma.get_config(config.action_expert_variant)
        return int(action_expert.width), "resolved from openpi pi0.5 action expert width"
    except Exception:
        return 1024, "resolved from openpi source defaults: pi0.5 action expert variant gemma_300m width=1024"


def list_universal_specs() -> Iterable[UniversalVLASpec]:
    return UNIVERSAL_VLA_SPECS


def code_length(num_actions: int) -> int:
    return int(math.ceil(math.log2(num_actions)))
