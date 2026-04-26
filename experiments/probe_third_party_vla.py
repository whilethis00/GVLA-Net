import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe third-party VLA repos from isolated envs.")
    parser.add_argument("--model-family", choices=("octo", "openpi"), required=True)
    parser.add_argument("--third-party-root", type=Path, required=True)
    parser.add_argument("--checkpoint-name", type=str, default="octo-base-1.5")
    return parser.parse_args()


def probe_octo(third_party_root: Path, checkpoint_name: str) -> dict[str, object]:
    sys.path.insert(0, str(third_party_root / "octo"))

    from octo.model.components.transformer import common_transformer_sizes

    transformer_size = "vit_s" if "small" in checkpoint_name else "vit_b"
    token_embedding_size, _ = common_transformer_sizes(transformer_size)
    return {
        "backbone_dim": int(token_embedding_size),
        "status": f"resolved from isolated octo_env source import ({checkpoint_name}, transformer_size={transformer_size})",
    }


def probe_openpi(third_party_root: Path) -> dict[str, object]:
    sys.path.insert(0, str(third_party_root / "openpi" / "src"))

    from openpi.training import config as train_config
    import openpi.models.gemma as gemma

    config = train_config.get_config("pi05_droid")
    width = gemma.get_config(config.model.action_expert_variant).width
    return {
        "backbone_dim": int(width),
        "status": (
            "resolved from isolated openpi_env training config import "
            f"(config={config.name}, action_expert_variant={config.model.action_expert_variant})"
        ),
    }


def main() -> None:
    args = parse_args()
    if args.model_family == "octo":
        result = probe_octo(args.third_party_root, args.checkpoint_name)
    else:
        result = probe_openpi(args.third_party_root)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
