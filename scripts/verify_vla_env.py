#!/usr/bin/env python3
"""Verify the vla env has all dependencies needed for Octo ALOHA finetune + eval.

This script intentionally runs each check in a fresh subprocess. Importing
TensorFlow and JAX-heavy Octo modules in a single long-lived process can trip
XLA / absl initialization issues even when the environment is otherwise fine.
"""

from __future__ import annotations

import os
import subprocess
import sys


PROJECT_ROOT = "/home/introai11/.agile/users/hsjung/projects/GVLA-Net"
OCTO_PATH = f"{PROJECT_ROOT}/third_party/octo"
PYTHON = sys.executable


def run_check(name: str, code: str) -> bool:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    env["PYTHONPATH"] = f"{OCTO_PATH}:{env.get('PYTHONPATH', '')}".rstrip(":")

    proc = subprocess.run(
        [PYTHON, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode == 0:
        output = proc.stdout.strip() or "OK"
        print(f"  {name}: {output} ✓")
        return True

    message = proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}"
    last_line = message.splitlines()[-1]
    print(f"  {name}: FAIL — {last_line}")
    return False


def main() -> int:
    errors: list[str] = []

    print("=== vla env dependency check ===")
    print()

    print("[JAX stack]")
    jax_checks = [
        ("jax", "import jax; print(jax.__version__)"),
        ("jaxlib", "import jaxlib; print(jaxlib.__version__)"),
        ("flax", "import flax; print(flax.__version__)"),
        ("optax", "import optax; print(optax.__version__)"),
        ("chex", "import chex; print(chex.__version__)"),
        ("orbax-checkpoint", "import orbax.checkpoint as ocp; print(ocp.__version__)"),
    ]
    for name, code in jax_checks:
        if not run_check(name, code):
            errors.append(name)

    print()
    print("[TensorFlow stack]")
    tf_checks = [
        ("tensorflow", "import tensorflow as tf; print(tf.__version__)"),
        ("tensorflow_text", "import tensorflow_text; print(tensorflow_text.__version__)"),
        ("tensorflow_datasets", "import tensorflow_datasets as tfds; print(tfds.__version__)"),
        ("tensorflow_hub", "import tensorflow_hub as hub; print(hub.__version__)"),
    ]
    for name, code in tf_checks:
        if not run_check(name, code):
            errors.append(name)

    print()
    print("[Octo model]")
    octo_checks = [
        ("octo.model", "import octo.model.octo_model; print('OctoModel importable')"),
        (
            "octo.tokenizers",
            "import octo.model.components.tokenizers; print('LowdimObsTokenizer importable')",
        ),
        (
            "octo.action_heads",
            "import octo.model.components.action_heads; print('L1ActionHead importable')",
        ),
    ]
    for name, code in octo_checks:
        if not run_check(name, code):
            errors.append(name)

    print()
    print("[ALOHA / MuJoCo]")
    aloha_checks = [
        ("mujoco", "import mujoco; print(mujoco.__version__)"),
        ("gym", "import gym; print(gym.__version__)"),
        ("gymnasium", "import gymnasium; print(gymnasium.__version__)"),
        ("dm_control", "import dm_control; print('importable')"),
        ("gym_aloha", "import gym_aloha; print('gym_aloha importable')"),
    ]
    for name, code in aloha_checks:
        if not run_check(name, code):
            errors.append(name)

    print()
    print("[dlimp]")
    if not run_check("dlimp", "import dlimp; print(getattr(dlimp, '__version__', 'installed'))"):
        errors.append("dlimp")

    print()
    if errors:
        print(f"FAILED modules: {errors}")
        return 1

    print("All checks passed. Environment is ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
