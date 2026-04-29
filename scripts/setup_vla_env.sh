#!/usr/bin/env bash
# =============================================================================
# setup_vla_env.sh
# Install Octo + ALOHA dependencies in the 'vla' conda env (Python 3.10).
# Handles dependency conflicts with careful install ordering.
# Run WITHOUT GPU — CPU-only setup for debugging and eval.
# =============================================================================
set -euo pipefail

PY=/home/introai11/.conda/envs/vla/bin/python
PIP=/home/introai11/.conda/envs/vla/bin/pip
PROJECT_ROOT=/home/introai11/.agile/users/hsjung/projects/GVLA-Net
OCTO_DIR="${PROJECT_ROOT}/third_party/octo"

export PYTHONNOUSERSITE=1

echo "=== Step 1: Fix numpy version (TF 2.15 needs <2.0) ==="
$PIP install "numpy==1.26.4" "protobuf==4.25.3" -q

echo "=== Step 2: Install TensorFlow 2.15 ecosystem ==="
$PIP install \
  "tensorflow==2.15.0" \
  "tensorflow-probability==0.23.0" \
  "tensorflow-datasets==4.9.2" \
  "tensorflow-hub==0.16.1" \
  "tensorflow-graphics==2021.12.3" \
  -q 2>&1 | grep -v "^$"

# tensorflow-text must exactly match — 2.15 for TF 2.15
echo "=== Step 3: Install tensorflow-text 2.15 ==="
$PIP install "tensorflow-text==2.15.0" -q 2>&1 | grep -v "^$" || \
  echo "WARN: tensorflow-text 2.15.0 not available, trying >=2.13"
$PIP show tensorflow-text 2>/dev/null | grep Version

echo "=== Step 4: Pin JAX to 0.4.20 (after TF overwrites it) ==="
$PIP install \
  "jax==0.4.20" \
  "jaxlib==0.4.20" \
  "ml_dtypes==0.2.0" \
  --force-reinstall --no-deps -q

echo "=== Step 5: JAX ecosystem ==="
$PIP install \
  "orbax-checkpoint==0.4.4" \
  "flax==0.7.5" \
  "optax==0.1.5" \
  "chex==0.1.85" \
  "distrax==0.1.5" \
  "ml_collections>=0.1.0" \
  --force-reinstall -q 2>&1 | grep -v "^$"

echo "=== Step 6: Re-pin JAX after flax/orbax install ==="
$PIP install "jax==0.4.20" "jaxlib==0.4.20" "ml_dtypes==0.2.0" --force-reinstall --no-deps -q

echo "=== Step 6.5: Pin scipy/tensorstore to Octo-compatible versions ==="
$PIP install "scipy==1.11.4" "tensorstore==0.1.45" --force-reinstall --no-deps -q

echo "=== Step 7: MuJoCo + ALOHA gym ==="
$PIP install "mujoco==3.3.7" --only-binary=:all: -q
$PIP install "dm-control>=1.0.15" -q
$PIP install "gymnasium>=0.29.1" "gym==0.26.2" -q
$PIP install "gym-aloha==0.1.3" --ignore-requires-python --no-deps -q

echo "=== Step 8: Misc deps ==="
$PIP install \
  "setuptools<81" \
  "scipy>=1.6.0" \
  "imageio>=2.31.1" \
  "moviepy>=1.0.3" \
  "einops>=0.6.1" \
  "matplotlib" \
  -q

echo "=== Step 9: dlimp (Octo data pipeline) ==="
$PIP install "dlimp @ git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972" \
  -q 2>&1 | grep -v "^$"

echo "=== Step 10: Install Octo package ==="
$PIP install -e "${OCTO_DIR}" --no-deps -q

echo "=== Step 11: Final JAX pin (in case anything overwrote it) ==="
$PIP install "jax==0.4.20" "jaxlib==0.4.20" "ml_dtypes==0.2.0" --force-reinstall --no-deps -q

echo ""
echo "=== Verifying installation ==="
$PY /home/introai11/.agile/users/hsjung/projects/GVLA-Net/scripts/verify_vla_env.py
