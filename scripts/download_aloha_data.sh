#!/usr/bin/env bash
# Download ALOHA sim dataset for Octo finetune
set -euo pipefail

DATA_DIR="/home/introai11/.agile/users/hsjung/projects/GVLA-Net/data/aloha_sim"
ZIP_URL="https://rail.eecs.berkeley.edu/datasets/example_sim_data.zip"
ZIP_PATH="/tmp/example_sim_data.zip"

mkdir -p "${DATA_DIR}"

if [ -d "${DATA_DIR}/aloha_sim_cube_scripted_dataset" ]; then
  echo "Dataset already exists at ${DATA_DIR}/aloha_sim_cube_scripted_dataset"
  echo "Skipping download."
  exit 0
fi

echo "Downloading ALOHA sim dataset (~500MB)..."
wget -q --show-progress -O "${ZIP_PATH}" "${ZIP_URL}"

echo "Extracting..."
unzip -q "${ZIP_PATH}" -d "${DATA_DIR}"
rm -f "${ZIP_PATH}"

echo "Done. Dataset at: ${DATA_DIR}"
ls "${DATA_DIR}"
