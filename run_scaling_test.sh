#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 "${PROJECT_ROOT}/experiments/scaling_test.py" "$@"
