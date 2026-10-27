#!/usr/bin/env bash
# ==============================================================================
# LM-KD: Single-GPU / CPU Training Launcher
# ==============================================================================
# Usage:
#   bash scripts/train_single_gpu.sh <recipe_path> [extra_lmkd_args...]
#
# Examples:
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train_single_gpu.sh recipes/same_tokenizer/fkl.yaml
#   bash scripts/train_single_gpu.sh recipes/same_tokenizer/fkl.yaml --set train.max_steps=50 optim.lr=1e-5
# ==============================================================================

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <recipe_path> [extra_lmkd_args...]"
    echo "Example: $0 recipes/same_tokenizer/fkl.yaml --set train.max_steps=100"
    exit 1
fi

RECIPE="$1"
shift

if [ ! -f "$RECIPE" ]; then
    # Try resolving relative to repo directory
    if [ -f "$REPO_DIR/$RECIPE" ]; then
        RECIPE="$REPO_DIR/$RECIPE"
    else
        echo "Error: Recipe file not found: $RECIPE" >&2
        exit 1
    fi
fi

echo "======================================================================"
echo "LM-KD Training: Single Device"
echo "Recipe:  $RECIPE"
echo "Python:  $PYTHON"
echo "Device:  ${CUDA_VISIBLE_DEVICES:-Default (auto)}"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"

