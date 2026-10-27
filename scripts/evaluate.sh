#!/usr/bin/env bash
# ==============================================================================
# LM-KD: Model Evaluation Launcher
# ==============================================================================
# Usage:
#   bash scripts/evaluate.sh <recipe_path> --ckpt <checkpoint_dir> [extra_args...]
#
# Examples:
#   bash scripts/evaluate.sh recipes/same_tokenizer/fkl.yaml --ckpt outputs/fkl/checkpoint-final
# ==============================================================================

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <recipe_path> [lmkd_eval_flags...]"
    echo "Example: $0 recipes/same_tokenizer/fkl.yaml --ckpt outputs/fkl/checkpoint-final"
    exit 1
fi

RECIPE="$1"
shift

if [ ! -f "$RECIPE" ]; then
    if [ -f "$REPO_DIR/$RECIPE" ]; then
        RECIPE="$REPO_DIR/$RECIPE"
    else
        echo "Error: Recipe file not found: $RECIPE" >&2
        exit 1
    fi
fi

echo "======================================================================"
echo "LM-KD Evaluation"
echo "Recipe:  $RECIPE"
echo "Python:  $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli eval -c "$RECIPE" "$@"

