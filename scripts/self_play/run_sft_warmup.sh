#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: sft_warmup (Stage 1 Warmup SFT for Self-Play)
# Recipe:          recipes/self_play/sft_warmup.yaml
# Category:        self_play
# Dataset:         HuggingFaceH4/ultrachat_200k
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/self_play/sft_warmup.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Stage 1 Warmup SFT: sft_warmup"
echo "Category:                   self_play"
echo "Recipe:                     recipes/self_play/sft_warmup.yaml"
echo "Python:                     $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"

