#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: spin (Stage 2 Self-Play Fine-Tuning)
# Recipe:          recipes/self_play/spin.yaml
# Category:        self_play
# Dataset:         UCLA-AGI/SPIN_iter0 (50k prompts)
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/self_play/spin.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: spin (Self-Play Fine-Tuning)"
echo "Category:         self_play"
echo "Recipe:           recipes/self_play/spin.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"

