#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: tspin (Stage 2 Triplet Self-Play Fine-Tuning)
# Recipe:          recipes/self_play/tspin.yaml
# Category:        self_play
# Dataset:         UCLA-AGI/SPIN_iter0 (50k prompts with proto)
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/self_play/tspin.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: tspin (Triplet Self-Play Fine-Tuning)"
echo "Category:         self_play"
echo "Recipe:           recipes/self_play/tspin.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"

