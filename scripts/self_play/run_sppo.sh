#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: sppo (Stage 2 Self-Play Preference Optimization)
# Recipe:          recipes/self_play/sppo.yaml
# Category:        self_play
# Dataset:         UCLA-AGI/SPIN_iter0 (50k prompts)
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/self_play/sppo.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: sppo (Self-Play Preference Optimization)"
echo "Category:         self_play"
echo "Recipe:           recipes/self_play/sppo.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"

