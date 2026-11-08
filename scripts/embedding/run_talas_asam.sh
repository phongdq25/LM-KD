#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: talas_asam
# Recipe:          recipes/embedding/talas_asam.yaml
# Category:        embedding
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/embedding/talas_asam.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: talas_asam"
echo "Category:         embedding"
echo "Recipe:           recipes/embedding/talas_asam.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
