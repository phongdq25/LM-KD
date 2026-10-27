#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: sgl_spectral_pipeline
# Recipe:          recipes/long_cot/sgl_spectral_pipeline.yaml
# Category:        long_cot
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/long_cot/sgl_spectral_pipeline.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: sgl_spectral_pipeline"
echo "Category:         long_cot"
echo "Recipe:           recipes/long_cot/sgl_spectral_pipeline.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
