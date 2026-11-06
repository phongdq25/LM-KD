#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: provenance_nll_dft_j8
# Recipe:          recipes/long_cot/provenance_nll_dft_j8.yaml
# Category:        long_cot
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/long_cot/provenance_nll_dft_j8.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: provenance_nll_dft_j8"
echo "Category:         long_cot"
echo "Recipe:           recipes/long_cot/provenance_nll_dft_j8.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
