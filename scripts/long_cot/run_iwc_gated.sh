#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: iwc_gated
# Recipe:          recipes/long_cot/iwc_gated.yaml
# Category:        long_cot
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/long_cot/iwc_gated.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: iwc_gated"
echo "Category:         long_cot"
echo "Recipe:           recipes/long_cot/iwc_gated.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
