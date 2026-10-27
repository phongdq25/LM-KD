#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: dskdv2_impact_ratio1to1
# Recipe:          recipes/cross_tokenizer/dskdv2_impact_ratio1to1.yaml
# Category:        cross_tokenizer
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/cross_tokenizer/dskdv2_impact_ratio1to1.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: dskdv2_impact_ratio1to1"
echo "Category:         cross_tokenizer"
echo "Recipe:           recipes/cross_tokenizer/dskdv2_impact_ratio1to1.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
