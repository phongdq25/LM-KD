#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: residualkd_impact
# Recipe:          recipes/cross_tokenizer/residualkd_impact.yaml
# Category:        cross_tokenizer
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/cross_tokenizer/residualkd_impact.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: residualkd_impact"
echo "Category:         cross_tokenizer"
echo "Recipe:           recipes/cross_tokenizer/residualkd_impact.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
