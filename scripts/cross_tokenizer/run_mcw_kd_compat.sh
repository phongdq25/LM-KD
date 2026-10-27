#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: mcw_kd_compat
# Recipe:          recipes/cross_tokenizer/mcw_kd_compat.yaml
# Category:        cross_tokenizer
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/cross_tokenizer/mcw_kd_compat.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: mcw_kd_compat"
echo "Category:         cross_tokenizer"
echo "Recipe:           recipes/cross_tokenizer/mcw_kd_compat.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
