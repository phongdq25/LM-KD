#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: fkl_zero2
# Recipe:          recipes/same_tokenizer/fkl_zero2.yaml
# Category:        same_tokenizer
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/same_tokenizer/fkl_zero2.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: fkl_zero2"
echo "Category:         same_tokenizer"
echo "Recipe:           recipes/same_tokenizer/fkl_zero2.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
