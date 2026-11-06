#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: distillcse
# Recipe:          recipes/embedding/distillcse.yaml
# Category:        embedding
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/embedding/distillcse.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: distillcse"
echo "Category:         embedding"
echo "Recipe:           recipes/embedding/distillcse.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"

