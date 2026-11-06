#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: jasper (2-stage distillation via pipeline)
# Recipe:          recipes/embedding/jasper.yaml
# Category:        embedding
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/embedding/jasper.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: jasper (2-stage pipeline)"
echo "Category:         embedding"
echo "Recipe:           recipes/embedding/jasper.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"


