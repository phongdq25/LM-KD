#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: grpo
# Recipe:          recipes/self_distill/grpo.yaml
# Category:        self_distill
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/self_distill/grpo.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: grpo"
echo "Category:         self_distill"
echo "Recipe:           recipes/self_distill/grpo.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
