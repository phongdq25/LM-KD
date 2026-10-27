#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: epsilon_dpo
# Recipe:          recipes/preference/epsilon_dpo.yaml
# Category:        preference
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/preference/epsilon_dpo.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: epsilon_dpo"
echo "Category:         preference"
echo "Recipe:           recipes/preference/epsilon_dpo.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
