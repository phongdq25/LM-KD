#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: ra_dpo
# Recipe:          recipes/preference/ra_dpo.yaml
# Category:        preference
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/preference/ra_dpo.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: ra_dpo"
echo "Category:         preference"
echo "Recipe:           recipes/preference/ra_dpo.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
