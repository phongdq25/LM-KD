#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: tbpo_q
# Recipe:          recipes/preference/tbpo_q.yaml
# Category:        preference
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/preference/tbpo_q.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: tbpo_q"
echo "Category:         preference"
echo "Recipe:           recipes/preference/tbpo_q.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.reproduce tbpo_q "$@"
