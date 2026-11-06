#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: ctpd
# Recipe:          recipes/preference/ctpd.yaml
# Category:        preference
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/preference/ctpd.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: ctpd"
echo "Category:         preference"
echo "Recipe:           recipes/preference/ctpd.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
