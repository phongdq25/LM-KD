#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: logits_sam
# Recipe:          recipes/preference/logits_sam.yaml
# Category:        preference
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/preference/logits_sam.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: logits_sam"
echo "Category:         preference"
echo "Recipe:           recipes/preference/logits_sam.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.reproduce logits_sam "$@"
