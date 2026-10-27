#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: csd_on_policy
# Recipe:          recipes/same_tokenizer/csd_on_policy.yaml
# Category:        same_tokenizer
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/same_tokenizer/csd_on_policy.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: csd_on_policy"
echo "Category:         same_tokenizer"
echo "Recipe:           recipes/same_tokenizer/csd_on_policy.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
