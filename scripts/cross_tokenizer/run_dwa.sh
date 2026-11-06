#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: dwa (Dynamic Window Alignment Knowledge Distillation)
# Recipe:          recipes/cross_tokenizer/dwa.yaml
# Category:        cross_tokenizer
# Source:          MTA/dwa_mta
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/cross_tokenizer/dwa.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: dwa (DWA-KD)"
echo "Category:         cross_tokenizer"
echo "Recipe:           recipes/cross_tokenizer/dwa.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"

