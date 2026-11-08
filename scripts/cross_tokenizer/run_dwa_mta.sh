#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: dwa_mta (DWA-KD + Multi-granularity Text Alignment)
# Recipe:          recipes/cross_tokenizer/dwa_mta.yaml
# Category:        cross_tokenizer
# Source:          MTA/dwa_mta
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/cross_tokenizer/dwa_mta.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: dwa_mta (DWA-KD + MTA)"
echo "Category:         cross_tokenizer"
echo "Recipe:           recipes/cross_tokenizer/dwa_mta.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"

