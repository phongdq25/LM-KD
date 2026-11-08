#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: distillm2 (Pure DistiLLM-2, epoch-level student generation)
# Recipe:          recipes/same_tokenizer/distillm2.yaml
# Category:        same_tokenizer
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/same_tokenizer/distillm2.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: distillm2 (Epoch-level generation)"
echo "Category:         same_tokenizer"
echo "Recipe:           $RECIPE"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
