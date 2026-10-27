#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: dskdv2_impact_first
# Recipe:          recipes/cross_tokenizer/ablation/dskdv2_impact_first.yaml
# Category:        cross_tokenizer/ablation
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/cross_tokenizer/ablation/dskdv2_impact_first.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: dskdv2_impact_first"
echo "Category:         cross_tokenizer/ablation"
echo "Recipe:           recipes/cross_tokenizer/ablation/dskdv2_impact_first.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
