#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: dskdv2_impact_bi_ratio
# Recipe:          recipes/cross_tokenizer/dskdv2_impact_bi_ratio.yaml
# Category:        cross_tokenizer
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/cross_tokenizer/dskdv2_impact_bi_ratio.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: dskdv2_impact_bi_ratio"
echo "Category:         cross_tokenizer"
echo "Recipe:           recipes/cross_tokenizer/dskdv2_impact_bi_ratio.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.cli train -c "$RECIPE" "$@"
