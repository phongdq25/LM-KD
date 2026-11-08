#!/usr/bin/env bash
# ==============================================================================
# Baseline Runner: residualkd_full
# Recipe:          recipes/cross_tokenizer/residualkd_full.yaml
# Category:        cross_tokenizer
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"
RECIPE="$REPO_DIR/recipes/cross_tokenizer/residualkd_full.yaml"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

echo "======================================================================"
echo "Running Baseline: residualkd_full"
echo "Category:         cross_tokenizer"
echo "Recipe:           recipes/cross_tokenizer/residualkd_full.yaml"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
# ResidualKD 2-phase pipeline (Projector Pretraining -> Residual Distillation):
STAGE1="$REPO_DIR/recipes/cross_tokenizer/residualkd_stage1.yaml"
STAGE2="$REPO_DIR/recipes/cross_tokenizer/residualkd_stage2.yaml"
PROJECTOR_CKPT="${PROJECTOR_CKPT:-outputs/residualkd/stage1/checkpoint-final/modules.pt}"

echo "======================================================================"
echo "ResidualKD Phase 1: Pretraining Projector"
echo "Recipe: $STAGE1"
echo "======================================================================"
"$PYTHON" -m lmkd.cli train -c "$STAGE1" "$@"

echo "======================================================================"
echo "ResidualKD Phase 2: Residual Distillation"
echo "Recipe:         $STAGE2"
echo "Projector Ckpt: $PROJECTOR_CKPT"
echo "======================================================================"
exec "$PYTHON" -m lmkd.cli train -c "$STAGE2" --set "loss.terms.0.params.projector_ckpt=$PROJECTOR_CKPT" "$@"
