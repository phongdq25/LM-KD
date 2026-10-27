#!/usr/bin/env bash
# ==============================================================================
# LM-KD: Two-Phase ResidualKD Execution Pipeline
# ==============================================================================
# ResidualKD consists of two distinct training stages:
#   Phase 1: Pretrain projector P_TA using teacher representations.
#   Phase 2: Train student backbone with residual distillation using Phase 1 checkpoint.
#
# Usage:
#   bash scripts/run_residualkd_full.sh [extra_lmkd_args...]
# ==============================================================================

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

STAGE1_RECIPE="${RESIDUALKD_STAGE1_RECIPE:-recipes/cross_tokenizer/residualkd_stage1.yaml}"
STAGE2_RECIPE="${RESIDUALKD_STAGE2_RECIPE:-recipes/cross_tokenizer/residualkd_stage2.yaml}"
PROJECTOR_CKPT="${RESIDUALKD_PROJECTOR_CKPT:-outputs/residualkd/stage1/checkpoint-final/modules.pt}"

echo "======================================================================"
echo "ResidualKD: Launching Phase 1 (Projector Pretraining)"
echo "Recipe: $STAGE1_RECIPE"
echo "======================================================================"

cd "$REPO_DIR"
"$PYTHON" -m lmkd.cli train -c "$STAGE1_RECIPE" "$@"

echo "======================================================================"
echo "ResidualKD: Launching Phase 2 (Residual Distillation)"
echo "Recipe:         $STAGE2_RECIPE"
echo "Projector Ckpt: $PROJECTOR_CKPT"
echo "======================================================================"

"$PYTHON" -m lmkd.cli train -c "$STAGE2_RECIPE" \
  --set "loss.terms.0.params.projector_ckpt=$PROJECTOR_CKPT" "$@"

echo "ResidualKD 2-phase pipeline completed successfully!"
