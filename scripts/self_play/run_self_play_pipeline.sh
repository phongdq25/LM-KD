#!/usr/bin/env bash
# ==============================================================================
# LM-KD: Two-Stage Self-Play Fine-Tuning Pipeline Launcher
# ==============================================================================
# Executes the end-to-end 2-stage Self-Play fine-tuning protocol:
#   Stage 1: Warmup SFT on HuggingFaceH4/ultrachat_200k -> generates base model M_0
#   Stage 2: Iterative Self-Play (SPIN/T-SPIN/SPACE/SPPO/S-SPPO/RSPO) on UCLA-AGI/SPIN_iter0
#
# Usage:
#   bash scripts/self_play/run_self_play_pipeline.sh --method <method> [options...]
#
# Options:
#   --method <spin|tspin|space|sppo|s_sppo|rspo>  (Required) Self-play algorithm
#   --model <qwen2.5|gemma2|qwen3>               (Default: qwen2.5) Base model architecture
#   --skip-sft                                   Skip Stage 1 and use existing SFT checkpoint
#   --sft-checkpoint <path>                      Custom checkpoint path to start Stage 2 from
#   --dry-run                                    Validate recipes and print commands without executing
#   --gpu <gpu_id>                               CUDA_VISIBLE_DEVICES to use (default: current or 0)
#
# Examples:
#   bash scripts/self_play/run_self_play_pipeline.sh --method spin --dry-run
#   bash scripts/self_play/run_self_play_pipeline.sh --method spin --model qwen2.5
#   bash scripts/self_play/run_self_play_pipeline.sh --method sppo --skip-sft --sft-checkpoint outputs/self_play/sft_warmup/qwen2.5/final
# ==============================================================================
set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

METHOD=""
MODEL_KEY="qwen2.5"
SKIP_SFT=false
CUSTOM_SFT_CKPT=""
DRY_RUN=false
GPU_ID="${CUDA_VISIBLE_DEVICES:-0}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --model)
            MODEL_KEY="$2"
            shift 2
            ;;
        --skip-sft)
            SKIP_SFT=true
            shift
            ;;
        --sft-checkpoint)
            CUSTOM_SFT_CKPT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "$METHOD" ]]; then
    echo "ERROR: --method is required. Choose one of: spin, tspin, space, sppo, s_sppo, rspo" >&2
    echo "Usage: $0 --method <method> [--model qwen2.5|gemma2|qwen3] [--dry-run]" >&2
    exit 1
fi

case "$MODEL_KEY" in
    qwen2.5)
        BASE_MODEL="Qwen/Qwen2.5-1.5B"
        ;;
    gemma2)
        BASE_MODEL="google/gemma-2-2b"
        ;;
    qwen3)
        BASE_MODEL="Qwen/Qwen3-4B-Base"
        ;;
    *)
        BASE_MODEL="$MODEL_KEY"
        ;;
esac

STAGE1_RECIPE="$REPO_DIR/recipes/self_play/sft_warmup.yaml"
STAGE2_RECIPE="$REPO_DIR/recipes/self_play/${METHOD}.yaml"

if [[ ! -f "$STAGE2_RECIPE" ]]; then
    echo "ERROR: Recipe for method '$METHOD' not found at: $STAGE2_RECIPE" >&2
    exit 1
fi

STAGE1_OUT_DIR="$REPO_DIR/outputs/self_play/sft_warmup/${MODEL_KEY}"
STAGE1_FINAL_CKPT="${CUSTOM_SFT_CKPT:-$STAGE1_OUT_DIR/final}"
STAGE2_OUT_DIR="$REPO_DIR/outputs/self_play/${METHOD}_${MODEL_KEY}"

echo "======================================================================"
echo "LM-KD: Self-Play Two-Stage Fine-Tuning Pipeline"
echo "Method:               $METHOD"
echo "Model Key:            $MODEL_KEY ($BASE_MODEL)"
echo "Stage 1 Recipe:       $STAGE1_RECIPE"
echo "Stage 1 Output:       $STAGE1_OUT_DIR"
echo "Stage 1 Final Checkpoint: $STAGE1_FINAL_CKPT"
echo "Stage 2 Recipe:       $STAGE2_RECIPE"
echo "Stage 2 Output:       $STAGE2_OUT_DIR"
echo "Skip Stage 1:         $SKIP_SFT"
echo "Dry Run:              $DRY_RUN"
echo "GPU ID:               $GPU_ID"
echo "Python:               $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

# ------------------------------------------------------------------------------
# STAGE 1: SFT Warmup on UltraChat200k
# ------------------------------------------------------------------------------
if [[ "$SKIP_SFT" == "true" ]]; then
    echo ""
    echo ">>> [Stage 1] Skipped (--skip-sft specified). Using checkpoint: $STAGE1_FINAL_CKPT"
else
    echo ""
    echo ">>> [Stage 1/2] Launching SFT Warmup on HuggingFaceH4/ultrachat_200k..."
    STAGE1_CMD=(
        "$PYTHON" -m lmkd.cli train
        -c "$STAGE1_RECIPE"
        --set "models.student.path=$BASE_MODEL" "train.output_dir=$STAGE1_OUT_DIR"
    )
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] Command: ${STAGE1_CMD[*]}"
        "$PYTHON" -m lmkd.cli check -c "$STAGE1_RECIPE" --set "models.student.path=$BASE_MODEL" "train.output_dir=$STAGE1_OUT_DIR"
    else
        "${STAGE1_CMD[@]}"
        echo ">>> [Stage 1/2] SFT Warmup completed. Base checkpoint saved at: $STAGE1_FINAL_CKPT"
    fi
fi

# ------------------------------------------------------------------------------
# STAGE 2: Self-Play Fine-Tuning on UCLA-AGI/SPIN_iter0
# ------------------------------------------------------------------------------
echo ""
echo ">>> [Stage 2/2] Launching Self-Play ($METHOD) on UCLA-AGI/SPIN_iter0..."

# If stage 1 was run or custom ckpt was provided, use it; otherwise fallback to base model
INPUT_MODEL="$BASE_MODEL"
if [[ -d "$STAGE1_FINAL_CKPT" || "$SKIP_SFT" == "true" || -n "$CUSTOM_SFT_CKPT" ]]; then
    INPUT_MODEL="$STAGE1_FINAL_CKPT"
fi

STAGE2_SETS=("models.student.path=$INPUT_MODEL" "train.output_dir=$STAGE2_OUT_DIR")
if [[ "$METHOD" != "tspin" ]]; then
    STAGE2_SETS+=("models.teacher.path=$INPUT_MODEL")
fi

STAGE2_CMD=(
    "$PYTHON" -m lmkd.cli train
    -c "$STAGE2_RECIPE"
    --set "${STAGE2_SETS[@]}"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
    STAGE2_CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] Command: ${STAGE2_CMD[*]}"
    "$PYTHON" -m lmkd.cli check -c "$STAGE2_RECIPE" --set "${STAGE2_SETS[@]}"
    echo ""
    echo "=== Pipeline verification passed successfully (DRY-RUN) ==="
else
    "${STAGE2_CMD[@]}"
    echo ""
    echo ">>> [Stage 2/2] Self-Play ($METHOD) training finished!"
    echo "Final checkpoints and metrics saved in: $STAGE2_OUT_DIR"
fi
