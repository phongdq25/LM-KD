#!/usr/bin/env bash
# ==============================================================================
# LM-KD: Multi-GPU Distributed Training with PyTorch Native FSDP2
# ==============================================================================
# Usage:
#   NUM_GPUS=2 bash scripts/train_multigpu_fsdp2.sh <recipe_path> [extra_args...]
#
# Examples:
#   NUM_GPUS=2 bash scripts/train_multigpu_fsdp2.sh recipes/same_tokenizer/fkl.yaml \
#     --set engine.type=fsdp2 engine.mixed_precision=bf16
# ==============================================================================

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-$((29600 + RANDOM % 1000))}"

if [ $# -lt 1 ]; then
    echo "Usage: NUM_GPUS=<N> $0 <recipe_path> [extra_args...]"
    echo "Example: NUM_GPUS=2 $0 recipes/same_tokenizer/fkl.yaml --set engine.type=fsdp2 engine.mixed_precision=bf16"
    exit 1
fi

RECIPE="$1"
shift

if [ ! -f "$RECIPE" ]; then
    if [ -f "$REPO_DIR/$RECIPE" ]; then
        RECIPE="$REPO_DIR/$RECIPE"
    else
        echo "Error: Recipe file not found: $RECIPE" >&2
        exit 1
    fi
fi

echo "======================================================================"
echo "LM-KD Multi-GPU Training (Native FSDP2)"
echo "GPUs:         $NUM_GPUS"
echo "Master Port:  $MASTER_PORT"
echo "Recipe:       $RECIPE"
echo "Python:       $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"

exec "$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    -m lmkd.cli train -c "$RECIPE" \
    --set engine.type=fsdp2 engine.mixed_precision=bf16 "$@"

