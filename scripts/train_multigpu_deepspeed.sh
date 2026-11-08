#!/usr/bin/env bash
# ==============================================================================
# LM-KD: Multi-GPU Distributed Training with DeepSpeed ZeRO
# ==============================================================================
# Usage:
#   NUM_GPUS=2 bash scripts/train_multigpu_deepspeed.sh <recipe_path> [extra_args...]
#
# Examples:
#   NUM_GPUS=2 bash scripts/train_multigpu_deepspeed.sh recipes/same_tokenizer/fkl_zero2.yaml
#   NUM_GPUS=4 bash scripts/train_multigpu_deepspeed.sh recipes/same_tokenizer/fkl_zero2.yaml \
#     --set train.batch_size=64 train.micro_batch_size=4
# ==============================================================================

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-$((29500 + RANDOM % 1000))}"

if [ $# -lt 1 ]; then
    echo "Usage: NUM_GPUS=<N> $0 <recipe_path> [extra_args...]"
    echo "Example: NUM_GPUS=2 $0 recipes/same_tokenizer/fkl_zero2.yaml"
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
echo "LM-KD Multi-GPU Training (DeepSpeed ZeRO)"
echo "GPUs:         $NUM_GPUS"
echo "Master Port:  $MASTER_PORT"
echo "Recipe:       $RECIPE"
echo "Python:       $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"

exec "$PYTHON" -m torch.distributed.run \
    --nproc_per_node="$NUM_GPUS" \
    --master_port="$MASTER_PORT" \
    -m lmkd.cli train -c "$RECIPE" "$@"

