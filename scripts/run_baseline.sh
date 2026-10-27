#!/usr/bin/env bash
# ==============================================================================
# LM-KD: Universal Baseline Reproduction Launcher
# ==============================================================================
# Runs any of the 90+ supported distillation baselines in LM-KD.
#
# Usage:
#   bash scripts/run_baseline.sh --list                       # List all available baselines
#   bash scripts/run_baseline.sh <name> [options...]          # Run or preview a baseline
#
# Common Options:
#   --student <path_or_repo>     Override student model
#   --teacher <path_or_repo>     Override teacher model
#   --reference <path_or_repo>   Override reference model (for DPO/PO)
#   --data <path.jsonl>          Override training data path
#   --dev-data <path.jsonl>      Override validation data path
#   --output-dir <path>          Output directory for logs and checkpoints
#   --dry-run                    Generate resolved YAML and plan without launching training
#   --check                      Perform static validation check on baseline recipes
#
# Examples:
#   bash scripts/run_baseline.sh --list
#   bash scripts/run_baseline.sh fkl --dry-run
#   bash scripts/run_baseline.sh fkl \
#       --student Qwen/Qwen3-0.6B \
#       --teacher Qwen/Qwen3-4B-Instruct-2507 \
#       --data data/train.jsonl \
#       --output-dir outputs/fkl_qwen
# ==============================================================================

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <baseline_name|--list|--all --check> [options...]"
    echo "Run '$0 --list' to see all 90+ supported baselines."
    exit 1
fi

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.reproduce "$@"

