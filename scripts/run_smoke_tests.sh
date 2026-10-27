#!/usr/bin/env bash
# ==============================================================================
# LM-KD: Offline Smoke Tests (All 6 Baseline Domains)
# ==============================================================================
# Runs synthetic tiny-model training across FKL, ULD, SGL, GRPO, DPO, and TALAS.
# Completely self-contained: requires 0 GPU memory and 0 external downloads.
# Finishes in ~20-30 seconds on CPU.
#
# Usage:
#   bash scripts/run_smoke_tests.sh [--dry-run]
# ==============================================================================

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

OUTPUT_DIR="/tmp/lmkd-smoke-test-$(date +%s)"

echo "======================================================================"
echo "LM-KD Offline Smoke Test Suite"
echo "Output Directory: $OUTPUT_DIR"
echo "Python:           $PYTHON"
echo "======================================================================"

cd "$REPO_DIR"
"$PYTHON" examples/reproduce_tiny.py --output-dir "$OUTPUT_DIR" "$@"

echo "======================================================================"
echo "Smoke test passed successfully! Artifacts written to $OUTPUT_DIR"
echo "======================================================================"

