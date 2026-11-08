#!/usr/bin/env bash
# ==============================================================================
# LM-KD: Baseline Reproduction CLI Shorthand
# ==============================================================================
# Equivalent to: python -m lmkd.reproduce "$@"
#
# Examples:
#   bash scripts/reproduce.sh --list
#   bash scripts/reproduce.sh --all --check
#   bash scripts/reproduce.sh fkl --student /models/student --teacher /models/teacher
# ==============================================================================

set -euo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${LMKD_PYTHON:-$REPO_DIR/.venv/bin/python}"

if [ ! -x "$PYTHON" ]; then
    PYTHON="$(command -v python3)"
fi

cd "$REPO_DIR"
exec "$PYTHON" -m lmkd.reproduce "$@"
