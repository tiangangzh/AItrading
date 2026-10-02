#!/usr/bin/env bash
# One-shot setup + demo for macOS / Linux.
#   ./scripts/run_demo.sh            -> offline demo on the built-in synthetic market (no keys needed)
#   ./scripts/run_demo.sh --free     -> also install the free real-data extras (Yahoo + SEC EDGAR)
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
if [ "${1:-}" = "--free" ]; then
  python -m pip install -e ".[free]"
else
  python -m pip install -e .
fi
aitrading demo
