#!/usr/bin/env bash
# Create a virtual environment and install AItrading with the free real-data extras (macOS / Linux).
#   ./scripts/install.sh
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python3}
[ -d .venv ] || "$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -e ".[free]"

cat <<'MSG'

Installed. Next:
  source .venv/bin/activate
  export SEC_USER_AGENT="Your Name you@example.com"
  export ANTHROPIC_API_KEY="sk-ant-..."        # optional: enables Claude reasoning
  aitrading run "your investment observation"
See docs/QUICKSTART_PC.md for details.
MSG
