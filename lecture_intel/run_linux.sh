#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Recorder is not installed. Run ./install_linux.sh first." >&2
  exit 1
fi
exec .venv/bin/python app.py
