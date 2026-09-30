#!/usr/bin/env bash
# Bootstrap a source-based Linux install. This is not a self-contained binary.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
ROOT="$(pwd)"
PYTHON="${PYTHON:-python3}"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python 3.10+ is required (set PYTHON=/path/to/python3)." >&2
  exit 1
fi
"$PYTHON" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "Python 3.10 or newer is required." >&2
  exit 1
}

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Warning: ffmpeg was not found. Install it with your Linux package manager for audio decoding." >&2
fi

NATIVE_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/Recorder/native"
mkdir -p "$NATIVE_DIR"
install -m 0755 native/SystemAudioRecorderLinux.py "$NATIVE_DIR/SystemAudioRecorderLinux.py"
if command -v pw-record >/dev/null 2>&1; then
  echo "Linux system-audio backend: pw-record"
elif command -v parec >/dev/null 2>&1; then
  echo "Linux system-audio backend: parec"
else
  echo "Warning: neither pw-record nor parec was found; system-audio capture is unavailable." >&2
  echo "Install pipewire-bin or pulseaudio-utils with your Linux package manager." >&2
fi

if ! "$PYTHON" -c 'import ensurepip, venv' >/dev/null 2>&1; then
  echo "Python's venv module is missing. On Debian/Ubuntu: sudo apt install python3-venv" >&2
  exit 1
fi
# A venv whose creation failed half-way has a python but no pip; rebuild it
# instead of failing on every later run.
if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  "$PYTHON" -m venv --clear .venv
fi
.venv/bin/python -m pip install --upgrade pip
# resemblyzer (speaker labels) runs torch on the CPU. PyPI's Linux torch wheel
# pulls in ~3 GB of CUDA libraries nothing here uses, so take the CPU build.
# Set RECORDER_TORCH_INDEX_URL= (empty) to keep PyPI's default wheel.
TORCH_INDEX_URL="${RECORDER_TORCH_INDEX_URL-https://download.pytorch.org/whl/cpu}"
if [[ -n "$TORCH_INDEX_URL" ]]; then
  .venv/bin/python -m pip install --index-url "$TORCH_INDEX_URL" "torch>=2.1.0" \
    || echo "Warning: CPU-only torch was unavailable from $TORCH_INDEX_URL; using PyPI's larger default build." >&2
fi
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python prune_qt_webengine.py || echo "Warning: could not remove unused Qt WebEngine files." >&2

MODEL_HINT=""
if [[ "${1:-}" == "--download-cpu-model" ]]; then
  # The app is usable without it (it downloads on first use), so a network
  # failure here must not leave the install without its launcher.
  if ! .venv/bin/python download_models.py --engine cpu small; then
    MODEL_HINT="The speech model could not be downloaded now; retry later with the command below."
  fi
fi

mkdir -p "$HOME/.local/share/applications"
"$PYTHON" - "$ROOT" "$HOME/.local/share/applications/recorder.desktop" <<'PY'
import sys
from pathlib import Path

root = Path(sys.argv[1])
desktop_file = Path(sys.argv[2])
def quote_exec(value: str) -> str:
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"') + '"'

def escape_string(value: str) -> str:
    return value.replace('\\', '\\\\').replace(' ', '\\s')

desktop_file.write_text(
    "[Desktop Entry]\n"
    "Type=Application\n"
    "Name=Recorder\n"
    "Comment=Local speech-to-text\n"
    f"Exec={quote_exec('/bin/bash')} {quote_exec(str(root / 'run_linux.sh'))}\n"
    f"Path={escape_string(str(root))}\n"
    "Terminal=false\n"
    "Categories=AudioVideo;Audio;Utility;\n",
    encoding="utf-8",
)
desktop_file.chmod(0o755)
PY
chmod +x run_linux.sh

echo "Recorder installed in: $ROOT"
echo "Launch from your desktop menu or run: $ROOT/run_linux.sh"
if [[ -n "$MODEL_HINT" ]]; then
  echo "Warning: $MODEL_HINT" >&2
fi
echo "Optional offline model download: $ROOT/.venv/bin/python $ROOT/download_models.py --engine cpu small"
echo "GPU setup instructions: $ROOT/docs/GPU_BACKENDS.md"
