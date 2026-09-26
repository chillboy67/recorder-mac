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

if [[ ! -x .venv/bin/python ]]; then
  "$PYTHON" -m venv .venv
fi
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [[ "${1:-}" == "--download-cpu-model" ]]; then
  .venv/bin/python download_models.py --engine cpu small
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
echo "Optional offline model download: $ROOT/.venv/bin/python $ROOT/download_models.py --engine cpu small"
echo "GPU setup instructions: $ROOT/docs/GPU_BACKENDS.md"
