#!/bin/bash
# =============================================================================
# Build a double-clickable Recorder.app for macOS.
#
# Why not py2app / PyInstaller? They try to freeze torch + mlx + whisper into
# the bundle — gigabytes, slow, and breaks on every dependency bump. Instead we
# build a tiny launcher .app that runs the project's existing virtualenv. You
# get the double-click experience, and `pip install -U ...` keeps working.
#
# Usage:   ./make_app.sh
# Output:  ./dist/Recorder.app   (drag it to /Applications)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PROJECT_DIR="$(pwd)"
VENV_PY="$PROJECT_DIR/.venv/bin/python3"
APP="dist/Recorder.app"

if [[ ! -x "$VENV_PY" ]]; then
  echo "✗ Virtualenv not found at $VENV_PY"
  echo "  Create it first:  uv venv && uv pip install -r requirements.txt"
  exit 1
fi

echo "=== Building $APP ==="
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

# --- Info.plist -------------------------------------------------------------
cat > "$APP/Contents/Info.plist" << PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>            <string>Recorder</string>
  <key>CFBundleDisplayName</key>     <string>Recorder</string>
  <key>CFBundleIdentifier</key>      <string>com.lucaslab.recorder</string>
  <key>CFBundleVersion</key>         <string>2.0.0</string>
  <key>CFBundleShortVersionString</key><string>2.0</string>
  <key>CFBundlePackageType</key>     <string>APPL</string>
  <key>CFBundleExecutable</key>      <string>Recorder</string>
  <key>CFBundleIconFile</key>        <string>icon.icns</string>
  <key>NSMicrophoneUsageDescription</key>
  <string>Recorder 需要使用麦克风进行录音转写。</string>
  <key>NSHighResolutionCapable</key> <true/>
  <key>LSMinimumSystemVersion</key>  <string>12.0</string>
</dict>
</plist>
PLIST

# --- Launcher executable ----------------------------------------------------
cat > "$APP/Contents/MacOS/Recorder" << LAUNCHER
#!/bin/bash
# Auto-generated launcher. Runs the Recorder GUI from the project venv.
PROJECT_DIR="$PROJECT_DIR"
LOG="\$HOME/Library/Logs/Recorder.log"
mkdir -p "\$(dirname "\$LOG")"
cd "\$PROJECT_DIR"
# Make Homebrew tools (ffmpeg) reachable when launched from Finder.
export PATH="/opt/homebrew/bin:/usr/local/bin:\$PATH"
exec "$VENV_PY" "\$PROJECT_DIR/app.py" >> "\$LOG" 2>&1
LAUNCHER
chmod +x "$APP/Contents/MacOS/Recorder"

# --- Icon (optional) --------------------------------------------------------
if [[ -f "Resources/icon.icns" ]]; then
  cp "Resources/icon.icns" "$APP/Contents/Resources/icon.icns"
fi

echo ""
echo "✓ Built $APP"
echo ""
echo "Double-click dist/Recorder.app to launch, or move it to /Applications:"
echo "    mv \"$PROJECT_DIR/$APP\" /Applications/"
echo ""
echo "First launch: right-click → Open (Gatekeeper), then grant microphone access."
echo "Logs: ~/Library/Logs/Recorder.log"
