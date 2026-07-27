#!/bin/bash
# =============================================================================
# Build + install a double-clickable Recorder.app for macOS.
#
# WHY AN INSTALLER (not just a launcher pointing at this folder)?
#   macOS TCC blocks Finder-launched apps from reading ~/Documents, ~/Desktop,
#   ~/Downloads. This project lives under ~/Documents, so an app that runs code
#   from here crashes at startup with a PermissionError. The fix is to run the
#   app from a NON-protected location. We install a runnable copy + the venv to
#   ~/Library/Application Support/Recorder, and build the .app to launch from
#   there — no Full Disk Access prompts, true double-click.
#
#   Dev editing still happens in this folder; re-run this script to re-sync.
#
# WHY NOT py2app / PyInstaller? They try to freeze torch + mlx into the bundle
# (gigabytes, fragile). The launcher approach keeps `pip install -U` working.
#
# Usage:   ./make_app.sh
# Output:  /Applications/Recorder.app  (and ~/Library/Application Support/Recorder)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

SRC_DIR="$(pwd)"
APP_HOME="$HOME/Library/Application Support/Recorder"
APP="/Applications/Recorder.app"

echo "=== Recorder installer ==="
echo "source : $SRC_DIR"
echo "install: $APP_HOME"

mkdir -p "$APP_HOME"

# --- 1) Copy the code (small) to the non-protected install location ----------
echo "→ syncing code…"
rsync -a --delete \
  --exclude '.venv' --exclude 'dist' --exclude '__pycache__' \
  --exclude '*.pyc' --exclude 'models' --exclude 'output' \
  core gui modules app.py transcribe.py download_models.py requirements.txt README.md \
  "$APP_HOME/" 2>/dev/null || {
    # fall back to cp if some optional paths don't exist
    for item in core gui modules app.py transcribe.py download_models.py requirements.txt; do
      [[ -e "$item" ]] && cp -R "$item" "$APP_HOME/"
    done
  }

# --- 2) Ensure a venv exists at the install location -------------------------
if [[ -x "$APP_HOME/.venv/bin/python3" ]]; then
  echo "→ reusing existing install venv"
elif [[ -x "$SRC_DIR/.venv/bin/python3" && ! -L "$SRC_DIR/.venv" ]]; then
  # Move the dev venv out of the protected folder (instant same-volume rename),
  # leaving a symlink behind so terminal/dev use keeps working.
  echo "→ relocating venv to install location (instant rename)…"
  mv "$SRC_DIR/.venv" "$APP_HOME/.venv"
  ln -s "$APP_HOME/.venv" "$SRC_DIR/.venv"
else
  echo "→ creating fresh venv (downloads dependencies, one time)…"
  if command -v uv >/dev/null 2>&1; then
    uv venv "$APP_HOME/.venv"
    (cd "$APP_HOME" && uv pip install --python "$APP_HOME/.venv/bin/python3" -r requirements.txt)
  else
    python3 -m venv "$APP_HOME/.venv"
    "$APP_HOME/.venv/bin/python3" -m pip install -r "$APP_HOME/requirements.txt"
  fi
fi

VENV_PY="$APP_HOME/.venv/bin/python3"

# --- 3) Build the .app launcher ---------------------------------------------
echo "→ building $APP …"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

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
  <key>NSMicrophoneUsageDescription</key>
  <string>Recorder 需要使用麦克风进行录音转写。</string>
  <key>NSHighResolutionCapable</key> <true/>
  <key>LSMinimumSystemVersion</key>  <string>12.0</string>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/Recorder" << LAUNCHER
#!/bin/bash
# Auto-generated launcher → runs Recorder from the non-protected install dir.
APP_HOME="$APP_HOME"
LOG="\$HOME/Library/Logs/Recorder.log"
mkdir -p "\$(dirname "\$LOG")"
cd "\$APP_HOME"
export PATH="/opt/homebrew/bin:/usr/local/bin:\$PATH"   # find ffmpeg from Finder
exec "$VENV_PY" "\$APP_HOME/app.py" >> "\$LOG" 2>&1
LAUNCHER
chmod +x "$APP/Contents/MacOS/Recorder"

[[ -f "Resources/icon.icns" ]] && cp "Resources/icon.icns" "$APP/Contents/Resources/icon.icns"

echo ""
echo "✓ Installed: $APP"
echo "  Code + venv: $APP_HOME"
echo ""
echo "Double-click Recorder in /Applications (or Launchpad)."
echo "First launch: right-click → Open to clear Gatekeeper, then allow the mic."
echo "Logs: ~/Library/Logs/Recorder.log"
