#!/bin/bash
# ================================================================
# Build macOS .app bundle for Lecture Intelligence System
# Uses py2app to create a double-clickable application.
#
# Usage:
#   chmod +x build_app.sh
#   ./build_app.sh
#
# Output: dist/Lecture Intelligence.app
# ================================================================
set -euo pipefail
cd "$(dirname "$0")"

echo "=== Installing py2app ==="
pip install py2app

echo "=== Generating setup_app.py ==="
cat > setup_app.py << 'PYEOF'
from setuptools import setup

APP = ["app.py"]
OPTIONS = {
    "argv_emulation": False,
    "packages": [
        "pipeline", "modules", "gui",
        "pyside6", "yaml", "numpy", "torch",
    ],
    "plist": {
        "CFBundleName":                "Lecture Intelligence",
        "CFBundleDisplayName":         "Lecture Intelligence",
        "CFBundleIdentifier":          "com.lucaslab.lectureintel",
        "CFBundleVersion":             "1.0.0",
        "CFBundleShortVersionString":  "1.0",
        "NSMicrophoneUsageDescription": "Used for recording lectures",
        "NSHighResolutionCapable":     True,
        "LSEnvironment": {
            "PYTHONPATH": "$(pwd)",
        },
    },
}

setup(
    app=APP,
    name="Lecture Intelligence",
    options={"py2app": OPTIONS},
    setup_requires=["py2app"],
)
PYEOF

echo "=== Building .app bundle ==="
python setup_app.py py2app

echo ""
echo "=== Done ==="
echo "App bundle: dist/Lecture Intelligence.app"
echo "You can drag it to /Applications for double-click launch."
