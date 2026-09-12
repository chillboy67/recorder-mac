#!/usr/bin/env python3
"""
Recorder — Desktop App Entry Point

A local, offline speech-to-text app with three modes (general / classroom /
IELTS coach). Double-click Recorder.app, or run:

    python app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable (core.*, modules.*, gui.*).
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _check_dependencies() -> list[str]:
    """Return a list of missing critical dependencies (empty = all good)."""
    from core.i18n import t
    missing = []
    for mod, hint in [
        ("PySide6", "pip install pyside6"),
        ("faster_whisper", "pip install faster-whisper"),
        ("numpy", "pip install numpy"),
    ]:
        try:
            __import__(mod)
        except Exception:
            missing.append(f"{mod}  →  {hint}")
    # at least one whisper engine must exist
    has_mlx = _can_import("mlx_whisper")
    has_faster = _can_import("faster_whisper")
    if not (has_mlx or has_faster):
        missing.append(t("deps_whisper"))
    return missing


def _can_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def main() -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication, QMessageBox

    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("Recorder")
    app.setOrganizationName("LucasLab")
    app.setApplicationVersion("2.0.0")

    from PySide6.QtCore import QSettings
    from core.i18n import detect_system_language, set_language, t
    from gui.theme import apply as apply_theme
    prefs = QSettings("LucasLab", "Recorder")
    # Apply the UI language before anything is drawn (or any dialog is shown),
    # defaulting to the system locale on first launch.
    lang_pref = prefs.value("ui_language")
    set_language(lang_pref if lang_pref else detect_system_language())
    mode = prefs.value("appearance", "auto")
    apply_theme(app, mode if mode in ("auto", "light", "dark") else "auto")

    missing = _check_dependencies()
    if missing:
        QMessageBox.critical(
            None, t("deps_title"), t("deps_body", missing="\n".join(missing)),
        )
        sys.exit(1)

    # Trigger macOS microphone permission prompt on first launch.
    import platform
    if platform.system() == "Darwin":
        try:
            from PySide6.QtMultimedia import QMediaDevices
            _ = QMediaDevices.audioInputs()
        except Exception:
            pass

    from gui.main_window import MainWindow
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
