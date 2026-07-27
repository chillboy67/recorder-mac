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
        missing.append("mlx-whisper 或 faster-whisper（二者至少装一个）")
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
    from gui.theme import apply as apply_theme
    mode = QSettings("LucasLab", "Recorder").value("appearance", "auto")
    apply_theme(app, mode if mode in ("auto", "light", "dark") else "auto")

    missing = _check_dependencies()
    if missing:
        QMessageBox.critical(
            None, "缺少依赖",
            "无法启动，缺少以下依赖：\n\n" + "\n".join(missing) +
            "\n\n请在项目目录运行：\n  uv pip install -r requirements.txt",
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
