"""
Single source of truth for where Recorder writes user data.

Everything the user produces — saved recordings and transcription outputs —
lives under ~/Documents/recorder/Recorder. Change DATA_ROOT here and the whole
app follows.

Note: ~/Documents is TCC-protected, so the first time a Finder-launched app
writes here macOS shows a one-time "allow access to Documents" prompt. That's
expected; once granted it persists. (The app bundle itself and the model cache
stay in ~/Library/Application Support/Recorder, which needs no prompt.)
"""
from __future__ import annotations

from pathlib import Path


def data_root() -> Path:
    """Root folder for all user-facing recordings and outputs."""
    root = Path.home() / "Documents" / "recorder" / "Recorder"
    root.mkdir(parents=True, exist_ok=True)
    return root


def record_dir() -> Path:
    """Folder for saved recordings."""
    d = data_root() / "record"
    d.mkdir(parents=True, exist_ok=True)
    return d
