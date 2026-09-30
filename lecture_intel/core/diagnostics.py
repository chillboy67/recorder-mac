"""
Local logs, crash traces, and the diagnostics bundle a user can attach to an issue.

Nothing here sends anything anywhere. The app and the pipeline subprocess each
log to their own rotating file in the platform log folder (``app.log``,
``worker.log``: two processes rotating one file would race), and ``faulthandler`` writes
the Python stack of a native crash (segfault, abort) to a crash file — the
case where the child dies without reporting and the GUI only sees an exit code.

``export_bundle`` zips those logs with a system summary and, when given the
last run's output folder, a *redacted* copy of its meta.json. The bundle never
contains audio or transcript text: meta.json's annotations carry the speaker's
original words, so every string outside a short allowlist of technical fields
(step names, verdicts, model names, hashes…) is replaced by its length, and
the user's home folder is replaced with ``~`` everywhere.
"""
from __future__ import annotations

import faulthandler
import json
import logging
import logging.handlers
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Optional

from core import APP_VERSION

_CRASH_PREFIX = "crash-"
_crash_files: dict[str, Any] = {}   # kept open for faulthandler's lifetime

# The packages whose versions decide how a transcription behaves.
_PACKAGES = ("mlx-whisper", "mlx", "faster-whisper", "ctranslate2", "torch",
             "resemblyzer", "numpy", "librosa", "soundfile", "PySide6", "httpx")

# meta.json fields whose string values are technical, never the speaker's words.
_META_KEEP = frozenset({
    "name", "type", "verdict", "method", "requested_model", "model", "engine",
    "backend", "language", "filter", "format", "sha256", "input_sha256",
    "output_sha256", "timestamp", "speaker_id", "mode", "oracle",
})


def log_dir() -> Path:
    """Platform log folder (``RECORDER_LOG_DIR`` overrides it)."""
    override = os.environ.get("RECORDER_LOG_DIR")
    if override:
        return Path(override).expanduser()
    home = Path.home()
    system = platform.system()
    if system == "Darwin":
        return home / "Library" / "Logs" / "Recorder"
    if system == "Windows":
        local = os.environ.get("LOCALAPPDATA")
        return (Path(local) if local else home / "AppData" / "Local") / "Recorder" / "Logs"
    state = os.environ.get("XDG_STATE_HOME")
    return (Path(state).expanduser() if state else home / ".local" / "state") / "recorder" / "logs"


def setup_logging(process: str) -> Path:
    """Log this process to ``<process>.log`` and trace native crashes.

    ``process`` ("app" / "worker") names both files. Safe to call more than
    once, and in a forked child: a handler inherited from the parent is
    swapped for this process's own. Returns the log folder."""
    folder = log_dir()
    folder.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    target = str(folder / f"{process}.log")
    for h in [h for h in root.handlers if getattr(h, "_recorder", False)]:
        if h.baseFilename == target:
            break
        root.removeHandler(h)
    else:
        handler = logging.handlers.RotatingFileHandler(
            target, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(process)d] %(levelname)s %(name)s: %(message)s"))
        handler._recorder = True
        root.addHandler(handler)
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)
    crash_path = str(folder / f"{_CRASH_PREFIX}{process}.log")
    if crash_path not in _crash_files:
        crash = open(crash_path, "a", encoding="utf-8")
        _crash_files[crash_path] = crash
        faulthandler.enable(file=crash, all_threads=True)
    return folder


def log_uncaught(exc_type, exc, tb) -> None:
    """``sys.excepthook``: keep the traceback in the log, then behave as usual."""
    logging.getLogger("recorder").critical("Uncaught exception", exc_info=(exc_type, exc, tb))
    sys.__excepthook__(exc_type, exc, tb)


def _redact_home(text: str) -> str:
    home = str(Path.home())
    return text.replace(home, "~") if home not in ("", "/") else text


def redact_meta(value: Any, key: Optional[str] = None) -> Any:
    """meta.json with every non-technical string replaced by its length."""
    if isinstance(value, dict):
        return {k: redact_meta(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_meta(v, key) for v in value]
    if isinstance(value, str):
        return _redact_home(value) if key in _META_KEEP else f"<{len(value)} chars>"
    return value


def _ffmpeg_version() -> Optional[str]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-version"], capture_output=True,
                             text=True, timeout=10).stdout
        return f"{out.splitlines()[0]} ({ffmpeg})" if out else ffmpeg
    except Exception as exc:
        return f"{ffmpeg}: {exc}"


def system_summary() -> dict:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    from core.i18n import current_language
    summary = {
        "app_version": APP_VERSION,
        "created": datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "ui_language": current_language(),
        "ffmpeg": _ffmpeg_version(),
        "packages": versions,
    }
    return json.loads(_redact_home(json.dumps(summary, ensure_ascii=False)))


def default_bundle_name() -> str:
    return f"Recorder-diagnostics-{datetime.now():%Y%m%d-%H%M%S}.zip"


def _extra_logs() -> list[Path]:
    """The macOS launcher's own log (first-launch installs, launch failures)."""
    launcher = Path.home() / "Library" / "Logs" / "Recorder.log"
    return [launcher] if launcher.is_file() else []


def export_bundle(dest: str | Path, output_dir: str | Path | None = None) -> Path:
    """Write the diagnostics zip to ``dest``; see the module docstring."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    folder = log_dir()
    logs = sorted(p for p in folder.glob("*.log*") if p.is_file())
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("system.json", json.dumps(system_summary(), ensure_ascii=False, indent=2))
        for path in logs + _extra_logs():
            name = "logs/launcher.log" if path in _extra_logs() else f"logs/{path.name}"
            text = path.read_text(encoding="utf-8", errors="replace")
            zf.writestr(name, _redact_home(text))
        meta = Path(output_dir) / "meta.json" if output_dir else None
        if meta is not None and meta.is_file():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                zf.writestr("meta.redacted.json",
                            json.dumps(redact_meta(data), ensure_ascii=False, indent=2))
            except (OSError, ValueError) as exc:
                zf.writestr("meta.redacted.json", json.dumps({"error": str(exc)}))
    return dest

