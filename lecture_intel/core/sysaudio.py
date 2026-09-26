"""Locate and drive the platform system-audio capture helper.

Each helper writes a PCM WAV file and implements a small line-based control
protocol on stdin: ``PAUSE``, ``RESUME`` and ``STOP``.  New macOS, Windows and
Linux helpers advertise that protocol with the embedded ``STDIN_CONTROL_V1``
marker.  Older macOS helpers remain supported through their SIGUSR1/SIGUSR2
control path until they are rebuilt.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import BinaryIO, Optional

from core.i18n import t

_PROTOCOL_MARKER = b"STDIN_CONTROL_V1"


def _platform_key() -> str:
    if sys.platform == "darwin":
        return "darwin"
    if sys.platform == "win32":
        return "win32"
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def _helper_name(platform_key: str) -> str:
    if platform_key == "win32":
        return "system_audio_recorder.exe"
    if platform_key == "linux":
        return "SystemAudioRecorderLinux.py"
    return "system_audio_recorder"


def _installed_native_dir(platform_key: str) -> Path:
    """Where the platform installer drops the helper it built."""
    if platform_key == "darwin":
        return Path.home() / "Library" / "Application Support" / "Recorder" / "native"
    if platform_key == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return base / "Recorder" / "native"
    base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "Recorder" / "native"


def _helper_candidates(platform_key: str, name: str) -> list[Path]:
    """Installed copy first (the app may run from a data dir), then the source
    tree so a dev checkout works without re-running the platform installer."""
    tree = Path(__file__).resolve().parent.parent / "native" / name
    return [_installed_native_dir(platform_key) / name, tree]


def _is_launchable(path: Path, platform_key: str) -> bool:
    if not path.is_file():
        return False
    if platform_key == "win32":
        return path.suffix.lower() == ".exe"
    if platform_key == "linux" and path.suffix.lower() == ".py":
        return True
    return os.access(path, os.X_OK)


def binary_path() -> Optional[Path]:
    """Locate the helper for the current platform."""
    platform_key = _platform_key()
    name = _helper_name(platform_key)
    for path in _helper_candidates(platform_key, name):
        if _is_launchable(path, platform_key):
            return path
    return None


def _linux_backend_available() -> bool:
    return shutil.which("pw-record") is not None or shutil.which("parec") is not None


def available() -> bool:
    helper = binary_path()
    if helper is None:
        return False
    return _platform_key() != "linux" or _linux_backend_available()


def _control_mode(helper: Path) -> Optional[str]:
    """Return ``stdin``, legacy macOS ``signal``, or None."""
    try:
        payload = helper.read_bytes()
    except OSError:
        return None
    if _PROTOCOL_MARKER in payload:
        return "stdin"
    if (_platform_key() == "darwin" and b"PAUSED" in payload
            and hasattr(signal, "SIGUSR1") and hasattr(signal, "SIGUSR2")):
        return "signal"
    return None


def _helper_command(helper: Path, output_path: str) -> list[str]:
    if _platform_key() == "linux" and helper.suffix.lower() == ".py":
        return [sys.executable, str(helper), output_path]
    return [str(helper), output_path]


class SystemAudioRecorder:
    """Start, pause, resume and stop system-audio capture to a temp WAV."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen[str]] = None
        self.output_path: Optional[str] = None
        self._err_path: Optional[str] = None
        self._err_handle: Optional[BinaryIO] = None
        self._paused = False
        self._mode: Optional[str] = None

    def start(self) -> str:
        if self.is_running:
            raise RuntimeError("system-audio capture is already running")
        helper = binary_path()
        if helper is None or (_platform_key() == "linux" and not _linux_backend_available()):
            raise RuntimeError(t("sys_component_missing"))

        out = tempfile.NamedTemporaryFile(
            suffix=".wav", prefix="recorder_sys_", delete=False)
        out.close()
        self.output_path = out.name

        errf = tempfile.NamedTemporaryFile(
            suffix=".log", prefix="recorder_sys_", delete=False)
        self._err_path = errf.name
        errf.close()
        self._err_handle = open(self._err_path, "wb", buffering=0)
        self._mode = _control_mode(helper)

        kwargs = {
            "stdin": subprocess.PIPE if self._mode == "stdin" else subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": self._err_handle,
            "text": True,
            "encoding": "utf-8",
            "bufsize": 1,
        }
        if _platform_key() == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW"):
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

        try:
            self._proc = subprocess.Popen(_helper_command(helper, self.output_path), **kwargs)
        except Exception:
            self._close_error_handle()
            raise
        self._paused = False
        return self.output_path

    @staticmethod
    def can_pause() -> bool:
        helper = binary_path()
        return helper is not None and _control_mode(helper) is not None

    def _write_command(self, command: str) -> bool:
        proc = self._proc
        if proc is None or proc.poll() is not None or proc.stdin is None:
            return False
        try:
            proc.stdin.write(command + "\n")
            proc.stdin.flush()
            return True
        except (BrokenPipeError, OSError, ValueError):
            return False

    def pause(self) -> None:
        """Pause capture; helpers keep draining audio but drop captured samples."""
        if not self.is_running or self._paused:
            return
        if self._mode == "stdin":
            changed = self._write_command("PAUSE")
        elif self._mode == "signal" and hasattr(signal, "SIGUSR1"):
            self._proc.send_signal(signal.SIGUSR1)
            changed = True
        else:
            changed = False
        if changed:
            self._paused = True

    def resume(self) -> None:
        if not self.is_running or not self._paused:
            return
        if self._mode == "stdin":
            changed = self._write_command("RESUME")
        elif self._mode == "signal" and hasattr(signal, "SIGUSR2"):
            self._proc.send_signal(signal.SIGUSR2)
            changed = True
        else:
            changed = False
        if changed:
            self._paused = False

    def stop(self) -> Optional[str]:
        """Stop capture; return the WAV path when it contains usable audio."""
        proc = self._proc
        if proc is not None:
            try:
                if proc.poll() is None:
                    if self._mode == "stdin":
                        if not self._write_command("STOP"):
                            proc.terminate()
                    elif self._mode == "signal":
                        proc.send_signal(signal.SIGTERM)
                    else:
                        proc.terminate()
                    proc.wait(timeout=8)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
            finally:
                try:
                    if proc.stdin is not None:
                        proc.stdin.close()
                except Exception:
                    pass
                self._proc = None
                self._paused = False
                self._close_error_handle()

        path = self.output_path
        if path and Path(path).exists() and Path(path).stat().st_size > 1024:
            return path
        return None

    def _close_error_handle(self) -> None:
        if self._err_handle is not None:
            try:
                self._err_handle.close()
            except OSError:
                pass
            self._err_handle = None

    def error_text(self) -> str:
        """Read helper stderr (permissions, missing session/backend, device errors)."""
        if self._err_path and Path(self._err_path).exists():
            try:
                return Path(self._err_path).read_text(encoding="utf-8", errors="ignore")
            except OSError:
                return ""
        return ""

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None
