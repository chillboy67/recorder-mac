"""
Drive the native ScreenCaptureKit helper to record the Mac's own audio output.

The compiled Swift binary (native/system_audio_recorder) does the actual capture;
this module just locates it, starts it on a temp WAV, and stops it (SIGTERM →
the helper finalizes the file). Requires the "Screen Recording" permission the
first time.
"""
from __future__ import annotations

import signal
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from core.i18n import t


def binary_path() -> Optional[Path]:
    """Locate the compiled helper (installed location first, then dev tree)."""
    candidates = [
        Path.home() / "Library" / "Application Support" / "Recorder" / "native" / "system_audio_recorder",
        Path(__file__).resolve().parent.parent / "native" / "system_audio_recorder",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_mode & 0o111:
            return p
    return None


def available() -> bool:
    return binary_path() is not None


class SystemAudioRecorder:
    """Start/stop a system-audio capture to a temp .wav."""

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen] = None
        self.output_path: Optional[str] = None
        self._err_path: Optional[str] = None
        self._paused = False

    def start(self) -> str:
        b = binary_path()
        if b is None:
            raise RuntimeError(t("sys_component_missing"))
        out = tempfile.NamedTemporaryFile(suffix=".wav", prefix="recorder_sys_", delete=False)
        out.close()
        self.output_path = out.name
        errf = tempfile.NamedTemporaryFile(suffix=".log", prefix="recorder_sys_", delete=False)
        self._err_path = errf.name
        self._proc = subprocess.Popen([str(b), self.output_path],
                                      stderr=errf, stdout=subprocess.DEVNULL)
        self._paused = False
        return self.output_path

    @staticmethod
    def can_pause() -> bool:
        """True if the compiled helper handles SIGUSR1/SIGUSR2.

        Older builds don't install handlers, and the default disposition of
        SIGUSR1 would kill the helper mid-recording — so probe the binary for
        the "PAUSED" log marker that only pause-capable builds contain.
        """
        b = binary_path()
        if b is None:
            return False
        try:
            return b"PAUSED" in b.read_bytes()
        except Exception:
            return False

    def pause(self) -> None:
        """Pause capture — the helper drops samples until resume()."""
        if self.is_running and not self._paused and self.can_pause():
            self._proc.send_signal(signal.SIGUSR1)
            self._paused = True

    def resume(self) -> None:
        if self.is_running and self._paused:
            self._proc.send_signal(signal.SIGUSR2)
            self._paused = False

    def stop(self) -> Optional[str]:
        """Stop capture; returns the wav path if it has audio, else None."""
        if self._proc is not None:
            try:
                self._proc.send_signal(signal.SIGTERM)
                self._proc.wait(timeout=8)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None
        path = self.output_path
        if path and Path(path).exists() and Path(path).stat().st_size > 1024:
            return path
        return None

    def error_text(self) -> str:
        """Read the helper's stderr (e.g. permission errors) for diagnostics."""
        if self._err_path and Path(self._err_path).exists():
            try:
                return Path(self._err_path).read_text(encoding="utf-8", errors="ignore")
            except Exception:
                return ""
        return ""

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None
