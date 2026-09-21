"""
System-audio capture — locating and driving the native ScreenCaptureKit helper.

The helper is a compiled Swift binary, so these tests exercise the Python side
against a stand-in: which files it will accept as "the helper", how it guards
pause against older builds that would die on SIGUSR1, and how it decides a
capture produced usable audio.
"""
from __future__ import annotations

import signal
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import sysaudio as S  # noqa: E402


@pytest.fixture
def no_helper(tmp_path, monkeypatch):
    """Neither candidate location exists: HOME is redirected and the dev-tree
    path is derived from a fake __file__."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(S, "__file__", str(tmp_path / "fake" / "core" / "sysaudio.py"))
    return tmp_path


def install_helper(root: Path, content: bytes = b"binary", mode: int = 0o755) -> Path:
    p = root / "fake" / "native" / "system_audio_recorder"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    p.chmod(mode)
    return p


def fake_proc(alive: bool = True):
    class FakeProc:
        def __init__(self):
            self.signals = []
            self._alive = alive

        def send_signal(self, sig):
            self.signals.append(sig)

        def wait(self, timeout=None):
            self._alive = False
            return 0

        def poll(self):
            return None if self._alive else 0

        def kill(self):
            self._alive = False

    return FakeProc()


# ── locating the helper ─────────────────────────────────────────────

def test_no_helper_means_unavailable(no_helper):
    assert S.binary_path() is None
    assert S.available() is False


def test_an_executable_helper_is_accepted(no_helper):
    p = install_helper(no_helper)
    assert S.binary_path() == p
    assert S.available() is True


def test_a_non_executable_file_is_not_the_helper(no_helper):
    """The 0o111 check keeps a stray data file from being launched."""
    install_helper(no_helper, mode=0o644)
    assert S.binary_path() is None


# ── pause capability probe ──────────────────────────────────────────

def test_can_pause_is_false_without_a_helper(no_helper):
    assert S.SystemAudioRecorder.can_pause() is False


def test_can_pause_detects_the_pause_marker(no_helper):
    install_helper(no_helper, content=b"...PAUSED...")
    assert S.SystemAudioRecorder.can_pause() is True


def test_can_pause_is_false_for_older_builds(no_helper):
    """Without the marker, SIGUSR1's default disposition would kill the helper
    mid-recording, so pause must stay hidden."""
    install_helper(no_helper, content=b"older build without the marker")
    assert S.SystemAudioRecorder.can_pause() is False


# ── start / is_running ──────────────────────────────────────────────

def test_start_without_the_helper_raises(no_helper):
    with pytest.raises(RuntimeError):
        S.SystemAudioRecorder().start()


def test_start_launches_the_helper_on_a_temp_wav(no_helper, monkeypatch, tmp_path):
    binary = install_helper(no_helper, content=b"PAUSED")
    monkeypatch.setattr(S.tempfile, "tempdir", str(tmp_path))
    seen = {}

    def fake_popen(argv, **kw):
        seen["argv"] = argv
        return fake_proc()

    monkeypatch.setattr(S.subprocess, "Popen", fake_popen)

    rec = S.SystemAudioRecorder()
    out = rec.start()

    try:
        assert out.endswith(".wav") and Path(out).exists()
        assert seen["argv"][0] == str(binary)
        assert seen["argv"][1] == out
        assert rec.is_running is True
    finally:
        rec._proc = None            # don't let the temp file linger as "running"


def test_is_running_is_false_before_start():
    assert S.SystemAudioRecorder().is_running is False


# ── pause / resume ──────────────────────────────────────────────────

def test_pause_and_resume_send_usr_signals(no_helper, monkeypatch):
    install_helper(no_helper, content=b"PAUSED")
    rec = S.SystemAudioRecorder()
    rec._proc = fake_proc()

    rec.pause()
    assert rec._proc.signals == [signal.SIGUSR1]
    rec.resume()
    assert rec._proc.signals == [signal.SIGUSR1, signal.SIGUSR2]


def test_pause_is_ignored_when_the_helper_cannot_pause(no_helper):
    install_helper(no_helper, content=b"no marker here")
    rec = S.SystemAudioRecorder()
    rec._proc = fake_proc()
    rec.pause()
    assert rec._proc.signals == []          # never signal a helper that would die


def test_pause_and_resume_do_nothing_when_not_running(no_helper):
    install_helper(no_helper, content=b"PAUSED")
    rec = S.SystemAudioRecorder()
    rec.pause()                             # _proc is None
    rec.resume()
    assert rec._paused is False


# ── stop ────────────────────────────────────────────────────────────

def test_stop_returns_none_without_a_recording():
    assert S.SystemAudioRecorder().stop() is None


def test_stop_returns_the_wav_once_it_holds_audio(tmp_path):
    wav = tmp_path / "capture.wav"
    wav.write_bytes(b"\0" * 4096)
    rec = S.SystemAudioRecorder()
    rec.output_path = str(wav)
    assert rec.stop() == str(wav)


def test_stop_discards_a_header_only_file(tmp_path):
    """≤1024 bytes is a WAV header with nothing behind it."""
    wav = tmp_path / "empty.wav"
    wav.write_bytes(b"\0" * 1024)
    rec = S.SystemAudioRecorder()
    rec.output_path = str(wav)
    assert rec.stop() is None


def test_stop_terminates_the_helper(tmp_path):
    wav = tmp_path / "capture.wav"
    wav.write_bytes(b"\0" * 4096)
    rec = S.SystemAudioRecorder()
    rec.output_path = str(wav)
    proc = fake_proc()
    rec._proc = proc
    rec.stop()
    assert proc.signals == [signal.SIGTERM]
    assert rec._proc is None
    assert rec.is_running is False


# ── error_text ──────────────────────────────────────────────────────

def test_error_text_reads_the_helper_stderr(tmp_path):
    log = tmp_path / "helper.log"
    log.write_text("Screen Recording permission denied", encoding="utf-8")
    rec = S.SystemAudioRecorder()
    rec._err_path = str(log)
    assert "permission denied" in rec.error_text()


def test_error_text_is_empty_when_there_is_nothing_to_read():
    rec = S.SystemAudioRecorder()
    assert rec.error_text() == ""
    rec._err_path = "/nonexistent/helper.log"
    assert rec.error_text() == ""
