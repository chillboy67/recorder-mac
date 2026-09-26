"""Platform-neutral tests for the system-audio helper driver."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import sysaudio as S  # noqa: E402


class FakeStdin:
    def __init__(self) -> None:
        self.writes: list[str] = []
        self.flushes = 0
        self.closed = False

    def write(self, value: str) -> None:
        self.writes.append(value)

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed = True


class FakeProc:
    def __init__(self, alive: bool = True, with_stdin: bool = True) -> None:
        self.signals = []
        self._alive = alive
        self.stdin = FakeStdin() if with_stdin else None
        self.terminated = False
        self.killed = False

    def send_signal(self, sig) -> None:
        self.signals.append(sig)

    def wait(self, timeout=None) -> int:
        self._alive = False
        return 0

    def poll(self):
        return None if self._alive else 0

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False

    def kill(self) -> None:
        self.killed = True
        self._alive = False


@pytest.fixture
def no_helper(tmp_path, monkeypatch):
    """Use Windows path rules so helper discovery is deterministic on every OS."""
    monkeypatch.setattr(S, "_platform_key", lambda: "win32")
    monkeypatch.setattr(S, "__file__", str(tmp_path / "fake" / "core" / "sysaudio.py"))
    return tmp_path


def install_helper(root: Path, content: bytes = b"binary") -> Path:
    path = root / "fake" / "native" / "system_audio_recorder.exe"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


# ── helper selection ─────────────────────────────────────────────────

def test_no_helper_means_unavailable(no_helper):
    assert S.binary_path() is None
    assert S.available() is False


def test_platform_helper_is_accepted(no_helper):
    path = install_helper(no_helper)
    assert S.binary_path() == path
    assert S.available() is True


def test_wrong_platform_helper_name_is_ignored(no_helper):
    path = no_helper / "fake" / "native" / "system_audio_recorder"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"binary")
    assert S.binary_path() is None


def test_linux_availability_requires_capture_backend(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_platform_key", lambda: "linux")
    monkeypatch.setattr(S, "__file__", str(tmp_path / "fake" / "core" / "sysaudio.py"))
    helper = tmp_path / "fake" / "native" / "SystemAudioRecorderLinux.py"
    helper.parent.mkdir(parents=True)
    helper.write_text("STDIN_CONTROL_V1", encoding="utf-8")

    monkeypatch.setattr(S, "_linux_backend_available", lambda: False)
    assert S.binary_path() == helper
    assert S.available() is False
    monkeypatch.setattr(S, "_linux_backend_available", lambda: True)
    assert S.available() is True


def test_darwin_prefers_the_installed_copy_over_the_tree(tmp_path, monkeypatch):
    """The .app bundle runs from ~/Library/Application Support, so a rebuilt
    tree helper must not shadow the installed one."""
    monkeypatch.setattr(S, "_platform_key", lambda: "darwin")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(S, "__file__", str(tmp_path / "fake" / "core" / "sysaudio.py"))
    tree = tmp_path / "fake" / "native" / "system_audio_recorder"
    tree.parent.mkdir(parents=True)
    tree.write_bytes(b"binary")
    tree.chmod(0o755)
    installed = (tmp_path / "Library" / "Application Support" / "Recorder"
                 / "native" / "system_audio_recorder")
    installed.parent.mkdir(parents=True)
    installed.write_bytes(b"binary")
    installed.chmod(0o755)
    assert S.binary_path() == installed


def test_linux_helper_runs_with_current_python(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "_platform_key", lambda: "linux")
    helper = tmp_path / "SystemAudioRecorderLinux.py"
    helper.write_text("STDIN_CONTROL_V1", encoding="utf-8")
    assert S._helper_command(helper, "out.wav") == [
        sys.executable, str(helper), "out.wav"]


# ── capability and launch ────────────────────────────────────────────

def test_can_pause_requires_stdin_protocol_marker(no_helper):
    install_helper(no_helper, content=b"older helper with PAUSED only")
    assert S.SystemAudioRecorder.can_pause() is False
    install_helper(no_helper, content=b"...STDIN_CONTROL_V1...")
    assert S.SystemAudioRecorder.can_pause() is True


def test_start_without_helper_raises(no_helper):
    with pytest.raises(RuntimeError):
        S.SystemAudioRecorder().start()


def test_start_launches_helper_with_control_pipe(no_helper, monkeypatch, tmp_path):
    binary = install_helper(no_helper, content=b"STDIN_CONTROL_V1")
    monkeypatch.setattr(S.tempfile, "tempdir", str(tmp_path))
    seen = {}
    proc = FakeProc()

    def fake_popen(argv, **kwargs):
        seen["argv"] = argv
        seen["kwargs"] = kwargs
        return proc

    monkeypatch.setattr(S.subprocess, "Popen", fake_popen)

    rec = S.SystemAudioRecorder()
    out = rec.start()
    try:
        assert out.endswith(".wav") and Path(out).exists()
        assert seen["argv"] == [str(binary), out]
        assert seen["kwargs"]["stdin"] == S.subprocess.PIPE
        assert seen["kwargs"]["text"] is True
        assert rec.is_running is True
    finally:
        rec._proc = None
        rec._close_error_handle()


def test_is_running_is_false_before_start():
    assert S.SystemAudioRecorder().is_running is False


# ── stdin controls ───────────────────────────────────────────────────

def test_pause_and_resume_write_protocol_commands(no_helper):
    install_helper(no_helper, content=b"STDIN_CONTROL_V1")
    rec = S.SystemAudioRecorder()
    rec._proc = FakeProc()
    rec._mode = "stdin"

    rec.pause()
    rec.resume()

    assert rec._proc.stdin.writes == ["PAUSE\n", "RESUME\n"]
    assert rec._proc.stdin.flushes == 2
    assert rec._paused is False


def test_pause_is_ignored_without_control_capability(no_helper):
    install_helper(no_helper, content=b"no marker")
    rec = S.SystemAudioRecorder()
    rec._proc = FakeProc(with_stdin=False)
    rec._mode = None
    rec.pause()
    assert rec._paused is False
    assert rec._proc.signals == []


def test_pause_and_resume_do_nothing_when_not_running(no_helper):
    install_helper(no_helper, content=b"STDIN_CONTROL_V1")
    rec = S.SystemAudioRecorder()
    rec.pause()
    rec.resume()
    assert rec._paused is False


# ── stop and diagnostics ─────────────────────────────────────────────

def test_stop_returns_none_without_recording():
    assert S.SystemAudioRecorder().stop() is None


def test_stop_returns_wav_once_it_holds_audio(tmp_path):
    wav = tmp_path / "capture.wav"
    wav.write_bytes(b"\0" * 4096)
    rec = S.SystemAudioRecorder()
    rec.output_path = str(wav)
    assert rec.stop() == str(wav)


def test_stop_discards_header_only_file(tmp_path):
    wav = tmp_path / "empty.wav"
    wav.write_bytes(b"\0" * 1024)
    rec = S.SystemAudioRecorder()
    rec.output_path = str(wav)
    assert rec.stop() is None


def test_stop_sends_protocol_command_and_closes_pipe(tmp_path):
    wav = tmp_path / "capture.wav"
    wav.write_bytes(b"\0" * 4096)
    rec = S.SystemAudioRecorder()
    rec.output_path = str(wav)
    proc = FakeProc()
    rec._proc = proc
    rec._mode = "stdin"

    assert rec.stop() == str(wav)
    assert proc.stdin.writes == ["STOP\n"]
    assert proc.stdin.closed is True
    assert rec._proc is None
    assert rec.is_running is False


def test_error_text_reads_helper_stderr(tmp_path):
    log = tmp_path / "helper.log"
    log.write_text("audio service unavailable", encoding="utf-8")
    rec = S.SystemAudioRecorder()
    rec._err_path = str(log)
    assert "service unavailable" in rec.error_text()


def test_error_text_is_empty_when_there_is_nothing_to_read():
    rec = S.SystemAudioRecorder()
    assert rec.error_text() == ""
    rec._err_path = "/nonexistent/helper.log"
    assert rec.error_text() == ""
