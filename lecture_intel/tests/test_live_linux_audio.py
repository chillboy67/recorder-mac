"""Live system-audio capture on a real Linux sound server — opt-in.

The other sysaudio tests use fakes. This one plays tones through the default
output and records them with the app's own ``SystemAudioRecorder``, so it
checks what users hear: the helper starts on the installed PipeWire or
PulseAudio, paused audio stays out of the file, and the end is not cut off.

Needs a running PipeWire (with pipewire-pulse) or PulseAudio session with a
default sink, plus ``pw-play`` or ``paplay``. A virtual sink is enough::

    pactl load-module module-null-sink sink_name=speakers
    pactl set-default-sink speakers
    RECORDER_LIVE_AUDIO_TESTS=1 .venv/bin/python -m pytest tests/test_live_linux_audio.py
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pytest

if not os.environ.get("RECORDER_LIVE_AUDIO_TESTS"):
    pytest.skip("live audio tests are opt-in: set RECORDER_LIVE_AUDIO_TESTS=1",
                allow_module_level=True)
if not sys.platform.startswith("linux"):
    pytest.skip("Linux system-audio capture only", allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import sysaudio  # noqa: E402

RATE = 48000
TONES = (300, 600, 900, 1200, 1500)   # one second each
WINDOW_S = 0.25


def _write_tones(path: Path) -> None:
    t = np.arange(RATE) / RATE
    signal = np.concatenate([0.3 * np.sin(2 * np.pi * f * t) for f in TONES])
    pcm = (signal * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(RATE)
        wav.writeframes(np.repeat(pcm, 2).tobytes())


# Resolved at import, before a test narrows PATH to one capture backend.
PLAYER = shutil.which("pw-play") or shutil.which("paplay")


def _tone_windows(path: Path) -> list[int]:
    """Dominant tone of each 0.25 s window (0 for silence)."""
    with wave.open(str(path)) as wav:
        rate, channels = wav.getframerate(), wav.getnchannels()
        samples = np.frombuffer(wav.readframes(wav.getnframes()), "<i2")
    mono = samples[::channels].astype(float)
    size = int(rate * WINDOW_S)
    found = []
    for start in range(0, len(mono) - size + 1, size):
        window = mono[start:start + size]
        if np.sqrt(np.mean(window ** 2)) < 300:
            found.append(0)
            continue
        spectrum = np.abs(np.fft.rfft(window * np.hanning(size)))
        peak = np.fft.rfftfreq(size, 1 / rate)[int(np.argmax(spectrum))]
        found.append(min(TONES, key=lambda f: abs(f - peak)))
    return found


@pytest.fixture(params=["pw-record", "parec"])
def only_backend(request, tmp_path, monkeypatch):
    """Expose a single capture backend on PATH so each one is exercised."""
    tool = shutil.which(request.param)
    if tool is None:
        pytest.skip(f"{request.param} is not installed")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / request.param).symlink_to(tool)
    monkeypatch.setenv("PATH", str(bin_dir))
    return request.param


def test_system_audio_capture_honours_pause_and_keeps_the_end(tmp_path, only_backend):
    if PLAYER is None:
        pytest.skip("neither pw-play nor paplay is installed")
    tones = tmp_path / "tones.wav"
    _write_tones(tones)
    assert sysaudio.available(), sysaudio.binary_path()

    recorder = sysaudio.SystemAudioRecorder()
    recorder.start()
    try:
        time.sleep(0.8)
        playback = subprocess.Popen([PLAYER, str(tones)])
        time.sleep(1.0)              # 300 Hz plays
        recorder.pause()
        time.sleep(2.0)              # 600 and 900 Hz play while paused
        recorder.resume()
        playback.wait(timeout=15)    # 1200 and 1500 Hz
        time.sleep(0.5)
    finally:
        path = recorder.stop()
    assert path is not None, recorder.error_text()

    windows = _tone_windows(Path(path))
    counts = {f: windows.count(f) for f in TONES}
    detail = f"{only_backend}: {windows}\n{recorder.error_text()}"
    assert counts[300] >= 3, detail
    assert counts[600] <= 1 and counts[900] <= 1, detail   # paused: left out
    assert counts[1200] >= 3, detail
    assert counts[1500] >= 3, detail                     # the end survives stop
    Path(path).unlink(missing_ok=True)
