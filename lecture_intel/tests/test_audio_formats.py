"""
Every input format the file picker offers, through the ffmpeg calls the app
really makes: probe + convert (AudioLoader), the classroom noise gate and
denoise chain, repeat arbitration's silencedetect, and the mic + system mix.

The release app ships its own audio-only ffmpeg (build_ffmpeg.sh). Point
RECORDER_FFMPEG_DIR at that build's bin/ to check it has every demuxer,
decoder and filter these paths need; otherwise the ffmpeg on PATH is used.

Fixtures (tests/fixtures/formats/) are 1.7 s: 0.6 s tone, 0.5 s silence,
0.6 s tone — so silencedetect has exactly one inner gap to find.
"""
from __future__ import annotations

import ast
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "formats"
DURATION = 1.7
BUNDLED_DIR = os.environ.get("RECORDER_FFMPEG_DIR")


def _picker_extensions() -> set[str]:
    """SUPPORTED_EXTENSIONS from the home screen, read without importing Qt."""
    src = Path(__file__).resolve().parent.parent / "gui" / "widgets" / "home_screen.py"
    for node in ast.parse(src.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "SUPPORTED_EXTENSIONS":
            return set(ast.literal_eval(node.value))
    raise AssertionError("SUPPORTED_EXTENSIONS not found")


@pytest.fixture(autouse=True)
def ffmpeg_on_path(monkeypatch):
    if BUNDLED_DIR:
        monkeypatch.setenv("PATH", BUNDLED_DIR + os.pathsep + os.environ["PATH"])
        for tool in ("ffmpeg", "ffprobe"):
            found = shutil.which(tool)
            assert found and Path(found).parent == Path(BUNDLED_DIR).resolve(), found
    elif not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not installed")


@pytest.fixture
def mono_wav():
    from modules.audio_loader import AudioLoader

    return AudioLoader({}).process(FIXTURES / "stereo.wav").path


@pytest.mark.parametrize("name", sorted(p.name for p in FIXTURES.iterdir()))
def test_loader_converts_every_offered_format(name):
    import soundfile as sf
    from modules.audio_loader import AudioLoader

    src = FIXTURES / name
    assert src.suffix in _picker_extensions()
    audio = AudioLoader({}).process(src)

    if src.suffix != ".aac":   # raw ADTS has no index; ffprobe estimates from bitrate
        assert audio.duration_sec == pytest.approx(DURATION, abs=0.1)
    data, sr = sf.read(str(audio.path))
    assert (sr, data.ndim) == (16000, 1)
    assert len(data) / sr == pytest.approx(DURATION, abs=0.1)
    assert abs(data).max() > 0.03         # decoded sound (fixtures peak at 0.06), not silence


def test_every_offered_extension_has_a_fixture():
    assert _picker_extensions() <= {p.suffix for p in FIXTURES.iterdir()}


def test_classroom_noise_gate_and_denoise(mono_wav):
    import soundfile as sf
    from core.denoise import denoise_audio, measure_noise_floor

    assert measure_noise_floor(mono_wav) is not None
    cleaned = denoise_audio(mono_wav)
    assert cleaned != mono_wav            # the chain ran rather than falling back
    data, sr = sf.read(str(cleaned))
    assert sr == 16000 and len(data) / sr == pytest.approx(DURATION, abs=0.1)


def test_repeat_arbitration_counts_voiced_bursts(mono_wav):
    from core.repeat_arbitration import _count_voiced_bursts

    assert _count_voiced_bursts(mono_wav, 0.0, DURATION) == 2


def test_mic_and_system_audio_mix():
    import soundfile as sf
    from core.sysaudio import mix_recordings

    mic = str(FIXTURES / "stereo.wav")
    mixed = mix_recordings(mic, mic)
    try:
        assert mixed not in (None, mic)
        data, sr = sf.read(mixed)
        assert (sr, data.ndim) == (48000, 1)
    finally:
        if mixed and mixed != mic:
            Path(mixed).unlink(missing_ok=True)
