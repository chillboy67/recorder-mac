"""
Denoise — the classroom-only cleanup step and its noise-floor gate.

`measure_noise_floor` is a volumedetect *heuristic* (documented as such): it
decides whether classroom mode cleans the audio at all, and both the value and
the decision are recorded in meta.json. These tests pin the parsing, the
"never break the pipeline" fallbacks, and the ffmpeg parameters actually used.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import denoise as D  # noqa: E402


@pytest.fixture
def tmp_tmpdir(tmp_path, monkeypatch):
    """Keep tempfile writes inside tmp_path (the sandbox rejects the default)."""
    monkeypatch.setattr(D.tempfile, "tempdir", str(tmp_path))
    return tmp_path


def fake_which(path):
    return lambda name: path


VOLUMEDETECT_OK = SimpleNamespace(
    returncode=0,
    stderr=("[Parsed_volumedetect_0 @ 0x7f] mean_volume: -38.2 dB\n"
            "[Parsed_volumedetect_0 @ 0x7f] max_volume: -6.0 dB\n"))


# ── measure_noise_floor ─────────────────────────────────────────────

def test_measure_noise_floor_reads_mean_volume(monkeypatch):
    monkeypatch.setattr(D.shutil, "which", fake_which("/fake/ffmpeg"))
    monkeypatch.setattr(D.subprocess, "run", lambda cmd, **kw: VOLUMEDETECT_OK)
    assert D.measure_noise_floor(Path("/tmp/any.wav")) == -38.2


def test_measure_noise_floor_returns_none_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(D.shutil, "which", fake_which(None))
    assert D.measure_noise_floor(Path("/tmp/any.wav")) is None


def test_measure_noise_floor_returns_none_when_unparseable(monkeypatch):
    """No volumedetect line → None, and the caller then defaults to denoising."""
    monkeypatch.setattr(D.shutil, "which", fake_which("/fake/ffmpeg"))
    monkeypatch.setattr(D.subprocess, "run",
                        lambda cmd, **kw: SimpleNamespace(returncode=1,
                                                          stderr="boom"))
    assert D.measure_noise_floor(Path("/tmp/any.wav")) is None


def test_measure_noise_floor_returns_none_when_ffmpeg_raises(monkeypatch):
    monkeypatch.setattr(D.shutil, "which", fake_which("/fake/ffmpeg"))

    def boom(cmd, **kw):
        raise OSError("exec format error")

    monkeypatch.setattr(D.subprocess, "run", boom)
    assert D.measure_noise_floor(Path("/tmp/any.wav")) is None


# ── denoise_audio fallbacks ─────────────────────────────────────────

def test_denoise_audio_without_ffmpeg_returns_the_input(monkeypatch, tmp_path):
    monkeypatch.setattr(D.shutil, "which", fake_which(None))
    src = tmp_path / "in.wav"
    src.write_bytes(b"x")
    assert D.denoise_audio(src) == src


def test_denoise_audio_returns_input_when_ffmpeg_fails(monkeypatch, tmp_path,
                                                      tmp_tmpdir):
    monkeypatch.setattr(D.shutil, "which", fake_which("/fake/ffmpeg"))
    monkeypatch.setattr(D.subprocess, "run",
                        lambda cmd, **kw: SimpleNamespace(returncode=1,
                                                          stderr="Invalid data"))
    src = tmp_path / "in.wav"
    src.write_bytes(b"x")
    assert D.denoise_audio(src) == src          # never sinks the run


def test_denoise_audio_returns_input_when_output_is_empty(monkeypatch, tmp_path,
                                                          tmp_tmpdir):
    monkeypatch.setattr(D.shutil, "which", fake_which("/fake/ffmpeg"))
    monkeypatch.setattr(D.subprocess, "run",
                        lambda cmd, **kw: SimpleNamespace(returncode=0,
                                                          stderr=""))
    src = tmp_path / "in.wav"
    src.write_bytes(b"x")
    assert D.denoise_audio(src) == src          # mkstemp left a 0-byte file


def test_denoise_audio_returns_the_cleaned_file_on_success(monkeypatch,
                                                           tmp_path, tmp_tmpdir):
    monkeypatch.setattr(D.shutil, "which", fake_which("/fake/ffmpeg"))
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        Path(cmd[-1]).write_bytes(b"RIFF" + b"\0" * 512)   # producer wrote output
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(D.subprocess, "run", fake_run)
    src = tmp_path / "in.wav"
    src.write_bytes(b"x")

    out = D.denoise_audio(src)

    assert out != src and out.exists() and out.stat().st_size > 0
    # the signal chain is part of the fidelity contract: content never altered
    assert D.FILTER_CHAIN in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("-ac") + 1] == "1"
    assert seen["cmd"][seen["cmd"].index("-ar") + 1] == "16000"


def test_denoise_audio_survives_an_unexpected_exception(monkeypatch, tmp_path,
                                                        tmp_tmpdir):
    monkeypatch.setattr(D.shutil, "which", fake_which("/fake/ffmpeg"))

    def boom(cmd, **kw):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(D.subprocess, "run", boom)
    src = tmp_path / "in.wav"
    src.write_bytes(b"x")
    assert D.denoise_audio(src) == src


def test_filter_chain_is_conservative_and_documented():
    """Rumble + broadband noise + gentle levelling: this exact chain is written
    into meta.json, so changing it silently would break the audit trail."""
    assert D.FILTER_CHAIN == ("highpass=f=80,"
                              "afftdn=nr=12:nf=-25,"
                              "dynaudnorm=f=200:g=5")
