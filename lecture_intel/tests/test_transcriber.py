"""
Transcriber helpers that do not need a model.

The Whisper wrapper itself needs real weights, but `_local_model_dir` is plain
path resolution — and it decides whether a run uses the model the user already
downloaded or reaches for the network. It is also the one place that prefers the
installed copy over the development tree.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import transcriber as T  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect ~ (install candidate) and __file__ (dev-tree candidate)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("RECORDER_MODEL_CACHE_DIR", str(tmp_path / "model-cache"))
    monkeypatch.setattr(T, "__file__", str(tmp_path / "fake" / "core" / "transcriber.py"))
    return tmp_path


def install_dir(home: Path, model: str) -> Path:
    return (home / "model-cache" / "mlx" / f"whisper-{model}-mlx")


def legacy_macos_install_dir(home: Path, model: str) -> Path:
    return (home / "Library" / "Application Support" / "Recorder" / "models"
            / f"whisper-{model}-mlx")


def dev_dir(home: Path, model: str) -> Path:
    return home / "fake" / "models" / f"whisper-{model}-mlx"


def make_model_dir(d: Path, weights: str = "weights.npz", config: bool = True) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    if config:
        (d / "config.json").write_text("{}", encoding="utf-8")
    if weights:
        (d / weights).write_bytes(b"\0")
    return d


def test_no_model_anywhere_resolves_to_none(sandbox):
    assert T._local_model_dir("large-v3") is None


def test_the_installed_copy_is_used(sandbox):
    d = make_model_dir(install_dir(sandbox, "large-v3"))
    assert T._local_model_dir("large-v3") == d


def test_legacy_macos_install_copy_is_still_used(sandbox):
    d = make_model_dir(legacy_macos_install_dir(sandbox, "large-v3"))
    assert T._local_model_dir("large-v3") == d


def test_the_dev_tree_is_used_when_nothing_is_installed(sandbox):
    d = make_model_dir(dev_dir(sandbox, "small"))
    assert T._local_model_dir("small") == d


def test_the_installed_copy_wins_over_the_dev_tree(sandbox):
    installed = make_model_dir(install_dir(sandbox, "turbo"))
    make_model_dir(dev_dir(sandbox, "turbo"))
    assert T._local_model_dir("turbo") == installed


def test_a_directory_without_a_config_is_not_a_model(sandbox):
    make_model_dir(install_dir(sandbox, "medium"), config=False)
    assert T._local_model_dir("medium") is None


def test_a_directory_without_weights_is_not_a_model(sandbox):
    make_model_dir(install_dir(sandbox, "medium"), weights="")
    assert T._local_model_dir("medium") is None


def test_any_weights_file_counts(sandbox):
    """mlx ships weights.npz; a converted copy may use .safetensors."""
    d = make_model_dir(install_dir(sandbox, "large-v3"), weights="weights.safetensors")
    assert T._local_model_dir("large-v3") == d


def test_the_model_name_is_part_of_the_path(sandbox):
    make_model_dir(install_dir(sandbox, "large-v3"))
    assert T._local_model_dir("large-v3") is not None
    assert T._local_model_dir("small") is None


def test_auto_profile_keeps_accuracy_model_on_mlx_and_uses_small_on_cpu():
    transcriber = T.Transcriber(model="auto")
    assert transcriber._resolved_model("mlx-whisper") == "large-v3"
    assert transcriber._resolved_model("faster-whisper") == "small"


def test_explicit_model_is_never_overridden_by_auto_profile():
    transcriber = T.Transcriber(model="large-v3-turbo")
    assert transcriber._resolved_model("faster-whisper") == "large-v3-turbo"


def test_auto_can_select_configured_whisper_cpp_vulkan_before_cpu(
        tmp_path, monkeypatch):
    model_dir = tmp_path / "models" / "whisper.cpp" / "small"
    model_dir.mkdir(parents=True)
    (model_dir / "ggml-small.bin").write_bytes(b"weights")
    binary = tmp_path / "whisper-cli-vulkan"
    binary.write_bytes(b"binary")
    monkeypatch.setenv("RECORDER_MODEL_CACHE_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("RECORDER_WHISPER_CPP_VULKAN", str(binary))
    monkeypatch.setattr(T.Transcriber, "_mlx_available", staticmethod(lambda: False))
    monkeypatch.setattr(T.Transcriber, "_faster_available", staticmethod(lambda: True))

    assert T.Transcriber(model="auto")._resolve_engine() == "whisper.cpp-vulkan"


def test_explicit_whisper_cpp_engine_requires_its_binary_and_model(monkeypatch):
    monkeypatch.delenv("RECORDER_WHISPER_CPP_OPENVINO", raising=False)
    transcriber = T.Transcriber(model="small", engine="whisper.cpp-openvino")
    with pytest.raises(RuntimeError, match="whisper.cpp openvino is not ready"):
        transcriber._resolve_engine()


def test_chunked_mlx_and_single_pass_failure_really_falls_back_to_cpu(
        tmp_path, monkeypatch):
    transcriber = T.Transcriber(model="auto")
    monkeypatch.setattr(transcriber, "_resolve_engine", lambda: "mlx-whisper")

    def fail_chunked(*args, **kwargs):
        raise RuntimeError("chunking failed")

    def fail_single(*args, **kwargs):
        raise RuntimeError("MLX unavailable")

    monkeypatch.setattr(transcriber, "_transcribe_mlx_chunked", fail_chunked)
    monkeypatch.setattr(transcriber, "_transcribe_mlx", fail_single)
    monkeypatch.setattr(transcriber, "_faster_available", lambda: True)
    monkeypatch.setattr(transcriber, "_transcribe_faster",
                        lambda *args, **kwargs: ([{"start": 0, "end": 1,
                                                  "text": "hello", "words": []}], "en"))

    result = transcriber.transcribe(tmp_path / "unused.wav", chunked=True)

    assert result.model_used == "faster-whisper:small"
    assert result.full_text == "hello"
    assert len(result.warnings) == 1
    assert "chunked MLX failed" in result.warnings[0]


def test_cpu_loader_uses_the_shared_offline_cache_and_bounded_threads(
        monkeypatch, tmp_path):
    captured = {}

    class FakeWhisperModel:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured.update(kwargs)

    monkeypatch.setitem(sys.modules, "faster_whisper",
                        SimpleNamespace(WhisperModel=FakeWhisperModel))
    cached = tmp_path / "small-snapshot"
    monkeypatch.setattr(T, "find_faster_whisper_model", lambda model: cached)
    monkeypatch.setattr(T, "faster_whisper_cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(T.os, "cpu_count", lambda: 16)
    transcriber = T.Transcriber(model="small", engine="faster-whisper")

    transcriber._load_faster()

    assert captured["model"] == str(cached)
    assert captured["device"] == "cpu"
    assert captured["compute_type"] == "int8"
    assert captured["cpu_threads"] == 8
    assert captured["download_root"] == str(tmp_path / "cache")
