from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import model_cache


def test_cache_dir_honors_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDER_MODEL_CACHE_DIR", str(tmp_path / "models"))
    assert model_cache.model_cache_dir() == tmp_path / "models"


def test_windows_uses_local_app_data(tmp_path, monkeypatch):
    monkeypatch.delenv("RECORDER_MODEL_CACHE_DIR", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.setattr(model_cache.platform, "system", lambda: "Windows")
    assert model_cache.model_cache_dir() == tmp_path / "Local" / "Recorder" / "models"


def test_linux_respects_xdg_cache_home(tmp_path, monkeypatch):
    monkeypatch.delenv("RECORDER_MODEL_CACHE_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setattr(model_cache.platform, "system", lambda: "Linux")
    assert model_cache.model_cache_dir() == tmp_path / "xdg" / "recorder" / "models"


def test_macos_cache_uses_library_caches(tmp_path, monkeypatch):
    monkeypatch.delenv("RECORDER_MODEL_CACHE_DIR", raising=False)
    monkeypatch.setattr(model_cache.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert model_cache.model_cache_dir() == tmp_path / "Library" / "Caches" / "Recorder" / "models"


def test_faster_whisper_model_requires_config_model_and_tokenizer(tmp_path):
    assert not model_cache.is_faster_whisper_model(tmp_path)
    tmp_path.mkdir(exist_ok=True)
    for name in ("config.json", "model.bin", "tokenizer.json"):
        (tmp_path / name).write_bytes(b"model")
    assert model_cache.is_faster_whisper_model(tmp_path)
    (tmp_path / "model.bin").unlink()
    assert not model_cache.is_faster_whisper_model(tmp_path)


def test_local_model_lookup_uses_offline_cache(tmp_path, monkeypatch):
    model_path = tmp_path / "snapshot"
    model_path.mkdir()
    for name in ("config.json", "model.bin", "tokenizer.json"):
        (model_path / name).write_bytes(b"model")
    monkeypatch.setattr(model_cache, "faster_whisper_cache_dir", lambda: tmp_path / "cache")
    fake_utils = SimpleNamespace(download_model=lambda *args, **kwargs: str(model_path))
    fake_package = SimpleNamespace(utils=fake_utils)
    monkeypatch.setitem(sys.modules, "faster_whisper", fake_package)
    monkeypatch.setitem(sys.modules, "faster_whisper.utils", fake_utils)

    assert model_cache.find_faster_whisper_model("small") == model_path


def test_whisper_cpp_cache_requires_backend_specific_files(tmp_path, monkeypatch):
    monkeypatch.setenv("RECORDER_MODEL_CACHE_DIR", str(tmp_path / "cache"))
    model_dir = model_cache.whisper_cpp_model_dir("small")
    model_dir.mkdir(parents=True)
    (model_dir / "ggml-small.bin").write_bytes(b"weights")

    assert model_cache.whisper_cpp_model_available("small", "vulkan")
    assert not model_cache.whisper_cpp_model_available("small", "openvino")
    (model_dir / "ggml-small-encoder-openvino.xml").write_text("xml")
    (model_dir / "ggml-small-encoder-openvino.bin").write_bytes(b"encoder")
    assert model_cache.whisper_cpp_model_available("small", "openvino")
