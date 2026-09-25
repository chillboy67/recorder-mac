"""Cross-platform model-cache locations and model discovery."""
from __future__ import annotations

import os
import platform
from pathlib import Path


def model_cache_dir() -> Path:
    """Return a user-writable, platform-appropriate root for downloaded models."""
    override = os.environ.get("RECORDER_MODEL_CACHE_DIR")
    if override:
        return Path(override).expanduser()

    home = Path.home()
    system = platform.system()
    if system == "Darwin":
        return home / "Library" / "Caches" / "Recorder" / "models"
    if system == "Windows":
        local = os.environ.get("LOCALAPPDATA")
        base = Path(local) if local else home / "AppData" / "Local"
        return base / "Recorder" / "models"

    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else home / ".cache"
    return base / "recorder" / "models"


def faster_whisper_cache_dir() -> Path:
    return model_cache_dir() / "faster-whisper"


def mlx_cache_dir() -> Path:
    return model_cache_dir() / "mlx"


def is_faster_whisper_model(path: str | Path) -> bool:
    directory = Path(path)
    return (directory.is_dir()
            and (directory / "config.json").is_file()
            and (directory / "model.bin").is_file()
            and (directory / "tokenizer.json").is_file())


def find_faster_whisper_model(model: str) -> Path | None:
    """Return an already-cached CTranslate2 snapshot without using the network."""
    try:
        from faster_whisper.utils import download_model

        path = download_model(
            model,
            cache_dir=str(faster_whisper_cache_dir()),
            local_files_only=True,
        )
    except (OSError, RuntimeError, ValueError):
        return None
    return Path(path) if is_faster_whisper_model(path) else None


def whisper_cpp_model_dir(model: str) -> Path:
    """Return the cache folder for a ggml-format whisper.cpp model."""
    if not model or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-_" for ch in model):
        raise ValueError(f"Invalid whisper.cpp model name: {model!r}")
    return model_cache_dir() / "whisper.cpp" / model


def whisper_cpp_model_path(model: str) -> Path:
    return whisper_cpp_model_dir(model) / f"ggml-{model}.bin"


def whisper_cpp_model_available(model: str, backend: str) -> bool:
    directory = whisper_cpp_model_dir(model)
    weights = whisper_cpp_model_path(model)
    if not weights.is_file() or weights.stat().st_size == 0:
        return False
    if backend == "vulkan":
        return True
    if backend == "openvino":
        return ((directory / f"ggml-{model}-encoder-openvino.xml").is_file()
                and (directory / f"ggml-{model}-encoder-openvino.bin").is_file())
    return False
