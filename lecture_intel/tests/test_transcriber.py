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

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import transcriber as T  # noqa: E402


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect ~ (install candidate) and __file__ (dev-tree candidate)."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(T, "__file__", str(tmp_path / "fake" / "core" / "transcriber.py"))
    return tmp_path


def install_dir(home: Path, model: str) -> Path:
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
