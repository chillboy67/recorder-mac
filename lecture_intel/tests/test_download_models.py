from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from download_models import REPOS, main


def test_auto_prepares_mlx_and_cpu_fallback_on_apple_silicon(monkeypatch):
    calls = []
    monkeypatch.setattr("download_models._default_engine", lambda: "mlx")
    monkeypatch.setattr("download_models.fetch_model",
                        lambda name, use_hf: calls.append(("mlx", name)) or True)
    monkeypatch.setattr("download_models.fetch_cpu_model",
                        lambda name, use_hf=False: calls.append(("cpu", name)) or True)

    assert main([]) == 0
    assert {name for backend, name in calls if backend == "mlx"} == set(REPOS)
    assert ("cpu", "small") in calls


def test_auto_downloads_cpu_model_on_non_mlx_platform(monkeypatch):
    calls = []
    monkeypatch.setattr("download_models._default_engine", lambda: "cpu")
    monkeypatch.setattr("download_models.fetch_cpu_model",
                        lambda name, use_hf=False: calls.append(name) or True)

    assert main([]) == 0
    assert calls == ["small"]
