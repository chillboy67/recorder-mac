"""First-run model downloads report real progress (core/model_download.py)."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import model_download as D  # noqa: E402
from core import transcriber as T  # noqa: E402
from core.i18n import set_language  # noqa: E402

CHUNK = 1000


def _slow_writer(folder: Path, chunks: int = 4, name: str = "blob.incomplete"):
    """A fake hub download: writes into blobs/ a chunk at a time."""
    def fetch() -> str:
        blob = folder / "blobs" / name
        blob.parent.mkdir(parents=True, exist_ok=True)
        with open(blob, "wb") as fh:
            for _ in range(chunks):
                fh.write(b"x" * CHUNK)
                fh.flush()
                time.sleep(0.06)
        blob.rename(blob.with_suffix(""))
        return str(folder / "snapshots" / "abc")
    return fetch


def test_progress_rises_to_the_total(tmp_path):
    reports = []
    path = D.download_with_progress(_slow_writer(tmp_path), tmp_path, 4 * CHUNK,
                                    lambda done, total: reports.append((done, total)),
                                    interval=0.02)
    assert path.endswith("abc")
    done = [d for d, _ in reports]
    assert done == sorted(done) and len(set(done)) >= 3
    assert all(d < 4 * CHUNK for d in done[:-1])     # never 100% before it finishes
    assert reports[-1] == (4 * CHUNK, 4 * CHUNK)


def test_files_already_in_the_cache_are_not_counted(tmp_path):
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / "old").write_bytes(b"y" * 50_000)
    reports = []
    D.download_with_progress(_slow_writer(tmp_path, 2), tmp_path, None,
                             lambda done, total: reports.append((done, total)), interval=0.02)
    assert reports[-1] == (2 * CHUNK, None)


def test_a_failed_download_raises_in_the_caller(tmp_path):
    def fetch():
        raise ConnectionError("offline")
    with pytest.raises(ConnectionError):
        D.download_with_progress(fetch, tmp_path, 10, lambda *a: None, interval=0.01)


@pytest.mark.skipif(os.name == "nt", reason="symlinks need privileges on Windows")
def test_snapshot_symlinks_are_not_counted_twice(tmp_path):
    (tmp_path / "blobs").mkdir()
    (tmp_path / "blobs" / "b").write_bytes(b"z" * 300)
    (tmp_path / "snapshots").mkdir()
    (tmp_path / "snapshots" / "model.bin").symlink_to(tmp_path / "blobs" / "b")
    assert D.bytes_on_disk(tmp_path) == 300


def test_expected_bytes_counts_only_the_fetched_files(monkeypatch):
    siblings = [SimpleNamespace(rfilename="model.bin", size=900),
                SimpleNamespace(rfilename="vocabulary.json", size=90),
                SimpleNamespace(rfilename="README.md", size=9)]

    class FakeApi:
        def model_info(self, repo_id, files_metadata):
            assert files_metadata
            return SimpleNamespace(siblings=siblings)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=FakeApi))
    assert D.expected_bytes("org/m") == 999
    assert D.expected_bytes("org/m", ["model.bin", "vocabulary.*"]) == 990


def test_expected_bytes_is_none_when_the_hub_is_unreachable(monkeypatch):
    class DownApi:
        def model_info(self, *a, **k):
            raise OSError("no network")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=DownApi))
    assert D.expected_bytes("org/m") is None


@pytest.fixture
def english():
    set_language("en")
    yield
    set_language("zh")


def test_cpu_model_first_run_reports_download_percent(monkeypatch, tmp_path, english):
    cache = tmp_path / "cache"
    repo = "Systran/faster-whisper-small"
    folder = D.repo_cache_dir(repo, cache)
    calls = []

    def download_model(model, cache_dir):
        calls.append((model, cache_dir))
        return _slow_writer(folder)()

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(
        utils=SimpleNamespace(_MODELS={"small": repo}, download_model=download_model)))
    monkeypatch.setattr(T, "find_faster_whisper_model", lambda model: None)
    monkeypatch.setattr(T, "faster_whisper_cache_dir", lambda: cache)
    monkeypatch.setattr(T, "expected_bytes", lambda repo_id, patterns=None: 4 * CHUNK)
    monkeypatch.setattr(D, "POLL_INTERVAL", 0.02)
    messages = []

    T.Transcriber(model="small", engine="faster-whisper")._prefetch_faster(
        lambda frac, msg: messages.append((frac, msg)))

    assert calls == [("small", str(cache))]
    assert messages[-1] == (0.04, "First use: downloading the small model, "
                                  "0.00 / 0.00 GB (100%)")
    assert any("(25%)" in m or "(50%)" in m or "(75%)" in m for _, m in messages)


def test_cpu_model_already_cached_is_not_downloaded(monkeypatch):
    monkeypatch.setattr(T, "find_faster_whisper_model", lambda model: Path("/cached"))
    monkeypatch.setattr(T, "download_with_progress",
                        lambda *a, **k: pytest.fail("downloaded a cached model"))
    T.Transcriber(model="small", engine="faster-whisper")._prefetch_faster(None)


def test_a_failed_first_run_download_leaves_the_error_to_the_loader(monkeypatch, tmp_path, caplog):
    def download_model(model, cache_dir):
        raise OSError("offline")
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(
        utils=SimpleNamespace(_MODELS={"small": "org/m"}, download_model=download_model)))
    monkeypatch.setattr(T, "find_faster_whisper_model", lambda model: None)
    monkeypatch.setattr(T, "faster_whisper_cache_dir", lambda: tmp_path)
    monkeypatch.setattr(T, "expected_bytes", lambda *a: None)

    T.Transcriber(model="small", engine="faster-whisper")._prefetch_faster(None)

    assert "offline" in caplog.text


def test_mlx_model_downloads_through_snapshot_download(monkeypatch, tmp_path):
    repo = "mlx-community/whisper-small-mlx"
    calls = []

    def snapshot_download(repo_id, local_files_only=False):
        calls.append((repo_id, local_files_only))
        if local_files_only:
            raise OSError("not cached")
        return "/snap"

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setattr(T, "_local_model_dir", lambda model: None)
    monkeypatch.setattr(T, "expected_bytes", lambda repo_id: 10)
    monkeypatch.setattr(T, "repo_cache_dir", lambda repo_id: tmp_path)

    T.Transcriber(model="small", engine="faster-whisper")._prefetch_mlx(None)

    assert calls == [(repo, True), (repo, False)]


def test_mlx_model_from_a_local_folder_is_not_downloaded(monkeypatch, tmp_path):
    monkeypatch.setattr(T, "_local_model_dir", lambda model: tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        snapshot_download=lambda **k: pytest.fail("downloaded a local model")))
    T.Transcriber(model="small", engine="faster-whisper")._prefetch_mlx(None)
