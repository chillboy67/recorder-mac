"""Local logs and the diagnostics bundle (core/diagnostics.py)."""
from __future__ import annotations

import json
import logging
import sys
import zipfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import diagnostics  # noqa: E402

SPOKEN = "I goed to the school yesterday"


@pytest.fixture
def logs(tmp_path, monkeypatch):
    folder = tmp_path / "logs"
    monkeypatch.setenv("RECORDER_LOG_DIR", str(folder))
    root = logging.getLogger()
    before = list(root.handlers)
    level = root.level
    yield folder
    for h in root.handlers:
        if h not in before:
            root.removeHandler(h)
            h.close()
    root.setLevel(level)


def _output_dir(tmp_path: Path) -> Path:
    out = tmp_path / "out"
    out.mkdir()
    (out / "meta.json").write_text(json.dumps({
        "original": {"file": "original.wav", "sha256": "ab" * 32, "duration_sec": 12.5},
        "steps": [
            {"name": "transcribe", "params": {"model": "large-v3", "engine": "mlx",
                                              "language": "en"}},
            {"name": "annotations", "oracle_audio": str(Path.home() / "x" / "original.wav"),
             "annotations": [{"type": "repeat_arbitration", "segment_id": 3,
                              "original": SPOKEN, "verdict": "real_speech",
                              "folded": False}]},
            {"name": "keep_main_speaker_only",
             "removed": [{"segment_id": 7, "speaker_id": "S1", "text": SPOKEN}]},
        ],
    }), encoding="utf-8")
    (out / "original.wav").write_bytes(b"RIFF" + b"\0" * 64)
    (out / "talk.txt").write_text(SPOKEN, encoding="utf-8")
    return out


def test_logging_goes_to_a_file_per_process(logs):
    diagnostics.setup_logging("app")
    diagnostics.setup_logging("app")            # idempotent
    logging.getLogger("recorder.test").info("hello from the app")
    for h in logging.getLogger().handlers:
        h.flush()

    assert "hello from the app" in (logs / "app.log").read_text(encoding="utf-8")
    assert (logs / "crash-app.log").exists()
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_recorder", False)]
    assert len(ours) == 1


def test_a_forked_worker_swaps_the_inherited_handler(logs):
    diagnostics.setup_logging("app")
    diagnostics.setup_logging("worker")
    ours = [h for h in logging.getLogger().handlers if getattr(h, "_recorder", False)]
    assert [Path(h.baseFilename).name for h in ours] == ["worker.log"]


def test_bundle_has_logs_system_and_redacted_meta(logs, tmp_path):
    diagnostics.setup_logging("worker")
    logging.getLogger("recorder.test").warning("reading %s", Path.home() / "Documents" / "a.m4a")
    for h in logging.getLogger().handlers:
        h.flush()

    bundle = diagnostics.export_bundle(tmp_path / "diag.zip", _output_dir(tmp_path))

    with zipfile.ZipFile(bundle) as zf:
        names = set(zf.namelist())
        assert {"system.json", "logs/worker.log", "logs/crash-worker.log",
                "meta.redacted.json"} <= names
        assert not any(n.endswith((".wav", ".txt")) for n in names)
        everything = "\n".join(zf.read(n).decode("utf-8") for n in names)
        system = json.loads(zf.read("system.json"))
        meta = json.loads(zf.read("meta.redacted.json"))

    assert SPOKEN not in everything
    assert str(Path.home()) not in everything
    assert "~/Documents/a.m4a" in everything
    assert system["app_version"] and "packages" in system
    transcribe, annotations, removed = meta["steps"]
    assert transcribe["params"] == {"model": "large-v3", "engine": "mlx", "language": "en"}
    ann = annotations["annotations"][0]
    assert ann["verdict"] == "real_speech" and ann["folded"] is False
    assert ann["original"] == f"<{len(SPOKEN)} chars>"
    assert removed["removed"][0] == {"segment_id": 7, "speaker_id": "S1",
                                     "text": f"<{len(SPOKEN)} chars>"}
    assert meta["original"]["sha256"] == "ab" * 32


def test_bundle_without_a_run_or_logs(logs, tmp_path):
    bundle = diagnostics.export_bundle(tmp_path / "sub" / "diag.zip")
    with zipfile.ZipFile(bundle) as zf:
        assert "system.json" in zf.namelist()
        assert "meta.redacted.json" not in zf.namelist()
