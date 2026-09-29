from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from prune_qt_webengine import prune


def test_prune_removes_webengine_and_keeps_multimedia(tmp_path):
    base = tmp_path / "PySide6"
    web = base / "Qt" / "lib" / "QtWebEngineCore.framework" / "Versions" / "A"
    web.mkdir(parents=True)
    (web / "QtWebEngineCore").write_bytes(b"x" * 32)
    (base / "QtWebEngineCore.abi3.so").write_bytes(b"so")
    multimedia = base / "Qt" / "lib" / "QtMultimedia.framework" / "Versions" / "A"
    multimedia.mkdir(parents=True)
    (multimedia / "QtMultimedia").write_bytes(b"mm")
    (base / "QtMultimedia.abi3.so").write_bytes(b"mmso")
    (base / "QtWidgets.abi3.so").write_bytes(b"w")

    removed = prune(base)

    assert removed >= 32
    assert not (base / "Qt" / "lib" / "QtWebEngineCore.framework").exists()
    assert not (base / "QtWebEngineCore.abi3.so").exists()
    assert (multimedia / "QtMultimedia").read_bytes() == b"mm"
    assert (base / "QtMultimedia.abi3.so").read_bytes() == b"mmso"
    assert (base / "QtWidgets.abi3.so").read_bytes() == b"w"
