#!/usr/bin/env python3
"""Remove Qt WebEngine files from the active PySide6 install.

Recording uses Qt Widgets and Qt Multimedia. The PySide6 Addons wheel also
installs WebEngine, which this app does not import.
"""
from __future__ import annotations

import os
import shutil
import sys
import sysconfig
from pathlib import Path


def _is_webengine(name: str) -> bool:
    return "webengine" in name.lower()


def webengine_paths(pyside_dir: Path) -> list[Path]:
    """Top-most files and directories under ``pyside_dir`` whose name is WebEngine."""
    if not pyside_dir.is_dir():
        return []
    found: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(pyside_dir):
        parent = Path(dirpath)
        kept: list[str] = []
        for name in dirnames:
            if _is_webengine(name):
                found.append(parent / name)
            else:
                kept.append(name)
        dirnames[:] = kept
        for name in filenames:
            if _is_webengine(name):
                found.append(parent / name)
    return found


def _nbytes(path: Path) -> int:
    if path.is_symlink() or path.is_file():
        try:
            return path.lstat().st_size
        except OSError:
            return 0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for name in filenames:
            try:
                total += (Path(dirpath) / name).lstat().st_size
            except OSError:
                pass
    return total


def prune(pyside_dir: Path) -> int:
    """Delete WebEngine paths inside ``pyside_dir``. Returns bytes removed."""
    root = pyside_dir.resolve()
    removed = 0
    for path in webengine_paths(pyside_dir):
        if not path.exists() and not path.is_symlink():
            continue
        resolved = path.resolve()
        if resolved != root and root not in resolved.parents:
            continue
        removed += _nbytes(path)
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            shutil.rmtree(path)
    return removed


def main() -> int:
    pyside_dir = Path(sysconfig.get_paths()["purelib"]) / "PySide6"
    if not pyside_dir.is_dir():
        print(f"PySide6 is not installed in {pyside_dir.parent}")
        return 0
    removed = prune(pyside_dir)
    print(f"removed {removed} bytes of Qt WebEngine from {pyside_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
