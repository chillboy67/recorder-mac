"""
Single source of truth for where Recorder writes user data.

Two roots, deliberately different:

* ``data_root()`` — recordings, and the output folder when the app is installed
  rather than run from a checkout.
* ``default_output_root()`` — transcripts/reports. When the code is inside a git
  checkout, this is ``output/`` inside that checkout, i.e. the exact folder the
  user cloned or downloaded the repo into. Output never goes to a system
  location (~/Library/Application Support, /Applications) by default: nobody
  looks there for their own files.

Note: ~/Documents is TCC-protected, so the first time a Finder-launched app
writes here macOS shows a one-time "allow access to Documents" prompt. That's
expected; once granted it persists. (The app bundle itself and the model cache
stay in ~/Library/Application Support/Recorder, which needs no prompt.)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional


def clone_root() -> Optional[Path]:
    """The folder this code was cloned/downloaded into, or ``None``.

    Found by walking up to the ``.git`` marker, so it is whatever path the user
    chose — never a location baked in on the author's machine. ``None`` when the
    code isn't inside a checkout: ``make_app.sh`` copies ``lecture_intel/`` into
    the support folder, which has no repo around it.
    """
    for parent in Path(__file__).resolve().parents:
        if (parent / ".git").exists():
            return parent
    return None


def default_output_root() -> Path:
    """Suggested output folder, before the user has confirmed anything.

    A checkout writes to ``<the folder git created>/output``. Installed without
    a repo around it, the fallback is ``data_root()`` — the Documents folder the
    user can see in Finder — and not the support folder the code happens to sit
    in.
    """
    clone = clone_root()
    root = (clone / "output") if clone else data_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def data_root() -> Path:
    """Root folder for all user-facing recordings and outputs."""
    root = Path.home() / "Documents" / "recorder" / "Recorder"
    root.mkdir(parents=True, exist_ok=True)
    return root


def record_dir() -> Path:
    """Folder for saved recordings."""
    d = data_root() / "record"
    d.mkdir(parents=True, exist_ok=True)
    return d
