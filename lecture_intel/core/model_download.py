"""
First-run model downloads with real progress.

mlx-whisper and faster-whisper both fetch a missing model from the Hugging
Face Hub the first time they load it, silently — a multi-gigabyte wait behind
a message that never changes. ``download_with_progress`` runs that same
download itself, in a thread, and reports how far it has got while it runs.

Progress is measured on disk rather than through the hub's progress bars:
both of its download paths (plain HTTP and hf-xet) write each file into the
cache's ``blobs/`` folder as ``*.incomplete`` while it arrives, so the bytes
added under the model's cache folder since the start, over the size the hub
lists for the repo, is the real fraction — without depending on hub internals.
When the size can't be listed the byte count is still reported, without a
percentage.
"""
from __future__ import annotations

import fnmatch
import logging
import threading
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

logger = logging.getLogger(__name__)

# (downloaded bytes, total bytes or None)
Report = Callable[[int, Optional[int]], None]

POLL_INTERVAL = 0.5   # seconds between progress reports


def repo_cache_dir(repo_id: str, cache_dir: Optional[str | Path] = None) -> Path:
    """Where the hub caches ``repo_id`` (``models--org--name``)."""
    if cache_dir is None:
        from huggingface_hub import constants
        cache_dir = constants.HF_HUB_CACHE
    return Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"


def bytes_on_disk(folder: Path) -> int:
    """Bytes in real files under ``folder`` (snapshot symlinks are not counted twice)."""
    total = 0
    if not folder.exists():
        return 0
    for p in folder.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
        except OSError:   # a *.incomplete renamed between listing and stat
            pass
    return total


def expected_bytes(repo_id: str, allow_patterns: Optional[Iterable[str]] = None) -> Optional[int]:
    """Total size of the files the download will fetch, or None if unknown."""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo_id, files_metadata=True)
    except Exception as exc:
        logger.info("Could not list %s for download progress: %s", repo_id, exc)
        return None
    patterns = list(allow_patterns or [])
    sizes = [s.size or 0 for s in (info.siblings or [])
             if not patterns or any(fnmatch.fnmatch(s.rfilename, pat) for pat in patterns)]
    return sum(sizes) or None


def download_with_progress(
    fetch: Callable[[], str],
    folder: Path,
    total: Optional[int],
    report: Report,
    interval: Optional[float] = None,
) -> str:
    """Run ``fetch`` (the library's own download call) and report its progress.

    ``folder`` is the model's cache folder, watched for new bytes. Returns what
    ``fetch`` returns and re-raises what it raises."""
    interval = POLL_INTERVAL if interval is None else interval
    baseline = bytes_on_disk(folder)
    result: dict = {}

    def run() -> None:
        try:
            result["path"] = fetch()
        except BaseException as exc:   # re-raised in the caller's thread
            result["error"] = exc

    worker = threading.Thread(target=run, name="model-download", daemon=True)
    worker.start()
    last = -1
    while worker.is_alive():
        worker.join(interval)
        # Never report less than before: a scan that races the library's
        # rename of *.incomplete misses that file and would show progress
        # falling back to zero for one tick.
        done = max(last, 0, bytes_on_disk(folder) - baseline)
        if done != last:
            report(min(done, total - 1) if total else done, total)
            last = done
    if "error" in result:
        raise result["error"]
    if total:
        report(total, total)
    return result["path"]
