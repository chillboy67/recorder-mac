"""
Pipeline driver.

The heavy ML runs in a SEPARATE PROCESS (core.runner). The GUI side does NOT use
a QThread at all — it just polls the result queue from the main thread with a
QTimer. This deliberately avoids QThread entirely, which removes two whole
classes of crash we hit:
  - SIGBUS: deep native recursion (mlx graph compile) overflowing a worker
    thread's small stack — there's no worker thread now;
  - SIGABRT "QThread: Destroyed while thread is still running" at shutdown.

A native crash or OOM in the child can't take down the GUI, and the user's
recording (already on disk) is never lost.
"""
from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import sys
from pathlib import Path

from PySide6.QtCore import QObject, QTimer, Signal

_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.i18n import t  # noqa: E402  (after sys.path is set)


class PipelineWorker(QObject):
    progress = Signal(dict)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, input_path: str, output_dir: str, settings: dict, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.output_dir = output_dir
        self.settings = settings
        self._proc = None
        self._queue = None
        self._timer = None
        self._running = False

    # -- public API (mirrors the old QThread surface used by MainWindow) --

    def start(self) -> None:
        ctx = mp.get_context("spawn")
        self._queue = ctx.Queue()
        from core.runner import run_pipeline_subprocess
        self._proc = ctx.Process(
            target=run_pipeline_subprocess,
            args=(self.input_path, self.output_dir, self.settings, self._queue),
            daemon=True,
        )
        self._proc.start()
        self._running = True
        self._timer = QTimer(self)
        self._timer.setInterval(100)
        self._timer.timeout.connect(self._poll)
        self._timer.start()

    def isRunning(self) -> bool:
        return self._running

    def cancel(self) -> None:
        """Non-blocking: SIGTERM the child (returns instantly) and reap later.
        Does NOT block the UI thread and does NOT emit an error dialog."""
        if not self._running:
            return
        self._running = False
        self._stop_timer()
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()        # SIGTERM — returns immediately
        # reap in the background a moment later so the click never freezes
        QTimer.singleShot(1500, self._reap)

    def _reap(self) -> None:
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc.join(timeout=2)    # already dead → returns at once

    def wait(self, _ms: int = 0) -> bool:
        """Used on app close: terminate first, then a quick join (no long block)."""
        self._stop_timer()
        self._running = False
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc.join(timeout=5)
        return True

    # -- internals --

    def _poll(self) -> None:
        if not self._running or self._queue is None:
            return
        # drain everything currently available
        try:
            while True:
                kind, payload = self._queue.get_nowait()
                if kind == "progress":
                    self.progress.emit(payload)
                elif kind == "result":
                    self._finish()
                    self._emit_result(payload)
                    return
                elif kind == "error":
                    self._finish()
                    self.error.emit(payload)
                    return
        except queue_mod.Empty:
            pass
        # if the child died without delivering anything → crash / OOM
        if self._proc is not None and not self._proc.is_alive():
            # give the queue a last chance (race between exit and final put)
            try:
                kind, payload = self._queue.get_nowait()
                if kind == "result":
                    self._finish(); self._emit_result(payload); return
                if kind == "error":
                    self._finish(); self.error.emit(payload); return
            except queue_mod.Empty:
                pass
            code = self._proc.exitcode
            self._finish()
            code_part = (t("worker_crash_code", code=code)
                         if code is not None else "")
            self.error.emit(t("worker_crash", code=code_part))

    def _emit_result(self, summary: dict) -> None:
        stem = Path(self.input_path).stem
        out_dir = Path(self.output_dir)
        files = sorted(str(f) for f in out_dir.glob(f"{stem}*.*"))
        self.finished.emit({
            "output_dir": self.output_dir,
            "files": files,
            "stats": summary,
            "mode": summary.get("mode", "general"),
            "ielts": summary.get("ielts"),
            "classroom": summary.get("classroom"),
        })

    def _finish(self) -> None:
        self._stop_timer()
        self._running = False
        if self._proc is not None:
            self._proc.join(timeout=5)

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _terminate(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)
