"""
QThread worker: runs the core engine in a background thread, emitting
progress and results via Qt Signals back to the UI thread.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

from PySide6.QtCore import QThread, Signal


class PipelineWorker(QThread):
    """Background worker that wraps core.engine.run()."""

    progress = Signal(dict)    # {"step", "message", "percent", "status"}
    finished = Signal(dict)    # full result summary from engine.run
    error = Signal(str)

    def __init__(self, input_path: str, output_dir: str, settings: dict, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.output_dir = output_dir
        self.settings = settings
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            root = str(Path(__file__).parent.parent.parent)
            if root not in sys.path:
                sys.path.insert(0, root)

            from core.engine import run as engine_run

            summary = engine_run(
                input_path=self.input_path,
                output_dir=self.output_dir,
                mode_key=self.settings.get("mode", "general"),
                model=self.settings.get("model", "large-v3"),
                formats=self.settings.get("formats"),
                progress=self._on_progress,
            )

            out_dir = Path(self.output_dir)
            stem = Path(self.input_path).stem
            output_files = sorted(str(f) for f in out_dir.glob(f"{stem}*.*"))

            self.finished.emit({
                "output_dir": self.output_dir,
                "files": output_files,
                "stats": summary,
                "mode": summary.get("mode", "general"),
                "ielts": summary.get("ielts"),
            })

        except InterruptedError:
            self.error.emit("已取消。")
        except Exception:
            self.error.emit(
                f"{sys.exc_info()[0].__name__}: {sys.exc_info()[1]}\n\n"
                f"{traceback.format_exc()}"
            )

    def _on_progress(self, info: dict) -> None:
        if self._cancelled:
            raise InterruptedError("cancelled")
        self.progress.emit(info)
