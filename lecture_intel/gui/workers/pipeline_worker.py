"""
Background worker: spawns the pipeline in a SEPARATE PROCESS and relays its
progress/result to the UI via Qt signals.

The QThread itself does only lightweight queue I/O — all the heavy, crash-prone
native ML work happens in the child process. So:
  - a native crash/OOM in the child can't take down the GUI (and the user's
    recording, already saved to disk, is safe);
  - there's no deep native recursion on the QThread stack (no SIGBUS);
  - quitting mid-run just terminates the child (no QThread-destroyed abort).
"""
from __future__ import annotations

import multiprocessing as mp
import queue as queue_mod
import sys
from pathlib import Path

from PySide6.QtCore import QThread, Signal

# Ensure the project root is importable so the child can find core.runner.
_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


class PipelineWorker(QThread):
    progress = Signal(dict)
    finished = Signal(dict)
    error = Signal(str)

    def __init__(self, input_path: str, output_dir: str, settings: dict, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.output_dir = output_dir
        self.settings = settings
        self._cancelled = False
        self._proc: mp.process.BaseProcess | None = None

    def cancel(self) -> None:
        self._cancelled = True
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()

    def run(self) -> None:
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        from core.runner import run_pipeline_subprocess
        self._proc = ctx.Process(
            target=run_pipeline_subprocess,
            args=(self.input_path, self.output_dir, self.settings, q),
            daemon=True,
        )
        self._proc.start()

        while True:
            if self._cancelled:
                self._terminate()
                self.error.emit("已取消。")
                return
            try:
                kind, payload = q.get(timeout=0.2)
            except queue_mod.Empty:
                if not self._proc.is_alive():
                    # Child died without delivering a result → native crash / OOM.
                    code = self._proc.exitcode
                    self.error.emit(
                        "处理进程意外退出"
                        + (f"（退出码 {code}）" if code is not None else "")
                        + "，很可能是内存不足或模型过大。\n\n"
                        "你的录音文件已安全保存，没有丢失。\n"
                        "建议换更小的模型（识别模型里选「均衡」或「最快」）后重试。"
                    )
                    return
                continue

            if kind == "progress":
                self.progress.emit(payload)
            elif kind == "result":
                stem = Path(self.input_path).stem
                out_dir = Path(self.output_dir)
                files = sorted(str(f) for f in out_dir.glob(f"{stem}*.*"))
                self.finished.emit({
                    "output_dir": self.output_dir,
                    "files": files,
                    "stats": payload,
                    "mode": payload.get("mode", "general"),
                    "ielts": payload.get("ielts"),
                    "classroom": payload.get("classroom"),
                })
                self._join()
                return
            elif kind == "error":
                self.error.emit(payload)
                self._join()
                return

    # -- process lifecycle helpers --

    def _terminate(self) -> None:
        if self._proc is not None and self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=5)

    def _join(self) -> None:
        if self._proc is not None:
            self._proc.join(timeout=5)
