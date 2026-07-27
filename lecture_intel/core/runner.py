"""
Subprocess entry point for the pipeline.

The heavy ML (mlx/torch/whisper) runs in a SEPARATE PROCESS, not a thread.
Why: (1) a native crash (stack overflow, OOM, segfault in a C library) kills
only this child — the GUI survives and can tell the user, and their recording
file (already on disk) is never lost; (2) the child gets a full main-thread
stack, avoiding the QThread stack-overflow crash; (3) quitting the app just
terminates the child — no "QThread destroyed while running" abort.

Communication is a multiprocessing.Queue of (kind, payload) messages:
  ("progress", {...})  ("result", {...})  ("error", "traceback")
"""
from __future__ import annotations

import sys
from pathlib import Path


def run_pipeline_subprocess(input_path, output_dir, settings, queue):
    """Runs in the CHILD process. Never raises — reports everything via queue."""
    try:
        root = str(Path(__file__).resolve().parent.parent)
        if root not in sys.path:
            sys.path.insert(0, root)

        from core.engine import run

        def on_progress(info):
            try:
                queue.put(("progress", info))
            except Exception:
                pass

        summary = run(
            input_path=input_path,
            output_dir=output_dir,
            mode_key=settings.get("mode", "general"),
            model=settings.get("model", "large-v3"),
            formats=settings.get("formats"),
            progress=on_progress,
        )
        queue.put(("result", summary))
    except Exception:
        import traceback
        queue.put(("error", traceback.format_exc()))
    finally:
        try:
            queue.close()
        except Exception:
            pass
