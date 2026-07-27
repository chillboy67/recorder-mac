"""
Progress tracking panel.

Shows a progress bar, current operation label, and a per-step list
with status icons and elapsed times.
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

STEP_LABELS: dict[str, str] = {
    "load":      "Load Audio",
    "enhance":   "Enhance Audio",
    "diarize":   "Speaker Diarization",
    "vad":       "Voice Detection",
    "asr":       "Transcription",
    "merge":     "Merge Segments",
    "classify":  "Course Classification",
    "correct":   "Terminology Correction",
    "llm":       "LLM Correction",
    "structure": "Structuring",
    "export":    "Export",
}

STATUS_ICONS: dict[str, str] = {
    "waiting": "○",   # ○
    "running": "●",   # ●
    "done":    "✓",   # ✓
    "error":   "✗",   # ✗
}

STATUS_COLORS: dict[str, str] = {
    "waiting": "#C7C7CC",
    "running": "#0A84FF",
    "done":    "#34C759",
    "error":   "#FF3B30",
}


class ProgressPanel(QWidget):
    """Shows live pipeline progress with per-step timing."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._step_widgets: dict[str, dict[str, QWidget | QLabel]] = {}
        self._step_start_times: dict[str, float] = {}
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self._layout = QVBoxLayout(self)
        self._layout.setSpacing(6)
        self._layout.setContentsMargins(0, 10, 0, 0)

        # Overall progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        # The bar is a thin 6px sliver — its built-in % text doesn't fit and
        # overlaps the label below. Hide it; show the percent in the label.
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setFixedHeight(6)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: none;
                background: #E5E5EA;
                border-radius: 3px;
            }
            QProgressBar::chunk {
                background: #0A84FF;
                border-radius: 3px;
            }
        """)
        self._layout.addWidget(self._progress_bar)

        # Current operation label
        self._current_label = QLabel("等待文件…")
        self._current_label.setStyleSheet(
            "color: gray; font-size: 12px; margin: 4px 0;"
        )
        self._layout.addWidget(self._current_label)

        # Step list container
        self._steps_frame = QFrame()
        self._steps_layout = QVBoxLayout(self._steps_frame)
        self._steps_layout.setSpacing(2)
        self._steps_layout.setContentsMargins(0, 0, 0, 0)
        self._layout.addWidget(self._steps_frame)
        self._layout.addStretch()

    # ── Public API ────────────────────────────────────────────

    def reset(self, enabled_steps: list[str]) -> None:
        """Clear all step widgets and rebuild for the given step list."""
        # Remove old widgets
        for w in self._step_widgets.values():
            if isinstance(w["row"], QWidget):
                w["row"].deleteLater()
        self._step_widgets.clear()
        self._step_start_times.clear()

        self._progress_bar.setValue(0)
        self._current_label.setText("开始…")

        # Create a row per step
        for step in enabled_steps:
            label = STEP_LABELS.get(step, step)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(8)

            icon_lbl = QLabel(STATUS_ICONS["waiting"])
            icon_lbl.setFixedWidth(16)
            icon_lbl.setStyleSheet(
                f"color: {STATUS_COLORS['waiting']}; font-size: 13px;"
            )

            name_lbl = QLabel(label)
            name_lbl.setStyleSheet("color: #8E8E93; font-size: 12px;")

            time_lbl = QLabel("")
            time_lbl.setStyleSheet("color: #C7C7CC; font-size: 11px;")
            time_lbl.setAlignment(Qt.AlignRight)

            row_layout.addWidget(icon_lbl)
            row_layout.addWidget(name_lbl, stretch=1)
            row_layout.addWidget(time_lbl)

            self._steps_layout.addWidget(row)
            self._step_widgets[step] = {
                "row": row,
                "icon": icon_lbl,
                "name": name_lbl,
                "time": time_lbl,
            }

    def update_progress(self, info: dict) -> None:
        """Update progress display from a pipeline progress dict."""
        step = info.get("step", "")
        message = info.get("message", "")
        percent = info.get("percent", 0)
        status = info.get("status", "running")

        self._progress_bar.setValue(percent)
        self._current_label.setText(f"{percent}%　{message}")

        if step not in self._step_widgets:
            return

        w = self._step_widgets[step]
        icon = STATUS_ICONS.get(status, "●")
        color = STATUS_COLORS.get(status, "#0A84FF")

        w["icon"].setText(icon)
        w["icon"].setStyleSheet(f"color: {color}; font-size: 13px;")

        if status == "running":
            self._step_start_times[step] = time.time()
            w["name"].setStyleSheet(
                "color: #0A84FF; font-size: 12px; font-weight: 500;"
            )
        elif status == "done":
            elapsed = time.time() - self._step_start_times.get(step, time.time())
            w["time"].setText(f"{elapsed:.1f}s")
            w["name"].setStyleSheet("color: #3A3A3C; font-size: 12px;")
        elif status == "error":
            w["name"].setStyleSheet("color: #FF3B30; font-size: 12px;")

    def set_total_time(self, seconds: float) -> None:
        """Display final completion message with total elapsed time."""
        self._current_label.setText(
            f"✓ 完成，用时 {seconds:.0f}s"
        )
        self._current_label.setStyleSheet(
            "color: #34C759; font-size: 12px; font-weight: 500;"
        )
