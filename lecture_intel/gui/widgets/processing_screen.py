"""
ProcessingScreen — progress ring + glass step card.

API mirrors the old ProgressPanel:
    reset(steps, filename, meta)
    update_progress(info dict from PipelineWorker)
Signal: cancel_requested
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from core.i18n import t
from gui import theme
from gui.widgets.visuals import ProgressRing

STEP_LABELS: dict[str, str] = {
    "load":     "step_load",
    "denoise":  "step_denoise",
    "asr":      "step_asr",
    "diarize":  "step_diarize",
    "analyze":  "step_analyze",
    "export":   "step_export",
}
STATUS_ICONS = {"waiting": "○", "running": "●", "done": "✓", "error": "✗"}

# Transcription is the long wait, so the line under the ring rotates through
# these instead of repeating one message; the pool shifts as the ring fills.
CHEER_INTERVAL_MS = 7000
CHEER_EARLY = ("tr_writing", "cheer_sip", "cheer_slow")
CHEER_HALF = ("cheer_half", "tr_writing", "cheer_slow")      # ring ≥ 50%
CHEER_NEARLY = ("cheer_nearly", "tr_writing")                # ring ≥ 67%


class ProcessingScreen(QWidget):
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._step_widgets: dict[str, dict] = {}
        self._step_start: dict[str, float] = {}
        self._cheer_tick = 0
        self._cheer_percent = 0
        self._cheer_timer = QTimer(self)
        self._cheer_timer.setInterval(CHEER_INTERVAL_MS)
        self._cheer_timer.timeout.connect(self._next_cheer)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(48, 12, 48, 24)
        root.setSpacing(48)
        root.setAlignment(Qt.AlignCenter)

        self._ring = ProgressRing()
        root.addWidget(self._ring)

        card = QFrame()
        card.setObjectName("glassCard")
        card.setFixedWidth(400)
        lay = QVBoxLayout(card)
        lay.setContentsMargins(28, 24, 28, 20)
        lay.setSpacing(13)

        head = QHBoxLayout()
        self._file_lbl = QLabel("")
        self._file_lbl.setStyleSheet("font-size: 14px; font-weight: 600;")
        self._meta_lbl = QLabel("")
        self._meta_lbl.setProperty("mono", True)
        head.addWidget(self._file_lbl, stretch=1)
        head.addWidget(self._meta_lbl)
        lay.addLayout(head)

        hr = QFrame()
        hr.setObjectName("hairline")
        hr.setFixedHeight(1)
        lay.addWidget(hr)

        self._steps_box = QVBoxLayout()
        self._steps_box.setSpacing(10)
        lay.addLayout(self._steps_box)

        foot = QHBoxLayout()
        foot.addStretch()
        self._cancel_btn = QPushButton(t("proc_cancel"))
        self._cancel_btn.setObjectName("cancelPill")
        self._cancel_btn.setCursor(Qt.PointingHandCursor)
        self._cancel_btn.clicked.connect(self.cancel_requested)
        foot.addWidget(self._cancel_btn)
        lay.addLayout(foot)

        root.addWidget(card)

    # ── public API ──────────────────────────────────────────

    def reset(self, enabled_steps: list[str], filename: str = "",
              meta: str = "") -> None:
        for w in self._step_widgets.values():
            w["row"].deleteLater()
        self._step_widgets.clear()
        self._step_start.clear()
        self._cheer_timer.stop()
        self._ring.set_progress(0, t("proc_preparing"))
        self._file_lbl.setText(filename)
        self._meta_lbl.setText(meta)

        c = theme.current_scheme()
        for step in enabled_steps:
            label = t(STEP_LABELS.get(step, step))
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(10)
            icon = QLabel(STATUS_ICONS["waiting"])
            icon.setFixedWidth(16)
            icon.setStyleSheet(f"color: {c['ink3']}; font-size: 13px;")
            name = QLabel(label)
            name.setStyleSheet(f"color: {c['ink3']}; font-size: 13px;")
            tlbl = QLabel("")
            tlbl.setStyleSheet(f"color: {c['ink3']}; font-size: 11px;")
            tlbl.setAlignment(Qt.AlignRight)
            rl.addWidget(icon)
            rl.addWidget(name, stretch=1)
            rl.addWidget(tlbl)
            self._steps_box.addWidget(row)
            self._step_widgets[step] = {"row": row, "icon": icon,
                                        "name": name, "time": tlbl, "key": step}

    def update_progress(self, info: dict) -> None:
        step = info.get("step", "")
        message = info.get("message", "")
        percent = info.get("percent", 0)
        status = info.get("status", "running")
        if info.get("cheer"):
            self._cheer_percent = int(percent)
            if not self._cheer_timer.isActive():
                self._cheer_tick = 0
                self._cheer_timer.start()
            message = self._cheer_line()
        else:
            self._cheer_timer.stop()
        self._ring.set_progress(int(percent), message[:22])

        if step not in self._step_widgets:
            return
        c = theme.current_scheme()
        colors = {"waiting": c["ink3"], "running": c["accent"],
                  "done": c["ok"], "error": c["danger"]}
        w = self._step_widgets[step]
        w["icon"].setText(STATUS_ICONS.get(status, "●"))
        w["icon"].setStyleSheet(
            f"color: {colors.get(status, c['accent'])}; font-size: 13px;")
        if status == "running":
            self._step_start[step] = time.time()
            w["name"].setStyleSheet(
                f"color: {c['accent']}; font-size: 13px; font-weight: 600;")
            w["time"].setText(t("proc_in_progress"))
            w["time"].setStyleSheet(f"color: {c['accent']}; font-size: 11px;")
        elif status == "done":
            elapsed = time.time() - self._step_start.get(step, time.time())
            w["name"].setStyleSheet(f"color: {c['ink2']}; font-size: 13px;")
            w["time"].setText(f"{elapsed:.1f}s")
            w["time"].setStyleSheet(f"color: {c['ink3']}; font-size: 11px;")
        elif status == "error":
            w["name"].setStyleSheet(f"color: {c['danger']}; font-size: 13px;")
            w["time"].setText("")

    def _cheer_line(self) -> str:
        if self._cheer_percent >= 67:
            pool = CHEER_NEARLY
        elif self._cheer_percent >= 50:
            pool = CHEER_HALF
        else:
            pool = CHEER_EARLY
        return t(pool[self._cheer_tick % len(pool)])

    def _next_cheer(self) -> None:
        self._cheer_tick += 1
        self._ring.set_progress(self._cheer_percent, self._cheer_line()[:22])

    def hideEvent(self, event) -> None:
        self._cheer_timer.stop()
        super().hideEvent(event)

    def retranslate(self) -> None:
        """Re-apply static text in the newly selected language."""
        self._cancel_btn.setText(t("proc_cancel"))
        for step, w in self._step_widgets.items():
            w["name"].setText(t(STEP_LABELS.get(step, step)))
            if w["icon"].text() == STATUS_ICONS["running"]:
                w["time"].setText(t("proc_in_progress"))
