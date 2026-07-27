"""
ProcessingScreen — progress ring + glass step card.

API mirrors the old ProgressPanel:
    reset(steps, filename, meta)
    update_progress(info dict from PipelineWorker)
Signal: cancel_requested
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.widgets.visuals import ProgressRing

STEP_LABELS: dict[str, str] = {
    "load":     "加载音频",
    "denoise":  "降噪处理",
    "asr":      "语音转写",
    "diarize":  "区分说话人",
    "analyze":  "分析 / AI 增强",
    "export":   "导出结果",
}
STATUS_ICONS = {"waiting": "○", "running": "●", "done": "✓", "error": "✗"}


class ProcessingScreen(QWidget):
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._step_widgets: dict[str, dict] = {}
        self._step_start: dict[str, float] = {}
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
        cancel = QPushButton("取消")
        cancel.setObjectName("cancelPill")
        cancel.setCursor(Qt.PointingHandCursor)
        cancel.clicked.connect(self.cancel_requested)
        foot.addWidget(cancel)
        lay.addLayout(foot)

        root.addWidget(card)

    # ── public API ──────────────────────────────────────────

    def reset(self, enabled_steps: list[str], filename: str = "",
              meta: str = "") -> None:
        for w in self._step_widgets.values():
            w["row"].deleteLater()
        self._step_widgets.clear()
        self._step_start.clear()
        self._ring.set_progress(0, "准备中…")
        self._file_lbl.setText(filename)
        self._meta_lbl.setText(meta)

        c = theme.current_scheme()
        for step in enabled_steps:
            label = STEP_LABELS.get(step, step)
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
                                        "name": name, "time": tlbl}

    def update_progress(self, info: dict) -> None:
        step = info.get("step", "")
        message = info.get("message", "")
        percent = info.get("percent", 0)
        status = info.get("status", "running")
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
            w["time"].setText("进行中")
            w["time"].setStyleSheet(f"color: {c['accent']}; font-size: 11px;")
        elif status == "done":
            elapsed = time.time() - self._step_start.get(step, time.time())
            w["name"].setStyleSheet(f"color: {c['ink2']}; font-size: 13px;")
            w["time"].setText(f"{elapsed:.1f}s")
            w["time"].setStyleSheet(f"color: {c['ink3']}; font-size: 11px;")
        elif status == "error":
            w["name"].setStyleSheet(f"color: {c['danger']}; font-size: 13px;")
            w["time"].setText("")
