"""
Left vertical step rail: 01 输入 · 02 模式 · 03 转写 · 04 结果.
Purely indicative — reflects the app's current stage.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from core.i18n import t
from gui import theme

_STEP_KEYS = ["rail_input", "rail_mode", "rail_transcribe", "rail_results"]


class StepRail(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("stepRail")
        self.setFixedWidth(96)
        self._items: list[dict] = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 28, 0, 28)
        layout.setSpacing(30)
        layout.setAlignment(Qt.AlignTop | Qt.AlignHCenter)

        self._active = 0
        for i, key in enumerate(_STEP_KEYS):
            num = QLabel(f"0{i + 1}")
            num.setAlignment(Qt.AlignCenter)
            name = QLabel(t(key))
            name.setAlignment(Qt.AlignCenter)
            bar = QFrame()
            bar.setFixedSize(18, 2)

            cell = QVBoxLayout()
            cell.setSpacing(5)
            cell.setAlignment(Qt.AlignHCenter)
            for w in (num, name, bar):
                cell.addWidget(w, alignment=Qt.AlignHCenter)
            layout.addLayout(cell)
            self._items.append({"num": num, "name": name, "bar": bar,
                                "key": key})

        self.set_stage(0)

    def set_stage(self, active: int) -> None:
        """Steps before `active` are done; `active` is current."""
        self._active = active
        for i, it in enumerate(self._items):
            state = "done" if i < active else "active" if i == active else "idle"
            it["num"].setProperty("railNum", state)
            it["name"].setProperty("railLabel", state)
            it["name"].setText(t(it["key"]) + (" ✓" if state == "done" else ""))
            it["bar"].setProperty("railBar", "on" if state == "active" else "off")
            for w in (it["num"], it["name"], it["bar"]):
                theme.repolish(w)

    def retranslate(self) -> None:
        self.set_stage(self._active)
