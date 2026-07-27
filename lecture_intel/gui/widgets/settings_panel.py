"""
Settings panel: pick a processing mode, model size, and output formats.

The mode encapsulates everything else (denoise, diarization, analysis), so the
UI stays simple — the user just picks what they're transcribing.
"""
from __future__ import annotations

from PySide6.QtCore import QSettings, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.modes import GENERAL, CLASSROOM, IELTS

MODES = [GENERAL, CLASSROOM, IELTS]

MODELS = [
    ("large-v3", "最准（large-v3）"),
    ("large-v3-turbo", "均衡（large-v3-turbo）"),
    ("small", "最快（small）"),
]


class SettingsPanel(QWidget):
    mode_changed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._prefs = QSettings("LucasLab", "Recorder")
        self._build_ui()
        self._load_prefs()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(0, 0, 0, 0)

        # -- Mode selection (radio buttons with descriptions) ----
        mode_box = QGroupBox("模式")
        mode_layout = QVBoxLayout(mode_box)
        mode_layout.setSpacing(2)
        self._mode_group = QButtonGroup(self)
        self._mode_buttons: dict[str, QRadioButton] = {}
        for m in MODES:
            rb = QRadioButton(m.label)
            rb.setStyleSheet("font-size: 13px; font-weight: 500;")
            self._mode_group.addButton(rb)
            self._mode_buttons[m.key] = rb
            rb.toggled.connect(lambda checked, k=m.key: self._on_mode(k, checked))
            mode_layout.addWidget(rb)

            # Description label. A QLabel stylesheet `margin` does NOT reserve
            # vertical space, so wrapped lines overlapped/clipped. Indent via a
            # container layout and let the label size to its wrapped content.
            desc = QLabel(m.description)
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #8E8E93; font-size: 11px;")
            desc.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.MinimumExpanding)
            row = QHBoxLayout()
            row.setContentsMargins(22, 0, 4, 6)
            row.addWidget(desc)
            mode_layout.addLayout(row)
        layout.addWidget(mode_box)

        # -- Model size ------------------------------------------
        model_box = QGroupBox("识别模型")
        model_layout = QVBoxLayout(model_box)
        self._model_combo = QComboBox()
        for value, label in MODELS:
            self._model_combo.addItem(label, userData=value)
        self._model_combo.currentIndexChanged.connect(self._save_prefs)
        model_layout.addWidget(self._model_combo)
        layout.addWidget(model_box)

        # -- Local LLM enhancement -------------------------------
        llm_box = QGroupBox("增强（可选）")
        llm_layout = QVBoxLayout(llm_box)
        self._cb_llm = QCheckBox("本地大模型增强")
        self._cb_llm.setChecked(False)
        self._cb_llm.stateChanged.connect(self._save_prefs)
        llm_hint = QLabel("中文用中文模型、英文用英文模型（自动判断）。通用：AI校对全文　"
                          "课堂：据内容校对+重点总结　雅思：AI考官点评。\n"
                          "全程本地离线。未安装 Ollama 时自动跳过。")
        llm_hint.setWordWrap(True)
        llm_hint.setStyleSheet("color: #8E8E93; font-size: 11px;")
        llm_layout.addWidget(self._cb_llm)
        llm_layout.addWidget(llm_hint)
        layout.addWidget(llm_box)

        # -- Output formats --------------------------------------
        fmt_box = QGroupBox("导出格式")
        fmt_layout = QHBoxLayout(fmt_box)
        self._cb_txt = QCheckBox(".txt")
        self._cb_md = QCheckBox(".md")
        self._cb_doc = QCheckBox(".doc")
        self._cb_docx = QCheckBox(".docx")
        for cb in (self._cb_txt, self._cb_md, self._cb_docx):
            cb.setChecked(True)
        for cb in (self._cb_txt, self._cb_md, self._cb_doc, self._cb_docx):
            cb.stateChanged.connect(self._save_prefs)
            fmt_layout.addWidget(cb)
        layout.addWidget(fmt_box)

        layout.addStretch()

    # ── events ────────────────────────────────────────────────

    def _on_mode(self, key: str, checked: bool) -> None:
        if checked:
            self._save_prefs()
            self.mode_changed.emit(key)

    # ── public API ────────────────────────────────────────────

    def current_mode(self) -> str:
        for key, rb in self._mode_buttons.items():
            if rb.isChecked():
                return key
        return "general"

    def get_settings(self) -> dict:
        formats = [f for f, cb in (
            ("txt", self._cb_txt), ("md", self._cb_md),
            ("doc", self._cb_doc), ("docx", self._cb_docx),
        ) if cb.isChecked()] or ["txt", "md"]
        return {
            "mode": self.current_mode(),
            "model": self._model_combo.currentData(),
            "formats": formats,
            "use_llm": self._cb_llm.isChecked(),
            "llm_model": "llama3.1:8b",      # English
            "chinese_model": "qwen-zh:7b",   # Chinese (auto-selected for zh audio)
        }

    # ── persistence ───────────────────────────────────────────

    def _save_prefs(self) -> None:
        s = self.get_settings()
        self._prefs.setValue("mode", s["mode"])
        self._prefs.setValue("model", s["model"])
        self._prefs.setValue("formats", s["formats"])
        self._prefs.setValue("use_llm", s["use_llm"])

    def _load_prefs(self) -> None:
        mode = self._prefs.value("mode", "general")
        self._mode_buttons.get(mode, self._mode_buttons["general"]).setChecked(True)

        model = self._prefs.value("model", "large-v3")
        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == model:
                self._model_combo.setCurrentIndex(i)
                break

        fmts = self._prefs.value("formats", ["txt", "md", "docx"])
        if isinstance(fmts, str):
            fmts = [fmts]
        self._cb_txt.setChecked("txt" in fmts)
        self._cb_md.setChecked("md" in fmts)
        self._cb_doc.setChecked("doc" in fmts)
        self._cb_docx.setChecked("docx" in fmts)

        self._cb_llm.setChecked(self._prefs.value("use_llm", False, type=bool))
