"""
HomeScreen — the AURA idle stage:

  · glass input card: drag & drop / 选择文件 / 实时录音 + 来源 + 麦克风
  · (after a file/recording is ready) selected-file card with 开始转写
  · three mode cards (通用 / 课堂 / 雅思)
  · compact settings row: 识别模型 · 导出格式 chips · 本地大模型增强

Signals:
    record_requested(str source_key, object mic_device)   # QAudioDevice | None
    process_requested(str input_path, dict settings)
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSettings, Qt, Signal
from PySide6.QtGui import QCursor, QDragEnterEvent, QDropEvent
from PySide6.QtMultimedia import QMediaDevices
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.modes import GENERAL, CLASSROOM, IELTS
from gui import theme
from gui.widgets.common import NoScrollComboBox
from gui.widgets.visuals import WaveGlyph

SUPPORTED_EXTENSIONS: set[str] = {
    ".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".webm",
}

MODES = [GENERAL, CLASSROOM, IELTS]
MODELS = [
    ("large-v3", "最准 large-v3"),
    ("large-v3-turbo", "均衡 large-v3-turbo"),
    ("small", "最快 small"),
]
SOURCES = [("mic", "麦克风"), ("system", "电脑声音"), ("both", "麦克风＋电脑声音")]


class ModeCard(QFrame):
    clicked = Signal(str)

    def __init__(self, key: str, title: str, desc: str,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.key = key
        self.setObjectName("modeCard")
        self.setFixedWidth(200)
        self.setCursor(QCursor(Qt.PointingHandCursor))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        lay.setSpacing(6)
        self._title = QLabel(title)
        self._desc = QLabel(desc)
        self._desc.setWordWrap(True)
        lay.addWidget(self._title)
        lay.addWidget(self._desc)
        self.set_selected(False)

    def set_selected(self, on: bool) -> None:
        self.setProperty("selected", "true" if on else "false")
        self._title.setProperty("cardTitle", "on" if on else "off")
        self._desc.setProperty("cardDesc", "on" if on else "off")
        for w in (self, self._title, self._desc):
            theme.repolish(w)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self.key)
        super().mousePressEvent(event)


class HomeScreen(QWidget):
    record_requested = Signal(str, object)
    process_requested = Signal(str, dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._prefs = QSettings("LucasLab", "Recorder")
        self._selected_path: str | None = None
        self.setAcceptDrops(True)
        self._build_ui()
        self._load_prefs()

    # ── UI ──────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 12, 48, 24)
        root.setSpacing(24)
        root.setAlignment(Qt.AlignCenter)

        # ---- input card (drop state) ----
        self._drop_card = QFrame()
        self._drop_card.setObjectName("glassCard")
        self._drop_card.setFixedWidth(640)
        drop = QVBoxLayout(self._drop_card)
        drop.setContentsMargins(40, 34, 40, 30)
        drop.setSpacing(14)
        drop.setAlignment(Qt.AlignHCenter)

        glyph = WaveGlyph()
        drop.addWidget(glyph, alignment=Qt.AlignHCenter)

        title = QLabel("把音频拖到这里")
        title.setStyleSheet("font-size: 15px; font-weight: 600;")
        drop.addWidget(title, alignment=Qt.AlignHCenter)

        exts = QLabel("m4a · mp3 · wav · webm · flac · aac")
        exts.setProperty("mono", True)
        drop.addWidget(exts, alignment=Qt.AlignHCenter)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.setAlignment(Qt.AlignHCenter)
        self._browse_btn = QPushButton("选择文件…")
        self._browse_btn.setObjectName("ghostPill")
        self._browse_btn.setCursor(Qt.PointingHandCursor)
        self._browse_btn.clicked.connect(self.browse)
        self._rec_btn = QPushButton("●  实时录音")
        self._rec_btn.setObjectName("primaryPill")
        self._rec_btn.setCursor(Qt.PointingHandCursor)
        self._rec_btn.clicked.connect(self._request_record)
        btn_row.addWidget(self._browse_btn)
        btn_row.addWidget(self._rec_btn)
        drop.addLayout(btn_row)

        # source segmented control
        seg_wrap = QFrame()
        seg_wrap.setObjectName("segWrap")
        seg_lay = QHBoxLayout(seg_wrap)
        seg_lay.setContentsMargins(3, 3, 3, 3)
        seg_lay.setSpacing(2)
        self._src_group = QButtonGroup(self)
        self._src_buttons: dict[str, QPushButton] = {}
        for key, label in SOURCES:
            b = QPushButton(label)
            b.setObjectName("seg")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.clicked.connect(lambda _=False, k=key: self._on_source(k))
            self._src_group.addButton(b)
            self._src_buttons[key] = b
            seg_lay.addWidget(b)
        self._src_buttons["mic"].setChecked(True)
        drop.addWidget(seg_wrap, alignment=Qt.AlignHCenter)

        # mic device picker
        self._mic_combo = NoScrollComboBox()
        self._mic_combo.setFixedWidth(280)
        self._mic_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self._refresh_devices()
        drop.addWidget(self._mic_combo, alignment=Qt.AlignHCenter)

        root.addWidget(self._drop_card, alignment=Qt.AlignHCenter)

        # ---- selected-file card ----
        self._file_card = QFrame()
        self._file_card.setObjectName("glassCardAccent")
        self._file_card.setFixedWidth(640)
        fc = QVBoxLayout(self._file_card)
        fc.setContentsMargins(40, 30, 40, 28)
        fc.setSpacing(10)
        fc.setAlignment(Qt.AlignHCenter)

        fglyph = WaveGlyph()
        fc.addWidget(fglyph, alignment=Qt.AlignHCenter)
        self._file_name = QLabel("")
        self._file_name.setStyleSheet("font-size: 16px; font-weight: 700;")
        self._file_name.setAlignment(Qt.AlignCenter)
        fc.addWidget(self._file_name)
        self._file_meta = QLabel("")
        self._file_meta.setProperty("mono", True)
        self._file_meta.setAlignment(Qt.AlignCenter)
        fc.addWidget(self._file_meta)

        act_row = QHBoxLayout()
        act_row.setSpacing(10)
        act_row.setAlignment(Qt.AlignHCenter)
        self._start_btn = QPushButton("开 始 转 写")
        self._start_btn.setObjectName("primaryPill")
        self._start_btn.setCursor(Qt.PointingHandCursor)
        self._start_btn.setMinimumWidth(180)
        self._start_btn.clicked.connect(self._start)
        remove_btn = QPushButton("移除")
        remove_btn.setObjectName("outlinePill")
        remove_btn.setCursor(Qt.PointingHandCursor)
        remove_btn.clicked.connect(self.clear_file)
        act_row.addWidget(self._start_btn)
        act_row.addWidget(remove_btn)
        fc.addLayout(act_row)

        self._file_card.setVisible(False)
        root.addWidget(self._file_card, alignment=Qt.AlignHCenter)

        # ---- mode cards ----
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        cards_row.setAlignment(Qt.AlignHCenter)
        self._mode_cards: dict[str, ModeCard] = {}
        descs = {
            "general": "最高精度、忠实原文，中英混合自动识别",
            "classroom": "降噪 · 聚焦主讲人 · 自动提取重点",
            "ielts": "区分教官/考生 · 标注读音与语法疑点",
        }
        for m in MODES:
            card = ModeCard(m.key, m.label, descs.get(m.key, m.description))
            card.clicked.connect(self._on_mode)
            self._mode_cards[m.key] = card
            cards_row.addWidget(card)
        root.addLayout(cards_row)

        # ---- settings row ----
        srow = QHBoxLayout()
        srow.setSpacing(18)
        srow.setAlignment(Qt.AlignHCenter)

        lbl_model = QLabel("识别模型")
        lbl_model.setProperty("tone", "hint")
        self._model_combo = NoScrollComboBox()
        for value, label in MODELS:
            self._model_combo.addItem(label, userData=value)
        self._model_combo.currentIndexChanged.connect(self._save_prefs)
        srow.addWidget(lbl_model)
        srow.addWidget(self._model_combo)

        srow.addWidget(self._divider())

        lbl_fmt = QLabel("导出")
        lbl_fmt.setProperty("tone", "hint")
        srow.addWidget(lbl_fmt)
        self._chips: dict[str, QPushButton] = {}
        for ext in ("txt", "md", "doc", "docx"):
            chip = QPushButton("." + ext)
            chip.setObjectName("chip")
            chip.setCheckable(True)
            chip.setCursor(Qt.PointingHandCursor)
            chip.toggled.connect(self._on_chip)
            self._chips[ext] = chip
            srow.addWidget(chip)

        srow.addWidget(self._divider())

        from PySide6.QtWidgets import QCheckBox
        self._cb_llm = QCheckBox("本地大模型增强")
        self._cb_llm.stateChanged.connect(self._save_prefs)
        self._cb_llm.setToolTip(
            "中文用中文模型、英文用英文模型（自动判断）。全程本地离线，"
            "未安装 Ollama 时自动跳过。")
        srow.addWidget(self._cb_llm)

        root.addLayout(srow)

    @staticmethod
    def _divider() -> QFrame:
        d = QFrame()
        d.setObjectName("hairline")
        d.setFixedSize(1, 16)
        return d

    # ── devices ─────────────────────────────────────────────

    def _refresh_devices(self) -> None:
        self._mic_combo.blockSignals(True)
        self._mic_combo.clear()
        devices = QMediaDevices.audioInputs()
        default = QMediaDevices.defaultAudioInput()
        default_idx = 0
        for i, dev in enumerate(devices):
            self._mic_combo.addItem(dev.description(), userData=dev)
            if dev.id() == default.id():
                default_idx = i
        self._mic_combo.setCurrentIndex(default_idx)
        self._mic_combo.blockSignals(False)

    def _on_source(self, key: str) -> None:
        self._src_buttons[key].setChecked(True)
        self._mic_combo.setVisible(key in ("mic", "both"))

    def _current_source(self) -> str:
        for key, b in self._src_buttons.items():
            if b.isChecked():
                return key
        return "mic"

    # ── mode / settings ─────────────────────────────────────

    def _on_mode(self, key: str) -> None:
        for k, card in self._mode_cards.items():
            card.set_selected(k == key)
        self._save_prefs()

    def current_mode(self) -> str:
        for k, card in self._mode_cards.items():
            if card.property("selected") == "true":
                return k
        return "general"

    def _on_chip(self, _checked: bool) -> None:
        for ext, chip in self._chips.items():
            chip.setText(("✓ ." if chip.isChecked() else ".") + ext)
        self._save_prefs()

    def get_settings(self) -> dict:
        formats = [e for e, chip in self._chips.items() if chip.isChecked()] \
            or ["txt", "md"]
        return {
            "mode": self.current_mode(),
            "model": self._model_combo.currentData(),
            "formats": formats,
            "use_llm": self._cb_llm.isChecked(),
            "llm_model": "llama3.1:8b",
            "chinese_model": "qwen-zh:7b",
        }

    def _save_prefs(self) -> None:
        s = self.get_settings()
        self._prefs.setValue("mode", s["mode"])
        self._prefs.setValue("model", s["model"])
        self._prefs.setValue("formats", s["formats"])
        self._prefs.setValue("use_llm", s["use_llm"])

    def _load_prefs(self) -> None:
        mode = self._prefs.value("mode", "general")
        self._on_mode(mode if mode in self._mode_cards else "general")

        model = self._prefs.value("model", "large-v3")
        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == model:
                self._model_combo.setCurrentIndex(i)
                break

        fmts = self._prefs.value("formats", ["txt", "md", "docx"])
        if isinstance(fmts, str):
            fmts = [fmts]
        for ext, chip in self._chips.items():
            chip.setChecked(ext in fmts)
        self._on_chip(False)

        self._cb_llm.setChecked(self._prefs.value("use_llm", False, type=bool))
        self._on_source(self._current_source())

    # ── file selection ──────────────────────────────────────

    def browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "选择音频文件", str(Path.home() / "Desktop"),
            "Audio Files (*.m4a *.mp3 *.wav *.flac *.aac *.ogg *.opus *.webm);;"
            "All Files (*)")
        if path:
            self.set_file(path)

    def set_file(self, path: str, meta: str | None = None) -> None:
        self._selected_path = path
        p = Path(path)
        if meta is None:
            size_mb = p.stat().st_size / 1_048_576
            meta = f"{size_mb:.1f} MB"
        self._file_name.setText(p.name)
        self._file_meta.setText(meta)
        self._drop_card.setVisible(False)
        self._file_card.setVisible(True)

    def clear_file(self) -> None:
        self._selected_path = None
        self._file_card.setVisible(False)
        self._drop_card.setVisible(True)

    @property
    def selected_path(self) -> str | None:
        return self._selected_path

    # ── actions ─────────────────────────────────────────────

    def _request_record(self) -> None:
        src = self._current_source()
        dev = self._mic_combo.currentData() if src in ("mic", "both") else None
        self.record_requested.emit(src, dev)

    def _start(self) -> None:
        if self._selected_path:
            self.process_requested.emit(self._selected_path, self.get_settings())

    # ── drag & drop ─────────────────────────────────────────

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if Path(url.toLocalFile()).suffix.lower() in SUPPORTED_EXTENSIONS:
                    event.acceptProposedAction()
                    self._drop_card.setProperty("drop", "hover")
                    theme.repolish(self._drop_card)
                    return
        event.ignore()

    def dragLeaveEvent(self, _event) -> None:
        self._drop_card.setProperty("drop", "")
        theme.repolish(self._drop_card)

    def dropEvent(self, event: QDropEvent) -> None:
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if Path(path).suffix.lower() in SUPPORTED_EXTENSIONS:
                self.set_file(path)
                break
        self._drop_card.setProperty("drop", "")
        theme.repolish(self._drop_card)
