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

from core.i18n import t
from core.languages import PICKER_LANGUAGES
from core.modes import GENERAL, CLASSROOM, IELTS
from gui import theme
from gui.widgets.common import NoScrollComboBox
from gui.widgets.visuals import WaveGlyph

SUPPORTED_EXTENSIONS: set[str] = {
    ".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".webm",
}

MODES = [GENERAL, CLASSROOM, IELTS]
# (value, i18n key) — labels resolve through t() so they follow the UI language.
MODELS = [
    ("large-v3", "model_large"),
    ("large-v3-turbo", "model_turbo"),
    ("small", "model_small"),
]
SOURCES = [("mic", "source_mic"), ("system", "source_system"),
           ("both", "source_both")]


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

    def set_texts(self, title: str, desc: str) -> None:
        self._title.setText(title)
        self._desc.setText(desc)

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

        self._drop_title = QLabel(t("home_drop_title"))
        self._drop_title.setStyleSheet("font-size: 15px; font-weight: 600;")
        drop.addWidget(self._drop_title, alignment=Qt.AlignHCenter)

        exts = QLabel("m4a · mp3 · wav · webm · flac · aac")
        exts.setProperty("mono", True)
        drop.addWidget(exts, alignment=Qt.AlignHCenter)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.setAlignment(Qt.AlignHCenter)
        self._browse_btn = QPushButton(t("home_browse"))
        self._browse_btn.setObjectName("ghostPill")
        self._browse_btn.setCursor(Qt.PointingHandCursor)
        self._browse_btn.clicked.connect(self.browse)
        self._rec_btn = QPushButton(t("home_record"))
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
        for key, label_key in SOURCES:
            b = QPushButton(t(label_key))
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
        self._start_btn = QPushButton(t("home_start"))
        self._start_btn.setObjectName("primaryPill")
        self._start_btn.setCursor(Qt.PointingHandCursor)
        self._start_btn.setMinimumWidth(180)
        self._start_btn.clicked.connect(self._start)
        self._remove_btn = QPushButton(t("home_remove"))
        self._remove_btn.setObjectName("outlinePill")
        self._remove_btn.setCursor(Qt.PointingHandCursor)
        self._remove_btn.clicked.connect(self.clear_file)
        act_row.addWidget(self._start_btn)
        act_row.addWidget(self._remove_btn)
        fc.addLayout(act_row)

        self._file_card.setVisible(False)
        root.addWidget(self._file_card, alignment=Qt.AlignHCenter)

        # ---- mode cards ----
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        cards_row.setAlignment(Qt.AlignHCenter)
        self._mode_cards: dict[str, ModeCard] = {}
        for m in MODES:
            card = ModeCard(m.key, t(f"mode_{m.key}_title"),
                            t(f"mode_{m.key}_desc"))
            card.clicked.connect(self._on_mode)
            self._mode_cards[m.key] = card
            cards_row.addWidget(card)
        root.addLayout(cards_row)

        # ---- settings row ----
        srow = QHBoxLayout()
        srow.setSpacing(18)
        srow.setAlignment(Qt.AlignHCenter)

        self._lbl_model = QLabel(t("home_model"))
        self._lbl_model.setProperty("tone", "hint")
        self._model_combo = NoScrollComboBox()
        for value, key in MODELS:
            self._model_combo.addItem(t(key), userData=value)
        self._model_combo.currentIndexChanged.connect(self._save_prefs)
        srow.addWidget(self._lbl_model)
        srow.addWidget(self._model_combo)

        srow.addWidget(self._divider())

        self._lbl_lang = QLabel(t("home_language"))
        self._lbl_lang.setProperty("tone", "hint")
        self._lang_combo = NoScrollComboBox()
        for value, label in PICKER_LANGUAGES:
            self._lang_combo.addItem(label, userData=value)
        self._lang_combo.setToolTip(t("home_lang_tooltip"))
        self._lang_combo.currentIndexChanged.connect(self._save_prefs)
        srow.addWidget(self._lbl_lang)
        srow.addWidget(self._lang_combo)

        srow.addWidget(self._divider())

        self._lbl_fmt = QLabel(t("home_export"))
        self._lbl_fmt.setProperty("tone", "hint")
        srow.addWidget(self._lbl_fmt)
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
        self._cb_llm = QCheckBox(t("home_llm"))
        self._cb_llm.stateChanged.connect(self._save_prefs)
        self._cb_llm.setToolTip(t("home_llm_tooltip"))
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
            "language": self._lang_combo.currentData(),
            "formats": formats,
            "use_llm": self._cb_llm.isChecked(),
            "llm_model": "llama3.1:8b",
            "chinese_model": "qwen-zh:7b",
        }

    def _save_prefs(self) -> None:
        s = self.get_settings()
        self._prefs.setValue("mode", s["mode"])
        self._prefs.setValue("model", s["model"])
        self._prefs.setValue("language", s["language"])
        self._prefs.setValue("formats", s["formats"])
        self._prefs.setValue("use_llm", s["use_llm"])

    def _load_prefs(self) -> None:
        # Read everything before applying any of it: applying the mode card
        # fires _save_prefs with the not-yet-restored widget state, which used
        # to overwrite the stored model/language/formats/LLM prefs with
        # defaults before they were read back — resetting them every launch.
        mode = self._prefs.value("mode", "general")
        model = self._prefs.value("model", "large-v3")
        lang = self._prefs.value("language", "auto")
        fmts = self._prefs.value("formats", ["txt", "md", "docx"])
        if isinstance(fmts, str):
            fmts = [fmts]
        use_llm = self._prefs.value("use_llm", False, type=bool)

        self._on_mode(mode if mode in self._mode_cards else "general")

        for i in range(self._model_combo.count()):
            if self._model_combo.itemData(i) == model:
                self._model_combo.setCurrentIndex(i)
                break

        for i in range(self._lang_combo.count()):
            if self._lang_combo.itemData(i) == lang:
                self._lang_combo.setCurrentIndex(i)
                break

        for ext, chip in self._chips.items():
            chip.setChecked(ext in fmts)
        self._on_chip(False)

        self._cb_llm.setChecked(use_llm)
        self._on_source(self._current_source())

    # ── file selection ──────────────────────────────────────

    def browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, t("home_browse_title"), str(Path.home() / "Desktop"),
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

    # ── live language switch ─────────────────────────

    def retranslate(self) -> None:
        self._drop_title.setText(t("home_drop_title"))
        self._browse_btn.setText(t("home_browse"))
        self._rec_btn.setText(t("home_record"))
        self._start_btn.setText(t("home_start"))
        self._remove_btn.setText(t("home_remove"))
        for key, label_key in SOURCES:
            if key in self._src_buttons:
                self._src_buttons[key].setText(t(label_key))
        for m in MODES:
            card = self._mode_cards.get(m.key)
            if card is not None:
                card.set_texts(t(f"mode_{m.key}_title"), t(f"mode_{m.key}_desc"))
        self._lbl_model.setText(t("home_model"))
        self._lbl_lang.setText(t("home_language"))
        self._lbl_fmt.setText(t("home_export"))
        self._lang_combo.setToolTip(t("home_lang_tooltip"))
        self._cb_llm.setText(t("home_llm"))
        self._cb_llm.setToolTip(t("home_llm_tooltip"))
        for i, (_value, key) in enumerate(MODELS):
            self._model_combo.setItemText(i, t(key))
