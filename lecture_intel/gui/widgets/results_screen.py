"""
ResultsScreen — two glass panes: transcript / report preview (left, tabbed)
and info or IELTS feedback rail (right).

API mirrors the old ResultsPanel: load_results(result), reset().
Signal: new_requested — user wants to start another transcription.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QGuiApplication
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from core.i18n import t
from core.languages import display_name
from gui import theme


class _GlassPane(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("glassCard")
        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(0, 0, 0, 0)
        self._lay.setSpacing(0)

        head = QWidget()
        hl = QHBoxLayout(head)
        hl.setContentsMargins(24, 16, 24, 14)
        self.title_lbl = QLabel(title)
        self.title_lbl.setStyleSheet("font-size: 14px; font-weight: 700;")
        self.meta_lbl = QLabel("")
        self.meta_lbl.setProperty("mono", True)
        hl.addWidget(self.title_lbl, stretch=1)
        hl.addWidget(self.meta_lbl)
        self._lay.addWidget(head)

        hr = QFrame()
        hr.setObjectName("hairline")
        hr.setFixedHeight(1)
        self._lay.addWidget(hr)

        self.body = QVBoxLayout()
        self.body.setContentsMargins(24, 16, 24, 16)
        self.body.setSpacing(12)
        body_host = QWidget()
        body_host.setLayout(self.body)
        self._lay.addWidget(body_host, stretch=1)

    def add_footer(self) -> QHBoxLayout:
        hr = QFrame()
        hr.setObjectName("hairline")
        hr.setFixedHeight(1)
        self._lay.addWidget(hr)
        foot = QWidget()
        fl = QHBoxLayout(foot)
        fl.setContentsMargins(24, 12, 24, 12)
        fl.setSpacing(8)
        self._lay.addWidget(foot)
        return fl


class ResultsScreen(QWidget):
    new_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_dir: str | None = None
        self._last_result: dict | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(28, 8, 28, 24)
        root.setSpacing(20)

        # ── left pane: tabbed previews ──
        self._left = _GlassPane(t("res_transcript"))
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._left.body.addWidget(self._tabs)

        foot = self._left.add_footer()
        self._btn_folder = QPushButton(t("res_open_folder"))
        self._btn_folder.setObjectName("primaryPill")
        self._btn_folder.setCursor(Qt.PointingHandCursor)
        self._btn_folder.clicked.connect(self._open_folder)
        self._btn_copy = QPushButton(t("res_copy"))
        self._btn_copy.setObjectName("outlinePill")
        self._btn_copy.setCursor(Qt.PointingHandCursor)
        self._btn_copy.clicked.connect(self._copy_current)
        self._btn_new = QPushButton(t("res_new"))
        self._btn_new.setObjectName("ghostPill")
        self._btn_new.setCursor(Qt.PointingHandCursor)
        self._btn_new.clicked.connect(self.new_requested)
        foot.addWidget(self._btn_folder)
        foot.addWidget(self._btn_copy)
        foot.addStretch()
        foot.addWidget(self._btn_new)

        root.addWidget(self._left, stretch=3)

        # ── right pane: info / feedback rail ──
        self._right = _GlassPane(t("res_info"))
        self._rail_scroll = QScrollArea()
        self._rail_scroll.setWidgetResizable(True)
        self._rail_scroll.setFrameShape(QScrollArea.NoFrame)
        self._rail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._rail_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        self._rail_host = QWidget()
        self._rail = QVBoxLayout(self._rail_host)
        self._rail.setContentsMargins(0, 0, 0, 0)
        self._rail.setSpacing(16)
        self._rail.setAlignment(Qt.AlignTop)
        self._rail_scroll.setWidget(self._rail_host)
        self._right.body.addWidget(self._rail_scroll)

        rfoot = self._right.add_footer()
        self._path_lbl = QLabel("")
        self._path_lbl.setProperty("mono", True)
        rfoot.addWidget(self._path_lbl)

        root.addWidget(self._right, stretch=2)

    # ── public API ──────────────────────────────────────────

    def load_results(self, result_info: dict) -> None:
        self._last_result = result_info
        self._output_dir = result_info["output_dir"]
        files = result_info.get("files", [])
        stats = result_info.get("stats", {})
        ielts = result_info.get("ielts")
        classroom = result_info.get("classroom")
        mode = result_info.get("mode", "general")

        file_by_ext: dict[str, str] = {}
        for f in files:
            name = Path(f).name
            if name.endswith((".ielts.md", ".summary.md", ".corrected.md")):
                continue
            file_by_ext[Path(f).suffix.lower().lstrip(".")] = f

        # left pane header + tabs
        self._left.title_lbl.setText(
            t("res_transcript_ielts") if mode == "ielts" else t("res_transcript"))
        lang = display_name(stats.get("language", "?"))
        dur = stats.get("duration_sec", 0)
        meta = f"{lang} · {dur / 60:.0f}:{dur % 60:02.0f}"
        if ielts:
            meta += f" · {ielts.get('wpm', 0):.0f} WPM"
        self._left.meta_lbl.setText(meta)

        while self._tabs.count():
            self._tabs.removeTab(0)
        if ielts and ielts.get("markdown"):
            self._add_text_tab(t("res_tab_ielts"), ielts["markdown"])
        if classroom and classroom.get("markdown"):
            title = (t("res_tab_summary_ai") if classroom.get("llm")
                     else t("res_tab_summary"))
            self._add_text_tab(title, classroom["markdown"])
        tidy = stats.get("tidy_markdown")
        if tidy:
            self._add_text_tab(t("res_tab_tidy"), tidy)
        for ext in ("md", "txt", "srt", "json"):
            if ext in file_by_ext:
                self._add_file_tab(ext, file_by_ext[ext])

        # right rail
        self._right.title_lbl.setText(t("res_feedback") if ielts else t("res_info"))
        self._clear_rail()
        c = theme.current_scheme()

        if ielts:
            stat_row = QHBoxLayout()
            stat_row.setSpacing(10)
            for value, label, key in (
                (str(ielts.get("pron_issue_count", 0)), t("res_stat_pron"), "danger"),
                (str(ielts.get("grammar_issue_count", 0)), t("res_stat_grammar"), "warn"),
                (f"{ielts.get('wpm', 0):.0f}", t("res_stat_wpm"), "ok"),
            ):
                cell = QFrame()
                cell.setObjectName("modeCard")
                cl = QVBoxLayout(cell)
                cl.setContentsMargins(12, 10, 12, 10)
                cl.setSpacing(2)
                v = QLabel(value)
                v.setFont(theme.display_font(18, QFont.Weight.Medium))
                v.setStyleSheet(f"color: {c[key]};")
                t = QLabel(label)
                t.setStyleSheet(f"font-size: 11px; color: {c['ink2']};")
                cl.addWidget(v)
                cl.addWidget(t)
                stat_row.addWidget(cell)
            host = QWidget()
            host.setLayout(stat_row)
            self._rail.addWidget(host)

        mode_title = (t(f"mode_{mode}_title")
                      if mode in ("general", "classroom", "ielts") else mode)
        self._add_rail_kv(t("res_key_mode"), mode_title)
        self._add_rail_kv(t("res_key_language"), str(lang))
        self._add_rail_kv(t("res_key_duration"), f"{dur:.0f}s")
        total_s = stats.get("total_time_s", 0)
        self._add_rail_kv(t("res_key_elapsed"), f"{total_s:.0f}s")
        if classroom:
            self._add_rail_kv(t("res_key_emphasis"), str(classroom.get("emphasis_count", 0)))
            self._add_rail_kv(t("res_key_definitions"), str(classroom.get("definition_count", 0)))

        names = QLabel("\n".join(Path(f).name for f in files))
        names.setProperty("mono", True)
        names.setWordWrap(True)
        self._rail.addWidget(self._rail_section(t("res_section_files")))
        self._rail.addWidget(names)

        self._path_lbl.setText(t("res_output", dir=str(self._output_dir)))
        self._btn_folder.setEnabled(True)
        self._btn_copy.setEnabled(True)

    def reset(self) -> None:
        self._last_result = None
        while self._tabs.count():
            self._tabs.removeTab(0)
        self._clear_rail()
        self._path_lbl.setText("")
        self._left.title_lbl.setText(t("res_transcript"))
        self._right.title_lbl.setText(t("res_info"))

    # ── helpers ─────────────────────────────────────────────

    def _rail_section(self, text: str) -> QLabel:
        lbl = QLabel(text)
        c = theme.current_scheme()
        lbl.setStyleSheet(
            f"font-size: 11px; letter-spacing: 2px; color: {c['ink3']};")
        return lbl

    def _add_rail_kv(self, key: str, value: str) -> None:
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        c = theme.current_scheme()
        k = QLabel(key)
        k.setStyleSheet(f"font-size: 12px; color: {c['ink3']};")
        v = QLabel(value)
        v.setStyleSheet(f"font-size: 12px; color: {c['ink2']};")
        v.setAlignment(Qt.AlignRight)
        rl.addWidget(k)
        rl.addWidget(v, stretch=1)
        self._rail.addWidget(row)

    def _clear_rail(self) -> None:
        while self._rail.count():
            item = self._rail.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _add_text_tab(self, title: str, content: str) -> None:
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(content)
        editor.setFont(QFont("SF Pro Text", 13))
        self._tabs.addTab(editor, title)

    def _add_file_tab(self, ext: str, filepath: str) -> None:
        try:
            content = Path(filepath).read_text(encoding="utf-8")
        except Exception:
            content = t("res_read_error", path=filepath)
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(content)
        editor.setFont(QFont("Menlo" if ext in {"json", "srt"} else "SF Pro Text",
                             12 if ext in {"json", "srt"} else 13))
        self._tabs.addTab(editor, f".{ext}")

    def _open_folder(self) -> None:
        if self._output_dir:
            subprocess.run(["open", self._output_dir])

    def _copy_current(self) -> None:
        current = self._tabs.currentWidget()
        if isinstance(current, QPlainTextEdit):
            QGuiApplication.clipboard().setText(current.toPlainText())

    # ── live language switch ─────────────────────────

    def retranslate(self) -> None:
        self._btn_folder.setText(t("res_open_folder"))
        self._btn_copy.setText(t("res_copy"))
        self._btn_new.setText(t("res_new"))
        if self._last_result is not None:
            keep = self._tabs.currentIndex()
            self.load_results(self._last_result)
            if 0 <= keep < self._tabs.count():
                self._tabs.setCurrentIndex(keep)
        else:
            self._left.title_lbl.setText(t("res_transcript"))
            self._right.title_lbl.setText(t("res_info"))
