"""
ResultsScreen — two glass panes: transcript / report preview (left, tabbed)
and info or IELTS feedback rail (right).

API mirrors the old ResultsPanel: load_results(result), reset().
Signal: new_requested — user wants to start another transcription.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal, QUrl
from PySide6.QtGui import QDesktopServices, QFont, QGuiApplication
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


class _ElidedLabel(QLabel):
    """One-line label that elides in the middle instead of demanding its full
    width. A plain QLabel makes a long output path the pane's minimum width,
    which squeezes the other pane until its footer buttons clip."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._full = ""

    def setText(self, text: str) -> None:  # noqa: N802 (Qt naming)
        self._full = text
        self.setToolTip(text)
        self._elide()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._elide()

    def minimumSizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(0, self.fontMetrics().height() + 4)

    def sizeHint(self) -> QSize:  # noqa: N802 (Qt naming)
        return QSize(self.fontMetrics().horizontalAdvance(self._full),
                     self.fontMetrics().height() + 4)

    def _elide(self) -> None:
        super().setText(self.fontMetrics().elidedText(
            self._full, Qt.ElideMiddle, self.width()))


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
        self._rail_scroll.setStyleSheet(
            "QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        self._rail_scroll.viewport().setAutoFillBackground(False)
        self._rail_host = QWidget()
        self._rail_host.setAutoFillBackground(False)
        self._rail = QVBoxLayout(self._rail_host)
        self._rail.setSpacing(4)
        self._rail.setAlignment(Qt.AlignTop)
        self._rail.setContentsMargins(0, 0, 8, 0)
        self._rail_scroll.setWidget(self._rail_host)
        self._right.body.addWidget(self._rail_scroll)

        rfoot = self._right.add_footer()
        self._path_lbl = _ElidedLabel()
        self._path_lbl.setProperty("mono", True)
        rfoot.addWidget(self._path_lbl, stretch=1)

        root.addWidget(self._right, stretch=2)

    # ── public API ──────────────────────────────────────────

    def load_results(self, result_info: dict) -> None:
        self._last_result = result_info
        self._output_dir = result_info["output_dir"]
        files = result_info.get("files", [])
        stats = result_info.get("stats", {})
        ielts = result_info.get("ielts")
        classroom = result_info.get("classroom")
        fidelity = result_info.get("fidelity")
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
            self._add_text_tab(t("res_tab_summary"), classroom["markdown"])
        # The fidelity audit trail: what the repeat arbitration flagged, and
        # which spans actually lost text. Same lines as the exported md/txt.
        if fidelity and fidelity.get("total"):
            self._add_text_tab(t("res_tab_fidelity"),
                               "\n".join(fidelity.get("lines") or []))
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
                # NOTE: never name these `t` — that shadows the i18n `t()`
                # imported at module level, which makes it a function-local
                # everywhere in load_results() and breaks every other t(...)
                # call in it with UnboundLocalError.
                val_lbl = QLabel(value)
                val_lbl.setFont(theme.display_font(18, QFont.Weight.Medium))
                val_lbl.setStyleSheet(f"color: {c[key]};")
                key_lbl = QLabel(label)
                key_lbl.setStyleSheet(f"font-size: 11px; color: {c['ink2']};")
                cl.addWidget(val_lbl)
                cl.addWidget(key_lbl)
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
        if stats.get("model"):
            self._add_rail_kv(t("res_key_model"), str(stats["model"]))
        warnings = stats.get("warnings") or []
        if warnings:
            self._rail.addWidget(self._rail_section(t("res_key_warnings")))
            warning_label = QLabel("\n".join(str(item) for item in warnings))
            warning_label.setWordWrap(True)
            warning_label.setProperty("tone", "danger")
            self._rail.addWidget(warning_label)
        if classroom:
            self._add_rail_kv(t("res_key_emphasis"), str(classroom.get("emphasis_count", 0)))
            self._add_rail_kv(t("res_key_definitions"), str(classroom.get("definition_count", 0)))
        if fidelity:
            self._add_fidelity_card(fidelity, c)

        self._rail.addWidget(self._rail_section(t("res_section_files")))
        self._add_files_card([Path(f).name for f in files])

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

    def _rail_section(self, text: str) -> QWidget:
        """Small caps-style heading with breathing room above it."""
        host = QWidget()
        hl = QVBoxLayout(host)
        hl.setContentsMargins(0, 8, 0, 0)
        lbl = QLabel(text)
        c = theme.current_scheme()
        lbl.setStyleSheet(
            f"font-size: 11px; font-weight: 700; letter-spacing: 1px; "
            f"color: {c['ink3']};")
        hl.addWidget(lbl)
        return host

    def _add_rail_kv(self, key: str, value: str, *, last: bool = False) -> None:
        row = QWidget()
        rl = QVBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)
        line = QHBoxLayout()
        line.setContentsMargins(0, 4, 0, 4)
        c = theme.current_scheme()
        k = QLabel(key)
        k.setStyleSheet(f"font-size: 12px; color: {c['ink3']};")
        v = _ElidedLabel()
        v.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {c['ink']};")
        v.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        v.setText(value)
        line.addWidget(k)
        line.addWidget(v, stretch=1)
        rl.addLayout(line)
        if not last:
            hr = QFrame()
            hr.setObjectName("hairline")
            hr.setFixedHeight(1)
            rl.addWidget(hr)
        self._rail.addWidget(row)

    def _add_files_card(self, names: list[str]) -> None:
        card = QFrame()
        card.setObjectName("modeCard")
        cl = QVBoxLayout(card)
        cl.setContentsMargins(12, 8, 12, 8)
        cl.setSpacing(2)
        for name in names:
            lbl = _ElidedLabel()
            lbl.setProperty("mono", True)
            lbl.setText(name)
            cl.addWidget(lbl)
        self._rail.addWidget(card)

    def _add_fidelity_card(self, fid: dict, c: dict) -> None:
        """At-a-glance counts for the fidelity layer.

        Until this existed the annotations reached only meta.json and the
        optional json export, so a user could not tell that e.g. "打打打打" had
        been flagged — or folded — by the classroom mode. The prose for every
        entry lives in the tab, produced by the same formatter the exported
        md/txt use (``core.export.fidelity_lines``).
        """
        self._rail.addWidget(self._rail_section(t("res_section_fidelity")))
        if not fid.get("total"):
            card = QFrame()
            card.setObjectName("modeCard")
            cl = QVBoxLayout(card)
            cl.setContentsMargins(12, 7, 12, 7)
            none_lbl = QLabel(t("res_fid_none"))
            none_lbl.setWordWrap(True)
            none_lbl.setStyleSheet(f"font-size: 12px; color: {c['ok']};")
            cl.addWidget(none_lbl)
            self._rail.addWidget(card)
            return
        row = QHBoxLayout()
        row.setSpacing(10)
        for value, label, key in (
            (str(fid.get("asr_loop", 0)), t("res_fid_artifact"), "danger"),
            (str(fid.get("real_speech", 0)), t("res_fid_real"), "ok"),
            (str(fid.get("uncertain", 0)), t("res_fid_uncertain"), "warn"),
        ):
            cell = QFrame()
            cell.setObjectName("modeCard")
            cl = QVBoxLayout(cell)
            cl.setContentsMargins(12, 10, 12, 10)
            cl.setSpacing(2)
            val_lbl = QLabel(value)
            val_lbl.setFont(theme.display_font(18, QFont.Weight.Medium))
            val_lbl.setStyleSheet(f"color: {c[key]};")
            key_lbl = QLabel(label)
            key_lbl.setStyleSheet(f"font-size: 11px; color: {c['ink2']};")
            cl.addWidget(val_lbl)
            cl.addWidget(key_lbl)
            row.addWidget(cell)
        host = QWidget()
        host.setLayout(row)
        self._rail.addWidget(host)

        if fid.get("folded_count"):
            self._add_rail_kv(t("res_fid_folded"), str(fid["folded_count"]))
        if fid.get("dropped_count"):
            self._add_rail_kv(t("res_fid_dropped"), str(fid["dropped_count"]))
        hint = QLabel(t("res_fid_hint", tab=t("res_tab_fidelity")))
        hint.setWordWrap(True)
        hint.setStyleSheet(f"font-size: 11px; color: {c['ink3']};")
        self._rail.addWidget(hint)

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
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._output_dir))

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
