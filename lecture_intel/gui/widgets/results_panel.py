"""
Results preview panel.

Displays output files in tabs (one per format), with copy-to-clipboard
and open-folder actions.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QClipboard, QFont, QGuiApplication
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class ResultsPanel(QWidget):
    """Tabbed preview of generated output files."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._output_dir: str | None = None
        self._files: dict[str, str] = {}  # ext → absolute path
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Status bar
        self._status_bar = QLabel("暂无结果")
        self._status_bar.setStyleSheet("color: #8E8E93; font-size: 12px;")
        layout.addWidget(self._status_bar)

        # Tab widget for per-format preview
        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        layout.addWidget(self._tabs)

        # Bottom buttons
        btn_layout = QHBoxLayout()

        self._btn_folder = QPushButton("打开输出文件夹")
        self._btn_copy = QPushButton("复制到剪贴板")
        self._btn_folder.setEnabled(False)
        self._btn_copy.setEnabled(False)

        self._btn_folder.clicked.connect(self._open_folder)
        self._btn_copy.clicked.connect(self._copy_current)

        btn_layout.addWidget(self._btn_folder)
        btn_layout.addWidget(self._btn_copy)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

    # ── Public API ────────────────────────────────────────────

    def load_results(self, result_info: dict) -> None:
        """Populate preview tabs from a completed pipeline result dict."""
        self._output_dir = result_info["output_dir"]
        files = result_info.get("files", [])
        stats = result_info.get("stats", {})
        ielts = result_info.get("ielts")
        classroom = result_info.get("classroom")
        mode = result_info.get("mode", "general")

        # Index files by extension (skip the special report .md, shown in a tab)
        self._files = {}
        for f in files:
            name = Path(f).name
            if name.endswith((".ielts.md", ".summary.md", ".corrected.md")):
                continue
            ext = Path(f).suffix.lower().lstrip(".")
            self._files[ext] = f

        # Update status bar
        total_s = stats.get("total_time_s", 0)
        lang = stats.get("language", "?")
        dur = stats.get("duration_sec", 0)
        bits = [f"模式：{mode}", f"语言：{lang}", f"时长 {dur:.0f}s",
                f"用时 {total_s:.0f}s", f"{len(self._files)} 个文件"]
        if ielts:
            bits.append(f"发音疑点 {ielts.get('pron_issue_count', 0)} · "
                        f"语法 {ielts.get('grammar_issue_count', 0)} · "
                        f"{ielts.get('wpm', 0):.0f} WPM")
        if classroom:
            bits.append(f"重点 {classroom.get('emphasis_count', 0)} · "
                        f"定义 {classroom.get('definition_count', 0)}")
        self._status_bar.setText("  ·  ".join(bits))

        # Clear old tabs
        while self._tabs.count():
            self._tabs.removeTab(0)

        # Special report tab first (rendered markdown)
        if ielts and ielts.get("markdown"):
            self._add_text_tab("📋 雅思反馈", ielts["markdown"], mono=False)
        if classroom and classroom.get("markdown"):
            title = "📋 重点总结" + ("（AI）" if classroom.get("llm") else "")
            self._add_text_tab(title, classroom["markdown"], mono=False)
        tidy = stats.get("tidy_markdown")
        if tidy:
            self._add_text_tab("📋 AI校对版", tidy, mono=False)

        # One tab per format
        for ext in ("md", "txt", "srt", "json"):
            if ext in self._files:
                self._add_tab(ext, self._files[ext])

        self._btn_folder.setEnabled(True)
        self._btn_copy.setEnabled(True)

    def _add_text_tab(self, title: str, content: str, mono: bool = False) -> None:
        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(content)
        editor.setFont(QFont("Menlo" if mono else "SF Pro Text", 12 if mono else 13))
        editor.setStyleSheet("border: none; background: transparent;")
        self._tabs.addTab(editor, title)

    def reset(self) -> None:
        """Clear all previews and restore empty state."""
        while self._tabs.count():
            self._tabs.removeTab(0)
        self._status_bar.setText("暂无结果")
        self._btn_folder.setEnabled(False)
        self._btn_copy.setEnabled(False)

    # ── Internal ──────────────────────────────────────────────

    def _add_tab(self, ext: str, filepath: str) -> None:
        try:
            content = Path(filepath).read_text(encoding="utf-8")
        except Exception:
            content = f"[无法读取 {filepath}]"

        editor = QPlainTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(content)

        if ext in {"json", "srt"}:
            editor.setFont(QFont("Menlo", 12))
        else:
            editor.setFont(QFont("SF Pro Text", 13))

        editor.setStyleSheet("border: none; background: transparent;")
        self._tabs.addTab(editor, f".{ext}")

    def _open_folder(self) -> None:
        if self._output_dir:
            subprocess.run(["open", self._output_dir])

    def _copy_current(self) -> None:
        current = self._tabs.currentWidget()
        if isinstance(current, QPlainTextEdit):
            QGuiApplication.clipboard().setText(current.toPlainText())
