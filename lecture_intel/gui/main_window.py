"""
Main window — assembles all widgets and manages the PipelineWorker lifecycle.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from gui.widgets.input_panel import InputPanel
from gui.widgets.progress_panel import ProgressPanel
from gui.widgets.results_panel import ResultsPanel
from gui.widgets.settings_panel import SettingsPanel
from gui.workers.pipeline_worker import PipelineWorker

# The steps the engine emits, per mode.
_STEPS_BY_MODE: dict[str, list[str]] = {
    "general":   ["load", "asr", "export"],
    "classroom": ["load", "denoise", "asr", "diarize", "export"],
    "ielts":     ["load", "asr", "diarize", "analyze", "export"],
}


class MainWindow(QMainWindow):
    """Top-level application window."""

    def __init__(self) -> None:
        super().__init__()
        self._worker: PipelineWorker | None = None
        self._prefs = QSettings("LucasLab", "Recorder")

        self.setWindowTitle("Recorder · 录音转文字")
        self.setMinimumSize(920, 680)
        self._restore_geometry()
        self._build_menu()
        self._build_ui()
        self._build_status_bar()

    # ── Menu bar ──────────────────────────────────────────────

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        # File
        file_menu = menu_bar.addMenu("文件")

        open_act = QAction("打开录音…", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(self._browse_file)
        file_menu.addAction(open_act)

        file_menu.addSeparator()

        reveal_act = QAction("显示输出文件夹", self)
        reveal_act.setShortcut("Cmd+Shift+O")
        reveal_act.triggered.connect(self._reveal_output)
        file_menu.addAction(reveal_act)

        # View
        view_menu = menu_bar.addMenu("视图")

        reset_act = QAction("重置", self)
        reset_act.setShortcut("Cmd+R")
        reset_act.triggered.connect(self._reset)
        view_menu.addAction(reset_act)

        # Help
        help_menu = menu_bar.addMenu("帮助")

        docs_act = QAction("打开说明", self)
        docs_act.triggered.connect(self._open_readme)
        help_menu.addAction(docs_act)

    # ── Central UI ────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)  # Use VBox for mobile-friendly layout
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(0)

        # --- Top bar: process button ---
        self._process_btn = QPushButton("开始转写")
        self._process_btn.setFixedHeight(40)
        self._process_btn.setEnabled(False)
        self._process_btn.setStyleSheet("""
            QPushButton {
                background: #007AFF; color: white;
                border-radius: 8px; font-size: 14px; font-weight: 500;
            }
            QPushButton:hover   { background: #0066DD; }
            QPushButton:pressed { background: #0055BB; }
            QPushButton:disabled { background: #C7C7CC; }
        """)
        self._process_btn.clicked.connect(self._start_processing)

        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.setFixedHeight(32)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.setStyleSheet("""
            QPushButton {
                background: #FF3B30; color: white;
                border-radius: 6px; font-size: 13px;
            }
            QPushButton:hover { background: #E02020; }
        """)
        self._cancel_btn.clicked.connect(self._cancel_processing)

        # Horizontal splitter: left (input+settings) | right (progress+results)
        splitter = QSplitter(Qt.Horizontal)

        # -- Left column --
        left_col = QWidget()
        left_col.setFixedWidth(300)
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        self._input_panel = InputPanel()
        self._input_panel.file_ready.connect(self._on_file_selected)
        left_layout.addWidget(self._input_panel)

        self._settings = SettingsPanel()
        left_layout.addWidget(self._settings)

        left_layout.addWidget(self._process_btn)
        left_layout.addWidget(self._cancel_btn)
        left_layout.addStretch()

        splitter.addWidget(left_col)

        # -- Right column --
        right_splitter = QSplitter(Qt.Vertical)

        self._progress = ProgressPanel()
        right_splitter.addWidget(self._progress)

        self._results = ResultsPanel()
        right_splitter.addWidget(self._results)

        right_splitter.setSizes([220, 480])
        splitter.addWidget(right_splitter)

        root.addWidget(splitter)

    def _build_status_bar(self) -> None:
        self._status_bar = self.statusBar()
        self._status_bar.showMessage("就绪")

    # ── File selection ────────────────────────────────────────

    def _on_file_selected(self, path: str) -> None:
        self._process_btn.setEnabled(True)
        self._status_bar.showMessage(f"已选择：{Path(path).name}")

    def _browse_file(self) -> None:
        """Switch to File tab and trigger its Browse dialog."""
        self._input_panel._tabs.setCurrentIndex(1)
        self._input_panel._file_tab._browse()

    # ── Processing lifecycle ──────────────────────────────────

    def _start_processing(self) -> None:
        input_path = self._input_panel.selected_path
        if not input_path:
            return

        settings = self._settings.get_settings()
        output_dir = str(Path(input_path).parent / "recorder_output")
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        active_steps = _STEPS_BY_MODE.get(settings["mode"], _STEPS_BY_MODE["general"])

        self._progress.reset(active_steps)
        self._results.reset()

        # Toggle buttons
        self._process_btn.setEnabled(False)
        self._process_btn.setVisible(False)
        self._cancel_btn.setVisible(True)
        self._status_bar.showMessage("处理中…")

        # Launch worker
        self._worker = PipelineWorker(
            input_path=input_path,
            output_dir=output_dir,
            settings=settings,
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, info: dict) -> None:
        self._progress.update_progress(info)
        self._status_bar.showMessage(info.get("message", ""))

    def _on_finished(self, result: dict) -> None:
        total_s = result.get("stats", {}).get("total_time_s", 0)
        self._progress.set_total_time(total_s)
        self._results.load_results(result)

        self._cancel_btn.setVisible(False)
        self._process_btn.setVisible(True)
        self._process_btn.setEnabled(True)
        self._status_bar.showMessage(
            f"✓ 完成，用时 {total_s:.0f}s  ·  输出：{result['output_dir']}"
        )

        self._send_notification(
            "Recorder",
            f"转写完成 — {Path(result['output_dir']).name}",
        )

    def _on_error(self, msg: str) -> None:
        self._cancel_btn.setVisible(False)
        self._process_btn.setVisible(True)
        self._process_btn.setEnabled(True)
        self._status_bar.showMessage("发生错误")
        QMessageBox.critical(self, "处理出错", msg)

    def _cancel_processing(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(3000)
            self._cancel_btn.setVisible(False)
            self._process_btn.setVisible(True)
            self._process_btn.setEnabled(True)
            self._status_bar.showMessage("已取消")

    # ── Utilities ─────────────────────────────────────────────

    def _reset(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._input_panel.reset()
        self._results.reset()
        self._process_btn.setEnabled(False)
        self._status_bar.showMessage("就绪")

    def _reveal_output(self) -> None:
        path = self._input_panel.selected_path
        if path:
            output_dir = Path(path).parent / "recorder_output"
        else:
            output_dir = Path.home() / "Desktop"
        output_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["open", str(output_dir)])

    def _open_readme(self) -> None:
        readme = Path(__file__).parent.parent / "README.md"
        if readme.exists():
            subprocess.run(["open", str(readme)])

    @staticmethod
    def _send_notification(title: str, message: str) -> None:
        script = (
            f'display notification "{message}"'
            f' with title "{title}"'
            f' sound name "Glass"'
        )
        subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
        )

    # ── Window geometry persistence ───────────────────────────

    def _restore_geometry(self) -> None:
        geometry = self._prefs.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.resize(960, 720)

    def closeEvent(self, event) -> None:
        self._prefs.setValue("geometry", self.saveGeometry())
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        super().closeEvent(event)
