"""
Main window — assembles all widgets and manages the PipelineWorker lifecycle.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import Qt, QSettings
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from gui import theme

from gui.widgets.input_panel import InputPanel
from gui.widgets.progress_panel import ProgressPanel
from gui.widgets.results_panel import ResultsPanel
from gui.widgets.settings_panel import SettingsPanel
from gui.workers.pipeline_worker import PipelineWorker

# The steps the engine emits, per mode.
_STEPS_BY_MODE: dict[str, list[str]] = {
    "general":   ["load", "asr", "export"],
    "classroom": ["load", "denoise", "asr", "diarize", "analyze", "export"],
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
        self._sized = False
        self._restore_geometry()
        self._build_menu()
        self._build_ui()
        self._build_status_bar()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._sized:
            self._sized = True
            # Pin the left pane's minimum to its real content width so the divider
            # can never shrink the viewport below it — that's what caused the
            # horizontal wiggle. Measured once widgets are realized.
            inner = self._splitter.widget(0).widget()
            need = max(inner.minimumSizeHint().width(), inner.sizeHint().width())
            self._splitter.widget(0).setMinimumWidth(need + 8)

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

        view_menu.addSeparator()
        appearance = view_menu.addMenu("外观")
        self._appearance_group = QActionGroup(self)
        cur = self._prefs.value("appearance", "auto")
        for key, label in (("auto", "跟随系统"), ("light", "浅色"), ("dark", "深色")):
            act = QAction(label, self, checkable=True)
            act.setChecked(cur == key)
            act.triggered.connect(lambda _=False, k=key: self._set_appearance(k))
            self._appearance_group.addAction(act)
            appearance.addAction(act)

        # Help
        help_menu = menu_bar.addMenu("帮助")

        docs_act = QAction("打开说明", self)
        docs_act.triggered.connect(self._open_readme)
        help_menu.addAction(docs_act)

    def _set_appearance(self, mode: str) -> None:
        self._prefs.setValue("appearance", mode)
        theme.apply(QApplication.instance(), mode)
        self._input_panel.refresh_theme()   # redraw scheme-colored glyphs

    # ── Central UI ────────────────────────────────────────────

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 14)
        root.setSpacing(0)

        # --- Top bar: process button (styled by the global theme) ---
        self._process_btn = QPushButton("开始转写")
        self._process_btn.setObjectName("primary")
        self._process_btn.setMinimumHeight(46)
        self._process_btn.setCursor(Qt.PointingHandCursor)
        self._process_btn.setEnabled(False)
        self._process_btn.clicked.connect(self._start_processing)

        self._cancel_btn = QPushButton("取消")
        self._cancel_btn.setObjectName("danger")
        self._cancel_btn.setMinimumHeight(40)
        self._cancel_btn.setCursor(Qt.PointingHandCursor)
        self._cancel_btn.setVisible(False)
        self._cancel_btn.clicked.connect(self._cancel_processing)

        # Horizontal splitter: left (input+settings) | right (progress+results)
        splitter = QSplitter(Qt.Horizontal)

        # -- Left column (scrollable so nothing clips on small windows) --
        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        # Generous right margin so the macOS overlay scrollbar never draws over
        # the buttons' right edge / rounded corner; a little bottom padding so
        # the last button never sits flush against the viewport edge.
        left_layout.setContentsMargins(2, 0, 26, 10)
        left_layout.setSpacing(12)

        self._input_panel = InputPanel()
        self._input_panel.file_ready.connect(self._on_file_selected)
        left_layout.addWidget(self._input_panel)

        self._settings = SettingsPanel()
        left_layout.addWidget(self._settings)

        left_layout.addWidget(self._process_btn)
        left_layout.addWidget(self._cancel_btn)
        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidget(left_col)
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        # Stop the drag right where everything is fully shown with NO horizontal
        # scroll. Content needs ~315px; 340 leaves comfortable room. widgetResizable
        # + this minimum means the inner content never exceeds the viewport.
        left_scroll.setMinimumWidth(340)
        left_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        self._splitter = splitter
        splitter.addWidget(left_scroll)

        # -- Right column --
        right_splitter = QSplitter(Qt.Vertical)

        self._progress = ProgressPanel()
        right_splitter.addWidget(self._progress)

        self._results = ResultsPanel()
        right_splitter.addWidget(self._results)

        right_splitter.setSizes([220, 480])
        splitter.addWidget(right_splitter)

        # Free-drag divider: left keeps its size, right takes extra space; neither
        # collapses to zero. The handle is a real grabbable divider.
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setCollapsible(0, False)
        splitter.setCollapsible(1, False)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)
        splitter.setSizes([340, 620])

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
        # All user data lives under ~/Documents/recorder/Recorder (see core.paths).
        from core.paths import data_root
        output_dir = str(data_root() / Path(input_path).stem)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        active_steps = list(_STEPS_BY_MODE.get(settings["mode"], _STEPS_BY_MODE["general"]))
        # general mode only runs the analyze step when the LLM is enabled
        if settings.get("use_llm") and "analyze" not in active_steps:
            active_steps.insert(active_steps.index("export"), "analyze")

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
            self._worker.cancel()        # non-blocking
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
        from core.paths import data_root
        subprocess.run(["open", str(data_root())])

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
            # Terminates the child process and stops the poll timer. No QThread
            # is involved, so there's nothing that can abort on destruction.
            self._worker.wait(8000)
        super().closeEvent(event)
