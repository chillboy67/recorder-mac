"""
Main window — AURA single-stage layout.

Structure:
    top strip   wordmark · (state text) · 输出文件夹 / 外观
    step rail   01 输入 · 02 模式 · 03 转写 · 04 结果   (left, fixed 96px)
    stage       QStackedWidget: HomeScreen / RecordingScreen /
                ProcessingScreen / ResultsScreen

Pipeline lifecycle is unchanged from the old MainWindow (PipelineWorker).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from gui import theme
from gui.widgets.home_screen import HomeScreen
from gui.widgets.processing_screen import ProcessingScreen, STEP_LABELS
from gui.widgets.recording_screen import RecordingScreen
from gui.widgets.results_screen import ResultsScreen
from gui.widgets.step_rail import StepRail
from gui.widgets.visuals import GlowBackground
from gui.workers.pipeline_worker import PipelineWorker

_STEPS_BY_MODE: dict[str, list[str]] = {
    "general":   ["load", "asr", "export"],
    "classroom": ["load", "denoise", "asr", "diarize", "analyze", "export"],
    "ielts":     ["load", "asr", "diarize", "analyze", "export"],
}

HOME, RECORDING, PROCESSING, RESULTS = range(4)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._worker: PipelineWorker | None = None
        self._prefs = QSettings("LucasLab", "Recorder")

        self.setWindowTitle("Recorder · 录音转文字")
        self.setMinimumSize(1080, 720)
        self._restore_geometry()
        self._build_menu()
        self._build_ui()
        self.statusBar().showMessage("就绪 · 全程离线")

    # ── menu ────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()

        file_menu = menu_bar.addMenu("文件")
        open_act = QAction("打开录音…", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(lambda: self._home.browse())
        file_menu.addAction(open_act)
        file_menu.addSeparator()
        reveal_act = QAction("显示输出文件夹", self)
        reveal_act.setShortcut("Cmd+Shift+O")
        reveal_act.triggered.connect(self._reveal_output)
        file_menu.addAction(reveal_act)

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

        help_menu = menu_bar.addMenu("帮助")
        docs_act = QAction("打开说明", self)
        docs_act.triggered.connect(self._open_readme)
        help_menu.addAction(docs_act)

    def _set_appearance(self, mode: str) -> None:
        self._prefs.setValue("appearance", mode)
        theme.apply(QApplication.instance(), mode)
        self._rail.set_stage(self._stage_for(self._stack.currentIndex()))
        self.update()

    # ── central UI ──────────────────────────────────────────

    def _build_ui(self) -> None:
        central = GlowBackground()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # top strip
        strip = QWidget()
        sl = QHBoxLayout(strip)
        sl.setContentsMargins(20, 10, 20, 10)
        wordmark = QLabel("R E C O R D E R  —  全程离线")
        wordmark.setObjectName("wordmark")
        self._state_lbl = QLabel("")
        theme.set_tone(self._state_lbl, "hint")
        btn_folder = QPushButton("输出文件夹")
        btn_folder.setObjectName("quiet")
        btn_folder.setCursor(Qt.PointingHandCursor)
        btn_folder.clicked.connect(self._reveal_output)
        btn_theme = QPushButton("外观")
        btn_theme.setObjectName("quiet")
        btn_theme.setCursor(Qt.PointingHandCursor)
        btn_theme.clicked.connect(self._cycle_appearance)
        sl.addWidget(wordmark)
        sl.addStretch()
        sl.addWidget(self._state_lbl)
        sl.addStretch()
        sl.addWidget(btn_folder)
        sl.addWidget(btn_theme)
        outer.addWidget(strip)

        # rail + stage
        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)

        self._rail = StepRail()
        body.addWidget(self._rail)

        self._stack = QStackedWidget()

        self._home = HomeScreen()
        self._home.record_requested.connect(self._start_recording)
        self._home.process_requested.connect(self._start_processing)
        self._stack.addWidget(self._home)

        self._recording = RecordingScreen()
        self._recording.recording_ready.connect(self._on_recording_ready)
        self._recording.recording_aborted.connect(lambda: self._go(HOME))
        self._stack.addWidget(self._recording)

        self._processing = ProcessingScreen()
        self._processing.cancel_requested.connect(self._cancel_processing)
        self._stack.addWidget(self._processing)

        self._results = ResultsScreen()
        self._results.new_requested.connect(self._reset)
        self._stack.addWidget(self._results)

        body.addWidget(self._stack, stretch=1)
        outer.addLayout(body, stretch=1)

    # ── navigation ──────────────────────────────────────────

    @staticmethod
    def _stage_for(screen: int) -> int:
        return {HOME: 0, RECORDING: 0, PROCESSING: 2, RESULTS: 3}[screen]

    def _go(self, screen: int, state_text: str = "",
            tone: str = "hint") -> None:
        self._stack.setCurrentIndex(screen)
        self._rail.set_stage(self._stage_for(screen))
        self._state_lbl.setText(state_text)
        theme.set_tone(self._state_lbl, tone)

    # ── recording ───────────────────────────────────────────

    def _start_recording(self, source: str, device) -> None:
        self._go(RECORDING, "", "hint")
        self._recording.start_capture(source, device)
        self.statusBar().showMessage("录音中…")

    def _on_recording_ready(self, path: str, meta: str) -> None:
        self._home.set_file(path, meta)
        self._go(HOME)
        self.statusBar().showMessage(f"已录制:{Path(path).name}")

    # ── processing ──────────────────────────────────────────

    def _start_processing(self, input_path: str, settings: dict) -> None:
        from core.paths import data_root
        output_dir = str(data_root() / Path(input_path).stem)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        active_steps = list(_STEPS_BY_MODE.get(settings["mode"],
                                               _STEPS_BY_MODE["general"]))
        if settings.get("use_llm") and "analyze" not in active_steps:
            active_steps.insert(active_steps.index("export"), "analyze")

        p = Path(input_path)
        size_mb = p.stat().st_size / 1_048_576
        self._processing.reset(active_steps, p.name, f"{size_mb:.1f} MB")
        self._results.reset()
        self._go(PROCESSING, "PROCESSING", "accent")
        self.statusBar().showMessage("处理中…")

        self._worker = PipelineWorker(
            input_path=input_path, output_dir=output_dir, settings=settings)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, info: dict) -> None:
        self._processing.update_progress(info)
        self.statusBar().showMessage(info.get("message", ""))

    def _on_finished(self, result: dict) -> None:
        total_s = result.get("stats", {}).get("total_time_s", 0)
        self._results.load_results(result)
        self._go(RESULTS, f"✓ 转写完成 · 用时 {total_s:.0f}s", "ok")
        self.statusBar().showMessage(
            f"✓ 完成，用时 {total_s:.0f}s  ·  输出：{result['output_dir']}")
        self._send_notification(
            "Recorder", f"转写完成 — {Path(result['output_dir']).name}")

    def _on_error(self, msg: str) -> None:
        self._go(HOME, "发生错误", "danger")
        self.statusBar().showMessage("发生错误")
        QMessageBox.critical(self, "处理出错", msg)

    def _cancel_processing(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
        self._go(HOME, "已取消", "hint")
        self.statusBar().showMessage("已取消")

    # ── utilities ───────────────────────────────────────────

    def _reset(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._home.clear_file()
        self._results.reset()
        self._go(HOME)
        self.statusBar().showMessage("就绪 · 全程离线")

    def _cycle_appearance(self) -> None:
        cur = self._prefs.value("appearance", "auto")
        nxt = {"auto": "dark", "dark": "light", "light": "auto"}[cur]
        for act in self._appearance_group.actions():
            act.setChecked(
                act.text() == {"auto": "跟随系统", "dark": "深色",
                               "light": "浅色"}[nxt])
        self._set_appearance(nxt)

    def _reveal_output(self) -> None:
        from core.paths import data_root
        subprocess.run(["open", str(data_root())])

    def _open_readme(self) -> None:
        readme = Path(__file__).parent.parent / "README.md"
        if readme.exists():
            subprocess.run(["open", str(readme)])

    @staticmethod
    def _send_notification(title: str, message: str) -> None:
        script = (f'display notification "{message}"'
                  f' with title "{title}" sound name "Glass"')
        subprocess.run(["osascript", "-e", script], capture_output=True)

    # ── geometry ────────────────────────────────────────────

    def _restore_geometry(self) -> None:
        geometry = self._prefs.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        else:
            self.resize(1180, 780)

    def closeEvent(self, event) -> None:
        self._prefs.setValue("geometry", self.saveGeometry())
        if self._worker and self._worker.isRunning():
            self._worker.wait(8000)
        super().closeEvent(event)
