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

from core.i18n import current_language, set_language, t, ui_language_choices
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
        self._status_key: str | None = None
        self._status_args: dict = {}
        self._state_key: str = ""
        self._state_args: dict = {}
        self._state_tone: str = "hint"

        # The UI language is applied app-wide before this window is built (see
        # app.py). Re-assert it from prefs so the window is still correct when
        # constructed directly (e.g. in tests), and persist the first choice.
        lang = self._prefs.value("ui_language")
        if lang:
            set_language(lang)
        else:
            self._prefs.setValue("ui_language", current_language())

        self.setWindowTitle(t("app_title"))
        self.setMinimumSize(1080, 720)
        self._restore_geometry()
        self._build_menu()
        self._build_ui()
        self._show_status("status_ready")

    # ── menu ────────────────────────────────────────────────

    def _build_menu(self) -> None:
        menu_bar = self.menuBar()
        menu_bar.clear()

        file_menu = menu_bar.addMenu(t("menu_file"))
        open_act = QAction(t("menu_file_open"), self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(lambda: self._home.browse())
        file_menu.addAction(open_act)
        file_menu.addSeparator()
        reveal_act = QAction(t("menu_file_reveal"), self)
        reveal_act.setShortcut("Cmd+Shift+O")
        reveal_act.triggered.connect(self._reveal_output)
        file_menu.addAction(reveal_act)

        view_menu = menu_bar.addMenu(t("menu_view"))
        reset_act = QAction(t("menu_view_reset"), self)
        reset_act.setShortcut("Cmd+R")
        reset_act.triggered.connect(self._reset)
        view_menu.addAction(reset_act)
        view_menu.addSeparator()

        appearance = view_menu.addMenu(t("menu_appearance"))
        self._appearance_group = QActionGroup(self)
        cur_appearance = self._prefs.value("appearance", "auto")
        for key, label_key in (("auto", "appearance_auto"),
                               ("light", "appearance_light"),
                               ("dark", "appearance_dark")):
            act = QAction(t(label_key), self, checkable=True)
            act.setData(key)
            act.setChecked(cur_appearance == key)
            act.triggered.connect(lambda _=False, k=key: self._set_appearance(k))
            self._appearance_group.addAction(act)
            appearance.addAction(act)

        language = view_menu.addMenu(t("menu_language"))
        self._language_group = QActionGroup(self)
        cur_lang = current_language()
        for code, native in ui_language_choices():
            act = QAction(native, self, checkable=True)
            act.setData(code)
            act.setChecked(cur_lang == code)
            act.triggered.connect(lambda _=False, c=code: self._set_language(c))
            self._language_group.addAction(act)
            language.addAction(act)

        help_menu = menu_bar.addMenu(t("menu_help"))
        docs_act = QAction(t("menu_help_docs"), self)
        docs_act.triggered.connect(self._open_readme)
        help_menu.addAction(docs_act)

    _APPEARANCE_LABEL_KEYS = {"auto": "appearance_btn_auto",
                              "dark": "appearance_btn_dark",
                              "light": "appearance_btn_light"}

    def _set_appearance(self, mode: str) -> None:
        self._prefs.setValue("appearance", mode)
        for act in self._appearance_group.actions():
            act.setChecked(act.data() == mode)
        theme.apply(QApplication.instance(), mode)
        self._update_theme_btn()
        self._rail.set_stage(self._stage_for(self._stack.currentIndex()))
        self.update()

    def _update_theme_btn(self) -> None:
        cur = self._prefs.value("appearance", "auto")
        self._btn_theme.setText(
            t(self._APPEARANCE_LABEL_KEYS.get(cur, "appearance_btn_auto")))

    # ── language ──────────────────────────────────

    def _set_language(self, code: str) -> None:
        applied = set_language(code)
        self._prefs.setValue("ui_language", applied)
        for act in self._language_group.actions():
            act.setChecked(act.data() == applied)
        self._retranslate()

    def _cycle_language(self) -> None:
        self._set_language("en" if current_language() == "zh" else "zh")

    def _update_lang_btn(self) -> None:
        self._btn_lang.setText(t("lang_btn"))

    def _retranslate(self) -> None:
        """Re-apply every string in the newly selected language, live."""
        self.setWindowTitle(t("app_title"))
        self._build_menu()
        self._wordmark.setText(t("wordmark"))
        self._btn_folder.setText(t("btn_output_folder"))
        self._btn_theme.setToolTip(t("theme_tooltip"))
        self._btn_lang.setToolTip(t("lang_btn_tooltip"))
        self._update_theme_btn()
        self._update_lang_btn()
        self._rail.retranslate()
        self._home.retranslate()
        self._recording.retranslate()
        self._processing.retranslate()
        self._results.retranslate()
        if self._state_key:
            self._state_lbl.setText(t(self._state_key, **self._state_args))
        if self._status_key:
            self.statusBar().showMessage(
                t(self._status_key, **self._status_args))

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
        self._wordmark = QLabel(t("wordmark"))
        self._wordmark.setObjectName("wordmark")
        self._state_lbl = QLabel("")
        theme.set_tone(self._state_lbl, "hint")
        self._btn_folder = QPushButton(t("btn_output_folder"))
        self._btn_folder.setObjectName("quiet")
        self._btn_folder.setCursor(Qt.PointingHandCursor)
        self._btn_folder.clicked.connect(self._reveal_output)
        self._btn_theme = QPushButton()
        self._btn_theme.setObjectName("quiet")
        self._btn_theme.setCursor(Qt.PointingHandCursor)
        self._btn_theme.setToolTip(t("theme_tooltip"))
        self._btn_theme.clicked.connect(self._cycle_appearance)
        self._update_theme_btn()
        self._btn_lang = QPushButton()
        self._btn_lang.setObjectName("quiet")
        self._btn_lang.setCursor(Qt.PointingHandCursor)
        self._btn_lang.setToolTip(t("lang_btn_tooltip"))
        self._btn_lang.clicked.connect(self._cycle_language)
        self._update_lang_btn()
        sl.addWidget(self._wordmark)
        sl.addStretch()
        sl.addWidget(self._state_lbl)
        sl.addStretch()
        sl.addWidget(self._btn_folder)
        sl.addWidget(self._btn_theme)
        sl.addWidget(self._btn_lang)
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

    def _go(self, screen: int, state_key: str = "",
            tone: str = "hint", **args) -> None:
        self._stack.setCurrentIndex(screen)
        self._rail.set_stage(self._stage_for(screen))
        self._state_key = state_key
        self._state_args = args
        self._state_tone = tone
        self._state_lbl.setText(t(state_key, **args) if state_key else "")
        theme.set_tone(self._state_lbl, tone)

    def _show_status(self, key: str | None, **args) -> None:
        """Set the status bar from a catalog key, remembering it so a live
        language switch can re-render it."""
        self._status_key = key
        self._status_args = args
        self.statusBar().showMessage(t(key, **args) if key else "")

    def _show_status_text(self, text: str) -> None:
        """Show a transient, already-composed message (e.g. from the pipeline
        subprocess). Not tracked, so a language switch leaves it until the next
        progress tick replaces it."""
        self._status_key = None
        self._status_args = {}
        self.statusBar().showMessage(text)

    # ── recording ───────────────────────────────────────────

    def _start_recording(self, source: str, device) -> None:
        self._go(RECORDING, "", "hint")
        self._recording.start_capture(source, device)
        self._show_status("status_recording")

    def _on_recording_ready(self, path: str, meta: str) -> None:
        self._home.set_file(path, meta)
        self._go(HOME)
        self._show_status("status_recorded", name=Path(path).name)

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
        self._go(PROCESSING, "state_processing", "accent")
        self._show_status("status_processing")

        # Tell the subprocess which language to report progress in.
        settings = {**settings, "ui_lang": current_language()}
        self._worker = PipelineWorker(
            input_path=input_path, output_dir=output_dir, settings=settings)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, info: dict) -> None:
        self._processing.update_progress(info)
        self._show_status_text(info.get("message", ""))

    def _on_finished(self, result: dict) -> None:
        total_s = result.get("stats", {}).get("total_time_s", 0)
        self._results.load_results(result)
        self._go(RESULTS, "state_done", "ok", seconds=f"{total_s:.0f}")
        self._show_status("status_done", seconds=f"{total_s:.0f}",
                          dir=result["output_dir"])
        self._send_notification(
            "Recorder", t("notify_done", name=Path(result["output_dir"]).name))

    def _on_error(self, msg: str) -> None:
        self._go(HOME, "state_error", "danger")
        self._show_status("status_error")
        QMessageBox.critical(self, t("error_box_title"), msg)

    def _cancel_processing(self) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
        self._go(HOME, "state_cancelled", "hint")
        self._show_status("status_cancelled")

    # ── utilities ───────────────────────────────────────────

    def _reset(self) -> None:
        if self._worker and self._worker.isRunning():
            return
        self._home.clear_file()
        self._results.reset()
        self._go(HOME)
        self._show_status("status_ready")

    def _cycle_appearance(self) -> None:
        cur = self._prefs.value("appearance", "auto")
        nxt = {"auto": "dark", "dark": "light", "light": "auto"}.get(cur, "auto")
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
