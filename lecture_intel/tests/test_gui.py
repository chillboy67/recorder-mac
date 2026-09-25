"""
GUI smoke tests — offscreen rendering only, never a real window.

Why these exist: a local variable named ``t`` shadowed the i18n ``t()`` helper
inside ``ResultsScreen.load_results``, so it raised UnboundLocalError for every
mode. The screen came up empty and the Qt-free suite could not notice, because
**Qt swallows exceptions raised inside a slot**. Rendering offscreen is the
sanctioned way to catch that class of bug without launching the app.

Opt-in on purpose
-----------------
Skipped unless ``RECORDER_GUI_TESTS=1``: initialising Qt changes what else may
run in the same process, because in a sandboxed shell a process that has created
a QApplication is killed on its next localhost HTTP request — and test_core
(LanguageTool on 127.0.0.1:1) and test_llm (Ollama on 127.0.0.1:11434) both open
localhost connections. Keeping Qt out of the default suite keeps
``python -m pytest`` reliable; run these explicitly:

    RECORDER_GUI_TESTS=1 .venv/bin/python3 -m pytest tests/test_gui.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

if not os.environ.get("RECORDER_GUI_TESTS"):
    pytest.skip("GUI tests are opt-in: set RECORDER_GUI_TESTS=1",
                allow_module_level=True)

# Must be set before Qt initialises its platform plugin.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QSettings  # noqa: E402
from PySide6.QtWidgets import QApplication, QLabel  # noqa: E402

from gui import theme  # noqa: E402
from gui.widgets.results_screen import ResultsScreen  # noqa: E402
from gui.widgets.home_screen import HomeScreen  # noqa: E402

FIDELITY = {
    "asr_loop": 1, "real_speech": 2, "uncertain": 1, "adjacent_duplicates": 0,
    "folded_count": 1, "dropped_count": 0, "total": 4,
    "lines": ["[0:12–0:15] 疑似转写伪影 · 已在正文中折叠 ｜原话「打打打打」",
              "[0:31–0:33] 真实重复 · 正文原样保留 ｜原话「我 我」"],
}


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def dark_theme(qapp):
    """Deterministic scheme: the widgets read their colours from it."""
    theme.apply(qapp, "dark")
    yield
    theme.apply(qapp, "auto")


def painted(widget):
    """Force a paint pass — this is what turns a paintEvent crash into a failure."""
    widget.resize(320, 200)
    pixmap = widget.grab()
    assert not pixmap.isNull()
    return pixmap


def _result(mode: str, **over) -> dict:
    res = {
        "output_dir": "/tmp",
        "files": [],
        "stats": {"mode": mode, "language": "zh", "duration_sec": 60,
                  "total_time_s": 12},
        "mode": mode,
        "ielts": None,
        "classroom": None,
        "fidelity": None,
    }
    res.update(over)
    return res


# ── engine selection ─────────────────────────────────────────────────

def test_home_screen_exposes_persistent_engine_setting(qapp):
    settings = QSettings("LucasLab", "Recorder")
    settings.remove("engine")
    home = HomeScreen()
    try:
        home._engine_combo.setCurrentIndex(
            home._engine_combo.findData("whisper.cpp-vulkan"))
        assert home.get_settings()["engine"] == "whisper.cpp-vulkan"
        assert settings.value("engine") == "whisper.cpp-vulkan"
        home._engine_combo.setCurrentIndex(
            home._engine_combo.findData("whisper.cpp-openvino"))
        assert home.get_settings()["engine"] == "whisper.cpp-openvino"
    finally:
        settings.remove("engine")
        home.close()


# ── theme ───────────────────────────────────────────────────────────

def test_scheme_tables_define_every_colour_the_widgets_read(qapp):
    theme.apply(qapp, "dark")
    dark = theme.current_scheme()
    theme.apply(qapp, "light")
    light = theme.current_scheme()
    assert dark is not light and dark != light
    for key in ("ink2", "ink3", "accent", "ok", "warn", "danger"):
        assert key in dark and key in light


def test_is_dark_follows_the_requested_mode(qapp):
    theme.apply(qapp, "dark")
    assert theme.is_dark() is True
    theme.apply(qapp, "light")
    assert theme.is_dark() is False


def test_apply_installs_a_stylesheet_that_differs_per_scheme(qapp):
    theme.apply(qapp, "light")
    light_qss = qapp.styleSheet()
    theme.apply(qapp, "dark")
    dark_qss = qapp.styleSheet()
    assert light_qss and dark_qss and light_qss != dark_qss


def test_apply_is_idempotent(qapp):
    theme.apply(qapp, "dark")
    theme.apply(qapp, "dark")          # must not raise on re-apply
    assert theme.is_dark() is True


def test_system_scheme_hook_is_connected_only_in_auto_mode(qapp):
    """Disconnecting a signal that was never connected raises *and* makes
    libpyside log a RuntimeWarning on every apply() — so the hook must be
    tracked rather than blindly disconnected."""
    theme.apply(qapp, "auto")
    assert theme._state["hooked"] is True
    theme.apply(qapp, "dark")
    assert theme._state["hooked"] is False
    theme.apply(qapp, "light")
    assert theme._state["hooked"] is False
    theme.apply(qapp, "auto")
    assert theme._state["hooked"] is True
    theme.apply(qapp, "auto")          # re-applying auto must not double-connect
    assert theme._state["hooked"] is True


def test_display_font_carries_the_fallback_stack_and_size():
    font = theme.display_font(18)
    assert font.pointSize() == 18
    assert theme.DISPLAY_FAMILIES[0] == "Space Grotesk"   # documented chain
    assert "Helvetica Neue" in theme.DISPLAY_FAMILIES


def test_set_tone_and_repolish_accept_a_label(qapp):
    label = QLabel("x")
    theme.set_tone(label, "danger")
    theme.repolish(label)
    assert label.text() == "x"


# ── self-painted visuals ────────────────────────────────────────────

def test_wave_bars_start_stop_and_paint(qapp):
    from gui.widgets.visuals import WaveBars
    bars = WaveBars(bars=5, height=20)
    bars.start()
    painted(bars)
    bars.stop()
    painted(bars)


def test_wave_glyph_paints(qapp):
    from gui.widgets.visuals import WaveGlyph
    painted(WaveGlyph())


def test_glow_ring_shows_the_clock_and_paints(qapp):
    from gui.widgets.visuals import GlowRing
    ring = GlowRing()
    ring.set_time("00:01:02")
    ring.set_sub("hint")
    painted(ring)


def test_progress_ring_accepts_the_whole_range(qapp):
    from gui.widgets.visuals import ProgressRing
    ring = ProgressRing()
    for percent in (0, 37, 100):
        ring.set_progress(percent, f"{percent}%")
        painted(ring)


def test_pulse_dot_paints(qapp):
    from gui.widgets.visuals import PulseDot
    painted(PulseDot())


# ── common widgets ──────────────────────────────────────────────────

def test_chip_button_and_no_scroll_combo(qapp):
    from gui.widgets.common import ChipButton, NoScrollComboBox
    chip = ChipButton("chip")
    assert chip.text() == "chip"
    painted(chip)
    combo = NoScrollComboBox()
    combo.addItems(["a", "b"])
    combo.setCurrentIndex(1)
    assert combo.currentText() == "b"


# ── step rail ───────────────────────────────────────────────────────

def test_step_rail_accepts_every_stage_and_retranslates(qapp):
    from gui.widgets.step_rail import StepRail
    rail = StepRail()
    for stage in range(4):
        rail.set_stage(stage)
        painted(rail)
    rail.retranslate()


# ── home screen ─────────────────────────────────────────────────────

def test_home_screen_builds_paints_and_retranslates(qapp):
    from gui.widgets.home_screen import HomeScreen
    home = HomeScreen()
    painted(home)
    home.set_file("/tmp/take.wav", "1.2 MB")
    home.retranslate()
    painted(home)


# ── recording screen ────────────────────────────────────────────────

def test_recording_screen_builds_and_cleans_up_without_temps(qapp):
    from gui.widgets.recording_screen import RecordingScreen
    rec = RecordingScreen()
    painted(rec)
    rec.cleanup_temps()          # nothing recorded → must be a no-op
    rec.retranslate()


# ── processing screen ───────────────────────────────────────────────

def test_processing_screen_resets_and_reports_progress(qapp):
    from gui.widgets.processing_screen import ProcessingScreen
    screen = ProcessingScreen()
    screen.reset(["load", "asr", "export"], filename="take.wav")
    painted(screen)
    for step, percent in (("load", 8), ("asr", 50), ("export", 100)):
        screen.update_progress({"step": step, "percent": percent,
                                "message": step, "status": "done"})
        painted(screen)
    screen.retranslate()


# ── results screen ──────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["general", "classroom", "ielts"])
def test_load_results_renders_every_mode(qapp, mode):
    """Regression: this used to raise UnboundLocalError for all three modes."""
    extra = {}
    if mode == "classroom":
        extra["classroom"] = {"markdown": "# s", "llm": False,
                              "emphasis_count": 1, "definition_count": 0}
    elif mode == "ielts":
        extra["ielts"] = {"markdown": "# r", "pron_issue_count": 1,
                          "grammar_issue_count": 2, "wpm": 120}
    screen = ResultsScreen()
    screen.load_results(_result(mode, **extra))          # must not raise
    assert screen._rail.count() > 0
    painted(screen)


def test_fidelity_tab_and_card_appear(qapp):
    """The annotations have to be visible, not just written to meta.json."""
    screen = ResultsScreen()
    screen.load_results(_result("general", fidelity=FIDELITY))
    tabs = [screen._tabs.tabText(i) for i in range(screen._tabs.count())]
    assert "忠实度标注" in tabs
    shown = screen._tabs.widget(tabs.index("忠实度标注")).toPlainText()
    assert "打打打打" in shown and "已在正文中折叠" in shown
    painted(screen)


def test_fidelity_card_states_nothing_was_flagged(qapp):
    """An empty trail still has to say so — silence would read as 'unknown'."""
    screen = ResultsScreen()
    screen.load_results(_result("general", fidelity={
        **FIDELITY, "total": 0, "asr_loop": 0, "real_speech": 0,
        "uncertain": 0, "folded_count": 0, "dropped_count": 0, "lines": []}))
    texts = [w.text() for w in screen._rail_host.findChildren(QLabel)]
    assert any("未发现疑似伪影" in text for text in texts)
    assert screen._tabs.count() == 0        # nothing to show as a tab


def test_results_screen_reset_clears_everything(qapp):
    screen = ResultsScreen()
    screen.load_results(_result("general", fidelity=FIDELITY))
    screen.reset()
    assert screen._tabs.count() == 0 and screen._last_result is None


def test_retranslate_uses_the_active_language(qapp):
    """A shadowed `t` also broke retranslate() → load_results()."""
    from core import i18n
    screen = ResultsScreen()
    screen.load_results(_result("ielts", ielts={
        "markdown": "# r", "pron_issue_count": 1, "grammar_issue_count": 2,
        "wpm": 120}, fidelity=FIDELITY))
    try:
        i18n.set_language("en")
        screen.retranslate()
        tabs = [screen._tabs.tabText(i) for i in range(screen._tabs.count())]
        assert "Fidelity" in tabs
    finally:
        i18n.set_language("zh")


# ── main window ─────────────────────────────────────────────────────

@pytest.fixture
def isolated_prefs(tmp_path, monkeypatch):
    """Keep QSettings out of the user's real preferences."""
    for scope in (QSettings.UserScope, QSettings.SystemScope):
        QSettings.setPath(QSettings.NativeFormat, scope, str(tmp_path))
    yield


def test_main_window_builds_every_screen_and_switches_language(qapp, isolated_prefs):
    from gui.main_window import MainWindow
    from core import i18n
    window = MainWindow()
    try:
        for widget in (window._home, window._recording, window._processing,
                       window._results, window._rail):
            assert widget is not None
        window._retranslate()               # re-apply every string
        window._set_language("en")          # the real user path
        assert "Recorder" in window.windowTitle()
        painted(window)
    finally:
        i18n.set_language("zh")
        window.close()
        window.deleteLater()


# ── worker ──────────────────────────────────────────────────────────

def test_pipeline_worker_starts_idle(qapp):
    from gui.workers.pipeline_worker import PipelineWorker
    worker = PipelineWorker("in.wav", "/tmp", {"mode": "general"})
    assert worker.isRunning() is False
    for signal_name in ("progress", "finished", "error"):
        assert hasattr(worker, signal_name)
