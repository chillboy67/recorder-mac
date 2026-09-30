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
(LanguageTool on 127.0.0.1:1) opens a localhost connection. Keeping Qt out of
the default suite keeps
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

from gui import settings as settings_mod, theme  # noqa: E402
from gui.settings import app_settings  # noqa: E402
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


@pytest.fixture(autouse=True)
def isolated_prefs(tmp_path, monkeypatch):
    """Keep every test out of the user's real preferences.

    QSettings.setPath() cannot do this: on macOS the native format is
    CFPreferences and ignores it, so the tests used to write the real
    ~/Library/Preferences/com.lucaslab.Recorder.plist. Point the app's one
    settings helper at a fresh INI file instead."""
    monkeypatch.setattr(settings_mod, "SETTINGS_FILE", str(tmp_path / "prefs.ini"))
    yield


def test_prefs_go_to_the_test_file_not_the_native_store(tmp_path):
    prefs = app_settings()
    assert prefs.format() == QSettings.IniFormat
    assert Path(prefs.fileName()) == tmp_path / "prefs.ini"


def test_app_opens_preferences_only_through_the_settings_helper():
    """A QSettings built anywhere else would bypass the isolation above."""
    helper = ROOT / "gui" / "settings.py"
    sources = [ROOT / "app.py", *(ROOT / "gui").rglob("*.py"),
               *(ROOT / "core").rglob("*.py"), *(ROOT / "modules").rglob("*.py")]
    offenders = [str(path.relative_to(ROOT)) for path in sources
                 if path != helper and "QSettings(" in path.read_text(encoding="utf-8")]
    assert offenders == []


# ── engine selection ─────────────────────────────────────────────────

def test_home_screen_exposes_persistent_engine_setting(qapp):
    settings = app_settings()
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


def test_mlx_backend_is_offered_only_on_apple_silicon(qapp, monkeypatch):
    from gui.widgets import home_screen
    settings = app_settings()
    try:
        for apple, enabled, current in ((False, False, "auto"),
                                        (True, True, "mlx-whisper")):
            settings.setValue("engine", "mlx-whisper")
            monkeypatch.setattr(home_screen, "_apple_silicon", lambda apple=apple: apple)
            home = HomeScreen()
            index = home._engine_combo.findData("mlx-whisper")
            assert home._engine_combo.model().item(index).isEnabled() is enabled
            assert home.get_settings()["engine"] == current
            home.close()
    finally:
        settings.remove("engine")


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


@pytest.mark.parametrize("source, channels", [("mic", "mono"), ("system", "stereo"),
                                              ("both", "mono")])
def test_recording_ring_shows_the_real_capture_format(qapp, source, channels):
    """It used to read a fixed "−14 dB · 16 kHz" placeholder while capturing
    48 kHz, contradicting the fidelity the app is built around."""
    from core import i18n
    from gui.widgets.recording_screen import RecordingScreen
    i18n.set_language("en")
    rec = RecordingScreen()
    try:
        rec._source = source
        rec._enter_recording_ui()
        assert rec._ring._sub == f"48 kHz · {channels} · WAV"
        rec._exit_recording_ui()
    finally:
        i18n.set_language("zh")
        rec.cleanup_temps()


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


def test_processing_screen_never_sits_still(qapp):
    """Lines rotate every 7s; a stalled number creeps, but never past its step."""
    from core import i18n
    from gui.widgets import processing_screen as P
    i18n.set_language("zh")
    screen = P.ProcessingScreen()
    screen.reset(["asr"])
    screen.update_progress({"step": "asr", "percent": 20, "status": "running",
                            "message": "叫醒识别小助手"})
    assert screen._timer.isActive()
    labels = []
    for _ in range(3 * P.ROTATE_TICKS):
        screen._tick()
        labels.append(screen._ring._label)
    assert {"叫醒识别小助手", "喝口水，不着急", "慢工出细活"} <= set(labels)
    assert screen._ring._percent > 20                 # crept while stalled

    for _ in range(3000):
        screen._tick()
    assert screen._ring._percent < 75                 # asr ends at 75

    # a real update below the crept number never pulls the ring backwards
    crept = screen._ring._percent
    screen.update_progress({"step": "asr", "percent": 40, "status": "running",
                            "message": "一字一句记下来"})
    assert screen._ring._percent == crept
    assert screen._ring._label == "一字一句记下来"      # new line shows at once

    screen.update_progress({"step": "export", "percent": 100, "status": "done",
                            "message": "好啦！"})
    assert not screen._timer.isActive()
    assert screen._ring._percent == 100


# ── results screen ──────────────────────────────────────────────────

@pytest.mark.parametrize("mode", ["general", "classroom", "ielts"])
def test_load_results_renders_every_mode(qapp, mode):
    """Regression: this used to raise UnboundLocalError for all three modes."""
    extra = {}
    if mode == "classroom":
        extra["classroom"] = {"markdown": "# s",
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


def _assert_full_height(widget):
    # Height only: offscreen Qt on Windows has no real font database, and its
    # oversized text metrics cannot fit the fixed-width card horizontally.
    # The semibold width has its own test below.
    hint = widget.sizeHint()
    assert widget.height() >= hint.height(), (widget.objectName(), widget.size(), hint)


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_home_controls_keep_their_size_at_the_default_window(qapp, isolated_prefs, lang):
    """Wider fonts (DejaVu Sans on Linux) wrap a mode-card description onto
    one more line. The page used to take that line out of the input card,
    squashing the source segments and the Start button until their labels
    were cut in half."""
    from gui.main_window import MainWindow
    from core import i18n
    theme.apply(qapp, "light")
    window = MainWindow()
    try:
        window._set_language(lang)
        window.resize(1068, 660)            # the default and minimum size
        window.show()
        qapp.processEvents()
        home = window._home
        for button in home._src_buttons.values():
            _assert_full_height(button)
        home.set_file(str(ROOT / "tests" / "missing.wav"), "00:00:01 · 1 KB")
        qapp.processEvents()
        _assert_full_height(home._start_btn)
        combo = home._lang_combo
        assert combo.width() >= combo.fontMetrics().horizontalAdvance(combo.currentText()) + 48
    finally:
        i18n.set_language("zh")
        window.close()
        window.deleteLater()


def test_segment_reserves_room_for_its_semibold_checked_label(qapp):
    from PySide6.QtGui import QFont, QFontMetrics
    from gui.widgets.home_screen import _SegButton
    button = _SegButton("Microphone")
    bold = QFont(button.font())
    bold.setWeight(QFont.Weight.DemiBold)
    assert button.sizeHint().width() >= QFontMetrics(bold).horizontalAdvance("Microphone")
    assert button.minimumSizeHint() == button.sizeHint()


def test_error_summary_keeps_the_final_message_of_a_traceback():
    from gui.main_window import summarize_error
    trace = (
        "Traceback (most recent call last):\n"
        '  File "core/transcriber.py", line 500, in _load_faster\n'
        "    raise RuntimeError(\n"
        "RuntimeError: Could not load the CPU Whisper model 'small'. "
        "Check the network connection.\n"
    )
    assert summarize_error(trace) == (
        "Could not load the CPU Whisper model 'small'. Check the network connection.")
    assert summarize_error("Processing stopped (exit code -9)") == (
        "Processing stopped (exit code -9)")


# ── worker ──────────────────────────────────────────────────────────

def test_pipeline_worker_starts_idle(qapp):
    from gui.workers.pipeline_worker import PipelineWorker
    worker = PipelineWorker("in.wav", "/tmp", {"mode": "general"})
    assert worker.isRunning() is False
    for signal_name in ("progress", "finished", "error"):
        assert hasattr(worker, signal_name)


# ── cross-platform desktop actions ──────────────────────────────────
# "open" and "osascript" exist only on macOS; the app also runs on Windows
# and Linux, so folders/files go through QDesktopServices and notifications
# must never raise out of the finished-job slot.

@pytest.fixture
def opened_urls(monkeypatch):
    from PySide6.QtGui import QDesktopServices
    calls = []
    monkeypatch.setattr(QDesktopServices, "openUrl",
                        lambda url: calls.append(url) or True)
    return calls


@pytest.fixture
def main_window(qapp, isolated_prefs):
    from gui.main_window import MainWindow
    window = MainWindow()
    yield window
    window.close()
    window.deleteLater()


def test_reveal_output_opens_the_saved_folder(main_window, opened_urls, tmp_path):
    out = tmp_path / "results"
    out.mkdir()
    main_window._prefs.setValue("output_dir", str(out))
    main_window._reveal_output()
    assert [u.toLocalFile() for u in opened_urls] == [out.as_posix()]


def test_open_readme_opens_the_bundled_readme(main_window, opened_urls):
    main_window._open_readme()
    readme = ROOT / "README.md"
    expected = [readme.as_posix()] if readme.exists() else []
    assert [u.toLocalFile() for u in opened_urls] == expected


def test_results_open_folder_opens_the_output_dir(qapp, opened_urls, tmp_path):
    screen = ResultsScreen()
    screen.load_results(_result("general", output_dir=str(tmp_path)))
    screen._open_folder()
    assert [u.toLocalFile() for u in opened_urls] == [tmp_path.as_posix()]


@pytest.mark.parametrize("platform", ["darwin", "win32", "linux"])
def test_notification_never_raises_without_osascript(main_window, monkeypatch,
                                                     platform):
    import subprocess
    from gui import main_window as mw

    def missing(*args, **kwargs):
        raise FileNotFoundError("osascript")

    monkeypatch.setattr(mw.sys, "platform", platform)
    monkeypatch.setattr(subprocess, "run", missing)
    main_window._send_notification("Recorder", 'done "quoted"')


def test_on_finished_survives_a_failing_notification(main_window, monkeypatch,
                                                     tmp_path):
    import subprocess

    def missing(*args, **kwargs):
        raise FileNotFoundError("osascript")

    monkeypatch.setattr(subprocess, "run", missing)
    main_window._on_finished(_result("general", output_dir=str(tmp_path)))
    assert main_window._results._output_dir == str(tmp_path)


# ── first-run strip ─────────────────────────────────────────────────

def test_peak_level_reads_every_sample_format():
    import struct
    from PySide6.QtMultimedia import QAudioFormat
    from gui.widgets.onboarding import peak_level
    F = QAudioFormat.SampleFormat
    assert peak_level(struct.pack("<3h", 0, -16384, 100), F.Int16) == pytest.approx(0.5)
    assert peak_level(struct.pack("<2f", 0.25, -0.75), F.Float) == pytest.approx(0.75)
    assert peak_level(struct.pack("<i", 2 ** 30), F.Int32) == pytest.approx(0.5)
    assert peak_level(bytes([128, 192]), F.UInt8) == pytest.approx(0.5)
    assert peak_level(b"\x01", F.Int16) == 0.0            # half a sample


class _FakeCapture:
    """Stands in for MicCapture: emits the given peak as soon as it starts."""

    def __init__(self, peak):
        from PySide6.QtCore import QObject, Signal

        class Capture(QObject):
            level = Signal(float)
            finished = Signal(float)
            failed = Signal(str)
            devices = []

            def start(self, device):
                Capture.devices.append(device)
                self.level.emit(peak)
                self.finished.emit(peak)

        self.cls = Capture


@pytest.mark.parametrize("peak, tone", [(0.2, "ok"), (0.0005, "danger")])
def test_mic_check_reports_sound_or_silence(qapp, peak, tone):
    from gui.widgets.onboarding import OnboardingStrip
    fake = _FakeCapture(peak)
    strip = OnboardingStrip(lambda: "the-selected-mic", capture_factory=fake.cls)
    strip.start_check()
    assert fake.cls.devices == ["the-selected-mic"]
    assert strip._steps.property("tone") == tone
    assert strip._check_btn.isEnabled()
    strip.retranslate()                    # keeps the result, not the steps
    assert strip._steps.property("tone") == tone
    painted(strip)


def test_onboarding_strip_shows_once_until_dismissed(qapp, isolated_prefs):
    from gui.main_window import MainWindow
    window = MainWindow()
    try:
        home = window._home
        assert not home._onboarding.isHidden()          # fresh preferences
        home._onboarding.dismissed.emit()
        assert home._onboarding.isHidden()
        again = HomeScreen()
        assert again._onboarding.isHidden()             # remembered
        home.show_onboarding()                          # Help → Show Getting Started
        assert not home._onboarding.isHidden()
        home.set_file(str(ROOT / "tests" / "missing.wav"), "1 KB")
        assert home._onboarding.isHidden()              # steps done; room for Start
        home.clear_file()
        assert not home._onboarding.isHidden()
        again._dismiss_onboarding()
        again.set_file(str(ROOT / "tests" / "missing.wav"), "1 KB")
        again.clear_file()
        assert again._onboarding.isHidden()             # dismissed stays dismissed
    finally:
        window.close()
        window.deleteLater()


def test_long_output_path_does_not_squeeze_the_footer_buttons(qapp):
    """A long output path used to set the right pane's minimum width, pushing
    the screen past the window and clipping the left pane's button labels."""
    screen = ResultsScreen()
    screen.resize(972, 560)               # what the 1068-wide window leaves it
    screen.load_results(_result(
        "general", output_dir="/Users/someone/Documents/recorder/Recorder/" + "x" * 60))
    painted(screen)
    assert screen.minimumSizeHint().width() <= 972
    for btn in (screen._btn_folder, screen._btn_copy, screen._btn_new):
        assert btn.width() >= btn.sizeHint().width()
    assert "x" * 60 in screen._path_lbl.toolTip()     # full path still reachable


def test_info_rail_has_no_opaque_slab(qapp):
    """The scroll host used to paint the default palette as a grey block."""
    screen = ResultsScreen()
    screen.load_results(_result("general", fidelity={**FIDELITY, "total": 0}))
    assert not screen._rail_scroll.viewport().autoFillBackground()
    assert not screen._rail_host.autoFillBackground()
