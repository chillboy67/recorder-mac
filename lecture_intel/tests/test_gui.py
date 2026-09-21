"""
Results-screen smoke tests — offscreen rendering only, never a real window.

These exist because a local variable named ``t`` shadowed the i18n ``t()``
helper inside ``ResultsScreen.load_results``, so the method raised
UnboundLocalError for **every** mode: the results screen came up empty and the
core-logic suite (which is Qt-free) could not notice. Rendering offscreen is the
sanctioned way to check GUI wiring without launching the app.

Opt-in on purpose
-----------------
They are skipped unless ``RECORDER_GUI_TESTS=1``, because initialising Qt
changes what else may run in the same process: in a sandboxed shell a process
that has created a QApplication gets killed on its next localhost HTTP request.
``test_core.py`` (LanguageTool on 127.0.0.1:1) and ``test_llm.py`` (Ollama on
127.0.0.1:11434) both open localhost connections, so mixing them makes the
default ``python -m pytest`` die mid-run. Keeping Qt out of the default suite
keeps that command reliable; run these explicitly:

    RECORDER_GUI_TESTS=1 .venv/bin/python3 -m pytest tests/test_gui_results.py
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

from PySide6.QtWidgets import QApplication  # noqa: E402

from gui.widgets.results_screen import ResultsScreen  # noqa: E402

FIDELITY = {
    "asr_loop": 1, "real_speech": 2, "uncertain": 1, "adjacent_duplicates": 0,
    "folded_count": 1, "dropped_count": 0, "total": 4,
    "lines": ["[0:12–0:15] 疑似转写伪影 · 已在正文中折叠 ｜原话「打打打打」",
              "[0:31–0:33] 真实重复 · 正文原样保留 ｜原话「我 我」"],
}


@pytest.fixture(scope="module")
def qapp():
    yield QApplication.instance() or QApplication([])


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


def test_fidelity_tab_and_card_appear(qapp):
    """The annotations have to be visible, not just written to meta.json."""
    screen = ResultsScreen()
    screen.load_results(_result("general", fidelity=FIDELITY))
    tabs = [screen._tabs.tabText(i) for i in range(screen._tabs.count())]
    assert "忠实度标注" in tabs
    shown = screen._tabs.widget(tabs.index("忠实度标注")).toPlainText()
    assert "打打打打" in shown and "已在正文中折叠" in shown


def test_fidelity_card_states_nothing_was_flagged(qapp):
    """An empty trail still has to say so — silence would read as 'unknown'."""
    from PySide6.QtWidgets import QLabel

    screen = ResultsScreen()
    screen.load_results(_result("general", fidelity={
        **FIDELITY, "total": 0, "asr_loop": 0, "real_speech": 0,
        "uncertain": 0, "folded_count": 0, "dropped_count": 0, "lines": []}))
    texts = [w.text() for w in screen._rail_host.findChildren(QLabel)]
    assert any("未发现疑似伪影" in text for text in texts)
    assert screen._tabs.count() == 0        # nothing to show as a tab


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
