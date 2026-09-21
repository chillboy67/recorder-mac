"""
Runner — the child-process entry point that keeps the GUI alive.

The heavy ML runs in a separate process precisely so a native crash cannot take
the app down, which means this function's contract is "report everything through
the queue, never raise". It also maps the GUI's settings dict onto the engine's
keyword arguments, so a silent mismatch here would change what every run does.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import engine, i18n, llm as llm_mod  # noqa: E402
from core.runner import run_pipeline_subprocess  # noqa: E402


class FakeQueue:
    """Stand-in for multiprocessing.Queue: records (kind, payload) messages."""

    def __init__(self, fail_progress: bool = False):
        self.items: list[tuple[str, object]] = []
        self.closed = False
        self._fail_progress = fail_progress

    def put(self, item):
        if self._fail_progress and item[0] == "progress":
            raise BrokenPipeError("parent went away")
        self.items.append(item)

    def close(self):
        self.closed = True


@pytest.fixture
def stub_engine(monkeypatch):
    """Replace engine.run with a recorder and capture the kwargs it receives."""
    captured: dict = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"mode": kwargs.get("mode_key"), "stub": True}

    monkeypatch.setattr(engine, "run", fake_run)
    return captured


# ── happy path ──────────────────────────────────────────────────────

def test_result_is_reported_with_progress_messages(tmp_path, monkeypatch):
    def fake_run(**kwargs):
        kwargs["progress"]({"step": "asr", "percent": 50})
        return {"mode": "general", "stub": True}

    monkeypatch.setattr(engine, "run", fake_run)
    q = FakeQueue()

    run_pipeline_subprocess("in.wav", str(tmp_path), {"mode": "general"}, q)

    assert q.items[0] == ("progress", {"step": "asr", "percent": 50})
    assert q.items[1] == ("result", {"mode": "general", "stub": True})
    assert q.closed is True


def test_settings_are_mapped_onto_the_engine_kwargs(tmp_path, stub_engine):
    settings = {
        "mode": "classroom", "model": "small", "language": "zh",
        "formats": ["txt", "md"], "use_llm": True,
        "llm_model": "mistral", "chinese_model": "qwen3", "ui_lang": "en",
    }
    run_pipeline_subprocess("take.m4a", str(tmp_path), settings, FakeQueue())

    assert stub_engine["input_path"] == "take.m4a"
    assert stub_engine["output_dir"] == str(tmp_path)
    assert stub_engine["mode_key"] == "classroom"
    assert stub_engine["model"] == "small"
    assert stub_engine["language"] == "zh"
    assert stub_engine["formats"] == ["txt", "md"]
    assert stub_engine["use_llm"] is True
    assert stub_engine["llm_model"] == "mistral"
    assert stub_engine["chinese_model"] == "qwen3"


def test_missing_settings_fall_back_to_documented_defaults(tmp_path, stub_engine):
    run_pipeline_subprocess("in.wav", str(tmp_path), {}, FakeQueue())

    assert stub_engine["mode_key"] == "general"
    assert stub_engine["model"] == "large-v3"
    assert stub_engine["use_llm"] is False
    assert stub_engine["language"] is None
    assert stub_engine["llm_model"] == llm_mod.DEFAULT_MODEL
    assert stub_engine["chinese_model"] == llm_mod.DEFAULT_CHINESE_MODEL


def test_empty_llm_model_names_use_the_defaults(tmp_path, stub_engine):
    """The GUI stores "" when the user never picked a model."""
    run_pipeline_subprocess("in.wav", str(tmp_path),
                            {"llm_model": "", "chinese_model": ""}, FakeQueue())
    assert stub_engine["llm_model"] == llm_mod.DEFAULT_MODEL
    assert stub_engine["chinese_model"] == llm_mod.DEFAULT_CHINESE_MODEL


# ── the UI language is applied before the run ───────────────────────

def test_ui_language_is_applied_before_the_engine_runs(tmp_path, monkeypatch):
    """The child is a fresh interpreter, so progress strings would otherwise
    come out in the wrong language."""
    order: list = []
    monkeypatch.setattr(i18n, "set_language", lambda lang: order.append(("lang", lang)))
    monkeypatch.setattr(engine, "run",
                        lambda **kw: order.append(("run", None)) or {})

    run_pipeline_subprocess("in.wav", str(tmp_path), {"ui_lang": "en"}, FakeQueue())

    assert order == [("lang", "en"), ("run", None)]


def test_missing_ui_language_resets_to_the_default(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(i18n, "set_language", lambda lang: seen.append(lang))
    monkeypatch.setattr(engine, "run", lambda **kw: {})

    run_pipeline_subprocess("in.wav", str(tmp_path), None, FakeQueue())

    assert seen == [None]        # set_language(None) → DEFAULT


# ── failure handling ────────────────────────────────────────────────

def test_an_engine_failure_is_reported_as_a_traceback(tmp_path, monkeypatch):
    """The GUI shows this instead of dying with the child."""
    def boom(**kwargs):
        raise RuntimeError("model weights missing")

    monkeypatch.setattr(engine, "run", boom)
    q = FakeQueue()

    run_pipeline_subprocess("in.wav", str(tmp_path), {}, q)   # must not raise

    (kind, payload), = q.items
    assert kind == "error"
    assert "RuntimeError: model weights missing" in payload
    assert "Traceback" in payload
    assert q.closed is True


def test_a_broken_progress_put_does_not_abort_the_run(tmp_path, monkeypatch):
    """Progress is best-effort; losing a message must not lose the result."""
    monkeypatch.setattr(engine, "run", lambda **kw: {"mode": "general"})
    q = FakeQueue(fail_progress=True)

    run_pipeline_subprocess("in.wav", str(tmp_path), {}, q)

    assert q.items == [("result", {"mode": "general"})]
    assert q.closed is True


def test_the_queue_is_closed_even_when_run_raises(tmp_path, monkeypatch):
    def boom(**kwargs):
        raise KeyboardInterrupt        # not an Exception: exercises `finally`

    monkeypatch.setattr(engine, "run", boom)
    q = FakeQueue()

    with pytest.raises(KeyboardInterrupt):
        run_pipeline_subprocess("in.wav", str(tmp_path), {}, q)

    assert q.closed is True
