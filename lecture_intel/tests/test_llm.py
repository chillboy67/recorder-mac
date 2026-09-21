"""
LLM prompt-language tests — no Ollama needed.

`_gen` is stubbed to capture the system prompt each function builds, so these
pin the contract added with the multilingual upgrade: prompts are written in the
transcript's own language, with Chinese kept verbatim for zh/mixed/unknown
(which is exactly the pre-upgrade behavior).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import llm  # noqa: E402
from core.languages import LANGUAGE_NAMES_EN, PICKER_LANGUAGES  # noqa: E402


@pytest.fixture
def spy(monkeypatch):
    calls: list[tuple[str, str]] = []

    def fake_gen(prompt: str, system: str = "", **kw):
        calls.append((prompt, system))
        return "stub"

    monkeypatch.setattr(llm, "_gen", fake_gen)
    return calls


def test_correct_prompt_follows_the_transcript_language(spy):
    llm.correct_transcript("Bonjour le monde", language="fr")
    assert "French" in spy[-1][1]

    llm.correct_transcript("你好世界", language="zh")
    assert "校对员" in spy[-1][1]

    # code-switching and unknown transcripts keep the Chinese prompt, as before
    llm.correct_transcript("mixed 内容 here", language="mixed")
    assert "校对员" in spy[-1][1]
    llm.correct_transcript("hello there", language=None)
    assert "校对员" in spy[-1][1]


def test_correct_context_line_matches_the_prompt_language(spy):
    llm.correct_transcript("x", context="一节课的课堂录音", language="zh")
    assert "背景" in spy[-1][1]

    llm.correct_transcript("x", context="a classroom lecture recording", language="de")
    assert "Context:" in spy[-1][1]


def test_summarize_prompts_follow_the_language(spy):
    llm.summarize_lecture("lecture text", language="ja")
    systems = [s for _, s in spy]
    assert any("Japanese" in s for s in systems)      # map step names it
    assert any("Japanese" in s for s in systems[1:])  # reduce step names it too

    llm.summarize_lecture("课堂内容", language="zh")
    assert "课堂笔记助手" in spy[-1][1]


def test_tidy_prompt_follows_the_language(spy):
    llm.tidy_transcript("text", language="zh")
    assert "文字整理助手" in spy[-1][1]

    llm.tidy_transcript("text", language="es")
    assert "text-tidying assistant" in spy[-1][1]


def test_every_picker_language_has_an_english_name():
    for code, _ in PICKER_LANGUAGES:
        if code == "auto":
            continue
        assert LANGUAGE_NAMES_EN.get(code), f"{code} has no English name for prompts"


def test_resolve_first_degrades_through_installed_candidates(monkeypatch):
    """A missing default must not kill enhancement: each role walks its
    candidate list and takes the first model the user actually pulled."""
    from core import llm

    monkeypatch.setattr(llm, "_installed",
                        lambda host=None: ["llama3.1:8b", "qwen2.5:7b"])
    # qwen is Chinese-role only: the non-Chinese chain skips it and lands on
    # the newest non-Qwen model installed, here llama3.1 (2024-07)
    assert llm.resolve_first(llm.NON_CHINESE_CANDIDATES) == "llama3.1:8b"
    # qwen3 absent → the Chinese list lands on qwen2.5
    assert llm.resolve_first(llm.CHINESE_CANDIDATES) == "qwen2.5:7b"
    # nothing pulled at all → None, and the caller skips enhancement
    monkeypatch.setattr(llm, "_installed", lambda host=None: [])
    assert llm.resolve_first(llm.NON_CHINESE_CANDIDATES) is None


# ── _gen's error reporting ──────────────────────────────────────────

class _FakeResponse:
    def __init__(self, status: int, text: str, payload: dict | None = None):
        self.status_code = status
        self.text = text
        self._payload = payload

    def json(self):
        assert self._payload is not None, "a failed body must never be parsed"
        return self._payload


def test_gen_surfaces_ollamas_error_body(monkeypatch, caplog):
    """A model that was never pulled answers 404 with `model 'x' not found`.

    `raise_for_status()` reports only "Client error '404 Not Found'", which
    hides the one fact that identifies the problem — measured on a real machine
    where the default `mistral` was absent and every generation looked like a
    broken endpoint.
    """
    import httpx
    from core import llm

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResponse(
        404, '{"error":"model \'mistral\' not found"}'))

    with caplog.at_level("WARNING"):
        assert llm._gen("hi", model="mistral") is None

    assert "404" in caplog.text
    assert "model 'mistral' not found" in caplog.text


def test_gen_returns_the_message_content(monkeypatch):
    import httpx
    from core import llm

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResponse(
        200, "{}", {"message": {"role": "assistant", "content": "  hello  "}}))
    assert llm._gen("hi", model="m") == "hello"


def test_gen_returns_empty_string_when_content_is_missing(monkeypatch):
    """A thinking model can return a response with no `content` at all; the
    callers distinguish None (failed) from "" (ran, said nothing)."""
    import httpx
    from core import llm

    monkeypatch.setattr(httpx, "post", lambda *a, **kw: _FakeResponse(
        200, "{}", {"message": {"role": "assistant", "thinking": "..."}}))
    assert llm._gen("hi", model="m") == ""


def test_gen_returns_none_when_the_server_is_unreachable(monkeypatch):
    import httpx
    from core import llm

    def boom(*a, **kw):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(httpx, "post", boom)
    assert llm._gen("hi", model="m") is None
