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
