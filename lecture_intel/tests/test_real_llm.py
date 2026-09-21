"""
Real-Ollama integration tests — opt-in.

The rest of the suite stubs ``_gen``, so the part that actually talks to Ollama
was never exercised. These tests do, because two failures found on a real
machine are invisible to stubs:

* ``resolve_model``/``resolve_first`` only return models that are *pulled* —
  measured here, the app's English default (``mistral``) was absent, so the
  documented degradation to whatever the user has (``qwen3:8b``) is what keeps
  enhancement working at all;
* a generation against a missing model answers **404 with Ollama's own body**
  (``model 'x' not found``), which is why ``_gen`` now logs the body instead of
  httpx's bare ``Client error '404 Not Found'``.

Requirements
------------
* Ollama running: ``ollama serve`` (started manually; these tests never start it)
* at least one model pulled from the app's candidate lists
* a shell WITHOUT the sandbox's proxy vars — a sandboxed shell routes
  localhost through a proxy that answers 502/404, so the fixture clears them

Skipped unless ``RECORDER_REAL_MODEL_TESTS=1``.

Cost note: qwen3:8b takes 25–80 s per generation on Apple Silicon (it is a
thinking model). This file deliberately keeps only **two** generations; measured
in this environment, three or four back-to-back generations get the process
killed for resources. The other two public functions (``correct_lecture``,
``summarize_lecture``) share the same ``_gen`` path and were verified by hand —
measured 27.3 s and 78.4 s, both returning faithful content (see HANDOVER §20.3).

Run with::

    RECORDER_REAL_MODEL_TESTS=1 .venv/bin/python3 -m pytest tests/test_real_llm.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

if not os.environ.get("RECORDER_REAL_MODEL_TESTS"):
    pytest.skip("real-model tests are opt-in: set RECORDER_REAL_MODEL_TESTS=1",
                allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import llm  # noqa: E402

PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
              "ALL_PROXY", "all_proxy")


@pytest.fixture(scope="module", autouse=True)
def direct_localhost():
    """httpx reads these per call, so clearing them is enough to reach Ollama."""
    saved = {k: os.environ.pop(k) for k in PROXY_VARS if k in os.environ}
    yield
    os.environ.update(saved)


@pytest.fixture(scope="module")
def live_model():
    """Whichever model the engine would actually pick on this machine."""
    name = llm.resolve_first((llm.DEFAULT_CHINESE_MODEL,)
                             + llm.CHINESE_CANDIDATES
                             + llm.NON_CHINESE_CANDIDATES)
    if not name:
        pytest.skip("Ollama is not running, or no candidate model is pulled")
    return name


# ── model resolution (no generation) ────────────────────────────────

def test_only_pulled_models_resolve(live_model):
    assert llm.resolve_model(live_model) == live_model
    assert llm.available(live_model) is True
    # a name nobody pulled must come back None, not a hopeful guess
    assert llm.resolve_model("definitely-not-pulled-xyz") is None


def test_the_engine_chain_lands_on_a_pulled_model(live_model):
    """The engine's rule: the caller's preferred model heads the list, then each
    role degrades through whatever else is installed."""
    for asian in (True, False):
        head = llm.DEFAULT_CHINESE_MODEL if asian else llm.DEFAULT_MODEL
        own = llm.CHINESE_CANDIDATES if asian else llm.NON_CHINESE_CANDIDATES
        other = llm.NON_CHINESE_CANDIDATES if asian else llm.CHINESE_CANDIDATES
        assert llm.resolve_first((head,) + own + other) is not None


def test_lang_name_maps_known_codes_and_passes_others_through():
    assert llm._lang_name("zh") == "Chinese"
    assert llm._lang_name("en") == "English"
    assert llm._lang_name("xx-klingon") == "xx-klingon"      # verbatim, no crash


# ── generation (two calls, on purpose) ──────────────────────────────

def test_ielts_feedback_guards_blank_input_without_calling_the_model(live_model):
    assert llm.ielts_feedback("   ", model=live_model) is None


def test_ielts_feedback_returns_real_feedback(live_model):
    """End-to-end: prompt → Ollama chat → non-empty report."""
    out = llm.ielts_feedback("I has an useful idea and I very much like it.",
                             model=live_model)
    assert out and out.strip(), "a real model must produce feedback here"
    assert len(out) > 40


def test_correct_lecture_preserves_the_original_when_nothing_is_wrong(live_model):
    """Fidelity: this function recovers mis-heard words, so already-clean text
    must come back essentially unchanged — not paraphrased."""
    original = "今天讲三件事，请重点记住：第一，算法复杂度；第二，数据结构。"
    out = llm.correct_lecture(original, model=live_model)
    assert out is not None
    # the content words survive; a rewrite/summary would drop some of them
    for token in ("算法复杂度", "数据结构"):
        assert token in out
