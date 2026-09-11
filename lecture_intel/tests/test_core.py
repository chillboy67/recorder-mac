"""
Core-logic regression tests that don't need a Whisper model.

They exercise the IELTS analysis, speaker-label handling, and exporters on a
synthetic ASRResult, so they run in milliseconds and catch wiring bugs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import ASRResult, ASRSegment, ASRWord  # noqa: E402
from core import ielts as I  # noqa: E402
from core import export as E  # noqa: E402
from core.diarize import CANDIDATE, EXAMINER  # noqa: E402
from core.modes import get_mode, MODES  # noqa: E402


def _sample_asr() -> ASRResult:
    segs = [
        ASRSegment(0, 0.0, 3.0, "Can you describe a folk tale you know?", "en", -0.2,
                   [ASRWord("Can", 0, 0.3, 0.99), ASRWord("describe", 0.5, 1.0, 0.95)]),
        ASRSegment(1, 3.2, 12.0,
                   "Well, I would talk about Bluebeard, I heard it at promary school.", "en", -0.3,
                   [ASRWord("Bluebeard", 5.0, 5.6, 0.88),
                    ASRWord("promary", 9.0, 9.6, 0.21),     # low-confidence → flagged
                    ASRWord("school", 9.6, 10.0, 0.97)]),
        ASRSegment(2, 12.5, 14.0, "What happens next?", "en", -0.2,
                   [ASRWord("What", 12.5, 12.8, 0.99)]),
        ASRSegment(3, 15.5, 22.0, "He have many wives and it is a very good story.", "en", -0.25,
                   [ASRWord("have", 14.4, 14.7, 0.6), ASRWord("wives", 15.0, 15.4, 0.4)]),
    ]
    return ASRResult(segments=segs, full_text=" ".join(s.text for s in segs),
                     language="en", model_used="test", audio_duration_sec=22.0)


@pytest.fixture
def labels():
    return {0: EXAMINER, 1: CANDIDATE, 2: EXAMINER, 3: CANDIDATE}


def test_pronunciation_flags_low_confidence(labels):
    asr = _sample_asr()
    # no LanguageTool server in CI → offline rules
    report = I.analyze(asr, labels, 17.0, 5.0, 0.0, languagetool_url="http://127.0.0.1:1/none")
    flagged = {p.word.lower() for p in report.pron_issues}
    assert "promary" in flagged          # the mispronounced word is caught
    assert all(p.confidence < I.PRON_CONFIDENCE_THRESHOLD for p in report.pron_issues)


def test_transcript_is_verbatim(labels):
    asr = _sample_asr()
    report = I.analyze(asr, labels, 17.0, 5.0, 0.0, languagetool_url="http://127.0.0.1:1/none")
    # candidate text must preserve the original errors exactly
    assert "promary" in report.transcript_candidate
    assert "He have" in report.transcript_candidate


def test_offline_grammar_and_naturalness(labels):
    asr = _sample_asr()
    report = I.analyze(asr, labels, 17.0, 5.0, 0.0, languagetool_url="http://127.0.0.1:1/none")
    msgs = " ".join(g.message for g in report.grammar_issues)
    assert "has" in msgs                 # "He have" → has
    assert any("very good" in n for n in report.naturalness)


def test_export_all_formats(tmp_path, labels):
    asr = _sample_asr()
    report = I.analyze(asr, labels, 17.0, 5.0, 0.0, languagetool_url="http://127.0.0.1:1/none")
    out = E.export_all(asr, tmp_path, "s", ["txt", "md", "doc", "docx"],
                       labels=labels, extra_markdown=report.markdown,
                       extra_markdown_suffix="ielts")
    for k in ("txt", "md", "docx", "report"):
        assert out[k].exists() and out[k].stat().st_size > 0
    # .doc relies on macOS textutil; assert when available
    import shutil
    if shutil.which("textutil"):
        assert out["doc"].exists() and out["doc"].stat().st_size > 0
    txt = out["txt"].read_text(encoding="utf-8")
    assert "考生" in txt and "教官" in txt          # speaker labels rendered
    assert "promary" in txt                          # verbatim in output


def test_docx_contains_report_text(tmp_path, labels):
    asr = _sample_asr()
    report = I.analyze(asr, labels, 17.0, 5.0, 0.0, languagetool_url="http://127.0.0.1:1/none")
    out = E.export_all(asr, tmp_path, "s", ["docx"], labels=labels,
                       extra_markdown=report.markdown)
    from docx import Document
    text = "\n".join(p.text for p in Document(str(out["docx"])).paragraphs)
    assert "promary" in text          # verbatim transcript inside the Word doc


def test_repetition_collapse():
    from core.transcriber import _collapse_repeats
    # Whisper hallucination loops collapse to one copy
    assert _collapse_repeats("about this " * 8).strip() == "about this"
    assert len(_collapse_repeats("no no no no no no").split()) <= 2   # loop gone
    # Chinese (no spaces): a runaway char/phrase loop collapses
    assert _collapse_repeats("時" * 200) == "時"
    assert "時" * 10 not in _collapse_repeats("算法" + "時" * 150 + "分析")
    # genuine emphasis (3x) and normal text are left alone
    assert _collapse_repeats("well no no no I disagree") == "well no no no I disagree"
    assert _collapse_repeats("I really like it a lot") == "I really like it a lot"


def test_modes_present():
    assert set(MODES) == {"general", "classroom", "ielts"}
    assert get_mode("ielts").analyze_ielts is True
    assert get_mode("classroom").denoise is True
    assert get_mode("general").diarize is False


# ── language-based examiner / candidate split ────────────────────────

def _seg(text: str, language: str = "en") -> ASRSegment:
    return ASRSegment(id=0, start=0.0, end=1.0, text=text,
                      language=language, confidence=-0.2)


def test_coach_split_matches_old_chinese_test_on_zh_en():
    """This split is what makes examiner/candidate separation usable, so the
    zh/en case must be bit-for-bit what the old hard-coded `_has_chinese` test
    produced. These segments carry the candidate's language, so the script check
    decides — exactly as it did before."""
    import re
    from core.diarize import _is_coach_segment

    old_has_chinese = lambda t: len(re.findall(r"[一-鿿]", t)) >= 2
    for text in [
        "Can you describe a folk tale you know?",
        "Well, I would talk about Bluebeard, I heard it at promary school.",
        "这里要注意时态，你用了过去式。",
        "这个 very good",
        "我 said something",          # one stray char is not a turn
        "What happens next?",
        "",
    ]:
        assert _is_coach_segment(_seg(text)) == old_has_chinese(text), text


@pytest.mark.parametrize("text", [
    "すみません、時制が間違っています",        # Japanese coach
    "시제를 잘못 쓰셨어요",                    # Korean coach
    "Обратите внимание на время глагола",      # Russian coach
    "คุณใช้ไวยากรณ์ผิดตรงนี้",                 # Thai coach
])
def test_coach_split_recognises_any_coach_language(text):
    """The coach no longer has to be Chinese-speaking."""
    from core.diarize import _is_coach_segment
    assert _is_coach_segment(_seg(text))


def test_latin_script_coach_needs_the_segment_language():
    """A Latin-script coach writes the same letters as the English candidate, so
    no script test can separate them — the per-chunk language the chunked ASR
    path attaches to each segment is the signal that does."""
    from core.diarize import _is_coach_segment

    french = "Attention, vous avez utilisé le passé composé."
    assert _is_coach_segment(_seg(french, "fr"))
    assert _is_coach_segment(_seg(french, "de"))
    # Without a per-segment language it is indistinguishable from the candidate.
    assert not _is_coach_segment(_seg(french, "en"))


def test_script_fallback_still_covers_what_language_cannot():
    """Short turns, and the single-pass / faster-whisper paths, carry no usable
    per-segment language — the script test has to keep doing the work there."""
    from core.diarize import _is_coach_segment
    assert _is_coach_segment(_seg("这里要注意时态"))              # language defaults to en
    assert not _is_coach_segment(_seg("a perfectly normal English answer"))


def test_mixed_segments_defer_to_the_script_check():
    """A zh/en code-switched turn is labelled "mixed"; it must not be swept to
    the coach's side merely for not being "en"."""
    from core.diarize import _is_coach_segment
    assert _is_coach_segment(_seg("这个 very good", "mixed"))
    assert not _is_coach_segment(_seg("we switched to English here", "mixed"))


def test_foreign_script_count_respects_the_reference_language():
    from core.languages import foreign_script_count
    assert foreign_script_count("日本語です", "en") >= 2       # foreign to English
    assert foreign_script_count("日本語です", "ja") == 0       # native to Japanese
    assert foreign_script_count("한국어입니다", "ja") >= 2     # Hangul is not Japanese
    assert foreign_script_count("hello there", "en") == 0
