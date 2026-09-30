"""
Reports and exported files follow the UI language; the speaker's words don't.

The pipeline subprocess sets the language from the GUI's choice
(core.runner), so the IELTS report, the classroom summary and the labels in
exported files come out in that language. The Chinese output is covered by
the per-module tests; these check the English one, and that quoted speech is
copied through untouched in either language.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules import ASRResult, ASRSegment  # noqa: E402
from core import classroom, export, ielts  # noqa: E402
from core.diarize import CANDIDATE, EXAMINER  # noqa: E402
from core.i18n import current_language, set_language  # noqa: E402

CJK = re.compile(r"[　-〿一-鿿＀-￯]")
DEAD_LT = "http://127.0.0.1:9/v2/check"


@pytest.fixture
def english():
    before = current_language()
    set_language("en")
    yield
    set_language(before)


def _seg(i, text, start, end, lang="en"):
    return ASRSegment(id=i, start=start, end=end, text=text, language=lang, confidence=-0.1)


def _without(text: str, *quoted: str) -> str:
    for q in quoted:
        text = text.replace(q, "")
    return text


def test_ielts_report_in_english(english):
    spoken = "I has an useful idea and very good food"
    coach = "你可以说 I had an idea"
    asr = ASRResult(segments=[_seg(0, spoken, 0.0, 6.0), _seg(1, coach, 6.0, 8.0, "zh")],
                    full_text="x", language="en", model_used="t")

    report = ielts.analyze(asr, {0: CANDIDATE, 1: EXAMINER}, 6.0, 2.0, 0.0,
                           languagetool_url=DEAD_LT)

    md = report.markdown
    assert md.startswith("# IELTS Speaking Feedback")
    assert "## Candidate transcript (verbatim, unedited)" in md
    assert spoken in md and coach in md                     # words untouched
    assert "Subject–verb agreement: I takes have." in md    # offline rule
    assert "“very good” could be more natural" in md
    assert not CJK.search(_without(md, coach))


def test_classroom_summary_in_english_keeps_the_lecture_verbatim(english):
    said = "注意力是指把资源集中到某处"
    report = classroom.summarize(ASRResult(
        segments=[_seg(0, "第一句重点内容", 0.0, 30.0, "zh"), _seg(1, said, 31.0, 60.0, "zh")],
        full_text="x", language="zh", model_used="t"))

    md = report.markdown
    assert md.startswith("# Lecture Key Points")
    assert "## Full transcript" in md and said in md
    lecture = [s for s in md.splitlines() if CJK.search(s)]
    assert all(("第一句重点内容" in s or "注意力" in s or "资源" in s) for s in lecture), lecture


def test_exported_labels_and_fidelity_notes_in_english(english, tmp_path):
    words = "我我我觉得"
    asr = ASRResult(segments=[_seg(0, "hello there", 0.0, 2.0), _seg(1, words, 2.0, 4.0, "zh")],
                    full_text="x", language="en", model_used="mlx:large-v3")
    asr.annotations.append({"type": "repeat_arbitration", "segment_id": 1, "start": 2.0,
                            "end": 4.0, "original": words, "verdict": "real_speech",
                            "folded": False, "evidence": {"level": 2, "voiced_bursts": 3, "k": 3}})

    out = export.export_all(asr, tmp_path, "s", ["md", "txt"],
                            labels={0: CANDIDATE, 1: EXAMINER})

    md = out["md"].read_text(encoding="utf-8")
    assert "**Candidate**" in md and "**Coach**" in md
    assert "- Language: en" in md and "## Transcript" in md
    assert "## Fidelity notes (transcript not rewritten)" in md
    assert "Real repetition · kept as spoken | said: “我我我觉得”" in md
    assert "L2 3 voiced bursts ≥ 0.8×3" in md
    assert not CJK.search(_without(md, words))
    assert "Coach: 我我我觉得" in out["txt"].read_text(encoding="utf-8")
