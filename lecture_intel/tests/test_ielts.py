"""
IELTS analysis — the feedback report, and the fidelity rules behind it.

The report is a headline feature, yet only `analyze` and two helpers had any
coverage. These tests pin each extractor's thresholds (they decide what a
learner is told to work on) and the guarantee that the candidate's transcript is
reproduced verbatim, never corrected in place.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import ASRResult, ASRSegment, ASRWord  # noqa: E402
from core import ielts as I  # noqa: E402
from core.diarize import CANDIDATE, EXAMINER  # noqa: E402

# No server here → LanguageTool is skipped and the offline rules take over,
# which is also what happens for a user without LanguageTool installed.
DEAD_LT = "http://127.0.0.1:1/none"


def w(word, start, end, conf):
    return ASRWord(word=word, start=start, end=end, confidence=conf)


def seg(i, text, start, end, words=(), lang="en"):
    return ASRSegment(id=i, start=start, end=end, text=text, language=lang,
                      confidence=-0.2, words=list(words))


# ── pronunciation ───────────────────────────────────────────────────

def test_pron_flags_only_uncertain_content_words():
    s = seg(0, "Consider ambiguous", 0.0, 2.0, words=[
        w(" Consider", 0.0, 0.8, 0.9),      # first word: boundary artifact, skipped
        w(" ambiguous", 0.8, 1.6, 0.30),    # uncertain → flagged
        w(" the", 1.6, 1.8, 0.20),          # stopword → skipped
        w(" good", 1.8, 2.0, 0.95),         # confident → skipped
    ])
    issues = I._pronunciation_issues([s])
    assert [i.word for i in issues] == ["ambiguous"]
    assert issues[0].note == I._pron_note(0.30)


def test_pron_skips_short_non_alphabetic_and_zero_confidence_tokens():
    s = seg(0, "x", 0.0, 4.0, words=[
        w(" lead", 0.0, 0.2, 0.9),          # filler so the next ones aren't "first"
        w(" com", 0.3, 0.5, 0.10),          # under 4 chars → skipped
        w("算法", 0.5, 0.7, 0.10),           # not alphabetic → skipped
        w(" 1234", 0.7, 0.9, 0.10),         # digits → skipped
        w(" subtle", 0.9, 1.1, 0.0),        # exactly 0.0 → "no probability", skipped
        w(" subtle", 1.1, 1.3, 0.02),       # boundary is exclusive → skipped
        w(" subtle", 1.3, 1.5, 0.03),       # just above → flagged
    ])
    assert [i.word for i in I._pronunciation_issues([s])] == ["subtle"]


def test_pron_reports_the_word_without_surrounding_punctuation():
    s = seg(0, "ok", 0.0, 2.0, words=[
        w(" fine", 0.0, 0.2, 0.9),
        w(" ambiguous,", 0.3, 0.6, 0.20),
    ])
    assert I._pronunciation_issues([s])[0].word == "ambiguous"


def test_pron_sorts_most_uncertain_first_and_caps_at_20():
    # 30 distinct alphabetic 4+ char words, each progressively more confident
    names = [f"term{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(30)]
    words = [w(" okay", 0.0, 0.1, 0.9)]
    words += [w(f" {name}", 0.1 + i, 0.2 + i, 0.10 + i * 0.01)
              for i, name in enumerate(names)]
    issues = I._pronunciation_issues([seg(0, "x", 0.0, 40.0, words=words)])
    assert len(issues) == 20                     # cap
    confidences = [i.confidence for i in issues]
    assert confidences == sorted(confidences)    # most uncertain first


@pytest.mark.parametrize("conf,band", [(0.10, "很低"), (0.30, "低，"),
                                       (0.40, "偏低")])
def test_pron_note_bands(conf, band):
    assert band in I._pron_note(conf)


def test_pron_ignores_a_single_word_segment_prefix_rule():
    """The boundary skip only applies when there are more words to flag."""
    s = seg(0, "ambiguous", 0.0, 1.0, words=[w("ambiguous", 0.0, 1.0, 0.30)])
    assert len(I._pronunciation_issues([s])) == 1


# ── fillers / pauses / wpm ──────────────────────────────────────────

def test_count_fillers_matches_whole_words_only():
    assert I._count_fillers("I like it, um, you know") == 3      # like, um, you know
    assert I._count_fillers("likely umbrella") == 0              # no false positives


def test_count_fillers_is_case_insensitive():
    assert I._count_fillers("Um, UH, ER") == 3


def test_count_long_pauses_counts_gaps_over_the_threshold():
    segs = [seg(0, "a", 0.0, 1.0), seg(1, "b", 1.5, 2.0),     # 0.5s gap
            seg(2, "c", 4.0, 5.0),                            # 2.0s gap → 1
            seg(3, "d", 6.25, 7.0)]                           # 1.25s gap → 2
    assert I._count_long_pauses(segs) == 2


def test_count_long_pauses_is_zero_for_a_single_segment():
    assert I._count_long_pauses([seg(0, "a", 0.0, 5.0)]) == 0


def test_wpm_uses_word_count_over_spoken_time():
    # 6 words in 6 seconds of speech → 60 WPM
    segs = [seg(0, "one two three", 0.0, 3.0), seg(1, "four five six", 3.0, 6.0)]
    assert I._wpm(segs) == 60.0


def test_wpm_is_zero_without_speech():
    assert I._wpm([]) == 0.0
    assert I._wpm([seg(0, "x", 5.0, 5.0)]) == 0.0        # zero duration


# ── grammar ─────────────────────────────────────────────────────────

def test_grammar_falls_back_to_offline_rules_when_languagetool_is_absent():
    issues = I._grammar_issues("I has a pen and an useful tool", DEAD_LT)
    assert [i.category for i in issues] == ["subject-verb", "article"]
    assert "have" in issues[0].message           # \bI has\b → "I 用 have"
    assert "辅音音开头" in issues[1].message      # \ban useful\b
    assert all(i.context for i in issues)


def test_languagetool_returns_none_when_unreachable():
    assert I._languagetool("text", DEAD_LT) is None


def test_grammar_is_empty_for_empty_text():
    assert I._grammar_issues("   ", DEAD_LT) == []


def test_offline_rules_carry_a_replacement_where_one_exists():
    (issue,) = I._offline_grammar("I has a pen")
    assert issue.replacement == "I have"
    assert issue.category == "subject-verb"
    assert "I has" in issue.context


def test_offline_rules_flag_missing_replacements_as_none():
    (issue,) = I._offline_grammar("he have a car")
    assert issue.replacement is None


def test_offline_rules_report_every_match_of_a_pattern():
    assert len(I._offline_grammar("I has one. I has two.")) == 2


# ── naturalness / examiner corrections ──────────────────────────────

def test_naturalness_suggests_a_better_phrase():
    out = I._naturalness("This is very good and a lot of fun")
    assert len(out) == 2
    assert any("very good" in s and "excellent" in s for s in out)
    assert all(s.startswith("「") for s in out)


def test_naturalness_is_empty_for_plain_speech():
    assert I._naturalness("the cat sat on the mat") == []


def test_examiner_corrections_keep_only_cue_bearing_turns():
    segs = [seg(0, "So what did you do last week?", 0.0, 2.0),
            seg(1, "You can say I went hiking instead.", 2.0, 4.0),
            seg(2, "正确的说法是 the day before yesterday", 4.0, 6.0)]
    out = I._examiner_corrections(segs)
    assert out == ["You can say I went hiking instead.",
                   "正确的说法是 the day before yesterday"]


def test_examiner_corrections_are_capped():
    segs = [seg(i, f"you can say thing {i}", float(i), i + 1.0)
            for i in range(15)]
    assert len(I._examiner_corrections(segs)) == 10


# ── markdown rendering ──────────────────────────────────────────────

def _report(**over) -> I.IELTSReport:
    base = dict(transcript_candidate="I am fine", transcript_examiner="",
                candidate_seconds=60.0, examiner_seconds=10.0, other_seconds=0.0,
                words_per_minute=120.0, filler_count=3, long_pause_count=1)
    base.update(over)
    return I.IELTSReport(**base)


def test_render_has_every_section_and_states_the_empty_ones():
    md = I.render_markdown(_report())
    for heading in ("# 雅思口语反馈", "## 概览", "## 疑似发音问题（基于识别置信度，原文未改动）",
                    "## 语法 / 用词问题（仅标注，不改原文）", "## 表达地道度",
                    "## 考生原文（逐字，未修改）"):
        assert heading in md
    assert "- 未发现明显低置信度单词，发音整体较清晰。" in md
    assert "- 未检测到明显语法问题。" in md
    assert "- 未发现明显中式表达。" in md
    assert "I am fine" in md                       # verbatim, untouched


def test_render_mentions_background_speech_only_when_present():
    assert "背景人声" not in I.render_markdown(_report())
    assert "背景人声" in I.render_markdown(_report(other_seconds=12.0))


def test_render_omits_the_examiner_transcript_and_section_when_empty():
    md = I.render_markdown(_report())
    assert "## 教官原文" not in md
    md = I.render_markdown(_report(transcript_examiner="What is your name?"))
    assert "## 教官原文" in md and "What is your name?" in md


def test_render_lists_pron_issues_with_their_note():
    p = I.PronIssue("ambiguous", 3.0, 3.4, 0.2, "识别置信度很低")
    md = I.render_markdown(_report(pron_issues=[p]))
    assert "- **ambiguous**（3.0s，置信度 20%）：识别置信度很低" in md


def test_render_includes_the_suggested_replacement_for_grammar():
    g = I.GrammarIssue("主谓一致：I 用 have。", "I has a pen", "I have", "subject-verb")
    md = I.render_markdown(_report(grammar_issues=[g]))
    assert "- [subject-verb] 主谓一致：I 用 have。　建议：`I have`" in md
    assert "- 原文片段：`I has a pen`" in md


# ── analyze (end to end on a synthetic result) ──────────────────────

def test_analyze_splits_the_two_speakers_and_keeps_the_candidate_verbatim():
    asr = ASRResult(
        segments=[seg(0, "I has an useful idea", 0.0, 4.0, lang="en"),
                  seg(1, "You can say I had an idea", 4.0, 6.0, lang="en"),
                  seg(2, "unrelated background chatter", 6.0, 8.0, lang="en")],
        full_text="x", language="en", model_used="t")
    labels = {0: CANDIDATE, 1: EXAMINER, 2: "other"}

    report = I.analyze(asr, labels, 4.0, 2.0, 2.0, languagetool_url=DEAD_LT)

    assert report.transcript_candidate == "I has an useful idea"   # not corrected
    assert report.transcript_examiner == "You can say I had an idea"
    assert "background" not in report.transcript_candidate
    # 5 words over 4s of candidate speech
    assert report.words_per_minute == pytest.approx(75.0)
    assert report.examiner_corrections == ["You can say I had an idea"]
    assert report.markdown.startswith("# 雅思口语反馈")


def test_analyze_with_no_labels_produces_an_empty_but_valid_report():
    asr = ASRResult(segments=[seg(0, "hello", 0.0, 1.0)],
                    full_text="hello", language="en", model_used="t")
    report = I.analyze(asr, {}, 0.0, 0.0, 0.0, languagetool_url=DEAD_LT)
    assert report.transcript_candidate == ""
    assert report.words_per_minute == 0.0
    assert "（无）" in report.markdown          # the transcript placeholder


def test_join_strips_and_joins_segments():
    assert I._join([seg(0, "  a  ", 0.0, 1.0), seg(1, " b ", 1.0, 2.0)]) == "a b"
    assert I._join([]) == ""
