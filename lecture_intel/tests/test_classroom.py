"""
Classroom analysis heuristics — pure text logic, no audio needed.

`summarize` is what the classroom report is built from, and its four extractors
were previously reached by only one incidental reference in the suite. These
tests pin the behaviour that decides what a student sees as "the key points".
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import ASRResult, ASRSegment  # noqa: E402
from core import classroom as C  # noqa: E402


def seg(i: int, text: str, start: float, end: float) -> ASRSegment:
    return ASRSegment(id=i, start=start, end=end, text=text, language="zh",
                      confidence=-0.1)


def asr(*segs: ASRSegment) -> ASRResult:
    return ASRResult(segments=list(segs),
                     full_text="\n".join(s.text for s in segs),
                     language="zh", model_used="t")


# ── emphasis ────────────────────────────────────────────────────────

def test_emphasis_cues_match_both_languages():
    segs = [
        seg(0, "这个是重点", 0.0, 2.0),
        seg(1, "note that this matters", 3.0, 5.0),
        seg(2, "今天天气不错", 6.0, 8.0),          # no cue
        seg(3, "记住", 9.0, 10.0),                # cue but shorter than 4 chars
    ]
    points = C._emphasis_points(segs)
    assert len(points) == 2
    assert any("这个是重点" in p for p in points)
    assert any("note that" in p for p in points)
    assert all("今天天气" not in p for p in points)
    assert all("] 记住" not in p for p in points)   # length guard holds
    assert points[0].startswith("[0:00] ")          # timestamp prefix


def test_emphasis_dedupes_on_the_first_20_chars_and_caps_at_30():
    # same opening 20 chars → collapsed to one entry
    segs = [seg(i, "重点内容" + "甲" * 30, float(i), i + 1.0) for i in range(3)]
    assert len(C._emphasis_points(segs)) == 1
    many = [seg(i, f"注意第{i}项内容够长了吧", float(i) * 2, i * 2 + 1.0)
            for i in range(40)]
    assert len(C._emphasis_points(many)) == 30      # hard cap


# ── definitions ─────────────────────────────────────────────────────

def test_definitions_need_a_cue_and_a_sane_length():
    full = "\n".join([
        "注意力是指把资源集中到某处",     # cue, subject and length both ok
        "短是指",                        # under 6 chars
        "无关的句子在这里",              # no cue
        "A is defined as b",             # English cue, length ok
    ])
    assert C._definitions(full) == ["注意力是指把资源集中到某处",
                                   "A is defined as b"]


def test_definitions_need_a_subject_of_at_least_two_chars():
    """The cue is preceded by `(.{2,30}?)`, so a one-character subject does not
    produce a definition — a real boundary of the heuristic, pinned here."""
    assert C._definitions("熵是指系统的混乱程度") == []
    assert C._definitions("熵值是指系统的混乱程度") == ["熵值是指系统的混乱程度"]


def test_definitions_are_unique_and_capped():
    line = "注意力是指把资源集中到某处"
    assert C._definitions("\n".join([line] * 3)) == [line]
    many = "\n".join(f"术语{i}是指某个概念的含义" for i in range(30))
    assert len(C._definitions(many)) == 20


# ── frequent terms ──────────────────────────────────────────────────

def test_frequent_terms_need_the_floor_of_four_occurrences():
    assert C._frequent_terms("algorithm " * 4) == [("algorithm", 4)]
    assert C._frequent_terms("algorithm " * 3) == []      # below the floor


def test_frequent_terms_skip_stopwords_and_short_words():
    # "the" is a stopword, "cat" is under 4 chars → neither is a term
    assert C._frequent_terms("the the the the cat cat cat cat") == []


def test_frequent_terms_find_chinese_bigrams_without_function_chars():
    terms = dict(C._frequent_terms("算法算法算法算法"))
    assert terms.get("算法") == 4
    # 的/了/是 are function chars: every bigram touching them is dropped
    assert C._frequent_terms("的话的话的话的话") == []


# ── long stretches ──────────────────────────────────────────────────

def test_long_stretches_split_on_gaps_and_keep_the_longest():
    # one 40s continuous topic, then a 6s gap, then a 10s topic
    a = [seg(0, "甲", 0.0, 10.0), seg(1, "乙", 10.0, 20.0),
         seg(2, "丙", 20.0, 30.0), seg(3, "丁", 30.0, 40.0)]
    b = [seg(4, "戊", 46.0, 56.0)]
    stretches = C._long_stretches(a + b)
    assert len(stretches) == 1                 # the 10s block is under min_sec
    assert stretches[0].startswith("[0:00–0:40，约40s]")
    assert stretches[0].endswith("…")


def test_long_stretches_are_capped_and_ordered_by_duration():
    segs, t = [], 0.0
    for i in range(7):                         # 7 separate 30s topics
        segs.append(seg(len(segs), f"话题{i}", t, t + 30.0))
        t += 40.0                              # >4s gap splits them
    stretches = C._long_stretches(segs)
    assert len(stretches) == 5                 # top 5 only
    assert "话题0" in stretches[0]             # all equal length → stable order


def test_long_stretches_empty_input():
    assert C._long_stretches([]) == []


# ── summarize / render ──────────────────────────────────────────────

def test_summarize_skips_blank_segments_and_renders_every_section():
    result = asr(seg(0, "第一句重点内容", 0.0, 30.0),
                 seg(1, "   ", 30.0, 31.0),          # blank → dropped
                 seg(2, "  注意力是指把资源集中到某处  ", 31.0, 60.0))
    report = C.summarize(result)
    assert "   " not in report.transcript.splitlines()
    assert report.transcript == "第一句重点内容\n注意力是指把资源集中到某处"
    assert report.emphasis_points and report.definitions
    for heading in ("# 课堂重点总结", "## 老师强调的重点", "## 重要定义",
                    "## 高频主题（反复出现，可能是核心）",
                    "## 讲解篇幅最长的部分", "## 全文转写"):
        assert heading in report.markdown


def test_summarize_states_when_nothing_was_detected():
    """Empty sections must say so, not render as an empty list."""
    report = C.summarize(asr(seg(0, "今天天气不错", 0.0, 5.0)))
    assert report.emphasis_points == [] and report.definitions == []
    assert "- 未检测到明显的「重点/注意/常考」等强调用语。" in report.markdown
    assert "- 未检测到明显的定义句。" in report.markdown
    assert "（无）" in report.markdown


def test_summarize_handles_a_result_with_no_segments():
    report = C.summarize(asr())
    assert report.transcript == "" and report.markdown
    assert "（无）" in report.markdown          # transcript placeholder


def test_mmss_formats_minutes_and_hours():
    assert C._mmss(0) == "0:00"
    assert C._mmss(65) == "1:05"
    assert C._mmss(3661) == "1:01:01"
