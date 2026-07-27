"""
IELTS speaking analysis.

The point of this module (per the user) is NOT the feedback template — it's the
*detection* of two things that ordinary transcription apps either can't do or
actively hide:

  1. **Likely pronunciation problems.** Most apps silently "fix" what they hear.
     We instead surface where Whisper's acoustic model was *unsure* — low
     per-word confidence is a strong proxy for unclear / mispronounced / heavily
     accented words. We keep the original text verbatim and just *flag* these.

  2. **Grammar / phrasing problems, without rewriting.** We report issues
     (tense, agreement, articles, collocations, Chinglish phrasings) with a
     suggested fix, but the transcript itself is never altered.

Everything is offline. Grammar uses a local LanguageTool server if one is
running (better), otherwise a built-in rule set. The examiner's turns are used
only as *reference corrections*, never mixed into the candidate's transcript.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from modules import ASRResult, ASRSegment
from core.diarize import CANDIDATE, EXAMINER, OTHER

logger = logging.getLogger(__name__)

# Whisper word probability below this = "the model wasn't sure it heard this".
PRON_CONFIDENCE_THRESHOLD = 0.45
# Don't flag ultra-common function words even if confidence dips — not useful.
# Common words are excluded: a flagged "the"/"what" is rarely an actionable
# pronunciation note, and they dominate false positives from boundary artifacts.
_PRON_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "so", "of", "to", "in", "on", "at",
    "is", "am", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "have", "has", "had", "i", "you", "he", "she", "it", "we", "they", "me",
    "him", "her", "them", "my", "your", "his", "its", "our", "their", "this",
    "that", "these", "those", "here", "there", "what", "when", "where", "who",
    "why", "how", "which", "all", "some", "any", "more", "most", "much", "many",
    "one", "two", "well", "like", "just", "now", "then", "than", "as", "if",
    "for", "with", "from", "by", "about", "can", "will", "would", "could",
    "should", "may", "might", "not", "no", "yes", "yeah", "okay", "ok", "oh",
    "uh", "um", "er", "erm", "mm", "hmm", "also", "very", "really", "kind",
    "sort", "thing", "things", "people", "because", "actually", "maybe",
}
_FILLERS = {"um", "uh", "er", "erm", "mm", "like", "you know", "kind of", "sort of"}
LONG_PAUSE_SEC = 1.2


@dataclass
class PronIssue:
    word: str
    start: float
    end: float
    confidence: float
    note: str


@dataclass
class GrammarIssue:
    message: str
    context: str
    replacement: Optional[str]
    category: str


@dataclass
class IELTSReport:
    transcript_candidate: str          # verbatim, never altered
    transcript_examiner: str
    candidate_seconds: float
    examiner_seconds: float
    other_seconds: float
    words_per_minute: float
    filler_count: int
    long_pause_count: int
    pron_issues: list[PronIssue] = field(default_factory=list)
    grammar_issues: list[GrammarIssue] = field(default_factory=list)
    naturalness: list[str] = field(default_factory=list)
    examiner_corrections: list[str] = field(default_factory=list)
    markdown: str = ""


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def analyze(
    asr: ASRResult,
    labels: dict[int, str],
    candidate_seconds: float,
    examiner_seconds: float,
    other_seconds: float,
    languagetool_url: str = "http://127.0.0.1:8010/v2/check",
) -> IELTSReport:
    cand_segs = [s for s in asr.segments if labels.get(s.id) == CANDIDATE]
    exam_segs = [s for s in asr.segments if labels.get(s.id) == EXAMINER]

    cand_text = _join(cand_segs)
    exam_text = _join(exam_segs)

    pron = _pronunciation_issues(cand_segs)
    fillers = _count_fillers(cand_text)
    pauses = _count_long_pauses(cand_segs)
    wpm = _wpm(cand_segs)
    grammar = _grammar_issues(cand_text, languagetool_url)
    natural = _naturalness(cand_text)
    corrections = _examiner_corrections(exam_segs)

    report = IELTSReport(
        transcript_candidate=cand_text,
        transcript_examiner=exam_text,
        candidate_seconds=candidate_seconds,
        examiner_seconds=examiner_seconds,
        other_seconds=other_seconds,
        words_per_minute=wpm,
        filler_count=fillers,
        long_pause_count=pauses,
        pron_issues=pron,
        grammar_issues=grammar,
        naturalness=natural,
        examiner_corrections=corrections,
    )
    report.markdown = render_markdown(report)
    return report


# ----------------------------------------------------------------------
# Pronunciation (confidence-based)
# ----------------------------------------------------------------------

def _pronunciation_issues(segs: list[ASRSegment]) -> list[PronIssue]:
    issues: list[PronIssue] = []
    for seg in segs:
        for i, w in enumerate(seg.words):
            # The first word of a segment routinely gets a spurious ~0
            # probability (an ASR boundary artifact, esp. faster-whisper), so
            # skip it — it's noise, not a real pronunciation signal.
            if i == 0 and len(seg.words) > 1:
                continue
            token = re.sub(r"[^\w']", "", w.word).strip().lower()
            if not token or token in _PRON_STOPWORDS:
                continue
            # only multi-syllable-ish alphabetic English words (skip Chinese,
            # numbers, and short fragments like "com"/"per" that ASR over-splits)
            if not re.fullmatch(r"[a-z']+", token) or len(token) < 4:
                continue
            # Exactly-zero = "no probability computed" (artifact), not "unsure".
            if 0.02 < w.confidence < PRON_CONFIDENCE_THRESHOLD:
                issues.append(PronIssue(
                    word=w.word.strip().strip(".,!?;:\"')("),
                    start=w.start,
                    end=w.end,
                    confidence=w.confidence,
                    note=_pron_note(w.confidence),
                ))
    # most-uncertain first, cap to keep the report focused
    issues.sort(key=lambda x: x.confidence)
    return issues[:20]


def _pron_note(conf: float) -> str:
    if conf < 0.25:
        return "识别置信度很低，发音可能很不清晰或读错，建议重点核对。"
    if conf < 0.35:
        return "识别置信度低，发音可能不够清楚或重音/元音有偏差。"
    return "识别置信度偏低，建议确认发音是否标准。"


def _count_fillers(text: str) -> int:
    low = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(f)}\b", low)) for f in _FILLERS)


def _count_long_pauses(segs: list[ASRSegment]) -> int:
    count = 0
    prev_end = None
    for seg in segs:
        if prev_end is not None and seg.start - prev_end > LONG_PAUSE_SEC:
            count += 1
        prev_end = seg.end
    return count


def _wpm(segs: list[ASRSegment]) -> float:
    if not segs:
        return 0.0
    words = sum(len(s.text.split()) for s in segs)
    dur = sum(s.end - s.start for s in segs)
    if dur <= 0:
        return 0.0
    return round(words / (dur / 60.0), 1)


# ----------------------------------------------------------------------
# Grammar (LanguageTool if available, else offline rules)
# ----------------------------------------------------------------------

def _grammar_issues(text: str, lt_url: str) -> list[GrammarIssue]:
    if not text.strip():
        return []
    issues = _languagetool(text, lt_url)
    if issues is not None:
        return issues
    return _offline_grammar(text)


def _languagetool(text: str, url: str) -> Optional[list[GrammarIssue]]:
    try:
        import httpx
        resp = httpx.post(url, data={"text": text, "language": "en-US"}, timeout=4.0)
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None  # server not running → caller falls back to offline rules

    out: list[GrammarIssue] = []
    for m in payload.get("matches", []):
        reps = m.get("replacements") or []
        out.append(GrammarIssue(
            message=m.get("message", "可能的语法问题"),
            context=m.get("context", {}).get("text", ""),
            replacement=reps[0]["value"] if reps else None,
            category=m.get("rule", {}).get("category", {}).get("id", "grammar").lower(),
        ))
    return out


_OFFLINE_RULES: list[tuple[str, str, Optional[str], str]] = [
    (r"\bI has\b", "主谓一致：I 用 have。", "I have", "subject-verb"),
    (r"\b(he|she|it) have\b", "第三人称单数用 has。", None, "subject-verb"),
    (r"\b(people|they|we) is\b", "复数主语用 are。", None, "subject-verb"),
    (r"\ban useful\b", "useful 以辅音音开头，用 a。", "a useful", "article"),
    (r"\ba hour\b", "hour 以元音音开头，用 an。", "an hour", "article"),
    (r"\bmore better\b", "避免双重比较级。", "better", "comparative"),
    (r"\bdiscuss about\b", "discuss 后不加 about。", "discuss", "collocation"),
    (r"\binformations\b", "information 不可数。", "information", "noun-form"),
    (r"\badvices\b", "advice 不可数。", "advice", "noun-form"),
    (r"\bknowledges\b", "knowledge 不可数。", "knowledge", "noun-form"),
    (r"\bvery much like\b", "更自然：really like。", "really like", "phrasing"),
    (r"\baccording to me\b", "中式表达：用 in my opinion / personally。", "in my opinion", "phrasing"),
    (r"\bcan able to\b", "can 与 be able to 不能连用。", "can / am able to", "modal"),
    (r"\bnowadays\b.*\bnowadays\b", "nowadays 重复使用，注意多样性。", None, "repetition"),
]


def _offline_grammar(text: str) -> list[GrammarIssue]:
    out: list[GrammarIssue] = []
    for pattern, msg, rep, cat in _OFFLINE_RULES:
        for m in re.finditer(pattern, text, flags=re.IGNORECASE):
            ctx = text[max(0, m.start() - 30): m.end() + 30]
            out.append(GrammarIssue(message=msg, context=ctx.strip(),
                                    replacement=rep, category=cat))
    return out


# ----------------------------------------------------------------------
# Naturalness (Chinglish / weak word choice)
# ----------------------------------------------------------------------

_NATURAL = {
    "very good": "impressive / excellent",
    "very important": "essential / crucial",
    "a lot of": "a great deal of / plenty of",
    "many many": "a great many",
    "make me happy": "lift my mood / cheer me up",
    "big problem": "serious issue",
    "i think i think": "(重复) I'd say / In my view",
    "how to say": "how can I put it",
    "more and more": "increasingly",
    "in my country": "where I'm from / back home",
    "delicious food": "great food / a wide variety of dishes",
}


def _naturalness(text: str) -> list[str]:
    low = text.lower()
    out = []
    for plain, better in _NATURAL.items():
        if plain in low:
            out.append(f"「{plain}」可换成更地道的表达，如「{better}」。")
    return out


# ----------------------------------------------------------------------
# Examiner corrections (reference only)
# ----------------------------------------------------------------------

def _examiner_corrections(exam_segs: list[ASRSegment]) -> list[str]:
    """Pull examiner turns that look like a correction / model phrasing."""
    cues = re.compile(
        r"(you can say|you should say|better to say|we say|the correct|"
        r"应该说|可以说|正确的说法|更地道|不是.*而是)", re.IGNORECASE,
    )
    out = []
    for s in exam_segs:
        if cues.search(s.text):
            out.append(s.text.strip())
    return out[:10]


# ----------------------------------------------------------------------
# Report rendering
# ----------------------------------------------------------------------

def render_markdown(r: IELTSReport) -> str:
    L: list[str] = []
    L.append("# 雅思口语反馈")
    L.append("")
    L.append("## 概览")
    L.append(f"- 考生发言时长：约 {r.candidate_seconds:.0f}s")
    L.append(f"- 教官发言时长：约 {r.examiner_seconds:.0f}s")
    if r.other_seconds > 0:
        L.append(f"- 背景人声（已排除）：约 {r.other_seconds:.0f}s")
    L.append(f"- 语速：约 {r.words_per_minute:.0f} WPM")
    L.append(f"- 填充词（um/uh/like 等）：{r.filler_count} 处")
    L.append(f"- 明显停顿（>{LONG_PAUSE_SEC:.1f}s）：{r.long_pause_count} 处")
    L.append("")

    L.append("## 疑似发音问题（基于识别置信度，原文未改动）")
    if r.pron_issues:
        L.append("> 以下单词识别置信度偏低，往往对应发音不清/读错/口音偏差，建议逐一核对录音。")
        L.append("")
        for p in r.pron_issues:
            L.append(f"- **{p.word}**（{p.start:.1f}s，置信度 {p.confidence:.0%}）：{p.note}")
    else:
        L.append("- 未发现明显低置信度单词，发音整体较清晰。")
    L.append("")

    L.append("## 语法 / 用词问题（仅标注，不改原文）")
    if r.grammar_issues:
        for g in r.grammar_issues[:20]:
            fix = f"　建议：`{g.replacement}`" if g.replacement else ""
            L.append(f"- [{g.category}] {g.message}{fix}")
            if g.context:
                L.append(f"  - 原文片段：`{g.context}`")
    else:
        L.append("- 未检测到明显语法问题。")
    L.append("")

    L.append("## 表达地道度")
    if r.naturalness:
        for n in r.naturalness:
            L.append(f"- {n}")
    else:
        L.append("- 未发现明显中式表达。")
    L.append("")

    if r.examiner_corrections:
        L.append("## 教官的纠正参考")
        for c in r.examiner_corrections:
            L.append(f"- {c}")
        L.append("")

    L.append("## 考生原文（逐字，未修改）")
    L.append("")
    L.append(r.transcript_candidate or "（无）")
    L.append("")
    if r.transcript_examiner.strip():
        L.append("## 教官原文")
        L.append("")
        L.append(r.transcript_examiner)
        L.append("")
    return "\n".join(L)


def _join(segs: list[ASRSegment]) -> str:
    return " ".join(s.text.strip() for s in segs).strip()
