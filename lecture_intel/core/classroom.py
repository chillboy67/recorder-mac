"""
Classroom analysis: key-point summary (+ light cleanup).

Unlike IELTS (faithful, never altered), a lecture transcript is most useful when
it surfaces what matters. We extract "key points" with offline heuristics — no
cloud, no LLM required — covering the cues the user described:

  - 老师明确强调的（"重点 / 注意 / 常考 / 必考 / 记住 / 关键"，"important / note / exam")
  - 重要定义（"X 是 / 指 / 称为"，"X is / means / refers to / defined as")
  - 高频主题（reappearing terms = the lecture's backbone）
  - 讲得最久/最细的部分（the longest continuous stretches = time spent = emphasis）

Genuine *factual* correction of a teacher's mistakes needs a language model to
understand the content; that isn't possible fully offline. If a local LLM
(Ollama) is installed we could plug it in here later. For now we keep the
transcript and add the summary, and note this in the report.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from modules import ASRResult, ASRSegment

# Emphasis cues — Chinese and English.
_EMPHASIS = re.compile(
    r"(重点|重要|注意|划重点|敲黑板|记住|记一下|关键|核心|必考|常考|考试会考|"
    r"一定要|务必|特别是|尤其|总结一下|归纳|易错|难点|"
    r"important|note that|remember|key point|keep in mind|crucial|essential|"
    r"make sure|in summary|to summarize|exam|tested|focus on)",
    re.IGNORECASE,
)

# Definition cues.
_DEFINITION = re.compile(
    r"(.{2,30}?)(就是指|是指|定义为|称为|叫做|指的是|就是说|"
    r"\bis defined as\b|\brefers to\b|\bmeans\b|\bis called\b)",
    re.IGNORECASE,
)

# Tokens to ignore when counting "frequent terms".
_STOP = set("的 了 是 我 你 他 她 它 们 这 那 个 就 也 都 和 与 在 有 不 我们 你们 "
            "他们 这个 那个 什么 怎么 这样 那样 一个 一些 可以 因为 所以 但是 如果 "
            "然后 就是 这种 the a an of to in on and or is are was were be this that "
            "it we you they i he she for with as at by from".split())

# Common Chinese function characters — a bigram containing any of these is
# almost always filler ("的话/一下/里面/时候/这边"), not a content term.
_ZH_FUNC = set("的了是我你他她它们这那个就也都和与在有不会要把被让从对到说很"
               "可以里面时候现在后面前面这边那边一下大家然后还有就是因为所以"
               "但是如果什么怎么我们你们他们之上下来去")


@dataclass
class ClassroomReport:
    transcript: str
    emphasis_points: list[str] = field(default_factory=list)
    definitions: list[str] = field(default_factory=list)
    frequent_terms: list[tuple[str, int]] = field(default_factory=list)
    long_stretches: list[str] = field(default_factory=list)
    markdown: str = ""


def summarize(asr: ASRResult) -> ClassroomReport:
    segs = [s for s in asr.segments if s.text.strip()]
    full = "\n".join(s.text.strip() for s in segs)

    emphasis = _emphasis_points(segs)
    definitions = _definitions(full)
    frequent = _frequent_terms(full)
    stretches = _long_stretches(segs)

    report = ClassroomReport(
        transcript=full,
        emphasis_points=emphasis,
        definitions=definitions,
        frequent_terms=frequent,
        long_stretches=stretches,
    )
    report.markdown = _render(report)
    return report


# ----------------------------------------------------------------------

def _emphasis_points(segs: list[ASRSegment]) -> list[str]:
    out = []
    for s in segs:
        t = s.text.strip()
        if _EMPHASIS.search(t) and len(t) >= 4:
            ts = _mmss(s.start)
            out.append(f"[{ts}] {t}")
    # dedupe consecutive near-duplicates, cap
    seen, uniq = set(), []
    for x in out:
        key = x.split("] ", 1)[-1][:20]
        if key not in seen:
            seen.add(key); uniq.append(x)
    return uniq[:30]


def _definitions(full: str) -> list[str]:
    out = []
    for line in full.splitlines():
        line = line.strip()
        if _DEFINITION.search(line) and 6 <= len(line) <= 160:
            out.append(line)
    # unique, cap
    return list(dict.fromkeys(out))[:20]


def _frequent_terms(full: str) -> list[tuple[str, int]]:
    # English words (>=4 chars) + Chinese bigrams as cheap "terms".
    counts: Counter = Counter()
    for w in re.findall(r"[A-Za-z][A-Za-z\-']{3,}", full):
        wl = w.lower()
        if wl not in _STOP:
            counts[wl] += 1
    zh = re.findall(r"[一-鿿]+", full)
    for run in zh:
        for i in range(len(run) - 1):
            bi = run[i:i + 2]
            # skip bigrams containing a common function character — those are
            # almost always junk like "的话/一下/里面/时候", not real terms.
            if bi[0] in _ZH_FUNC or bi[1] in _ZH_FUNC or bi in _STOP:
                continue
            counts[bi] += 1
    # keep terms that recur a lot (the lecture's backbone)
    common = [(t, c) for t, c in counts.most_common(40) if c >= 4]
    return common[:15]


def _long_stretches(segs: list[ASRSegment], min_sec: float = 25.0) -> list[str]:
    """Topics the teacher spent the most continuous time on (split on big gaps)."""
    if not segs:
        return []
    blocks, cur = [], [segs[0]]
    for prev, s in zip(segs, segs[1:]):
        if s.start - prev.end > 4.0:        # a real topic break
            blocks.append(cur); cur = [s]
        else:
            cur.append(s)
    blocks.append(cur)
    blocks = [b for b in blocks if (b[-1].end - b[0].start) >= min_sec]
    blocks.sort(key=lambda b: -(b[-1].end - b[0].start))
    out = []
    for b in blocks[:5]:
        dur = b[-1].end - b[0].start
        head = " ".join(x.text.strip() for x in b)[:60]
        out.append(f"[{_mmss(b[0].start)}–{_mmss(b[-1].end)}，约{dur:.0f}s] {head}…")
    return out


def _render(r: ClassroomReport) -> str:
    L = ["# 课堂重点总结", ""]
    L.append("> 自动提取，供复习参考。完整内容见转写原文。")
    L.append("")

    L.append("## 老师强调的重点")
    if r.emphasis_points:
        L += [f"- {p}" for p in r.emphasis_points]
    else:
        L.append("- 未检测到明显的「重点/注意/常考」等强调用语。")
    L.append("")

    L.append("## 重要定义")
    if r.definitions:
        L += [f"- {d}" for d in r.definitions]
    else:
        L.append("- 未检测到明显的定义句。")
    L.append("")

    L.append("## 高频主题（反复出现，可能是核心）")
    if r.frequent_terms:
        L.append("、".join(f"{t}（{c}次）" for t, c in r.frequent_terms))
    else:
        L.append("（无）")
    L.append("")

    L.append("## 讲解篇幅最长的部分")
    if r.long_stretches:
        L += [f"- {s}" for s in r.long_stretches]
    else:
        L.append("（无）")
    L.append("")

    L.append("## 全文转写")
    L.append("")
    L.append(r.transcript or "（无）")
    L.append("")
    return "\n".join(L)


def _mmss(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
