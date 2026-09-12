"""
Optional local-LLM enhancement via Ollama.

Everything here is best-effort: if Ollama isn't running or the model is missing,
each function returns None and the caller falls back to the offline heuristics.
Nothing leaves the machine.

Default model: llama3.1:8b (Meta) — ~4.9GB / ~6GB RAM, runs on the M-series GPU.
Non-Chinese model, handles English well and Chinese acceptably. Heavy enough to
be useful, light enough not to cook a 16GB Mac for occasional batch jobs.
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from core.i18n import t
from core.languages import LANGUAGE_NAMES_EN

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_HOST = "http://127.0.0.1:11434"
_NUM_CTX = 8192


def _installed(host: str = DEFAULT_HOST) -> list[str]:
    try:
        import httpx
        r = httpx.get(f"{host}/api/tags", timeout=2.0)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return []


def resolve_model(preferred: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> Optional[str]:
    """Return an installed model name matching `preferred`.

    Tolerates the ModelScope-mirror name (e.g. modelscope.cn/Qwen/Qwen2.5-7B-
    Instruct-GGUF) when the user pulled via the China mirror without aliasing."""
    names = _installed(host)
    if not names:
        return None
    base = preferred.split(":")[0].lower()                 # "llama3.1"
    family = base.replace(".", "").replace("-", "")        # "llama31"
    # exact / same base first
    for n in names:
        if n == preferred or n.split(":")[0].lower() == base:
            return n
    # then any model whose name looks like the same family (e.g. the ModelScope
    # mirror name Meta-Llama-3.1-8B-Instruct-GGUF)
    for n in names:
        nl = n.lower().replace(".", "").replace("_", "").replace("-", "")
        if family and family in nl:
            return n
    return None


def available(model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST) -> bool:
    """True if the Ollama server is up and a matching model is pulled."""
    return resolve_model(model, host) is not None


def _gen(prompt: str, system: str = "", *, model: str = DEFAULT_MODEL,
         host: str = DEFAULT_HOST, temperature: float = 0.2,
         num_predict: int = 1024, timeout: float = 180.0) -> Optional[str]:
    """Call Ollama's CHAT endpoint (applies the instruct template, so the model
    follows instructions instead of doing raw completion) and cap output length."""
    try:
        import httpx
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_ctx": _NUM_CTX,
                        "num_predict": num_predict},
        }
        r = httpx.post(f"{host}/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return (r.json().get("message", {}).get("content") or "").strip()
    except Exception as exc:
        logger.warning("LLM chat failed: %s", exc)
        return None


def _chunks(text: str, size: int) -> list[str]:
    """Split text into ~size-char chunks at line/sentence boundaries."""
    out, cur = [], []
    n = 0
    for line in text.splitlines():
        cur.append(line)
        n += len(line) + 1
        if n >= size:
            out.append("\n".join(cur)); cur, n = [], 0
    if cur:
        out.append("\n".join(cur))
    return out or [text]


def _prompt_language(language: Optional[str]) -> str:
    """The language prompts are written in: the transcript's own. Code-switching
    transcripts default to Chinese, matching how their model is routed."""
    if not language or language == "mixed":
        return "zh"
    return language


def _lang_name(lang: str) -> str:
    """English name used inside prompts; unknown codes are used verbatim."""
    return LANGUAGE_NAMES_EN.get(lang, lang)


# ----------------------------------------------------------------------
# Classroom: correction + summary
# ----------------------------------------------------------------------

_CORRECT_SYS = (
    "你是课堂录音转写的校对助手。你会收到一段语音识别文本，可能有识别错误、"
    "口误、错别字、缺标点。请改正明显的识别/语法错误并补全标点，保持讲课原意和"
    "信息不变，不要增删内容、不要总结、不要解释。只输出改正后的文本本身。"
)


def correct_lecture(text: str, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                    progress: Optional[Callable[[float, str], None]] = None) -> Optional[str]:
    chunks = _chunks(text, 1500)
    out = []
    for i, ch in enumerate(chunks):
        res = _gen(f"原文：\n{ch}", _CORRECT_SYS, model=model, host=host, temperature=0.1)
        if res is None:
            return None
        out.append(res)
        if progress:
            progress((i + 1) / len(chunks), t("llm_proofreading", i=i + 1, n=len(chunks)))
    return "\n".join(out)


_CORRECT_SYS_ZH = (
    "你是中文语音转写校对员。下面是语音识别的结果，可能有同音字、识别错误、"
    "缺标点。请根据上下文判断说话人真正说的内容，改正明显的识别错误并补全标点。"
    "严格要求：只改错别字/同音字/标点，不要增加或删减信息、不要改写语气、"
    "不要总结、不要解释、不要输出任何额外说明。只输出改正后的文本本身。"
)
_CORRECT_SYS_EN = (
    "You are a speech-transcription proofreader for {name} text. Below is the "
    "recognition output; it may contain misheard words and missing punctuation. "
    "Judge from the context what the speaker actually said, fix obvious "
    "recognition errors and complete the punctuation. Strict requirements: only "
    "fix wrong words and punctuation; do not add or remove information, do not "
    "rephrase the tone, do not summarize, do not explain, and output nothing "
    "beyond the corrected text itself."
)


def correct_transcript(text: str, context: str = "", model: str = DEFAULT_MODEL,
                       host: str = DEFAULT_HOST, language: Optional[str] = None,
                       progress: Optional[Callable[[float, str], None]] = None
                       ) -> Optional[str]:
    """Use the LLM to recover what was *actually* said: fix homophone/recognition
    errors using context. Returns the corrected full transcript (verbatim meaning,
    nothing added/removed/summarized). Per-chunk fallback keeps the original text
    if a chunk's output looks wrong (too short), so it never collapses to junk.

    `language` is the transcript's language code: the prompt is written in it
    (code-switching transcripts use Chinese), so a non-Chinese transcript is
    proofread in its own language instead of being addressed by a Chinese
    proofreader prompt."""
    lang = _prompt_language(language)
    if lang == "zh":
        sys = _CORRECT_SYS_ZH
        if context:
            sys += f"\n背景：{context}。可据此判断专业术语的正确写法。"
    else:
        sys = _CORRECT_SYS_EN.format(name=_lang_name(lang))
        if context:
            sys += ("\nContext: " + context + ". Use it to judge the correct "
                    "spelling of technical terms.")

    chunks = _chunks(text, 1200)
    out = []
    for i, ch in enumerate(chunks):
        # allow enough output tokens to cover a same-length rewrite
        npred = min(4096, max(256, int(len(ch) * 2.0)))
        res = _gen(f"识别文本：\n{ch}", sys, model=model, host=host,
                   temperature=0.1, num_predict=npred)
        # Fallback: if the model returned almost nothing (the "对" failure mode),
        # keep the original chunk rather than destroying the transcript.
        if not res or len(res) < 0.4 * len(ch.strip()):
            out.append(ch)
        else:
            out.append(res)
        if progress:
            progress((i + 1) / len(chunks), t("llm_correcting", i=i + 1, n=len(chunks)))
    return "\n".join(out)


_MAP_SYS_ZH = (
    "你是课堂笔记助手。请从这段课堂转写中提取要点，用简洁中文分条列出："
    "老师强调的重点、重要概念与定义、常考/易错提醒。不要复述全文，只列要点。"
)
_REDUCE_SYS_ZH = (
    "你是课堂笔记助手。下面是一节课各部分的要点笔记，请整合、去重，输出结构化的"
    "课堂重点总结，使用以下小标题：\n## 本节重点\n## 重要概念与定义\n"
    "## 常考 / 易错提醒\n## 一句话总结\n保持简洁，用中文。"
)
_MAP_SYS_EN = (
    "You are a lecture-notes assistant. Extract the key points from this lecture "
    "transcript as a concise bullet list written in {name}: what the teacher "
    "emphasized, important concepts and definitions, and exam-prone or "
    "error-prone reminders. Do not restate the full text; list points only."
)
_REDUCE_SYS_EN = (
    "You are a lecture-notes assistant. Below are the key-point notes from each "
    "part of one lecture. Merge and deduplicate them into a structured summary "
    "written in {name}, using exactly these section headings:\n"
    "## Key Points\n## Key Concepts & Definitions\n"
    "## Exam-Prone / Error-Prone Reminders\n## One-Sentence Summary\n"
    "Keep it concise."
)


def summarize_lecture(text: str, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                      language: Optional[str] = None,
                      progress: Optional[Callable[[float, str], None]] = None
                      ) -> Optional[str]:
    """Summarize a lecture in the transcript's own language (Chinese for
    code-switching transcripts) — a Japanese lecture gets a Japanese summary."""
    lang = _prompt_language(language)
    map_sys = _MAP_SYS_ZH if lang == "zh" else _MAP_SYS_EN.format(name=_lang_name(lang))
    reduce_sys = (_REDUCE_SYS_ZH if lang == "zh"
                  else _REDUCE_SYS_EN.format(name=_lang_name(lang)))
    chunks = _chunks(text, 3000)
    notes = []
    for i, ch in enumerate(chunks):
        res = _gen(f"课堂转写：\n{ch}", map_sys, model=model, host=host, temperature=0.2)
        if res is None:
            return None
        notes.append(res)
        if progress:
            progress(0.1 + 0.7 * (i + 1) / len(chunks), t("llm_extracting", i=i + 1, n=len(chunks)))
    combined = "\n".join(notes)
    if progress:
        progress(0.9, t("llm_merging"))
    final = _gen(f"要点笔记：\n{combined}", reduce_sys, model=model, host=host, temperature=0.3)
    return final


# ----------------------------------------------------------------------
# IELTS: richer feedback on the candidate's English
# ----------------------------------------------------------------------

# The examiner report stays in Chinese on purpose: the candidate's answers are
# always English (this is IELTS), and Chinese is the UI's language. Per-language
# reports are the roadmap's explicitly-optional coach-side translation, so
# ielts_feedback takes no language parameter.
_IELTS_SYS = (
    "你是雅思口语考官，只做【语言诊断】。不要评价是否切题、不要扩展话题、"
    "不要复述题目。只基于考生英文原文找语言问题，严格按下面格式输出，用中文说明、"
    "英文例句保留英文；每节最多 3 条，没有问题就写“无明显问题”。除这些小标题外不要输出别的内容。\n\n"
    "## 语法错误\n- 原句「…」→ 改为「…」（原因）\n\n"
    "## 用词 / 搭配\n- 「…」→「…」\n\n"
    "## 不地道 / 中式表达\n- 「…」→ 更自然：「…」\n\n"
    "## 升级示范\n（把其中一两句改写得更高分，给出英文）"
)


def ielts_feedback(candidate_text: str, model: str = DEFAULT_MODEL,
                   host: str = DEFAULT_HOST) -> Optional[str]:
    if not candidate_text.strip():
        return None
    # candidate answers are usually short enough to fit; cap to be safe
    text = candidate_text[:6000]
    return _gen(f"考生回答：\n{text}", _IELTS_SYS, model=model, host=host, temperature=0.3)


# ----------------------------------------------------------------------
# General: light readability cleanup (punctuation + paragraphs, no rewriting)
# ----------------------------------------------------------------------

_TIDY_SYS_ZH = (
    "你是文字整理助手。请给这段语音转写补全标点、分段，纠正明显的识别错别字，"
    "但不要改写措辞、不要增删内容、不要翻译、不要总结。只输出整理后的文本。"
)
_TIDY_SYS_EN = (
    "You are a text-tidying assistant. Complete the punctuation and paragraphing "
    "of this speech transcript and fix obvious recognition typos, but do not "
    "rephrase the wording, do not add or remove content, do not translate, do "
    "not summarize. Output only the tidied text."
)


def tidy_transcript(text: str, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                    language: Optional[str] = None,
                    progress: Optional[Callable[[float, str], None]] = None
                    ) -> Optional[str]:
    sys = _TIDY_SYS_ZH if _prompt_language(language) == "zh" else _TIDY_SYS_EN
    chunks = _chunks(text, 2000)
    out = []
    for i, ch in enumerate(chunks):
        res = _gen(f"原文：\n{ch}", sys, model=model, host=host, temperature=0.1)
        if res is None:
            return None
        out.append(res)
        if progress:
            progress((i + 1) / len(chunks), t("llm_tidying", i=i + 1, n=len(chunks)))
    return "\n\n".join(out)
