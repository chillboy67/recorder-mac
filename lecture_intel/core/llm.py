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
            progress((i + 1) / len(chunks), f"校对中 {i+1}/{len(chunks)}")
    return "\n".join(out)


_MAP_SYS = (
    "你是课堂笔记助手。请从这段课堂转写中提取要点，用简洁中文分条列出："
    "老师强调的重点、重要概念与定义、常考/易错提醒。不要复述全文，只列要点。"
)
_REDUCE_SYS = (
    "你是课堂笔记助手。下面是一节课各部分的要点笔记，请整合、去重，输出结构化的"
    "课堂重点总结，使用以下小标题：\n## 本节重点\n## 重要概念与定义\n"
    "## 常考 / 易错提醒\n## 一句话总结\n保持简洁，用中文。"
)


def summarize_lecture(text: str, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                      progress: Optional[Callable[[float, str], None]] = None) -> Optional[str]:
    chunks = _chunks(text, 3000)
    notes = []
    for i, ch in enumerate(chunks):
        res = _gen(f"课堂转写：\n{ch}", _MAP_SYS, model=model, host=host, temperature=0.2)
        if res is None:
            return None
        notes.append(res)
        if progress:
            progress(0.1 + 0.7 * (i + 1) / len(chunks), f"提炼要点 {i+1}/{len(chunks)}")
    combined = "\n".join(notes)
    if progress:
        progress(0.9, "整合总结…")
    final = _gen(f"要点笔记：\n{combined}", _REDUCE_SYS, model=model, host=host, temperature=0.3)
    return final


# ----------------------------------------------------------------------
# IELTS: richer feedback on the candidate's English
# ----------------------------------------------------------------------

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

_TIDY_SYS = (
    "你是文字整理助手。请给这段语音转写补全标点、分段，纠正明显的识别错别字，"
    "但不要改写措辞、不要增删内容、不要翻译、不要总结。只输出整理后的文本。"
)


def tidy_transcript(text: str, model: str = DEFAULT_MODEL, host: str = DEFAULT_HOST,
                    progress: Optional[Callable[[float, str], None]] = None) -> Optional[str]:
    chunks = _chunks(text, 2000)
    out = []
    for i, ch in enumerate(chunks):
        res = _gen(f"原文：\n{ch}", _TIDY_SYS, model=model, host=host, temperature=0.1)
        if res is None:
            return None
        out.append(res)
        if progress:
            progress((i + 1) / len(chunks), f"整理中 {i+1}/{len(chunks)}")
    return "\n\n".join(out)
