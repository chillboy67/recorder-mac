"""
Language table and script-based language labelling.

Whisper already recognises ~99 languages, so nothing here teaches the app a new
language. What it adds is the curated list the UI offers, display names, and the
labeller that turns Whisper's output into the ``language`` tag stored on
``ASRResult`` / ``ASRSegment``.

Why character-level analysis is still needed: Whisper reports **one** language
per pass, so on code-switching audio (the zh/en case this app was built for) the
minority language is lost. Labelling by Unicode script recovers the mix.

Scripts fall into two groups:

* **Decisive** — the script identifies the language on its own: kana → ja,
  Hangul → ko, Han → zh, Thai → th, Greek → el …
* **Ambiguous** — many languages share the script. Latin cannot distinguish
  en/fr/de/es, Cyrillic cannot distinguish ru/uk/bg, Arabic cannot distinguish
  ar/fa/ur. For these, Whisper's own detection decides.

That split is the whole point of this module. The previous heuristic compared
CJK characters against Latin characters only, which is why Japanese (mostly
kanji) was labelled ``zh`` and Korean (Hangul, neither CJK-unified nor Latin)
was labelled ``en``.
"""
from __future__ import annotations

import re
from typing import Optional

# ── UI ───────────────────────────────────────────────────────────────
# "auto" lets Whisper decide and is what enables per-chunk code-switching
# detection. A forced language trades that away for reliability on audio
# Whisper keeps misdetecting.
AUTO = "auto"

PICKER_LANGUAGES: tuple[tuple[str, str], ...] = (
    (AUTO, "自动检测"),
    ("zh", "中文"),
    ("en", "英语"),
    ("ja", "日语"),
    ("ko", "韩语"),
    ("fr", "法语"),
    ("de", "德语"),
    ("es", "西班牙语"),
    ("it", "意大利语"),
    ("pt", "葡萄牙语"),
    ("ru", "俄语"),
    ("ar", "阿拉伯语"),
    ("th", "泰语"),
    ("vi", "越南语"),
    ("id", "印尼语"),
    ("hi", "印地语"),
)

# Display names for anything Whisper may report, not just what the picker
# offers. Unknown codes fall back to the code itself rather than guessing.
LANGUAGE_NAMES: dict[str, str] = {
    "mixed": "多语混合",
    "zh": "中文", "en": "英语", "ja": "日语", "ko": "韩语",
    "fr": "法语", "de": "德语", "es": "西班牙语", "it": "意大利语",
    "pt": "葡萄牙语", "nl": "荷兰语", "ru": "俄语", "uk": "乌克兰语",
    "pl": "波兰语", "cs": "捷克语", "sk": "斯洛伐克语", "sv": "瑞典语",
    "da": "丹麦语", "no": "挪威语", "fi": "芬兰语", "is": "冰岛语",
    "hu": "匈牙利语", "ro": "罗马尼亚语", "bg": "保加利亚语",
    "hr": "克罗地亚语", "sl": "斯洛文尼亚语", "sr": "塞尔维亚语",
    "el": "希腊语", "tr": "土耳其语", "ar": "阿拉伯语", "fa": "波斯语",
    "he": "希伯来语", "ur": "乌尔都语", "hi": "印地语", "bn": "孟加拉语",
    "ta": "泰米尔语", "te": "泰卢固语", "mr": "马拉地语", "gu": "古吉拉特语",
    "kn": "卡纳达语", "ml": "马拉雅拉姆语", "pa": "旁遮普语",
    "ne": "尼泊尔语", "si": "僧伽罗语", "th": "泰语", "lo": "老挝语",
    "my": "缅甸语", "km": "高棉语", "vi": "越南语", "id": "印尼语",
    "ms": "马来语", "tl": "他加禄语", "ka": "格鲁吉亚语",
    "hy": "亚美尼亚语", "az": "阿塞拜疆语", "kk": "哈萨克语",
    "uz": "乌兹别克语", "mn": "蒙古语", "sw": "斯瓦希里语",
    "af": "南非荷兰语", "ca": "加泰罗尼亚语", "eu": "巴斯克语",
    "gl": "加利西亚语", "cy": "威尔士语", "ga": "爱尔兰语", "la": "拉丁语",
    "eo": "世界语", "lv": "拉脱维亚语", "lt": "立陶宛语", "et": "爱沙尼亚语",
    "mk": "马其顿语", "sq": "阿尔巴尼亚语", "be": "白俄罗斯语",
}

# Languages whose model should be the Asian one (Qwen family), per
# docs/LLM_MODELS.md §2 — CJK text needs a CJK-trained model.
CJK_LANGS: frozenset[str] = frozenset({"zh", "ja", "ko"})

# ── scripts ──────────────────────────────────────────────────────────

_SCRIPT_PATTERNS: tuple[tuple[str, str], ...] = (
    ("kana", r"[\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff\uff66-\uff9f]"),
    ("hangul", r"[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]"),
    ("han", r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]"),
    ("cyrillic", r"[\u0400-\u04ff]"),
    ("greek", r"[\u0370-\u03ff]"),
    ("hebrew", r"[\u0590-\u05ff]"),
    ("arabic", r"[\u0600-\u06ff\u0750-\u077f]"),
    ("devanagari", r"[\u0900-\u097f]"),
    ("thai", r"[\u0e00-\u0e7f]"),
    ("georgian", r"[\u10a0-\u10ff]"),
    ("armenian", r"[\u0530-\u058f]"),
    ("latin", r"[A-Za-z\u00c0-\u00d6\u00d8-\u00f6\u00f8-\u024f\u1e00-\u1eff]"),
)

# Script → the language to report when Whisper's detection is unavailable or
# belongs to a different script. Only used for ambiguous scripts.
_SCRIPT_LANG: dict[str, str] = {
    "cyrillic": "ru", "greek": "el", "hebrew": "he", "arabic": "ar",
    "devanagari": "hi", "thai": "th", "georgian": "ka", "armenian": "hy",
    "latin": "en",
}

# Languages sharing each ambiguous script. When Whisper detected one of these,
# its answer beats the script default — Ukrainian text should not be reported
# as Russian just because both use Cyrillic.
_SCRIPT_FAMILY: dict[str, frozenset[str]] = {
    "cyrillic": frozenset({"ru", "uk", "bg", "sr", "mk", "be", "kk", "ky", "tg", "mn"}),
    "arabic": frozenset({"ar", "fa", "ur", "ps", "ku", "sd", "ug"}),
    "devanagari": frozenset({"hi", "mr", "ne", "sa"}),
    "latin": frozenset({
        "en", "fr", "de", "es", "it", "pt", "nl", "pl", "cs", "sk", "sv",
        "da", "no", "fi", "is", "hu", "ro", "hr", "sl", "tr", "vi", "id",
        "ms", "tl", "af", "ca", "eu", "gl", "cy", "ga", "la", "eo", "lv",
        "lt", "et", "sq", "sw", "uz", "az",
    }),
}

# A script needs at least this share of the text to count as the language's own
# — below it we treat the characters as incidental (a loanword, a name).
_DECISIVE = 0.02
# Han+Latin above this share each means genuine code-switching, not one
# language with a few foreign terms. Matches the long-standing zh/en threshold.
_MIXED = 0.15
_DOMINANT = 0.8


def script_counts(text: str) -> dict[str, int]:
    """Count characters per Unicode script, ignoring punctuation and digits."""
    counts: dict[str, int] = {}
    for script, pattern in _SCRIPT_PATTERNS:
        n = len(re.findall(pattern, text or ""))
        if n:
            counts[script] = n
    return counts


# Scripts each language is normally written in; anything outside them is
# foreign to that language. Languages not listed are treated as Latin-script.
_NATIVE_SCRIPTS: dict[str, frozenset[str]] = {
    "zh": frozenset({"han"}),
    "ja": frozenset({"han", "kana"}),
    "ko": frozenset({"hangul"}),
    "ru": frozenset({"cyrillic"}), "uk": frozenset({"cyrillic"}),
    "bg": frozenset({"cyrillic"}), "sr": frozenset({"cyrillic"}),
    "mk": frozenset({"cyrillic"}), "be": frozenset({"cyrillic"}),
    "el": frozenset({"greek"}),
    "he": frozenset({"hebrew"}),
    "ar": frozenset({"arabic"}), "fa": frozenset({"arabic"}),
    "ur": frozenset({"arabic"}),
    "hi": frozenset({"devanagari"}), "mr": frozenset({"devanagari"}),
    "ne": frozenset({"devanagari"}),
    "th": frozenset({"thai"}),
    "ka": frozenset({"georgian"}),
    "hy": frozenset({"armenian"}),
}


def foreign_script_count(text: str, lang: str = "en") -> int:
    """Count characters written in a script `lang` is not normally written in.

    This is how a turn is recognised as having left a language: in a coaching
    session the coach instructs in their own language while the candidate
    answers in the exam language, so non-native script characters mark the
    coach's turns — for any coach language, not just Chinese.

    Known limit: Latin-script languages share a script, so a French or German
    turn scores 0 against English and cannot be spotted this way. That needs
    per-segment language detection (Whisper only reports one language per
    pass/chunk), which the diarizer does not have yet.
    """
    native = _NATIVE_SCRIPTS.get(lang, frozenset({"latin"}))
    return sum(n for script, n in script_counts(text).items() if script not in native)


def _family_member(script: str, detected: Optional[str]) -> Optional[str]:
    if detected and detected in _SCRIPT_FAMILY.get(script, frozenset()):
        return detected
    return None


def detect_language(text: str, detected: Optional[str] = None) -> str:
    """Label a piece of text, combining its script with Whisper's detection.

    ``detected`` is Whisper's own language code for the audio. It is required to
    tell Latin-script languages apart and is otherwise only a fallback.
    """
    counts = script_counts(text)
    total = sum(counts.values())
    if not total:
        return detected or "en"
    share = {script: n / total for script, n in counts.items()}

    # Kana and Hangul settle it immediately, and must be checked before any
    # Han or mixing logic: ordinary Japanese is kanji + kana, which would
    # otherwise read as "mixed".
    if share.get("kana", 0.0) >= _DECISIVE:
        return "ja"
    if share.get("hangul", 0.0) >= _DECISIVE:
        return "ko"

    for script in ("cyrillic", "greek", "hebrew", "arabic",
                   "devanagari", "thai", "georgian", "armenian"):
        if share.get(script, 0.0) >= _DOMINANT:
            return _family_member(script, detected) or _SCRIPT_LANG[script]

    han = share.get("han", 0.0)
    latin = share.get("latin", 0.0)
    # Kanji-only lines in audio Whisper identified as Japanese: no kana to go
    # on, so the detection is the only evidence that this isn't Chinese.
    if han and detected == "ja":
        return "ja"
    if han >= _DOMINANT:
        return "zh"
    if latin >= _DOMINANT:
        return _family_member("latin", detected) or "en"
    if min(han, latin) >= _MIXED:
        return "mixed"
    if han > latin:
        return "zh"
    if latin:
        return _family_member("latin", detected) or "en"
    return detected or "en"


def prefers_asian_model(language: str) -> bool:
    """True when the transcript should go to the CJK-trained model."""
    return language in CJK_LANGS or language == "mixed"


def normalize_language(value) -> Optional[str]:
    """Map a UI/CLI choice to a Whisper language code; ``None`` means auto."""
    if value is None:
        return None
    v = str(value).strip().lower()
    if v in ("", AUTO, "none", "null"):
        return None
    return v


def display_name(code: Optional[str]) -> str:
    """Human-readable name for a language code, for the UI and exports."""
    if not code:
        return "未知"
    return LANGUAGE_NAMES.get(code, code)
