"""
Language detection tests — no Whisper model needed.

The bulk of these guard two regressions the old CJK-vs-Latin heuristic had:
Japanese text (mostly kanji) was labelled "zh", and Korean (Hangul, which is
neither CJK-unified nor Latin) was labelled "en". They also pin down the
behaviour the rest of the app depends on — that zh/en code-switching still
reports "mixed", and that Latin-script languages defer to Whisper's detection
because characters alone cannot tell en/fr/de/es apart.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.languages import (  # noqa: E402
    AUTO,
    LANGUAGE_NAMES,
    PICKER_LANGUAGES,
    detect_language,
    disambiguate_latin_language,
    display_name,
    normalize_language,
    prefers_asian_model,
    script_counts,
)
from core.transcriber import Transcriber  # noqa: E402


# ── decisive scripts ─────────────────────────────────────────────────

def test_japanese_is_japanese_not_chinese():
    """Regression: kanji-heavy Japanese used to be labelled "zh"."""
    assert detect_language("今日はいい天気ですね。日本語のテストです。") == "ja"


def test_japanese_with_heavy_kanji_still_japanese():
    assert detect_language("東京都内の地下鉄は本日平常通り運行しています") == "ja"


def test_korean_is_korean_not_english():
    """Regression: Hangul matched neither CJK nor Latin, so it fell through to "en"."""
    assert detect_language("안녕하세요 오늘 날씨가 정말 좋아요") == "ko"


def test_kanji_only_line_trusts_whisper():
    """A kanji-only line has no kana to prove it's Japanese, so Whisper decides."""
    assert detect_language("本日休診", detected="ja") == "ja"
    # Without that evidence it is genuinely indistinguishable from Chinese.
    assert detect_language("本日休診") == "zh"


@pytest.mark.parametrize("text,detected,expected", [
    ("Привет, как дела сегодня?", None, "ru"),
    ("Привет, як справи сьогодні?", "uk", "uk"),   # Cyrillic, but Whisper heard Ukrainian
    ("مرحبا كيف حالك اليوم", None, "ar"),
    ("سلام حال شما چطور است", "fa", "fa"),          # Arabic script, but Persian
    ("สวัสดีครับ วันนี้เป็นอย่างไรบ้าง", None, "th"),
    ("Γεια σας, τι κάνετε σήμερα;", None, "el"),
    ("שלום, איך אתה מרגיש היום?", None, "he"),
    ("नमस्ते, आज आप कैसे हैं?", None, "hi"),
    ("გამარჯობა, როგორ ხარ?", None, "ka"),
])
def test_non_latin_scripts(text, detected, expected):
    assert detect_language(text, detected) == expected


# ── Latin script defers to Whisper ───────────────────────────────────

@pytest.mark.parametrize("text,detected,expected", [
    ("Bonjour, comment allez-vous aujourd'hui ?", "fr", "fr"),
    ("Schöne Grüße aus München, wie geht es Ihnen?", "de", "de"),
    ("Hola, ¿cómo estás hoy? Muy bien, gracias.", "es", "es"),
    ("Buongiorno, come sta oggi?", "it", "it"),
    ("Olá, como você está hoje?", "pt", "pt"),
    ("Xin chào, hôm nay bạn thế nào?", "vi", "vi"),
    ("The quick brown fox jumps over the lazy dog", "en", "en"),
])
def test_latin_languages_come_from_whisper(text, detected, expected):
    assert detect_language(text, detected) == expected


def test_latin_without_detection_defaults_to_english():
    """Characters alone can't identify a Latin-script language."""
    assert detect_language("Bonjour, comment allez-vous ?") == "en"


# ── parity with the long-standing zh/en behaviour ────────────────────

def test_pure_chinese():
    assert detect_language("机器学习是人工智能的一个重要分支") == "zh"


def test_pure_english():
    assert detect_language("This is a plain English sentence.", "en") == "en"


def test_zh_en_code_switching_is_mixed():
    text = "我们用 Python 来实现 gradient descent 算法"
    assert detect_language(text, "zh") == "mixed"


def test_chinese_with_a_few_latin_terms_stays_mixed():
    """Matches the old threshold: >15% Latin alongside Chinese reads as mixed."""
    assert detect_language("今天的作业是使用 Python 完成", "zh") == "mixed"


def test_mostly_chinese_with_one_short_token_is_zh():
    han = "今天我们讨论深度学习中的反向传播算法以及它在神经网络训练过程中的应用"
    assert detect_language(han + " PyTorch", "zh") == "zh"


# ── degenerate input ─────────────────────────────────────────────────

@pytest.mark.parametrize("text,detected,expected", [
    ("", None, "en"),
    ("", "fr", "fr"),
    (None, None, "en"),
    ("123 ... !!!", "de", "de"),      # no letters at all → trust Whisper
    ("   ", "ja", "ja"),
])
def test_empty_or_unlettered_text(text, detected, expected):
    assert detect_language(text, detected) == expected


def test_script_counts_ignores_punctuation_and_digits():
    assert script_counts("Hello, 世界! 123") == {"han": 2, "latin": 5}
    assert script_counts("") == {}


# ── per-segment labelling through the transcriber ────────────────────

def _raw(text: str) -> dict:
    return {"start": 0.0, "end": 1.0, "text": text, "avg_logprob": -0.2, "words": []}


def test_segments_inherit_whisper_language_for_latin_text():
    segs = Transcriber._to_segments(
        [_raw("Bonjour tout le monde"), _raw("这是中文")], detected="fr")
    assert [s.language for s in segs] == ["fr", "zh"]


def test_segments_keep_zh_en_code_switching():
    """A Chinese-majority file must still tag its English segments as English."""
    segs = Transcriber._to_segments(
        [_raw("我们用中文交流"), _raw("and then we switch to English")], detected="zh")
    assert [s.language for s in segs] == ["zh", "en"]


def test_segments_label_japanese_and_english_separately():
    segs = Transcriber._to_segments(
        [_raw("すみません、もう一度お願いします"), _raw("Could you repeat that?")],
        detected="ja")
    assert [s.language for s in segs] == ["ja", "en"]


def test_segments_use_their_own_chunk_language():
    """Two Latin-script languages in one file share a script, so the file-level
    code can only pick one of them. The per-chunk language carried on each
    segment is what labels both correctly."""
    raw = [
        {**_raw("Attention, vous avez utilisé le passé."), "chunk_language": "fr"},
        {**_raw("I have been studying English for years."), "chunk_language": "en"},
    ]
    segs = Transcriber._to_segments(raw, detected="en")   # file level says English
    assert [s.language for s in segs] == ["fr", "en"]


def test_chunk_language_falls_back_to_the_file_level_code():
    assert Transcriber._to_segments(
        [_raw("Bonjour tout le monde")], detected="fr")[0].language == "fr"
    # Neither signal available: Latin text still defaults to English.
    assert Transcriber._to_segments([_raw("Bonjour tout le monde")])[0].language == "en"


def test_zh_en_code_switching_unaffected_by_chunk_language():
    raw = [
        {**_raw("我们用中文交流"), "chunk_language": "zh"},
        {**_raw("and then we switch to English"), "chunk_language": "en"},
    ]
    assert [s.language for s in Transcriber._to_segments(raw, detected="zh")] == ["zh", "en"]


# ── routing + UI helpers ─────────────────────────────────────────────

@pytest.mark.parametrize("code,expected", [
    ("zh", True), ("ja", True), ("ko", True), ("mixed", True),
    ("en", False), ("fr", False), ("de", False), ("ru", False), ("th", False),
])
def test_prefers_asian_model(code, expected):
    assert prefers_asian_model(code) is expected


@pytest.mark.parametrize("value,expected", [
    (None, None), ("", None), ("  ", None), (AUTO, None), ("AUTO", None),
    ("none", None), ("zh", "zh"), (" fr ", "fr"), ("EN", "en"),
])
def test_normalize_language(value, expected):
    assert normalize_language(value) == expected


def test_display_name():
    assert display_name("fr") == "法语"
    assert display_name("mixed") == "多语混合"
    assert display_name("zz") == "zz"      # unknown codes are shown verbatim
    assert display_name(None) == "未知"


def test_picker_languages_are_well_formed():
    codes = [c for c, _ in PICKER_LANGUAGES]
    assert codes[0] == AUTO, "the picker must default to auto-detect"
    assert len(codes) == len(set(codes)), "duplicate language in the picker"
    for code in codes[1:]:
        assert code in LANGUAGE_NAMES, f"{code} has no display name"


# ── forced-language plumbing through the transcriber ─────────────────

@pytest.fixture
def mlx_spy(monkeypatch):
    """Route Transcriber to a fake mlx engine and record which path runs."""
    calls = {"chunked": None, "single": False}

    def fake_chunked(self, audio_path, initial_prompt, progress, chunk_sec, language):
        calls["chunked"] = language
        return [], language or "en"

    def fake_single(self, *args):
        calls["single"] = True
        return [], "en"

    monkeypatch.setattr(Transcriber, "_resolve_engine", lambda self: "mlx-whisper")
    monkeypatch.setattr(Transcriber, "_transcribe_mlx_chunked", fake_chunked)
    monkeypatch.setattr(Transcriber, "_transcribe_mlx", fake_single)
    return calls


def test_pinned_language_still_chunks(tmp_path, mlx_spy):
    """Chunking is what bounds memory on a 2-hour recording; pinning a language
    disables per-chunk detection, not chunking itself."""
    result = Transcriber(model="large-v3").transcribe(
        tmp_path / "a.wav", language="ja", chunked=True)
    assert mlx_spy["chunked"] == "ja"
    assert not mlx_spy["single"], "a pinned language must not fall back to one mlx pass"
    assert result.language == "ja"


def test_auto_language_chunks_with_detection(tmp_path, mlx_spy):
    Transcriber(model="large-v3").transcribe(tmp_path / "a.wav", chunked=True)
    assert mlx_spy["chunked"] is None
    assert not mlx_spy["single"]


def test_chunking_off_uses_single_pass(tmp_path, mlx_spy):
    Transcriber(model="large-v3").transcribe(tmp_path / "a.wav", chunked=False)
    assert mlx_spy["chunked"] is None
    assert mlx_spy["single"]


# ── function-word Latin disambiguation (#11) ─────────────────────────

def test_function_words_correct_mislabelled_chunk():
    """English text inside a French-labelled chunk must be corrected to English."""
    assert disambiguate_latin_language(
        "I think we should talk about this issue", "fr") == "en"


def test_function_words_keep_correct_chunk_language():
    """French text in a French chunk stays French."""
    assert disambiguate_latin_language(
        "Attention, vous avez utilisé le passé composé.", "fr") == "fr"


def test_function_words_dont_override_non_latin():
    """Chinese text is never touched by the Latin function-word check."""
    # detected="fr" is not a realistic input for Chinese text, but the
    # function must be a no-op when the text has no Latin function words.
    assert disambiguate_latin_language("这里要注意时态", "fr") == "fr"
    # Non-Latin detected language: no-op regardless of text.
    assert disambiguate_latin_language("hello world", "zh") == "zh"


def test_function_words_short_text_unchanged():
    """Fewer than 3 words: not enough signal, leave unchanged."""
    assert disambiguate_latin_language("the book", "fr") == "fr"
    assert disambiguate_latin_language("hello", "de") == "de"


def test_function_words_via_to_segments(tmp_path):
    """End-to-end through _to_segments: English text in a French chunk is
    corrected to English at the segment level."""
    raw = [
        {**_raw("I think we should talk about this"), "chunk_language": "fr"},
        {**_raw("vous avez utilisé le passé composé"), "chunk_language": "fr"},
    ]
    segs = Transcriber._to_segments(raw, detected="fr")
    assert segs[0].language == "en"   # corrected from fr
    assert segs[1].language == "fr"   # stays fr
