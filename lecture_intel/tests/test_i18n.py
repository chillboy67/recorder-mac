"""
Localization tests — no Qt, no Whisper model needed.

These guard the bilingual (zh ↔ en) catalogue the whole UI now runs through:
every key must exist in both languages, ``t`` must follow the active language
and format its placeholders, and the language-aware helpers (``display_name``)
must switch with it. The autouse fixture resets the active language after each
test so module state never leaks into the other test files.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import i18n  # noqa: E402
from core.i18n import (  # noqa: E402
    DEFAULT,
    EN,
    SUPPORTED,
    ZH,
    current_language,
    detect_system_language,
    set_language,
    t,
    ui_language_choices,
)
from core.languages import display_name  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_language():
    """Keep the module-level language global from leaking between tests."""
    before = current_language()
    try:
        yield
    finally:
        set_language(before)


# ── catalogue integrity ────────────────────────────────────────────────

def test_catalog_has_both_languages_and_no_blanks():
    for key, entry in i18n._CATALOG.items():
        assert set(entry) == {"zh", "en"}, f"{key} must define zh and en"
        assert entry["zh"].strip(), f"{key} has an empty zh string"
        assert entry["en"].strip(), f"{key} has an empty en string"


def test_catalog_keys_are_unique_identifiers():
    # dict keys are unique by construction; assert they are non-empty strings
    for key in i18n._CATALOG:
        assert isinstance(key, str) and key.strip()


# ── t() basics ─────────────────────────────────────────────────────────

def test_default_language_is_chinese():
    assert DEFAULT == ZH
    set_language(ZH)
    assert t("app_title") == "Recorder · 录音转文字"


def test_t_follows_active_language():
    set_language(ZH)
    zh = t("btn_output_folder")
    set_language(EN)
    en = t("btn_output_folder")
    assert zh == "输出文件夹"
    assert en == "Output Folder"
    assert zh != en


def test_t_formats_placeholders():
    set_language(EN)
    assert t("status_done", seconds=12, dir="/tmp/out") == \
        "✓ Done in 12s · Output: /tmp/out"
    set_language(ZH)
    assert t("eng_asr_done", count=7) == "转写完成（7 段）"


def test_t_bad_format_args_do_not_raise():
    set_language(EN)
    # missing kwargs → the raw template is returned rather than crashing
    assert "{seconds}" in t("status_done")


def test_unknown_key_returns_itself():
    assert t("definitely_not_a_key") == "definitely_not_a_key"


def test_set_language_rejects_unknown_codes():
    assert set_language("klingon") == DEFAULT
    assert set_language(None) == DEFAULT
    assert set_language("") == DEFAULT
    assert set_language("EN") == EN          # normalized (case/space)
    assert current_language() == EN


# ── switcher + detection ───────────────────────────────────────────────

def test_ui_language_choices_use_native_names():
    choices = dict(ui_language_choices())
    assert set(choices) == set(SUPPORTED)
    # native names, independent of the active UI language
    assert choices[ZH] == "中文"
    assert choices[EN] == "English"
    set_language(EN)
    assert dict(ui_language_choices())[ZH] == "中文"


def test_detect_system_language_is_supported():
    assert detect_system_language() in SUPPORTED


# ── language-aware helpers ─────────────────────────────────────────────

def test_display_name_switches_with_ui_language():
    set_language(ZH)
    assert display_name("fr") == "法语"
    assert display_name("mixed") == "多语混合"
    assert display_name(None) == "未知"
    set_language(EN)
    assert display_name("fr") == "French"
    assert display_name("mixed") == "Mixed"
    assert display_name(None) == "Unknown"
    # unknown codes fall back to the code in both languages
    assert display_name("zz") == "zz"


def test_display_name_explicit_lang_override():
    # the lang= argument wins regardless of the active UI language
    set_language(ZH)
    assert display_name("en", lang=EN) == "English"
    assert display_name("en", lang=ZH) == "英语"
