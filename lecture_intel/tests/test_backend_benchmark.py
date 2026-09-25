from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

from backend_benchmark import error_rate, normalize_words, summarize  # noqa: E402


def test_normalize_words_uses_cjk_character_units_and_latin_words():
    assert normalize_words("Hello, 世界! 안녕") == ["hello", "世", "界", "안녕"]


def test_error_rate_is_word_level_for_english_and_character_level_for_cjk():
    assert error_rate("we test audio", "we test audio") == 0.0
    assert error_rate("we test audio", "we test") == 1 / 3
    assert error_rate("录音测试", "录音") == 0.5
    assert error_rate("", "") == 0.0
    assert error_rate("", "not empty") == 1.0


def test_summary_ignores_backend_mismatches_and_uses_medians():
    rows = [
        {"status": "ok", "requested_engine": "whisper.cpp-vulkan",
         "elapsed_sec": 2.0, "rtf": 0.2, "peak_rss_mb": 120.0, "wer": 0.1},
        {"status": "ok", "requested_engine": "whisper.cpp-vulkan",
         "elapsed_sec": 4.0, "rtf": 0.4, "peak_rss_mb": 140.0, "wer": 0.3},
        {"status": "backend-mismatch", "requested_engine": "whisper.cpp-vulkan",
         "elapsed_sec": 1.0, "rtf": 0.1, "peak_rss_mb": 100.0, "wer": 0.0},
    ]
    assert summarize(rows) == {
        "whisper.cpp-vulkan": {
            "successful_runs": 2,
            "median_elapsed_sec": 3.0,
            "elapsed_range_sec": [2.0, 4.0],
            "median_rtf": 0.3,
            "rtf_range": [0.2, 0.4],
            "median_peak_rss_mb": 130.0,
            "median_wer": 0.2,
        }
    }
