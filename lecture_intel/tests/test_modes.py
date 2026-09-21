"""
Mode presets — the table that decides what every run actually does.

`get_mode` was covered, but the presets themselves were not: which mode denoises,
which one drops background voices, which one reports. Those flags are the whole
difference between the three modes, and one flipped boolean silently changes a
user's output.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.modes import GENERAL, CLASSROOM, IELTS, MODES, Mode, get_mode  # noqa: E402


def test_the_three_documented_modes_exist():
    assert set(MODES) == {"general", "classroom", "ielts"}
    assert MODES["general"] is GENERAL
    assert MODES["classroom"] is CLASSROOM
    assert MODES["ielts"] is IELTS


def test_unknown_mode_falls_back_to_general():
    """A stale saved preference must not break a run."""
    assert get_mode("lecture") is GENERAL
    assert get_mode("") is GENERAL


@pytest.mark.parametrize("mode,denoise,summarize,main_only,diarize,ielts,expected", [
    (GENERAL, False, False, False, False, False, 0),
    (CLASSROOM, True, True, True, False, False, 0),
    (IELTS, False, False, False, True, True, 2),
])
def test_each_mode_enables_exactly_its_own_steps(
        mode, denoise, summarize, main_only, diarize, ielts, expected):
    assert mode.denoise is denoise
    assert mode.summarize is summarize
    assert mode.keep_main_speaker_only is main_only
    assert mode.diarize is diarize
    assert mode.analyze_ielts is ielts
    assert mode.expected_speakers == expected


def test_only_classroom_cleans_the_audio():
    """Denoise is the one step that alters the signal, so exactly one mode does it."""
    assert [m.key for m in MODES.values() if m.denoise] == ["classroom"]


def test_only_classroom_drops_other_speakers():
    """Everywhere else the transcript must keep every voice."""
    assert [m.key for m in MODES.values() if m.keep_main_speaker_only] == ["classroom"]


def test_condition_on_previous_is_off_for_every_mode():
    """Fidelity decision (Batch A): conditioning lets the decoder invent text, so
    it stays off unless a caller explicitly opts in."""
    assert all(m.condition_on_previous is False for m in MODES.values())
    assert Mode(key="x", label="x", description="x").condition_on_previous is False


def test_ielts_uses_finer_chunks_for_turn_separation():
    """Smaller chunks give diarization and per-chunk language detection more
    boundaries to work with."""
    assert IELTS.chunk_sec == 60.0
    assert IELTS.chunk_sec < Mode(key="x", label="x", description="x").chunk_sec


def test_no_mode_pins_a_language_by_default():
    assert all(m.language is None for m in MODES.values())


def test_mode_defaults_are_the_documented_ones():
    d = Mode(key="x", label="x", description="x")
    assert d.engine == "auto"
    assert d.chunked_language is True
    assert d.chunk_sec == 300.0
    assert d.initial_prompt == ""
    assert d.denoise_noise_floor_db == -45.0
    assert d.formats == ["txt", "md", "srt", "json"]


def test_every_mode_omits_the_json_export():
    """This is why the fidelity trail needed batch ②: without json there was
    nowhere to see the annotations."""
    for mode in MODES.values():
        assert "json" not in mode.formats
        assert {"txt", "md", "docx"} <= set(mode.formats)


def test_formats_are_not_shared_between_instances():
    """`field(default_factory=...)` — a plain default would alias one list, so
    appending to one mode's formats would leak into every other mode."""
    a, b = (Mode(key=k, label=k, description=k) for k in ("a", "b"))
    a.formats.append("sentinel")
    assert a.formats is not b.formats
    assert "sentinel" not in b.formats
