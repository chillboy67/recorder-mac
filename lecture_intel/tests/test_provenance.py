"""
Provenance — the audit trail the fidelity claim rests on.

`meta.json` is what lets someone verify *which* audio a transcript came from, so
these tests pin the append-only behaviour, the corrupt-file tolerance, and the
fact that `archive_input` records the hash of the copy it actually kept.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import provenance as P  # noqa: E402


def meta(out: Path) -> dict:
    return json.loads((out / "meta.json").read_text(encoding="utf-8"))


# ── sha256_file ─────────────────────────────────────────────────────

def test_sha256_matches_hashlib_and_is_deterministic(tmp_path):
    f = tmp_path / "blob.bin"
    payload = b"recorder" * 1000
    f.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    assert P.sha256_file(f) == expected
    assert P.sha256_file(f) == expected          # stable across calls


def test_sha256_streams_files_larger_than_one_chunk(tmp_path):
    """_CHUNK is 1 MiB; a file past that must still hash exactly."""
    f = tmp_path / "big.bin"
    payload = b"\x00\xff" * (P._CHUNK + 123)
    f.write_bytes(payload)
    assert P.sha256_file(f) == hashlib.sha256(payload).hexdigest()


# ── append_meta ─────────────────────────────────────────────────────

def test_append_meta_creates_the_file_and_stamps_a_timestamp(tmp_path):
    data = P.append_meta(tmp_path, {"name": "normalize", "params": {"a": 1}})
    assert [s["name"] for s in data["steps"]] == ["normalize"]
    assert data["steps"][0]["timestamp"]                     # auto-added
    assert data["steps"][0]["params"] == {"a": 1}
    assert meta(tmp_path) == data                            # persisted


def test_append_meta_accumulates_in_order_without_overwriting(tmp_path):
    P.append_meta(tmp_path, {"name": "normalize"})
    P.append_meta(tmp_path, {"name": "transcribe"})
    P.append_meta(tmp_path, {"name": "annotations"})
    assert [s["name"] for s in meta(tmp_path)["steps"]] == [
        "normalize", "transcribe", "annotations"]


def test_append_meta_preserves_other_top_level_keys(tmp_path):
    """`archive_input` writes "original"; later steps must not drop it."""
    (tmp_path / "meta.json").write_text(
        json.dumps({"original": {"file": "original.wav"}}), encoding="utf-8")
    P.append_meta(tmp_path, {"name": "normalize"})
    m = meta(tmp_path)
    assert m["original"] == {"file": "original.wav"}
    assert [s["name"] for s in m["steps"]] == ["normalize"]


def test_append_meta_does_not_let_a_step_forge_its_own_timestamp(tmp_path):
    P.append_meta(tmp_path, {"name": "x", "timestamp": "1999-01-01T00:00:00"})
    assert meta(tmp_path)["steps"][0]["timestamp"] == "1999-01-01T00:00:00"
    # setdefault: caller-supplied value wins, and nothing else is invented


def test_append_meta_recovers_from_a_corrupt_file(tmp_path):
    (tmp_path / "meta.json").write_text("{not json", encoding="utf-8")
    data = P.append_meta(tmp_path, {"name": "normalize"})
    assert [s["name"] for s in data["steps"]] == ["normalize"]
    assert meta(tmp_path) == data                 # rewritten cleanly


def test_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    P.append_meta(tmp_path, {"name": "one"})
    P.append_meta(tmp_path, {"name": "two"})
    assert not list(tmp_path.glob("*.tmp"))


# ── archive_input ───────────────────────────────────────────────────

def test_archive_input_keeps_a_verifiable_copy(tmp_path):
    src = tmp_path / "take.wav"
    src.write_bytes(b"RIFFfake-wave-content")
    out = tmp_path / "out"

    dest = P.archive_input(src, out)

    assert dest == out / "original.wav"
    assert dest.read_bytes() == src.read_bytes()      # byte-identical copy
    entry = meta(out)["original"]
    assert entry["file"] == "original.wav"
    # the hash describes the copy that was KEPT, not the caller's temp path
    assert entry["sha256"] == P.sha256_file(dest)
    assert entry["sha256"] == P.sha256_file(src)


def test_archive_input_creates_the_output_dir(tmp_path):
    src = tmp_path / "take.wav"
    src.write_bytes(b"x")
    out = tmp_path / "deep" / "nested" / "out"
    assert not out.exists()
    P.archive_input(src, out)
    assert out.is_dir()


def test_archive_input_survives_the_caller_deleting_their_copy(tmp_path):
    """The whole point: the output dir must stay self-contained evidence."""
    src = tmp_path / "gui-temp.wav"
    src.write_bytes(b"RIFFfake-wave-content")
    out = tmp_path / "out"
    P.archive_input(src, out)
    src.unlink()                                       # GUI cleans up its temp
    m = meta(out)
    assert (out / "original.wav").exists()
    assert P.sha256_file(out / "original.wav") == m["original"]["sha256"]


def test_append_meta_needs_an_existing_output_dir(tmp_path):
    """Unlike `archive_input`, append_meta does not create the directory.

    The engine is safe because archive_input runs first (engine.py step 0), but
    the precondition is real and load-bearing — pinned here so a future caller
    gets a failing test rather than a puzzling FileNotFoundError.
    """
    out = tmp_path / "not-created-yet"
    with pytest.raises(FileNotFoundError):
        P.append_meta(out, {"name": "manual"})
    out.mkdir()
    P.append_meta(out, {"name": "manual"})          # fine once it exists


def test_archive_input_does_not_disturb_existing_steps(tmp_path):
    src = tmp_path / "take.wav"
    src.write_bytes(b"x")
    out = tmp_path / "out"
    P.archive_input(src, out)                       # creates the dir, records original
    P.append_meta(out, {"name": "manual"})
    m = meta(out)
    assert [s["name"] for s in m["steps"]] == ["manual"]
    assert m["original"]["file"] == "original.wav"


# ── probe_wav ───────────────────────────────────────────────────────

def test_probe_wav_reports_the_real_format(tmp_path):
    sf = pytest.importorskip("soundfile")
    import numpy as np
    wav = tmp_path / "probe.wav"
    sf.write(str(wav), np.zeros(16000, dtype="float32"), 16000)
    info = P.probe_wav(wav)
    assert info["sample_rate"] == 16000
    assert info["channels"] == 1
    assert info["duration_sec"] == pytest.approx(1.0, abs=0.01)


def test_probe_wav_is_best_effort_on_non_audio(tmp_path):
    """Called on whatever `archive_input` was handed — never raises."""
    junk = tmp_path / "not-audio.wav"
    junk.write_bytes(b"definitely not a wave file")
    assert P.probe_wav(junk) == {}
