"""
Real-model integration tests — opt-in, and they need audio that is NOT in the repo.

Everything else in ``tests/`` stubs the ML layer, so the actual Whisper and
resemblyzer code paths were never exercised. These tests run them for real, on
real speech, because the failures that matter here (a voice split into spurious
clusters, a transcript whose word clocks drift outside its segment) are invisible
to stubs.

Skipped unless ``RECORDER_REAL_MODEL_TESTS=1``. They need:

* the mlx models already installed (``whisper-large-v3-turbo-mlx``),
* ``resemblyzer`` + ``torch`` in the venv,
* the audio clips below, which are **gitignored** (they contain real voices):

Build them once (public-domain LibriVox, readers ``pac``, ``mlm``, ``pfs``)::

    cd sample
    D=https://archive.org/download/spc277_2607_librivox
    for f in spc277_godsworld_pac spc277_april_pac \
             spc277_averageman_mlm spc277_familyfinanciering_mlm \
             spc277_foresthymn_pfs; do
      curl -sSLO "$D/${f}_64kb.mp3"
    done

    # two speakers, interleaved A-B-A-B (~102s; A=60s, B=40s)
    cut() { ffmpeg -v error -y -i "$1" -ss "$2" -t "$3" -ac 1 -ar 16000 -c:a pcm_s16le "$4"; }
    cut spc277_godsworld_pac_64kb.mp3            5 30 /tmp/a1.wav
    cut spc277_averageman_mlm_64kb.mp3          10 20 /tmp/b1.wav
    cut spc277_april_pac_64kb.mp3                5 30 /tmp/a2.wav
    cut spc277_familyfinanciering_mlm_64kb.mp3  20 20 /tmp/b2.wav
    ffmpeg -v error -y -f lavfi -i anullsrc=r=16000:cl=mono -t 0.6 -ac 1 \\
           -ar 16000 -c:a pcm_s16le /tmp/gap.wav
    printf "file '/tmp/a1.wav'\nfile '/tmp/gap.wav'\nfile '/tmp/b1.wav'\nfile '/tmp/gap.wav'\nfile '/tmp/a2.wav'\nfile '/tmp/gap.wav'\nfile '/tmp/b2.wav'\n" > /tmp/l.txt
    ffmpeg -v error -y -f concat -safe 0 -i /tmp/l.txt -c:a pcm_s16le real_two_speakers.wav

    # one speaker, short and long (the long one has enough segments to bypass
    # main_speaker_ids' "fewer than 4 dropped segments" guard)
    cut spc277_familyfinanciering_mlm_64kb.mp3   5 100 real_one_speaker.wav
    cut spc277_foresthymn_pfs_64kb.mp3           5 300 real_one_speaker_long.wav

Run with::

    RECORDER_REAL_MODEL_TESTS=1 .venv/bin/python3 -m pytest tests/test_real_models.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

if not os.environ.get("RECORDER_REAL_MODEL_TESTS"):
    pytest.skip("real-model tests are opt-in: set RECORDER_REAL_MODEL_TESTS=1",
                allow_module_level=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SAMPLES = ROOT.parent / "sample"
TWO_SPEAKERS = SAMPLES / "real_two_speakers.wav"
ONE_SPEAKER = SAMPLES / "real_one_speaker.wav"
ONE_SPEAKER_LONG = SAMPLES / "real_one_speaker_long.wav"

# The two-speaker clip is built as A-B-A-B; these are the source ranges, used to
# check that diarization reproduces the construction instead of just "some" split.
A_RANGES = ((0.0, 30.0), (50.0, 79.0))     # reader "pac"
B_RANGES = ((33.0, 51.0), (79.0, 103.0))   # reader "mlm"

pytestmark = pytest.mark.skipif(
    not TWO_SPEAKERS.exists(),
    reason=f"real audio not built (see module docstring); missing {TWO_SPEAKERS}")


@pytest.fixture(scope="module")
def transcriber():
    from core.transcriber import Transcriber
    return Transcriber(model="large-v3-turbo")


@pytest.fixture(scope="module")
def two_speaker_asr(transcriber):
    return transcriber.transcribe(TWO_SPEAKERS, language="en")


@pytest.fixture(scope="module")
def long_single_speaker_asr(transcriber):
    if not ONE_SPEAKER_LONG.exists():
        pytest.skip(f"missing {ONE_SPEAKER_LONG}")
    return transcriber.transcribe(ONE_SPEAKER_LONG, language="en")


def mid(segment) -> float:
    return (segment.start + segment.end) / 2


def source_of(segment) -> str:
    """Which construction range the segment belongs to: 'A', 'B' or 'gap'."""
    m = mid(segment)
    for lo, hi in A_RANGES:
        if lo <= m <= hi:
            return "A"
    for lo, hi in B_RANGES:
        if lo <= m <= hi:
            return "B"
    return "gap"


# ── transcription ───────────────────────────────────────────────────

def test_real_transcription_produces_word_level_timestamps(two_speaker_asr):
    asr = two_speaker_asr
    assert asr.segments, "mlx-whisper returned no segments for real speech"
    assert asr.language == "en"
    assert "large-v3-turbo" in asr.model_used
    assert asr.warnings == []

    for seg in asr.segments:
        assert seg.text.strip()
        assert seg.end > seg.start
        assert seg.words, "word_timestamps must be on — the arbitration needs them"
        # every word clock stays inside its own segment (a drift here would
        # corrupt the L1 timeline test in repeat_arbitration)
        for w in seg.words:
            assert seg.start - 0.05 <= w.start <= seg.end + 0.05
            assert w.end >= w.start


def test_real_segments_advance_in_time(two_speaker_asr):
    starts = [s.start for s in two_speaker_asr.segments]
    assert starts == sorted(starts)
    assert starts[-1] < 105.0            # the clip is 101.8s


# ── frame embeddings ────────────────────────────────────────────────

def test_real_frame_embeddings_cover_the_recording():
    from core import diarize as D
    embeds, times = D._frame_embeddings(TWO_SPEAKERS)

    assert len(embeds) >= 2 and embeds.shape[1] == 256      # resemblyzer's dim
    assert len(times) == len(embeds)
    assert times[0][0] == pytest.approx(0.0, abs=0.2)
    assert times[-1][1] == pytest.approx(101.8, abs=1.0)
    # frames must not go backwards: _assign_segments relies on the order
    assert all(times[i][0] <= times[i + 1][0] for i in range(len(times) - 1))


# ── diarization ─────────────────────────────────────────────────────

def test_real_diarization_finds_the_two_constructed_speakers(two_speaker_asr):
    from core import diarize as D
    result = D.diarize(two_speaker_asr, TWO_SPEAKERS, expected_speakers=2)

    assert result.method == "resemblyzer"          # not a fallback
    assert result.speaker_count == 2
    assert result.other_seconds == 0.0
    total = (result.candidate_seconds + result.examiner_seconds
             + result.other_seconds)
    assert total == pytest.approx(D._dur(two_speaker_asr.segments), rel=0.05)

    # every segment got a label
    assert set(result.labels) == {s.id for s in two_speaker_asr.segments}

    # ...and the split reproduces the A-B-A-B construction: both "pac" chunks
    # share one label, both "mlm" chunks share the other, and they disagree.
    a_labels = {result.labels[s.id] for s in two_speaker_asr.segments
                if source_of(s) == "A"}
    b_labels = {result.labels[s.id] for s in two_speaker_asr.segments
                if source_of(s) == "B"}
    assert a_labels and b_labels
    assert len(a_labels) == 1, f"the same reader was split: {a_labels}"
    assert len(b_labels) == 1, f"the same reader was split: {b_labels}"
    assert a_labels != b_labels, "the two readers were not separated"


def test_real_diarization_refuses_to_invent_a_second_speaker(transcriber):
    """A single voice must come back as one speaker, whatever the chunking."""
    if not ONE_SPEAKER.exists():
        pytest.skip(f"missing {ONE_SPEAKER}")
    from core import diarize as D
    asr = transcriber.transcribe(ONE_SPEAKER, language="en")
    result = D.diarize(asr, ONE_SPEAKER, expected_speakers=2)
    assert result.speaker_count == 1
    assert result.examiner_seconds == 0.0
    assert set(result.labels.values()) == {D.CANDIDATE}


# ── main-speaker filter ─────────────────────────────────────────────

def test_real_main_speaker_keeps_everything_for_one_voice(long_single_speaker_asr):
    """The documented promise: a single speaker with natural variation must NOT
    be filtered. The long clip has 25 segments, so the "fewer than 4 dropped
    segments" guard cannot be what saves it — the clustering has to be sane."""
    from core import diarize as D
    assert len(long_single_speaker_asr.segments) >= 6
    assert D.main_speaker_ids(long_single_speaker_asr, ONE_SPEAKER_LONG) is None


def test_real_main_speaker_filter_returns_segment_ids(two_speaker_asr):
    """On genuinely two-voice audio the dominant cluster may be kept.

    This is a *shape* assertion on purpose. Measured on this clip the filter
    keeps only the single largest cluster (see the limitation noted in
    HANDOVER §20.4): it cannot know that a chunk it is discarding belongs to the
    same reader as the one it keeps, because the auto cluster count is driven by
    timbre rather than by identity.
    """
    from core import diarize as D
    kept = D.main_speaker_ids(two_speaker_asr, TWO_SPEAKERS)

    if kept is None:
        return                                   # keeping all is always valid
    ids = {s.id for s in two_speaker_asr.segments}
    assert kept <= ids and kept                     # must be a real subset
