"""
Speaker separation — the parts the suite was not reaching.

`_candidate_score` was covered; these tests cover the rest of the module, above
all `main_speaker_ids`, whose whole job is to be *conservative*: it must keep a
monologue intact and only drop a substantial, sustained second voice. Every
early-return path here is a case where "keeping everything" is the right answer,
which is exactly the kind of branch that silently rots.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from modules import ASRResult, ASRSegment  # noqa: E402
from core import diarize as D  # noqa: E402


def seg(i, start, end, text="hello there", lang="en"):
    return ASRSegment(id=i, start=start, end=end, text=text, language=lang,
                      confidence=-0.1)


def asr(*segs) -> ASRResult:
    return ASRResult(segments=list(segs), full_text="x", language="en",
                     model_used="t")


def frames_for(segs, labels):
    """One frame per segment spanning it exactly, with the given cluster ids."""
    times = np.array([[s.start, s.end] for s in segs], dtype=float)
    embeds = np.zeros((len(segs), 4), dtype=float)
    return embeds, times, list(labels)


# ── _nearest_label ──────────────────────────────────────────────────

def test_nearest_label_takes_the_closest_labelled_neighbour():
    segs = [seg(0, 0.0, 2.0), seg(1, 10.0, 12.0), seg(2, 30.0, 32.0)]
    labels = {0: D.EXAMINER, 2: D.CANDIDATE}
    assert D._nearest_label(segs[1], segs, labels) == D.EXAMINER


def test_nearest_label_defaults_to_the_candidate():
    segs = [seg(0, 0.0, 2.0), seg(1, 10.0, 12.0)]
    assert D._nearest_label(segs[1], segs, {}) == D.CANDIDATE


def test_nearest_label_ignores_the_segment_itself():
    segs = [seg(0, 0.0, 2.0), seg(1, 10.0, 12.0)]
    labels = {1: D.EXAMINER, 0: D.CANDIDATE}
    # seg 1's own label must not win: the only *other* labelled segment is seg 0
    assert D._nearest_label(segs[1], segs, labels) == D.CANDIDATE


# ── _smooth ─────────────────────────────────────────────────────────

def test_smooth_flips_a_lone_segment_between_two_of_the_other_speaker():
    segs = [seg(0, 0.0, 1.0), seg(1, 1.0, 2.0), seg(2, 2.0, 3.0)]
    out = D._smooth(segs, {0: 0, 1: 1, 2: 0})
    assert out == {0: 0, 1: 0, 2: 0}


def test_smooth_leaves_a_genuine_speaker_change_alone():
    """Two consecutive segments of the other speaker are a real turn, not noise."""
    segs = [seg(0, 0.0, 1.0), seg(1, 1.0, 2.0), seg(2, 2.0, 3.0),
            seg(3, 3.0, 4.0)]
    out = D._smooth(segs, {0: 0, 1: 1, 2: 1, 3: 0})
    assert out == {0: 0, 1: 1, 2: 1, 3: 0}


def test_smooth_never_touches_the_first_or_last_segment():
    segs = [seg(0, 0.0, 1.0), seg(1, 1.0, 2.0), seg(2, 2.0, 3.0)]
    out = D._smooth(segs, {0: 1, 1: 0, 2: 1})
    assert out[0] == 1 and out[2] == 1


def test_smooth_uses_time_order_not_id_order():
    segs = [seg(5, 2.0, 3.0), seg(6, 0.0, 1.0), seg(7, 1.0, 2.0)]
    out = D._smooth(segs, {6: 0, 7: 1, 5: 0})
    assert out == {6: 0, 7: 0, 5: 0}


# ── _assign_segments ────────────────────────────────────────────────

def test_assign_segments_votes_by_overlap_duration():
    segs = [seg(0, 0.0, 4.0)]
    times = np.array([[0.0, 1.0], [1.0, 4.0]])
    # 1s of cluster 1, 3s of cluster 0 → segment 0 belongs to cluster 0
    assert D._assign_segments(segs, [1, 0], times) == {0: 0}


def test_assign_segments_skips_segments_no_frame_covers():
    segs = [seg(0, 0.0, 1.0), seg(1, 50.0, 51.0)]
    times = np.array([[0.0, 1.0]])
    assert D._assign_segments(segs, [0], times) == {0: 0}


# ── _dur ────────────────────────────────────────────────────────────

def test_dur_sums_segment_durations():
    assert D._dur([seg(0, 0.0, 2.5), seg(1, 3.0, 4.5)]) == 4.0
    assert D._dur([]) == 0.0


# ── DiarizationResult ───────────────────────────────────────────────

def test_diarization_result_carries_the_three_time_buckets():
    r = D.DiarizationResult(labels={0: D.CANDIDATE}, speaker_count=2,
                            candidate_seconds=30.0, examiner_seconds=8.0,
                            other_seconds=0.0, method="resemblyzer")
    assert r.labels == {0: "candidate"} and r.method == "resemblyzer"
    assert r.candidate_seconds + r.examiner_seconds + r.other_seconds == 38.0


# ── main_speaker_ids: every "keep everything" path ──────────────────

@pytest.fixture
def eight_segs():
    """8 × 5s segments: four per cluster when split down the middle."""
    return [seg(i, i * 5.0, i * 5.0 + 5.0) for i in range(8)]


def test_main_speaker_keeps_all_for_a_short_recording(monkeypatch):
    monkeypatch.setattr(D, "_frame_embeddings", lambda p: (_ for _ in ()).throw(
        AssertionError("must not embed for fewer than 6 segments")))
    assert D.main_speaker_ids(asr(*[seg(i, i * 2.0, i * 2.0 + 2.0)
                                    for i in range(5)]), Path("x.wav")) is None


def test_main_speaker_ignores_empty_or_zero_length_segments(monkeypatch):
    """They do not count towards the 6-segment minimum."""
    monkeypatch.setattr(D, "_frame_embeddings", lambda p: (_ for _ in ()).throw(
        AssertionError("must not embed")))
    segs = [seg(i, i * 2.0, i * 2.0 + 2.0) for i in range(4)]
    segs += [seg(4, 10.0, 10.0), seg(5, 11.0, 12.0, text="   ")]
    assert D.main_speaker_ids(asr(*segs), Path("x.wav")) is None


def test_main_speaker_keeps_all_when_embedding_fails(eight_segs, monkeypatch):
    def boom(_path):
        raise RuntimeError("no resemblyzer")

    monkeypatch.setattr(D, "_frame_embeddings", boom)
    assert D.main_speaker_ids(asr(*eight_segs), Path("x.wav")) is None


def test_main_speaker_keeps_all_with_too_few_frames(eight_segs, monkeypatch):
    monkeypatch.setattr(D, "_frame_embeddings",
                        lambda p: (np.zeros((1, 4)), np.zeros((1, 2)), None))
    assert D.main_speaker_ids(asr(*eight_segs), Path("x.wav")) is None


def test_main_speaker_keeps_all_when_clustering_finds_one_voice(
        eight_segs, monkeypatch):
    embeds, times, _ = frames_for(eight_segs, [0] * 8)
    monkeypatch.setattr(D, "_frame_embeddings", lambda p: (embeds, times))
    monkeypatch.setattr(D, "_cluster", lambda e, expected=0: [0] * 8)
    assert D.main_speaker_ids(asr(*eight_segs), Path("x.wav")) is None


def test_main_speaker_keeps_all_when_the_other_voice_is_too_few_segments(
        eight_segs, monkeypatch):
    """3 dropped segments is under the floor of 4 → not a real second speaker."""
    embeds, times, _ = frames_for(eight_segs, [0, 0, 0, 0, 0, 1, 1, 1])
    monkeypatch.setattr(D, "_frame_embeddings", lambda p: (embeds, times))
    monkeypatch.setattr(D, "_cluster", lambda e, expected=0: [0] * 5 + [1] * 3)
    assert D.main_speaker_ids(asr(*eight_segs), Path("x.wav")) is None


def test_main_speaker_keeps_all_when_the_other_voice_is_too_short(
        monkeypatch):
    """Enough dropped segments, but under max(12s, 10% of the recording).

    Four 20s turns (the dominant voice) plus four 2s turns: 8s of "other" is
    above the 10% share (8.8s of 88s) yet below the 12s floor, so it is treated
    as noise rather than a second speaker.
    """
    segs = [seg(i, i * 20.0, i * 20.0 + (20.0 if i < 4 else 2.0))
            for i in range(8)]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    embeds, times, _ = frames_for(segs, labels)
    monkeypatch.setattr(D, "_frame_embeddings", lambda p: (embeds, times))
    monkeypatch.setattr(D, "_cluster", lambda e, expected=0: labels)

    assert D.main_speaker_ids(asr(*segs), Path("x.wav")) is None


def test_main_speaker_drops_a_substantial_sustained_second_voice(
        eight_segs, monkeypatch):
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    embeds, times, _ = frames_for(eight_segs, labels)
    monkeypatch.setattr(D, "_frame_embeddings", lambda p: (embeds, times))
    monkeypatch.setattr(D, "_cluster", lambda e, expected=0: labels)

    kept = D.main_speaker_ids(asr(*eight_segs), Path("x.wav"))

    assert kept == {0, 1, 2, 3}                 # the dominant cluster only


def test_main_speaker_returns_none_when_only_one_cluster_survives_smoothing(
        eight_segs, monkeypatch):
    """One stray segment is smoothed away, leaving a single voice → keep all."""
    labels = [0, 0, 0, 0, 0, 0, 0, 1]
    embeds, times, _ = frames_for(eight_segs, labels)
    monkeypatch.setattr(D, "_frame_embeddings", lambda p: (embeds, times))
    monkeypatch.setattr(D, "_cluster", lambda e, expected=0: labels)
    assert D.main_speaker_ids(asr(*eight_segs), Path("x.wav")) is None
