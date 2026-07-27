"""
Token-free speaker diarization for the IELTS / classroom modes.

pyannote needs a gated HuggingFace model + token, which is friction for an
offline double-click app. Instead we use Resemblyzer's pretrained GE2E voice
encoder (≈17 MB, no token, downloads once).

The quality-critical detail: we do NOT embed each short ASR segment in isolation
(a 1–3s clip gives a noisy speaker vector, which scatters one speaker across
clusters). Instead we run the standard Resemblyzer diarization recipe:

  1. Slide a 1.6s window across the WHOLE audio → a stream of d-vectors
     ("frames"), each with a time span.
  2. Cluster the frames (cosine). For a fixed speaker count (IELTS = 2) we force
     k clusters; otherwise a distance threshold finds the natural number.
  3. Assign each ASR segment to the cluster whose frames cover most of its time.
  4. Temporal smoothing: a lone segment flanked by the other speaker is almost
     always a mis-assignment, so flip it.
  5. Rank clusters by total speech time → top 2 are the conversation
     (examiner/candidate), the rest is background "OTHER" and is excluded.
  6. Label which of the two is the candidate (more English, longer turns, asks
     fewer questions). The UI can swap if the guess is wrong.

If anything fails we degrade to a single speaker — transcription still works.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from modules import ASRResult, ASRSegment

logger = logging.getLogger(__name__)

CANDIDATE = "candidate"   # 考生
EXAMINER = "examiner"     # 教官
OTHER = "other"           # 背景人声

# A second cluster smaller than this is treated as noise / mis-split, not a
# real second speaker — prevents a monologue from showing a phantom examiner.
MIN_SECOND_SPEAKER_SEC = 6.0
MIN_SECOND_SPEAKER_SEGS = 2

_QUESTION_CUES = re.compile(
    r"\b(what|why|how|where|when|who|which|can you|could you|do you|"
    r"tell me|describe|would you|have you|are you)\b", re.IGNORECASE,
)


@dataclass
class DiarizationResult:
    labels: dict[int, str]          # speaker label per ASR segment id
    speaker_count: int
    candidate_seconds: float
    examiner_seconds: float
    other_seconds: float
    method: str


# ----------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------

def diarize(
    asr: ASRResult,
    wav_path: Path,
    expected_speakers: int = 2,
) -> DiarizationResult:
    segments = [s for s in asr.segments if (s.end - s.start) > 0.0 and s.text.strip()]
    if len(segments) < 2:
        labels = {s.id: CANDIDATE for s in asr.segments}
        return DiarizationResult(labels, 1, _dur(asr.segments), 0.0, 0.0, "single")

    try:
        embeds, ftimes = _frame_embeddings(wav_path)
    except Exception as exc:
        logger.warning("Speaker embedding failed (%s); single-speaker fallback", exc)
        labels = {s.id: CANDIDATE for s in asr.segments}
        return DiarizationResult(labels, 1, _dur(asr.segments), 0.0, 0.0, "fallback")

    if len(embeds) < 2:
        labels = {s.id: CANDIDATE for s in asr.segments}
        return DiarizationResult(labels, 1, _dur(asr.segments), 0.0, 0.0, "single")

    frame_labels = _cluster(embeds, expected_speakers)
    seg_label = _assign_segments(segments, frame_labels, ftimes)
    seg_label = _smooth(segments, seg_label)

    # group by cluster
    groups: dict[int, list[ASRSegment]] = {}
    for seg in segments:
        cid = seg_label.get(seg.id)
        if cid is None:
            continue
        groups.setdefault(cid, []).append(seg)
    if not groups:
        labels = {s.id: CANDIDATE for s in asr.segments}
        return DiarizationResult(labels, 1, _dur(asr.segments), 0.0, 0.0, "fallback")

    ranked = sorted(groups.items(), key=lambda kv: -_dur(kv[1]))
    main = ranked[:2]
    background = ranked[2:]

    total_speech = _dur(segments)
    acoustic_minority = _dur(main[1][1]) if len(main) == 2 else 0.0
    acoustic_quality = acoustic_minority / max(total_speech, 1e-9)

    # In a bilingual IELTS coaching session the student answers in English and
    # the coach corrects/instructs in Chinese — so any Chinese in a segment is a
    # very strong "this is the coach" signal. Two same-gender voices are often
    # acoustically too similar for the embedder to split (it collapses ~everything
    # into one cluster). When that happens AND there's meaningful Chinese, the
    # language signal is far more reliable than the degenerate acoustic split.
    zh_dur = sum(s.end - s.start for s in segments if _has_chinese(s.text))
    zh_share = zh_dur / max(total_speech, 1e-9)
    use_language = zh_share >= 0.10 and acoustic_quality < 0.18

    labels: dict[int, str] = {}

    if use_language:
        cand_s = exam_s = 0.0
        for s in asr.segments:
            if _has_chinese(s.text):
                labels[s.id] = EXAMINER
                exam_s += s.end - s.start
            else:
                labels[s.id] = CANDIDATE
                cand_s += s.end - s.start
        background = []
        other_s = 0.0
        method = "language"
        logger.info(
            "Diarization (language split): candidate=%.0fs examiner=%.0fs "
            "(zh_share=%.0f%%, acoustic split too weak %.0f%%)",
            cand_s, exam_s, zh_share * 100, acoustic_quality * 100,
        )
    else:
        method = "resemblyzer"
        second_is_real = (
            len(main) == 2
            and _dur(main[1][1]) >= MIN_SECOND_SPEAKER_SEC
            and len(main[1][1]) >= MIN_SECOND_SPEAKER_SEGS
        )
        if len(main) == 1 or not second_is_real:
            for _, segs in main:
                for s in segs:
                    labels[s.id] = CANDIDATE
            cand_s = _dur([s for _, segs in main for s in segs])
            exam_s = 0.0
        else:
            (_, segs0), (_, segs1) = main[0], main[1]
            if _candidate_score(segs0) >= _candidate_score(segs1):
                cand_segs, exam_segs = segs0, segs1
            else:
                cand_segs, exam_segs = segs1, segs0
            for s in cand_segs:
                labels[s.id] = CANDIDATE
            for s in exam_segs:
                labels[s.id] = EXAMINER
            cand_s, exam_s = _dur(cand_segs), _dur(exam_segs)

        other_s = 0.0
        for _, segs in background:
            for s in segs:
                labels[s.id] = OTHER
            other_s += _dur(segs)

        for s in asr.segments:
            labels.setdefault(s.id, _nearest_label(s, asr.segments, labels))
        logger.info(
            "Diarization: candidate=%.0fs examiner=%.0fs other=%.0fs (%d raw groups)",
            cand_s, exam_s, other_s, len(ranked),
        )

    speaker_count = (2 if exam_s > 0 else 1) + (1 if background else 0)
    return DiarizationResult(labels, speaker_count, cand_s, exam_s, other_s, method)


def main_speaker_ids(asr: ASRResult, wav_path: Path) -> Optional[set[int]]:
    """Classroom helper: return segment ids belonging to the dominant speaker.

    Conservative — only drops a *substantial, sustained* secondary voice
    (background chatter); otherwise keeps everything (one speaker with normal
    acoustic variation). Returns None to mean "keep all"."""
    segments = [s for s in asr.segments if (s.end - s.start) > 0.0 and s.text.strip()]
    if len(segments) < 6:
        return None
    try:
        embeds, ftimes = _frame_embeddings(wav_path)
    except Exception as exc:
        logger.warning("main-speaker embedding failed (%s); keeping all", exc)
        return None
    if len(embeds) < 2:
        return None

    frame_labels = _cluster(embeds, expected=0)  # auto
    seg_label = _smooth(segments, _assign_segments(segments, frame_labels, ftimes))

    groups: dict[int, list[ASRSegment]] = {}
    for seg in segments:
        cid = seg_label.get(seg.id)
        if cid is not None:
            groups.setdefault(cid, []).append(seg)
    if len(groups) < 2:
        return None

    dominant = max(groups.items(), key=lambda kv: _dur(kv[1]))
    others = [(c, s) for c, s in groups.items() if c != dominant[0]]
    dropped_dur = sum(_dur(s) for _, s in others)
    dropped_segs = sum(len(s) for _, s in others)
    total_dur = _dur(segments)

    if (dropped_dur < max(12.0, 0.10 * total_dur)) or (dropped_segs < 4):
        logger.info("Main-speaker filter: non-dominant voice too small "
                    "(%.0fs / %d segs) → keeping all", dropped_dur, dropped_segs)
        return None

    kept = {s.id for s in dominant[1]}
    logger.info("Main-speaker filter: kept %d segs, dropped ~%.0fs / %d segs of others",
                len(kept), dropped_dur, dropped_segs)
    return kept


# ----------------------------------------------------------------------
# Continuous frame embeddings
# ----------------------------------------------------------------------

def _frame_embeddings(wav_path: Path):
    """Return (frame_embeddings [N,256], frame_times [N,2] in seconds)."""
    import soundfile as sf
    from resemblyzer import VoiceEncoder

    wav, sr = sf.read(str(wav_path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000
    peak = float(np.abs(wav).max())
    if peak > 0:
        wav = wav / peak * 0.95   # normalize amplitude, but DON'T trim silence
                                  # (trimming would break time alignment)

    encoder = VoiceEncoder(verbose=False)
    # rate=1.3 → a 1.6s window every ~0.77s; min_coverage keeps the tail frame.
    _, partials, slices = encoder.embed_utterance(
        wav, return_partials=True, rate=1.3, min_coverage=0.5,
    )
    partials = np.asarray(partials)
    times = np.array([[sl.start / sr, sl.stop / sr] for sl in slices], dtype=float)
    return partials, times


def _assign_segments(segments, frame_labels, frame_times) -> dict[int, int]:
    """Assign each ASR segment the cluster of the frames covering it most."""
    out: dict[int, int] = {}
    fl = np.asarray(frame_labels)
    for seg in segments:
        a, b = seg.start, seg.end
        # overlap of each frame with [a, b]
        ov = np.minimum(b, frame_times[:, 1]) - np.maximum(a, frame_times[:, 0])
        mask = ov > 0
        if not mask.any():
            continue
        votes: dict[int, float] = {}
        for lab, w in zip(fl[mask], ov[mask]):
            votes[int(lab)] = votes.get(int(lab), 0.0) + float(w)
        out[seg.id] = max(votes, key=votes.get)
    return out


def _smooth(segments, seg_label: dict[int, int]) -> dict[int, int]:
    """Flip a lone segment whose both time-neighbors are the other speaker."""
    ordered = sorted(segments, key=lambda s: s.start)
    ids = [s.id for s in ordered]
    out = dict(seg_label)
    for i in range(1, len(ids) - 1):
        prev_l = out.get(ids[i - 1])
        cur_l = out.get(ids[i])
        next_l = out.get(ids[i + 1])
        if cur_l is not None and prev_l is not None and prev_l == next_l and cur_l != prev_l:
            out[ids[i]] = prev_l
    return out


def _cluster(embeds: np.ndarray, expected: int) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering

    n = len(embeds)
    if n <= 1:
        return [0] * n
    if expected and expected > 0:
        k = min(expected, n)
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        return list(model.fit_predict(embeds))
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.30,
        metric="cosine", linkage="average",
    )
    return list(model.fit_predict(embeds))


# ----------------------------------------------------------------------
# Labeling heuristics
# ----------------------------------------------------------------------

def _candidate_score(segs: list[ASRSegment]) -> float:
    """Higher = more likely the candidate (long English answers, few questions)."""
    text = " ".join(s.text for s in segs)
    words = text.split()
    n_words = len(words)
    latin = len(re.findall(r"[a-zA-Z]", text))
    cjk = len(re.findall(r"[一-鿿]", text))
    english_share = latin / max(latin + cjk, 1)
    questions = len(_QUESTION_CUES.findall(text))
    avg_turn = n_words / max(len(segs), 1)
    return n_words * english_share + avg_turn * 3.0 - questions * 8.0


def _nearest_label(seg, all_segs, labels) -> str:
    best, best_gap = CANDIDATE, 1e9
    for other in all_segs:
        if other.id in labels and other.id != seg.id:
            gap = abs(other.start - seg.start)
            if gap < best_gap:
                best_gap, best = gap, labels[other.id]
    return best


def _has_chinese(text: str) -> bool:
    """True if the segment contains a meaningful amount of Chinese (≥2 chars)."""
    return len(re.findall(r"[一-鿿]", text)) >= 2


def _dur(segs) -> float:
    return float(sum(s.end - s.start for s in segs))
