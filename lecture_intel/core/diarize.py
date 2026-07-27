"""
Token-free speaker diarization for the IELTS mode.

pyannote needs a gated HuggingFace model + token, which is friction for an
offline double-click app. Instead we use Resemblyzer's pretrained GE2E voice
encoder (≈17 MB, no token, downloads once) to embed each Whisper segment, then
cluster.

For an IELTS practice recording the structure is:
  - two main voices: the examiner (coach) and the candidate
  - occasionally background chatter from other people in the room

So the algorithm is:
  1. Embed every ASR segment's audio slice → 256-d d-vector.
  2. Agglomerative clustering (cosine) with a distance threshold → natural
     speaker groups.
  3. Rank groups by total speech time. The top 2 are examiner & candidate; any
     remaining groups are background "OTHER" and get excluded from analysis.
  4. Heuristically label which of the top-2 is the candidate (the one with the
     most English speech / longest answers) vs the examiner. The UI lets the
     user swap if the guess is wrong.

If anything fails, we degrade gracefully to a single speaker — transcription
still works.
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
    # speaker label per ASR segment id
    labels: dict[int, str]
    speaker_count: int
    candidate_seconds: float
    examiner_seconds: float
    other_seconds: float
    method: str


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
        embeds, idx_with_embed = _embed_segments(segments, wav_path)
    except Exception as exc:
        logger.warning("Speaker embedding failed (%s); single-speaker fallback", exc)
        labels = {s.id: CANDIDATE for s in asr.segments}
        return DiarizationResult(labels, 1, _dur(asr.segments), 0.0, 0.0, "fallback")

    if len(idx_with_embed) < 2:
        labels = {s.id: CANDIDATE for s in asr.segments}
        return DiarizationResult(labels, 1, _dur(asr.segments), 0.0, 0.0, "single")

    cluster_ids = _cluster(embeds, expected_speakers)

    # group segment indices by cluster
    groups: dict[int, list[ASRSegment]] = {}
    for seg_idx, cid in zip(idx_with_embed, cluster_ids):
        groups.setdefault(cid, []).append(segments[seg_idx])

    # rank clusters by total duration; top-2 are the conversation, rest = OTHER
    ranked = sorted(groups.items(), key=lambda kv: -sum(s.end - s.start for s in kv[1]))
    main = ranked[:2]
    background = ranked[2:]

    labels: dict[int, str] = {}

    # If the 2nd-biggest group is tiny, it's a mis-split of one voice, not a
    # real second speaker → collapse to a single speaker (candidate).
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
        (c0, segs0), (c1, segs1) = main[0], main[1]
        # candidate = the group that speaks more English / longer answers
        score0 = _candidate_score(segs0)
        score1 = _candidate_score(segs1)
        if score0 >= score1:
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

    # any segment we skipped (too short) → nearest in time gets same label
    for s in asr.segments:
        labels.setdefault(s.id, _nearest_label(s, asr.segments, labels))

    speaker_count = (2 if exam_s > 0 else 1) + (1 if background else 0)
    logger.info(
        "Diarization: candidate=%.0fs examiner=%.0fs other=%.0fs (%d raw groups)",
        cand_s, exam_s, other_s, len(ranked),
    )
    return DiarizationResult(labels, speaker_count, cand_s, exam_s, other_s, "resemblyzer")


# ----------------------------------------------------------------------
# Embedding & clustering
# ----------------------------------------------------------------------

def _embed_segments(segments: list[ASRSegment], wav_path: Path):
    """Return (embeddings array, list of segment indices that were embedded)."""
    import soundfile as sf
    from resemblyzer import VoiceEncoder

    wav, sr = sf.read(str(wav_path))
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)
    if sr != 16000:
        # AudioLoader already gives 16k, but stay safe.
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
        sr = 16000

    encoder = VoiceEncoder(verbose=False)
    embeds, idxs = [], []
    for i, seg in enumerate(segments):
        a = int(max(0, seg.start) * sr)
        b = int(min(len(wav) / sr, seg.end) * sr)
        slice_ = wav[a:b]
        if len(slice_) < int(0.4 * sr):   # too short to embed reliably
            continue
        try:
            emb = encoder.embed_utterance(slice_)
        except Exception:
            continue
        embeds.append(emb)
        idxs.append(i)
    if not embeds:
        return np.zeros((0, 256)), []
    return np.vstack(embeds), idxs


def _cluster(embeds: np.ndarray, expected: int) -> list[int]:
    from sklearn.cluster import AgglomerativeClustering

    n = len(embeds)
    if n <= 1:
        return [0] * n

    if expected and expected > 0:
        k = min(expected, n)
        model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
        return list(model.fit_predict(embeds))

    # auto: distance-threshold clustering on cosine
    model = AgglomerativeClustering(
        n_clusters=None, distance_threshold=0.35,
        metric="cosine", linkage="average",
    )
    return list(model.fit_predict(embeds))


# ----------------------------------------------------------------------
# Labeling heuristics
# ----------------------------------------------------------------------

def main_speaker_ids(asr: ASRResult, wav_path: Path) -> Optional[set[int]]:
    """Classroom helper: return segment ids belonging to the dominant speaker.

    Clusters voices and keeps only the one that talks the most (the lecturer),
    dropping background chatter. Returns None if clustering isn't possible
    (caller then keeps everything).
    """
    segments = [s for s in asr.segments if (s.end - s.start) > 0.0 and s.text.strip()]
    if len(segments) < 4:
        return None
    try:
        embeds, idxs = _embed_segments(segments, wav_path)
    except Exception as exc:
        logger.warning("main-speaker embedding failed (%s); keeping all", exc)
        return None
    if len(idxs) < 4:
        return None
    cluster_ids = _cluster(embeds, expected=0)  # auto
    groups: dict[int, list[ASRSegment]] = {}
    for seg_idx, cid in zip(idxs, cluster_ids):
        groups.setdefault(cid, []).append(segments[seg_idx])
    if len(groups) < 2:
        return None  # only one speaker → nothing to drop
    dominant = max(groups.items(), key=lambda kv: sum(s.end - s.start for s in kv[1]))
    kept = {s.id for s in dominant[1]}
    # also keep short segments adjacent in time to the dominant speaker
    dropped = sum(s.end - s.start for cid, segs in groups.items()
                  if cid != dominant[0] for s in segs)
    logger.info("Main-speaker filter: kept %d segs, dropped ~%.0fs of others",
                len(kept), dropped)
    return kept


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
    # candidate: lots of English words, long turns, asks few questions
    return (
        n_words * english_share
        + avg_turn * 3.0
        - questions * 8.0
    )


def _nearest_label(seg, all_segs, labels) -> str:
    best, best_gap = CANDIDATE, 1e9
    for other in all_segs:
        if other.id in labels and other.id != seg.id:
            gap = abs(other.start - seg.start)
            if gap < best_gap:
                best_gap, best = gap, labels[other.id]
    return best


def _dur(segs) -> float:
    return float(sum(s.end - s.start for s in segs))
