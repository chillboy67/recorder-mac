"""
Repeat arbitration — separate a speaker's real stutter from an ASR loop.

Text alone cannot tell "no no no no" (a stutter) from the same string produced
by a decoding loop, so nothing here folds text unconditionally. Each suspected
run is judged on the *word timeline* plus the *audio*:

  L1 (free)    timeline shape — do word clocks actually advance across copies?
  L2 (ffmpeg)  silencedetect — does each copy carry its own voicing burst?
  L3 (model)   isolated re-decode of the window (only when L2 is inconclusive)

Every verdict lands in ``ASRResult.annotations`` with its evidence. Only
classroom mode folds a confirmed ``asr_loop`` — and the original text always
survives in the annotation, so the edit is reviewable after the fact.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import statistics
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from modules import ASRResult, ASRSegment

logger = logging.getLogger(__name__)

MAX_NGRAM = 4          # longest repeating unit considered ("no" = 1, "the fact" = 2)
MIN_REPEATS = 4        # k ≥ 4 — genuine emphasis ("no, no, no") stays untouched
TIMELINE_FACTOR = 0.6  # speech occupies ~60% of its span; the rest is pauses
TIMELINE_ASR_LOOP = 0.5
# A frozen-clock loop spans about ONE copy no matter how many times the decoder
# re-emits it; a real rapid stutter still consumes k× real time. The absolute
# cap keeps long real runs (e.g. 200 copies of "不" in 12 s, each far faster
# than the file-wide median word) from being misjudged by the relative test.
FROZEN_SPAN_MAX = 0.8  # seconds
SILENCE_PAD = 0.3      # clip padding around the run for the audio probes
SILENCE_NOISE_DB = -35.0
# silencedetect only takes one duration: the minimum *silence* length that
# counts as a separator. Bursts shorter than ~120 ms are not reliable copies.
SILENCE_MIN_DUR = 0.08
RECODE_OK_RATIO = 0.75
# L3 re-decodes the run window at several temperatures: real stutter survives
# across decoding strategies (the n-gram is in the audio), while a decode-loop
# artifact is unstable. The greedy seed (0.0) keeps the historic semantics;
# the sampling seeds act as a rescue oracle — e.g. a shouted phone-call window
# that greedy hears as nothing can still reproduce the n-gram under sampling.
L3_SEEDS = (0.0, 0.2, 0.8)


# ----------------------------------------------------------------------
# Runs
# ----------------------------------------------------------------------

@dataclass
class Run:
    """One n-gram repeated k× consecutively inside a segment."""
    segment_id: int
    n: int                     # words per copy
    k: int                     # consecutive copies
    gram: tuple[str, ...]      # normalized words of a single copy
    start: float               # first copy start (s)
    end: float                 # last copy end (s)
    word_start: int            # index of the run's first word in segment.words
    word_end: int              # index one past the run's last word
    original: str              # exact text spanned by the run
    words: list = field(default_factory=list, repr=False)
    verified: bool = True      # False = text-regex fallback (no word timestamps)


@dataclass
class Verdict:
    segment_id: int
    start: float
    end: float
    original: str
    verdict: str               # "asr_loop" | "real_speech" | "uncertain"
    evidence: dict = field(default_factory=dict)
    run: Optional[Run] = field(default=None, repr=False)


def _norm(word: str) -> str:
    """Comparison key: case-folded, punctuation stripped (keeps CJK)."""
    return re.sub(r"[^\w\s]", "", word, flags=re.UNICODE).lower().strip()


def find_repeat_runs(segment: ASRSegment) -> list[Run]:
    """Detect n-grams (n ≤ 4) repeated k ≥ 4 times in a row.

    Prefers the word timeline (word_timestamps is always on); without word
    timestamps falls back to the old text regex, but every run found that way
    is marked unverified so it can only ever be annotated, never folded.
    """
    words = segment.words
    runs: list[Run] = []
    if words:
        toks = [_norm(w.word) for w in words]
        i, total = 0, len(toks)
        while i < total:
            matched = False
            for n in range(1, MAX_NGRAM + 1):
                if i + n * MIN_REPEATS > total:
                    continue
                gram = tuple(toks[i:i + n])
                if any(not g for g in gram):
                    continue            # punctuation-only tokens: not a phrase
                k = 1
                while toks[i + k * n:i + (k + 1) * n] == list(gram):
                    k += 1
                if k >= MIN_REPEATS:
                    j = i + k * n
                    runs.append(Run(
                        segment_id=segment.id, n=n, k=k, gram=gram,
                        start=words[i].start, end=words[j - 1].end,
                        word_start=i, word_end=j,
                        original="".join(w.word for w in words[i:j]),
                        words=words[i:j]))
                    i = j
                    matched = True
                    break
            if not matched:
                i += 1
        return runs
    return _text_fallback_runs(segment)


def _text_fallback_runs(segment: ASRSegment) -> list[Run]:
    """No word timestamps: locate loops by text alone, all unverified."""
    runs = []
    for m in re.finditer(r"(.{1,8}?)\1{3,}", segment.text):
        unit, whole = m.group(1), m.group(0)
        if not unit.strip():
            continue
        k = whole.count(unit)
        runs.append(Run(segment_id=segment.id, n=1, k=k, gram=(_norm(unit),),
                        start=segment.start, end=segment.end,
                        word_start=0, word_end=0, original=whole,
                        verified=False))
    return runs


def median_word_duration(segments: list[ASRSegment]) -> float:
    durs = [w.end - w.start for s in segments for w in s.words
            if w.end > w.start]
    return statistics.median(durs) if durs else 0.25


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------

def classify_run(run: Run, wav_path, transcriber, segment_lang: str,
                 median_dur: float, oracle_path=None) -> Verdict:
    """Three-level arbitration; earlier levels are cheaper and decisive.

    `oracle_path` (defaults to `wav_path`) is the audio L3 re-decodes. The
    engine passes the *pre-denoise* original: denoise can induce decode loops,
    and re-hearing the same denoised audio only reproduces the artifact.
    Re-hearing the original is the truthful test — real stutter is in the
    acoustics whether or not it was denoised.
    """
    if not run.verified:
        return Verdict(run.segment_id, run.start, run.end, run.original,
                       "uncertain",
                       {"reason": "no word timestamps — text fallback only"},
                       run=run)

    # L1 — timeline: a real stutter consumes real time per copy. In an ASR
    # loop the decoder freezes and re-emits the phrase on a near-zero clock.
    expected = run.k * median_dur * TIMELINE_FACTOR
    span = run.end - run.start
    copy_durs = [run.words[(i + 1) * run.n - 1].end - run.words[i * run.n].start
                 for i in range(run.k)]
    suspect = len(copy_durs) >= 3 and statistics.pstdev(copy_durs) < 0.02
    if span < TIMELINE_ASR_LOOP * expected and span < FROZEN_SPAN_MAX:
        return Verdict(run.segment_id, run.start, run.end, run.original,
                       "asr_loop",
                       {"level": 1, "span_sec": round(span, 3),
                        "expected_sec": round(expected, 3)}, run=run)
    # L2 — silencedetect: each real copy is voiced separately, so a genuine
    # k-fold stutter shows ~k voiced bursts. A high count is trusted as
    # real_speech; a LOW count is not trusted as asr_loop, because rapid real
    # stutter (or a noisy line) can merge every copy into one continuous burst
    # — the annotation on real phone-call speech proved this. Low counts
    # escalate to L3; only the model's re-decode may fold.
    voiced = _count_voiced_bursts(wav_path, run.start - SILENCE_PAD,
                                  run.end + SILENCE_PAD)
    if voiced is not None and voiced >= 0.8 * run.k:
        return Verdict(run.segment_id, run.start, run.end, run.original,
                       "real_speech",
                       {"level": 2, "voiced_bursts": voiced, "k": run.k,
                        "suspect": suspect}, run=run)
    if voiced is not None:
        suspect = True   # low burst count on live audio: treat as suspect
    # L3 — isolated re-decode (greedy + sampling seeds), no cross-window
    # context, judged against the pre-denoise original when the engine gives
    # it. Any seed that reproduces most of the run proves the n-gram is in
    # the audio (real_speech); a greedy single-copy collapse confirms asr_loop.
    # Zero reproductions on the pipeline audio stays uncertain (fidelity-first
    # — the model may simply not hear a real stutter in degraded audio); but
    # zero reproductions on ALL seeds over the PRE-DENOISE original means the
    # repetition is absent from the recording → asr_loop (foldable in
    # classroom, original kept in the annotation).
    counts = _redecode_copies(oracle_path or wav_path, run, transcriber,
                              segment_lang)
    oracle = "original" if oracle_path else "pipeline"
    ok = [c for c in counts if c is not None]
    if not ok:
        return Verdict(run.segment_id, run.start, run.end, run.original,
                       "uncertain",
                       {"level": 3, "reason": "re-decode failed",
                        "oracle": oracle, "seeds": list(counts),
                        "suspect": suspect}, run=run)
    primary = counts[0]
    if any(c >= RECODE_OK_RATIO * run.k for c in ok):
        verdict = "real_speech"
    elif primary == 1:
        verdict = "asr_loop"
    elif oracle == "original" and len(ok) == len(counts) and all(
            c == 0 for c in ok):
        # Every decoding strategy heard NOTHING of the n-gram in the
        # pre-denoise original — yet the pipeline (possibly denoised) decode
        # produced k copies. The repetition is not in the recording; it was
        # invented downstream (denoise/context). Strongest artifact evidence
        # available; classroom folds it (original kept in the annotation).
        verdict = "asr_loop"
    else:
        verdict = "uncertain"
    return Verdict(run.segment_id, run.start, run.end, run.original, verdict,
                   {"level": 3, "redecoded_copies": primary, "k": run.k,
                    "oracle": oracle, "seeds": list(counts),
                    "reason": ("stable reproduction on original audio"
                               if verdict == "real_speech" and oracle == "original"
                               else "zero copies on all seeds over the original — "
                                    "repetition absent from the recording"
                               if verdict == "asr_loop" and oracle == "original"
                               else "re-decode could not reproduce the n-gram"
                               if verdict == "uncertain" else None),
                    "suspect": suspect}, run=run)


def _extract_clip(wav_path, a: float, b: float) -> Optional[str]:
    """Cut [a, b] seconds out of the wav into a temp file; None on failure."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="recorder_arb_")
    os.close(fd)
    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
           "-ss", f"{max(0.0, a):.3f}", "-to", f"{b:.3f}", "-i", str(wav_path),
           "-ac", "1", tmp]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode == 0 and Path(tmp).stat().st_size > 44:
            return tmp
    except Exception:
        pass
    Path(tmp).unlink(missing_ok=True)
    return None


def _count_voiced_bursts(wav_path, a: float, b: float) -> Optional[int]:
    """Voiced-burst count in [a, b] via ffmpeg silencedetect (None if ffmpeg
    is unavailable). Silence touching the clip edges is pad, not separator."""
    clip = _extract_clip(wav_path, a, b)
    if clip is None:
        return None
    try:
        proc = subprocess.run(
            ["ffmpeg", "-hide_banner", "-nostats", "-i", clip,
             "-af", f"silencedetect=noise={SILENCE_NOISE_DB}dB:"
                     f"d={SILENCE_MIN_DUR}", "-f", "null", "-"],
            capture_output=True, text=True, check=False)
        starts, ends = [], []
        for line in proc.stderr.splitlines():
            m = re.search(r"silence_start: ([\d.]+)", line)
            if m:
                starts.append(float(m.group(1)))
            m = re.search(r"silence_end: ([\d.]+)", line)
            if m:
                ends.append(float(m.group(1)))
        dur = b - a
        inner = [(s, e) for s, e in zip(starts, ends)
                 if s > 0.02 and e < dur - 0.02]
        return len(inner) + 1
    except Exception:
        return None
    finally:
        Path(clip).unlink(missing_ok=True)


def _redecode_copies(wav_path, run: Run, transcriber, segment_lang: str,
                     seeds: tuple = L3_SEEDS) -> list[Optional[int]]:
    """Re-transcribe the run's window in isolation at several temperatures
    (no context) and count how many copies of the n-gram the model still hears
    per seed. A seed that returns nothing is None (inconclusive), never zero —
    an empty greedy pass on a shouted window is not evidence of a loop.
    """
    clip = _extract_clip(wav_path, run.start - SILENCE_PAD, run.end + SILENCE_PAD)
    if clip is None:
        return [None] * len(seeds)
    try:
        lang = segment_lang if segment_lang in ("en", "zh", "ja", "ko", "fr",
                                                "de", "es") else None
        counts: list[Optional[int]] = []
        for seed in seeds:
            result = transcriber.transcribe(
                Path(clip), language=lang, condition_on_previous=False,
                temperature=(seed,))
            toks = [w.word for s in result.segments for w in s.words]
            if not toks:
                toks = result.full_text.split()
            if not toks:
                counts.append(None)
                continue
            toks = [_norm(w) for w in toks]
            n, count, i = len(run.gram), 0, 0
            while i + n <= len(toks):
                if tuple(toks[i:i + n]) == run.gram:
                    count += 1
                    i += n
                else:
                    i += 1
            counts.append(count)
        return counts
    except Exception as exc:
        logger.warning("L3 re-decode failed: %s", exc)
        return [None] * len(seeds)
    finally:
        Path(clip).unlink(missing_ok=True)


# ----------------------------------------------------------------------
# Orchestration + application
# ----------------------------------------------------------------------

def arbitrate(asr: ASRResult, wav_path, transcriber,
              oracle_path=None) -> list[Verdict]:
    """Find every repeat run in the result and classify it.

    `oracle_path` is the audio L3 re-decodes (the engine passes the
    pre-denoise original, see ``classify_run``); defaults to ``wav_path``.
    """
    med = median_word_duration(asr.segments)
    verdicts = []
    for seg in asr.segments:
        for run in find_repeat_runs(seg):
            verdicts.append(classify_run(run, wav_path, transcriber,
                                         seg.language, med,
                                         oracle_path=oracle_path))
    return verdicts


def apply_verdicts(asr: ASRResult, verdicts: list[Verdict], mode_key: str) -> None:
    """Record every verdict as an annotation (all modes). Classroom also folds
    confirmed asr_loop runs — at word level, so the surviving text is the
    speaker's own tokens; the original is always kept in the annotation."""
    for v in verdicts:
        asr.annotations.append({
            "type": "repeat_arbitration",
            "segment_id": v.segment_id,
            "start": round(v.start, 3),
            "end": round(v.end, 3),
            "original": v.original,
            "verdict": v.verdict,
            "evidence": v.evidence,
        })
    if mode_key != "classroom":
        return
    drops: dict[int, list[Run]] = {}
    for v in verdicts:
        if v.verdict == "asr_loop" and v.run is not None and v.run.verified:
            drops.setdefault(v.segment_id, []).append(v.run)
    for seg in asr.segments:
        runs = drops.get(seg.id)
        if not runs or not seg.words:
            continue
        keep: set[int] = set()
        for run in sorted(runs, key=lambda r: r.word_start):
            keep.update(range(run.word_start, run.word_start + run.n))
        seg.words = [w for i, w in enumerate(seg.words) if i in keep
                     or not any(r.word_start <= i < r.word_end
                                for r in runs)]
        seg.text = "".join(w.word for w in seg.words)
    asr.full_text = " ".join(s.text for s in asr.segments).strip()


def adjacent_duplicate_segments(asr: ASRResult) -> list[dict]:
    """Consecutive segments whose normalized text is identical and ≤ 8 words —
    the segment-level tail of the old _suppress_repetition. Annotated (or, in
    classroom, dropped) by the caller; here we only report."""
    dups: list[dict] = []
    prev_norm, prev_id, prev_text = None, None, None
    for s in asr.segments:
        norm = re.sub(r"\s+", " ", s.text.strip().lower())
        if norm and norm == prev_norm and len(norm.split()) <= 8:
            dups.append({"segment_id": s.id, "start": round(s.start, 3),
                         "end": round(s.end, 3), "text": s.text,
                         "duplicate_of": prev_id})
        if norm:
            prev_norm, prev_id, prev_text = norm, s.id, s.text
    return dups
