"""
Transcriber — the accuracy core.

Design decisions that matter for accuracy (these fix the old pipeline's
"效果不好"):

1. **Feed the WHOLE file to Whisper.** The old pipeline pre-split audio into
   independent 30s chunks with Silero VAD and transcribed each in isolation.
   That throws away cross-window context and causes dropped/duplicated words at
   the seams. Whisper already does internal 30s windowing *with* context
   carry-over (``condition_on_previous_text``) and its own VAD/no-speech
   handling — so we just hand it the entire file.

2. **mlx-whisper large-v3 first** (Apple-Silicon Metal acceleration → roughly
   real-time on an M-series), **faster-whisper as a CPU fallback** so the app
   still works if MLX is unavailable.

3. **Word-level timestamps + probabilities always on.** Confidence is what lets
   the IELTS mode flag *likely* mispronunciations, and the per-word timing is
   what the diarizer needs.

4. **Faithful output.** temperature=0 (greedy, deterministic) with Whisper's
   standard fallback ladder; no paraphrasing anywhere.
"""
from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Callable, Optional

from modules import ASRResult, ASRSegment, ASRWord

from core.i18n import t
from core.languages import detect_language, disambiguate_latin_language, _LATIN_LANGS

logger = logging.getLogger(__name__)

# mlx model repos keyed by the whisper model name
_MLX_REPOS = {
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v2": "mlx-community/whisper-large-v2-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
}

# Common decode options shared by both engines.
_NO_SPEECH_THRESHOLD = 0.6
_LOGPROB_THRESHOLD = -1.0
_COMPRESSION_RATIO_THRESHOLD = 2.4
# Whisper's standard temperature fallback ladder: only escalates above 0 when a
# window looks broken (low logprob / high compression). Keeps clean audio
# greedy/deterministic while still recovering from hard spots.
# Short ladder: a hard 30s window is retried at most twice instead of 6×. The
# full 6-step ladder triples-to-sextuples the compute on noisy lectures (lots of
# heat) for little gain, and runaway loops are cleaned afterwards anyway.
_TEMPERATURE_LADDER = (0.0, 0.4)


class Transcriber:
    """Mode-aware Whisper transcription with MLX → faster-whisper fallback."""

    def __init__(
        self,
        model: str = "large-v3",
        engine: str = "auto",        # auto | mlx-whisper | faster-whisper
        beam_size: int = 5,
        compute_type: str = "int8",  # faster-whisper CPU precision
    ):
        self.model = model
        self.requested_engine = engine
        self.beam_size = beam_size
        self.compute_type = compute_type
        self._engine: Optional[str] = None
        self._faster_model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(
        self,
        audio_path: Path,
        *,
        language: Optional[str] = None,
        initial_prompt: str = "",
        condition_on_previous: bool = True,
        chunked: bool = False,
        chunk_sec: float = 90.0,
        duration_sec: float = 0.0,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> ASRResult:
        """Transcribe a (normalized 16k mono wav) file.

        chunked=True (every mode uses it): split at silences and transcribe
        chunk by chunk, each with its own language — detected when `language`
        is None (keeps zh/en code-switching), or the caller's pinned language
        otherwise. Either way chunking bounds memory on long recordings and
        stays on the GPU (mlx's single pass locks to one global language).
        """
        engine = self._resolve_engine()
        logger.info("Transcribing with %s (model=%s, chunked=%s)", engine, self.model, chunked)
        if progress:
            progress(0.0, t("tr_load_model", model=self.model, engine=engine))

        warnings: list[str] = []
        t0 = time.time()
        if engine == "mlx-whisper" and chunked:
            # Chunking stays on even for a pinned language: it is what bounds
            # memory on long recordings. Only per-chunk *detection* needs the
            # language to be unset.
            try:
                raw_segments, detected_lang = self._transcribe_mlx_chunked(
                    audio_path, initial_prompt, progress, chunk_sec, language
                )
            except Exception as exc:
                logger.warning("chunked mlx failed (%s); falling back to single-pass", exc)
                raw_segments, detected_lang = self._transcribe_mlx(
                    audio_path, language, initial_prompt, condition_on_previous, progress
                )
        elif engine == "mlx-whisper":
            try:
                raw_segments, detected_lang = self._transcribe_mlx(
                    audio_path, language, initial_prompt, condition_on_previous, progress
                )
            except Exception as exc:
                # mlx failed mid-run (e.g. model download interrupted) — don't
                # leave the user stranded; fall back to the CPU engine.
                logger.warning("mlx-whisper failed (%s); falling back to faster-whisper", exc)
                if not self._faster_available():
                    raise
                warnings.append(t("tr_warn_fallback", exc=exc))
                self._engine = engine = "faster-whisper"
                if progress:
                    progress(0.05, t("tr_mlx_fallback"))
                raw_segments, detected_lang = self._transcribe_faster(
                    audio_path, language, initial_prompt, condition_on_previous, progress
                )
        else:
            raw_segments, detected_lang = self._transcribe_faster(
                audio_path, language, initial_prompt, condition_on_previous, progress
            )
        dt = time.time() - t0

        segments = self._to_segments(raw_segments, detected_lang)
        segments = _suppress_repetition(segments)
        full_text = " ".join(s.text for s in segments).strip()
        language_label = detect_language(full_text, detected_lang)

        logger.info(
            "ASR done: engine=%s lang=%s segs=%d chars=%d in %.1fs",
            engine, language_label, len(segments), len(full_text), dt,
        )
        if progress:
            progress(1.0, t("tr_done"))

        return ASRResult(
            segments=segments,
            full_text=full_text,
            language=language_label,
            model_used=f"{engine}:{self.model}",
            audio_duration_sec=duration_sec,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Engine resolution
    # ------------------------------------------------------------------

    def _resolve_engine(self) -> str:
        if self._engine:
            return self._engine
        if self.requested_engine in ("auto", "mlx-whisper") and self._mlx_available():
            self._engine = "mlx-whisper"
        elif self._faster_available():
            self._engine = "faster-whisper"
            if self.requested_engine == "mlx-whisper":
                logger.warning("mlx-whisper unavailable; using faster-whisper")
        else:
            raise RuntimeError(
                "No ASR engine available. Install one of:\n"
                "  pip install mlx-whisper     (Apple Silicon, recommended)\n"
                "  pip install faster-whisper  (CPU, cross-platform)"
            )
        return self._engine

    @staticmethod
    def _mlx_available() -> bool:
        try:
            import mlx_whisper  # noqa: F401
            return True
        except Exception:
            return False

    @staticmethod
    def _faster_available() -> bool:
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # MLX
    # ------------------------------------------------------------------

    def _mlx_repo(self) -> str:
        # Prefer a locally downloaded model (via download_models.py). This makes
        # transcription fully offline and sidesteps networks where HuggingFace
        # is throttled. Falls back to the HF repo id if no local copy exists.
        local = _local_model_dir(self.model)
        if local is not None:
            logger.info("Using local model: %s", local)
            return str(local)
        return _MLX_REPOS.get(self.model, f"mlx-community/whisper-{self.model}-mlx")

    def _transcribe_mlx(
        self, audio_path, language, initial_prompt, condition_on_previous, progress,
    ):
        import mlx_whisper

        if progress:
            progress(0.05, t("tr_mlx"))
        result = mlx_whisper.transcribe(
            str(audio_path),
            path_or_hf_repo=self._mlx_repo(),
            language=language,
            initial_prompt=initial_prompt or None,
            word_timestamps=True,
            condition_on_previous_text=condition_on_previous,
            temperature=_TEMPERATURE_LADDER,
            no_speech_threshold=_NO_SPEECH_THRESHOLD,
            logprob_threshold=_LOGPROB_THRESHOLD,
            compression_ratio_threshold=_COMPRESSION_RATIO_THRESHOLD,
        )
        segs = result.get("segments", [])
        lang = result.get("language", language or "en")
        return segs, lang

    def _transcribe_mlx_chunked(self, audio_path, initial_prompt, progress,
                                chunk_sec=90.0, language: Optional[str] = None):
        """GPU code-switching: split at silences, transcribe chunk by chunk.

        mlx runs on the GPU but does ONE global language pass, so it translates
        the minority language away on mixed zh/en audio. Splitting at natural
        pauses and transcribing each chunk with its own language detection keeps
        both languages — and stays on the GPU (fast). Chunk boundaries are at
        silences, so accuracy at the seams is preserved.

        When the caller pins `language`, every chunk is transcribed in that
        language instead (no per-chunk detection); chunking then serves memory
        bounding rather than code-switching.
        """
        import mlx_whisper
        import numpy as np
        import soundfile as sf

        wav, sr = sf.read(str(audio_path))
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav = wav.astype(np.float32)
        if sr != 16000:
            import librosa
            wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            sr = 16000

        chunks = self._silence_chunks(wav, sr, max_sec=chunk_sec)
        repo = self._mlx_repo()
        logger.info("Chunked mlx: %d chunk(s)", len(chunks))

        all_segs: list[dict] = []
        langs: list[str] = []
        for i, (a, b) in enumerate(chunks):
            clip = wav[a:b]
            if len(clip) < int(0.2 * sr):
                continue
            r = mlx_whisper.transcribe(
                clip, path_or_hf_repo=repo, language=language,
                initial_prompt=initial_prompt or None, word_timestamps=True,
                condition_on_previous_text=False, temperature=_TEMPERATURE_LADDER,
                no_speech_threshold=_NO_SPEECH_THRESHOLD,
                logprob_threshold=_LOGPROB_THRESHOLD,
                compression_ratio_threshold=_COMPRESSION_RATIO_THRESHOLD,
            )
            offset = a / sr
            # The chunk's own detected language, carried onto each of its
            # segments. It is the only signal that can separate two languages
            # sharing a script (fr vs en), but note the limit found on real
            # audio: one chunk covers ~60s of conversation, so a chunk that
            # mixes languages yields a single label for both. Script-based
            # languages (zh/ja/ko/ru/th…) are unaffected — see _to_segments.
            chunk_lang = r.get("language")
            for seg in r.get("segments", []):
                seg = dict(seg)
                seg["start"] = seg.get("start", 0.0) + offset
                seg["end"] = seg.get("end", 0.0) + offset
                seg["words"] = [
                    {**w, "start": w.get("start", 0.0) + offset,
                     "end": w.get("end", 0.0) + offset}
                    for w in seg.get("words", [])
                ]
                seg["chunk_language"] = chunk_lang
                all_segs.append(seg)
            if chunk_lang:
                langs.append(chunk_lang)
            if progress:
                progress(0.05 + 0.9 * (i + 1) / len(chunks), t("tr_gpu_chunk"))

        # overall language label = most common per-chunk detection
        lang = max(set(langs), key=langs.count) if langs else "en"
        return all_segs, lang

    @staticmethod
    def _silence_chunks(wav, sr, max_sec: float = 28.0, top_db: int = 30):
        """Return [(start_sample, end_sample)] chunks split on silence.

        Speech regions are merged (absorbing short gaps) up to max_sec, breaking
        only at silences so no word is cut. Falls back to one chunk if VAD finds
        nothing."""
        import librosa
        import numpy as np

        try:
            intervals = librosa.effects.split(wav, top_db=top_db,
                                               frame_length=2048, hop_length=512)
        except Exception:
            intervals = np.array([[0, len(wav)]])
        if len(intervals) == 0:
            return [(0, len(wav))]

        max_len = int(max_sec * sr)
        chunks: list[tuple[int, int]] = []
        cur_a, cur_b = int(intervals[0][0]), int(intervals[0][1])
        for s, e in intervals[1:]:
            s, e = int(s), int(e)
            if e - cur_a <= max_len:
                cur_b = e               # extend (absorbs the silent gap)
            else:
                chunks.append((cur_a, cur_b))
                cur_a, cur_b = s, e
        chunks.append((cur_a, cur_b))
        # small padding so onsets/codas aren't clipped
        pad = int(0.15 * sr)
        return [(max(0, a - pad), min(len(wav), b + pad)) for a, b in chunks]

    # ------------------------------------------------------------------
    # faster-whisper (CPU fallback)
    # ------------------------------------------------------------------

    def _load_faster(self):
        if self._faster_model is None:
            from faster_whisper import WhisperModel
            logger.info("Loading faster-whisper %s (%s, CPU)…", self.model, self.compute_type)
            self._faster_model = WhisperModel(
                self.model, device="cpu", compute_type=self.compute_type,
                cpu_threads=max(4, 0), num_workers=1,
            )
        return self._faster_model

    def _transcribe_faster(
        self, audio_path, language, initial_prompt, condition_on_previous, progress,
    ):
        model = self._load_faster()
        if progress:
            progress(0.05, t("tr_cpu"))
        seg_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=self.beam_size,
            word_timestamps=True,
            initial_prompt=initial_prompt or None,
            condition_on_previous_text=condition_on_previous,
            temperature=_TEMPERATURE_LADDER,
            no_speech_threshold=_NO_SPEECH_THRESHOLD,
            log_prob_threshold=_LOGPROB_THRESHOLD,
            compression_ratio_threshold=_COMPRESSION_RATIO_THRESHOLD,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        total = info.duration or 0.0
        segs = []
        for s in seg_iter:
            segs.append({
                "start": s.start, "end": s.end, "text": s.text,
                "avg_logprob": s.avg_logprob,
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end,
                     "probability": w.probability}
                    for w in (s.words or [])
                ],
            })
            if progress and total:
                progress(min(0.95, 0.05 + 0.9 * (s.end / total)), t("tr_cpu"))
        return segs, info.language

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _to_segments(raw_segments: list[dict],
                     detected: Optional[str] = None) -> list[ASRSegment]:
        """`detected` is Whisper's file-level language code. Per-segment labels
        are derived from script, but Latin script cannot tell en/fr/de/es apart,
        so the labeller needs a language to break the tie — the segment's own
        chunk language when the chunked path supplied one, otherwise the
        file-level `detected`."""
        segments: list[ASRSegment] = []
        for i, s in enumerate(raw_segments):
            words = [
                ASRWord(
                    word=(w.get("word") or "").strip(),
                    start=float(w.get("start", 0.0) or 0.0),
                    end=float(w.get("end", 0.0) or 0.0),
                    confidence=float(w.get("probability", 0.0) or 0.0),
                )
                for w in s.get("words", [])
            ]
            text = (s.get("text") or "").strip()
            lang = detect_language(text, s.get("chunk_language") or detected)
            # Function-word disambiguation for Latin-script languages: when
            # the chunk language and the detected language are both Latin
            # (en/fr/de/es), the per-segment function words can correct a
            # chunk-level mislabel — e.g. an English segment inside a
            # French-labelled chunk.
            chunk_lang = s.get("chunk_language")
            if chunk_lang in _LATIN_LANGS and lang in _LATIN_LANGS:
                lang = disambiguate_latin_language(text, chunk_lang)
            segments.append(ASRSegment(
                id=i,
                start=float(s.get("start", 0.0) or 0.0),
                end=float(s.get("end", 0.0) or 0.0),
                text=text,
                language=lang,
                confidence=float(s.get("avg_logprob", 0.0) or 0.0),
                words=words,
            ))
        return segments


def _suppress_repetition(segments):
    """Collapse Whisper hallucination loops (e.g. "about this about this …" ×30).

    These are ASR artifacts, not speech, so removing them isn't "changing the
    speaker's words" — the speaker never said the phrase 30 times. We collapse a
    phrase repeated ≥3× in a row to one copy, then drop consecutive segments that
    became identical (a loop often spans several segments)."""
    cleaned = []
    prev_norm = None
    for s in segments:
        s.text = _collapse_repeats(s.text)
        norm = re.sub(r"\s+", " ", s.text.strip().lower())
        if norm and norm == prev_norm and len(norm.split()) <= 8:
            continue   # consecutive duplicate from a loop
        cleaned.append(s)
        if norm:
            prev_norm = norm
    # renumber ids so downstream stays consistent
    for i, s in enumerate(cleaned):
        s.id = i
    return cleaned


def _collapse_repeats(text: str) -> str:
    """Collapse runaway ASR repetition loops down to a single copy.

    Two passes, because Chinese has no spaces:
      1. character/substring level — a 1–8 char unit repeated ≥4× (catches
         "時時時時…" and "啊啊啊", which word-splitting misses);
      2. word level — for space-separated languages ("no no no no").
    ≥4 repeats so genuine emphasis ("no, no, no") is preserved."""
    if not text:
        return text
    # 1) substring-level (handles no-space scripts). Non-greedy 1–8 char unit
    #    repeated 4+ times → keep one copy.
    text = re.sub(r"(.{1,8}?)\1{3,}", r"\1", text)

    # 2) word-level
    words = text.split()
    if len(words) < 6:
        return text
    for n in (1, 2, 3, 4):
        out, i = [], 0
        while i < len(words):
            gram = words[i:i + n]
            if len(gram) < n:
                out.extend(words[i:])
                break
            reps, j = 1, i + n
            while words[j:j + n] == gram:
                reps += 1
                j += n
            if reps >= 4:
                out.extend(gram)   # keep a single copy
                i = j
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)


def _local_model_dir(model: str):
    """Return a local mlx model directory for `model` if present, else None.

    Looks in the install dir and the project tree under models/whisper-<name>-mlx
    (populated by download_models.py)."""
    from pathlib import Path as _P
    name = f"whisper-{model}-mlx"
    candidates = [
        _P.home() / "Library" / "Application Support" / "Recorder" / "models" / name,
        _P(__file__).resolve().parent.parent / "models" / name,
    ]
    for d in candidates:
        if (d / "config.json").exists() and any(d.glob("weights.*")):
            return d
    return None
