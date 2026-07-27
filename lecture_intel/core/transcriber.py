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
import time
from pathlib import Path
from typing import Callable, Optional

from modules import ASRResult, ASRSegment, ASRWord

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
_TEMPERATURE_LADDER = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


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
        duration_sec: float = 0.0,
        progress: Optional[Callable[[float, str], None]] = None,
    ) -> ASRResult:
        """Transcribe a (normalized 16k mono wav) file in one pass."""
        engine = self._resolve_engine()
        logger.info("Transcribing with %s (model=%s)", engine, self.model)
        if progress:
            progress(0.0, f"加载 {self.model} 模型 ({engine})…")

        warnings: list[str] = []
        t0 = time.time()
        if engine == "mlx-whisper":
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
                warnings.append(f"mlx-whisper 失败，已回退 faster-whisper：{exc}")
                self._engine = engine = "faster-whisper"
                if progress:
                    progress(0.05, "MLX 失败，改用 CPU 引擎…")
                raw_segments, detected_lang = self._transcribe_faster(
                    audio_path, language, initial_prompt, condition_on_previous, progress
                )
        else:
            raw_segments, detected_lang = self._transcribe_faster(
                audio_path, language, initial_prompt, condition_on_previous, progress
            )
        dt = time.time() - t0

        segments = self._to_segments(raw_segments)
        full_text = " ".join(s.text for s in segments).strip()
        language_label, _ = self._language_mix(full_text, detected_lang)

        logger.info(
            "ASR done: engine=%s lang=%s segs=%d chars=%d in %.1fs",
            engine, language_label, len(segments), len(full_text), dt,
        )
        if progress:
            progress(1.0, "转写完成")

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
            progress(0.05, "转写中（MLX 加速）…")
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
            progress(0.05, "转写中（CPU）…")
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
                progress(min(0.95, 0.05 + 0.9 * (s.end / total)), "转写中（CPU）…")
        return segs, info.language

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _to_segments(raw_segments: list[dict]) -> list[ASRSegment]:
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
            segments.append(ASRSegment(
                id=i,
                start=float(s.get("start", 0.0) or 0.0),
                end=float(s.get("end", 0.0) or 0.0),
                text=text,
                language=_segment_lang(text),
                confidence=float(s.get("avg_logprob", 0.0) or 0.0),
                words=words,
            ))
        return segments

    @staticmethod
    def _language_mix(text: str, detected: str) -> tuple[str, float]:
        import re
        stripped = re.sub(r"\s+", "", text or "")
        if not stripped:
            return detected or "en", 0.0
        cjk = len(re.findall(r"[一-鿿]", stripped))
        latin = len(re.findall(r"[a-zA-Z]", stripped))
        meaningful = max(cjk + latin, 1)
        ratio = cjk / meaningful
        if ratio > 0.8:
            return "zh", ratio
        if ratio > 0.15:
            return "mixed", ratio
        return "en", ratio


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


def _segment_lang(text: str) -> str:
    import re
    if not text:
        return "en"
    cjk = len(re.findall(r"[一-鿿]", text))
    latin = len(re.findall(r"[a-zA-Z]", text))
    meaningful = max(cjk + latin, 1)
    ratio = cjk / meaningful
    if ratio > 0.8:
        return "zh"
    if ratio > 0.15:
        return "mixed"
    return "en"
