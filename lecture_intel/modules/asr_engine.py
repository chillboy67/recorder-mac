"""
Module 5 — ASREngine

Automatic Speech Recognition using Whisper.
Primary: mlx-whisper (Apple Silicon MLX acceleration)
Fallback: faster-whisper (CTranslate2, CPU)
Future: FunASR Paraformer-zh for pure Chinese segments.

Output: ASRResult with segmented transcriptions, word timestamps,
language detection, and confidence scores.

Usage:
    asr = ASREngine(config)
    result = asr.process(segments)
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Optional

from modules import ASRResult, ASRSegment, ASRWord, Segment

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

DEFAULT_MODEL = "large-v3"
DEFAULT_BEAM_SIZE = 5
DEFAULT_INITIAL_PROMPT = (
    "This is a university lecture transcript. Technical terms may include: "
    "machine learning, neural network, attention mechanism, seq2seq, transformer, "
    "gradient descent, backpropagation, R language, Python, NumPy, PyTorch, "
    "p-value, hypothesis testing, linear regression, Bayesian, MCMC."
)

# Chinese Unicode range
CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")


class ASREngine:
    """Speech recognition with automatic engine selection and fallback."""

    def __init__(self, config: dict):
        """
        Args:
            config: The 'asr' section from config.yaml.
        """
        self.preferred_engine = config.get("primary_engine", "mlx-whisper")
        self.model_name = config.get("model", DEFAULT_MODEL)
        self.beam_size = config.get("beam_size", DEFAULT_BEAM_SIZE)
        self.word_timestamps = config.get("word_timestamps", True)
        self.condition_on_prev = config.get("condition_on_previous", True)
        self.compute_type = config.get("compute_type", "int8")
        self.initial_prompt = config.get("initial_prompt", DEFAULT_INITIAL_PROMPT)
        self.chinese_threshold = config.get("chinese_threshold", 0.8)
        self.paraformer_enabled = config.get("paraformer_enabled", False)

        self._engine: Optional[str] = None
        self._mlx_model = None
        self._faster_model = None
        self._warnings: list[str] = []

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------

    def process(self, segments: list[Segment]) -> ASRResult:
        """
        Transcribe all segments and return a unified ASRResult.

        If no segments provided, returns empty result.
        """
        if not segments:
            return ASRResult(
                segments=[],
                full_text="",
                language="unknown",
                model_used="none",
                warnings=["No segments to transcribe"],
            )

        self._warnings = []
        engine = self._resolve_engine()
        logger.info("ASR engine: %s", engine)

        all_segments: list[ASRSegment] = []
        total_segments = len(segments)

        for i, seg in enumerate(segments):
            logger.info("Transcribing segment %d/%d (%.1fs - %.1fs)...",
                        i + 1, total_segments, seg.start, seg.end)

            try:
                if engine == "mlx-whisper":
                    raw_segments = self._transcribe_mlx(seg.audio_path)
                else:
                    raw_segments = self._transcribe_faster(seg.audio_path)
            except Exception as exc:
                logger.error("Transcription failed for segment %d: %s", i, exc)
                if engine == "mlx-whisper":
                    logger.info("Falling back to faster-whisper for this segment")
                    try:
                        raw_segments = self._transcribe_faster(seg.audio_path)
                        self._warnings.append(
                            f"Segment {i}: mlx-whisper failed, used faster-whisper"
                        )
                    except Exception as exc2:
                        logger.error("Fallback also failed: %s", exc2)
                        self._warnings.append(
                            f"Segment {i}: both engines failed ({exc})"
                        )
                        continue
                else:
                    self._warnings.append(f"Segment {i}: transcription failed ({exc})")
                    continue

            # Normalize segment IDs
            for j, raw in enumerate(raw_segments):
                seg_id = len(all_segments)
                all_segments.append(ASRSegment(
                    id=seg_id,
                    start=seg.start + raw.get("start", 0),
                    end=seg.start + raw.get("end", 0),
                    text=raw.get("text", "").strip(),
                    language=raw.get("language", "en"),
                    confidence=raw.get("confidence", 0.0),
                    words=[
                        ASRWord(
                            word=w.get("word", ""),
                            start=seg.start + w.get("start", 0),
                            end=seg.start + w.get("end", 0),
                            confidence=w.get("confidence", w.get("prob", 0.0)),
                        )
                        for w in raw.get("words", [])
                    ],
                ))

        full_text = " ".join(s.text for s in all_segments)
        language, chinese_ratio = self._detect_language_mix(full_text)

        logger.info("Language detection: %s (Chinese ratio: %.2f)", language, chinese_ratio)
        logger.info("Total segments: %d, total text length: %d chars",
                    len(all_segments), len(full_text))

        if self.paraformer_enabled and chinese_ratio > self.chinese_threshold:
            self._warnings.append(
                "Chinese-dominant audio detected. Consider using FunASR Paraformer "
                "for better Chinese recognition (not yet implemented in MVP)."
            )

        return ASRResult(
            segments=all_segments,
            full_text=full_text,
            language=language,
            model_used=engine,
            warnings=self._warnings,
        )

    # -----------------------------------------------------------
    # Engine resolution
    # -----------------------------------------------------------

    def _resolve_engine(self) -> str:
        """Determine which ASR engine to use."""
        if self._engine:
            return self._engine

        if self.preferred_engine == "mlx-whisper":
            if self._mlx_available():
                self._engine = "mlx-whisper"
                return self._engine
            logger.warning("mlx-whisper not available, falling back to faster-whisper")

        if self._faster_available():
            self._engine = "faster-whisper"
            return self._engine

        raise RuntimeError(
            "No ASR engine available. Install one of:\n"
            "  pip install mlx-whisper    (Apple Silicon, recommended)\n"
            "  pip install faster-whisper (CTranslate2, cross-platform)"
        )

    def _mlx_available(self) -> bool:
        """Check if mlx-whisper can be imported."""
        try:
            import mlx_whisper  # noqa: F401
            return True
        except ImportError:
            return False

    def _faster_available(self) -> bool:
        """Check if faster-whisper can be imported."""
        try:
            from faster_whisper import WhisperModel  # noqa: F401
            return True
        except ImportError:
            return False

    # -----------------------------------------------------------
    # MLX Whisper transcription
    # -----------------------------------------------------------

    def _load_mlx_model(self):
        """Lazy-load MLX Whisper model."""
        if self._mlx_model is not None:
            return self._mlx_model

        import mlx_whisper

        logger.info("Loading mlx-whisper model: %s ...", self.model_name)
        t0 = time.time()
        try:
            # mlx_whisper auto-caches the model
            self._mlx_model = mlx_whisper
            dt = time.time() - t0
            logger.info("mlx-whisper model loaded in %.1fs", dt)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load mlx-whisper model '{self.model_name}': {exc}"
            ) from exc

        return self._mlx_model

    def _transcribe_mlx(self, audio_path: Path) -> list[dict]:
        """
        Transcribe a single audio file with mlx-whisper.
        Returns list of segment dicts with keys: start, end, text, language,
        confidence, words.
        """
        mlx = self._load_mlx_model()

        t0 = time.time()
        try:
            result = mlx.transcribe(
                str(audio_path),
                path_or_hf_repo=f"mlx-community/whisper-{self.model_name}-mlx",
                language=None,                    # auto-detect
                initial_prompt=self.initial_prompt,
                word_timestamps=self.word_timestamps,
                beam_size=self.beam_size,
                condition_on_previous_text=self.condition_on_prev,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6,
            )
        except Exception:
            # Try without the mlx-community prefix
            logger.info("Retrying with direct model name...")
            result = mlx.transcribe(
                str(audio_path),
                path_or_hf_repo=self.model_name,
                language=None,
                initial_prompt=self.initial_prompt,
                word_timestamps=self.word_timestamps,
                beam_size=self.beam_size,
                condition_on_previous_text=self.condition_on_prev,
                temperature=0.0,
            )

        dt = time.time() - t0
        segments = result.get("segments", [])
        logger.debug("mlx-whisper: %d segments in %.1fs", len(segments), dt)

        normalized = []
        for seg in segments:
            words = seg.get("words", [])
            lang = self._detect_segment_language(seg.get("text", ""))
            normalized.append({
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", "").strip(),
                "language": lang,
                "confidence": seg.get("avg_logprob", 0.0),
                "words": [
                    {
                        "word": w.get("word", ""),
                        "start": w.get("start", 0.0),
                        "end": w.get("end", 0.0),
                        "confidence": w.get("probability", 0.0),
                    }
                    for w in words
                ],
            })
        return normalized

    # -----------------------------------------------------------
    # Faster-Whisper transcription (fallback)
    # -----------------------------------------------------------

    def _load_faster_model(self):
        """Lazy-load faster-whisper model."""
        if self._faster_model is not None:
            return self._faster_model

        from faster_whisper import WhisperModel

        logger.info("Loading faster-whisper model: %s (int8, CPU)...", self.model_name)
        t0 = time.time()
        try:
            self._faster_model = WhisperModel(
                self.model_name,
                device="cpu",
                compute_type=self.compute_type,
                cpu_threads=4,
                num_workers=1,
            )
            dt = time.time() - t0
            logger.info("faster-whisper model loaded in %.1fs", dt)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load faster-whisper model '{self.model_name}': {exc}"
            ) from exc

        return self._faster_model

    def _transcribe_faster(self, audio_path: Path) -> list[dict]:
        """
        Transcribe a single audio file with faster-whisper.
        Returns list of segment dicts in the same format as _transcribe_mlx.
        """
        model = self._load_faster_model()

        t0 = time.time()
        seg_iter, info = model.transcribe(
            str(audio_path),
            language=None,
            beam_size=self.beam_size,
            word_timestamps=self.word_timestamps,
            initial_prompt=self.initial_prompt,
            condition_on_previous_text=self.condition_on_prev,
            temperature=0.0,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )

        detected_lang = info.language
        dt = time.time() - t0

        normalized = []
        for seg in seg_iter:
            words_data = []
            if seg.words:
                for w in seg.words:
                    words_data.append({
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end,
                        "confidence": w.probability,
                    })

            lang = self._detect_segment_language(seg.text)
            normalized.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "language": lang or detected_lang or "en",
                "confidence": seg.avg_logprob,
                "words": words_data,
            })

        logger.debug("faster-whisper: %d segments in %.1fs", len(normalized), dt)
        return normalized

    # -----------------------------------------------------------
    # Language detection
    # -----------------------------------------------------------

    def _detect_language_mix(self, text: str) -> tuple[str, float]:
        """
        Detect the language mix of a text.
        Returns (language_label, chinese_character_ratio).

        Labels: "zh" (>80% Chinese), "en" (<20% Chinese), "mixed" (20-80%)
        """
        if not text or not text.strip():
            return "en", 0.0

        # Remove whitespace for counting
        stripped = re.sub(r"\s+", "", text)
        if not stripped:
            return "en", 0.0

        cjk_chars = len(CJK_RE.findall(stripped))
        # Count ASCII letters as non-CJK
        ascii_letters = len(re.findall(r"[a-zA-Z]", stripped))
        total_chars = len(stripped)

        # Chinese + ASCII ratio
        meaningful = max(cjk_chars + ascii_letters, 1)
        cjk_ratio = cjk_chars / meaningful if meaningful > 0 else 0.0

        if cjk_ratio > 0.8:
            return "zh", cjk_ratio
        elif cjk_ratio > 0.2:
            return "mixed", cjk_ratio
        else:
            return "en", cjk_ratio

    @staticmethod
    def _detect_segment_language(text: str) -> str:
        """Quick language label for a single segment."""
        if not text or not text.strip():
            return "en"
        cjk = len(CJK_RE.findall(text))
        ascii_letters = len(re.findall(r"[a-zA-Z]", text))
        meaningful = max(cjk + ascii_letters, 1)
        ratio = cjk / meaningful
        if ratio > 0.8:
            return "zh"
        elif ratio > 0.2:
            return "mixed"
        return "en"


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("ASREngine — Self Test")
    print("=" * 60)

    config = {
        "primary_engine": "mlx-whisper",
        "model": "large-v3",
        "beam_size": 5,
        "word_timestamps": True,
        "condition_on_previous": True,
        "compute_type": "int8",
        "initial_prompt": DEFAULT_INITIAL_PROMPT,
        "chinese_threshold": 0.8,
        "paraformer_enabled": False,
    }

    asr = ASREngine(config)

    # Check what's available
    print(f"mlx-whisper available:  {asr._mlx_available()}")
    print(f"faster-whisper available: {asr._faster_available()}")

    # Try to find audio segments to test with
    test_path = None
    if len(sys.argv) > 1:
        test_path = Path(sys.argv[1])
    else:
        # Look for any audio file in data/
        data_dir = Path(__file__).resolve().parents[2] / "data"
        if data_dir.exists():
            for subdir in sorted(data_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                for ext in (".wav", ".webm", ".m4a", ".mp3"):
                    candidates = list(subdir.glob(f"*{ext}"))
                    if candidates:
                        test_path = candidates[0]
                        break
                if test_path:
                    break

    if not test_path or not test_path.exists():
        print("\nNo test audio found. Pass a path to an audio file.")
        print("Example: python asr_engine.py /path/to/audio.wav")
        sys.exit(0)

    print(f"\nTesting with: {test_path}")
    segments = [Segment(start=0.0, end=9999.0, audio_path=test_path)]

    try:
        result = asr.process(segments)
        print(f"\n✓ Engine:     {result.model_used}")
        print(f"✓ Language:   {result.language}")
        print(f"✓ Segments:   {len(result.segments)}")
        print(f"✓ Text length: {len(result.full_text)} chars")
        if result.warnings:
            print(f"⚠ Warnings:   {len(result.warnings)}")
        print(f"\nFirst 300 chars of transcript:")
        print(result.full_text[:300])
        if len(result.full_text) > 300:
            print("...")
    except Exception as exc:
        print(f"✗ Error: {exc}")
        import traceback
        traceback.print_exc()

    print("\n=== Test Complete ===")
