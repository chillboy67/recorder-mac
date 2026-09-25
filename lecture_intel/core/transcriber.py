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

2. **MLX on Apple Silicon when available**, **faster-whisper on CPU everywhere
   else**. The automatic CPU profile uses a smaller model to keep first-run
   latency and memory practical; users can still select a larger model.

3. **Word-level timestamps + probabilities always on.** Confidence is what lets
   the IELTS mode flag *likely* mispronunciations, and the per-word timing is
   what the diarizer needs.

4. **Faithful output.** First pass at temperature 0 with beam search; only
   windows Whisper itself flags as unstable (low avg logprob or high compression
   ratio) are retried at temperature 0.4, which samples — those windows are not
   byte-for-byte reproducible. No paraphrasing anywhere.
"""
from __future__ import annotations

import logging
import os
import platform
import shutil
import time
from pathlib import Path
from typing import Callable, Optional

from modules import ASRResult, ASRSegment, ASRWord

from core.i18n import t
from core.model_cache import (faster_whisper_cache_dir, find_faster_whisper_model,
                              mlx_cache_dir, whisper_cpp_model_available,
                              whisper_cpp_model_path)
from core.languages import (detect_language, disambiguate_latin_language,
                            english_function_words_dominate, _LATIN_LANGS)

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
TEMPERATURE_LADDER = (0.0, 0.4)


class Transcriber:
    """Mode-aware Whisper transcription with MLX → faster-whisper fallback."""

    def __init__(
        self,
        model: str = "auto",
        engine: str = "auto",        # auto | mlx-whisper | faster-whisper
        beam_size: int = 5,
        compute_type: str = "int8",  # faster-whisper CPU precision
    ):
        self.requested_model = model
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
        condition_on_previous: bool = False,
        chunked: bool = False,
        chunk_sec: float = 90.0,
        duration_sec: float = 0.0,
        temperature: tuple = TEMPERATURE_LADDER,
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
        self.model = self._resolved_model(engine)
        logger.info("Transcribing with %s (model=%s, chunked=%s)", engine, self.model, chunked)
        if progress:
            progress(0.0, t("tr_load_model", model=self.model, engine=engine))

        warnings: list[str] = []
        if engine == "faster-whisper" and self.requested_engine != "faster-whisper":
            if self._mlx_supported_host():
                warnings.append(t("tr_warn_mlx_unavailable"))
            elif self.requested_engine == "mlx-whisper":
                warnings.append(t("tr_warn_mlx_unsupported"))
        t0 = time.time()
        if engine == "mlx-whisper" and chunked:
            # Chunking stays on even for a pinned language: it is what bounds
            # memory on long recordings. Only per-chunk *detection* needs the
            # language to be unset.
            try:
                raw_segments, detected_lang = self._transcribe_mlx_chunked(
                    audio_path, initial_prompt, progress, chunk_sec, language,
                    temperature,
                )
            except Exception as chunk_exc:
                logger.warning("chunked mlx failed (%s); retrying single-pass", chunk_exc)
                try:
                    raw_segments, detected_lang = self._transcribe_mlx(
                        audio_path, language, initial_prompt, condition_on_previous,
                        progress, temperature,
                    )
                except Exception as mlx_exc:
                    raw_segments, detected_lang = self._fallback_to_faster(
                        audio_path, language, initial_prompt, condition_on_previous,
                        progress, temperature, warnings,
                        RuntimeError(f"chunked MLX failed: {chunk_exc}; single-pass MLX failed: {mlx_exc}"),
                    )
        elif engine == "mlx-whisper":
            try:
                raw_segments, detected_lang = self._transcribe_mlx(
                    audio_path, language, initial_prompt, condition_on_previous,
                    progress, temperature,
                )
            except Exception as exc:
                logger.warning("mlx-whisper failed (%s); falling back to faster-whisper", exc)
                raw_segments, detected_lang = self._fallback_to_faster(
                    audio_path, language, initial_prompt, condition_on_previous,
                    progress, temperature, warnings, exc, backend="MLX",
                )
        elif engine in ("whisper.cpp-vulkan", "whisper.cpp-openvino"):
            try:
                raw_segments, detected_lang = self._transcribe_whisper_cpp(
                    engine, audio_path, language, initial_prompt,
                    condition_on_previous,
                )
                if chunked and language is None:
                    warnings.append(t("tr_cpp_language_warning"))
            except Exception as exc:
                logger.warning("%s failed (%s); falling back to CPU", engine, exc)
                raw_segments, detected_lang = self._fallback_to_faster(
                    audio_path, language, initial_prompt, condition_on_previous,
                    progress, temperature, warnings, exc, backend=engine,
                )
        else:
            raw_segments, detected_lang = self._transcribe_faster(
                audio_path, language, initial_prompt, condition_on_previous,
                progress, temperature,
            )
        engine = self._engine or engine
        dt = time.time() - t0

        segments = self._to_segments(raw_segments, detected_lang)
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
        elif self.requested_engine in ("auto", "whisper.cpp-vulkan", "whisper.cpp-openvino"):
            requested_cpp = (self.requested_engine.removeprefix("whisper.cpp-")
                             if self.requested_engine.startswith("whisper.cpp-") else None)
            backends = (requested_cpp,) if requested_cpp else ("vulkan", "openvino")
            for backend in backends:
                if self._whisper_cpp_available(backend, configured_only=requested_cpp is None):
                    self._engine = f"whisper.cpp-{backend}"
                    break
            if self._engine:
                return self._engine
            if requested_cpp:
                raise RuntimeError(
                    f"whisper.cpp {requested_cpp} is not ready. Configure "
                    f"RECORDER_WHISPER_CPP_{requested_cpp.upper()} and download its "
                    f"model with 'python download_models.py --engine cpp-{requested_cpp} {self._resolved_model('faster-whisper')}'."
                )
            if self._faster_available():
                self._engine = "faster-whisper"
            else:
                raise RuntimeError("No configured whisper.cpp GPU backend or CPU ASR engine is available.")
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
    def _mlx_supported_host() -> bool:
        return (platform.system() == "Darwin"
                and platform.machine().lower() in ("arm64", "aarch64"))

    def _resolved_model(self, engine: str) -> str:
        if self.requested_model != "auto":
            return self.requested_model
        return "large-v3" if engine == "mlx-whisper" else "small"

    def _whisper_cpp_available(self, backend: str, *, configured_only: bool) -> bool:
        variable = f"RECORDER_WHISPER_CPP_{backend.upper()}"
        configured = os.environ.get(variable)
        if configured:
            binary = Path(configured).expanduser()
            if not binary.is_file():
                return False
        elif configured_only:
            return False
        else:
            found = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
            if not found:
                return False
            binary = Path(found)
        model = self._resolved_model(f"whisper.cpp-{backend}")
        return whisper_cpp_model_available(model, backend)

    def _whisper_cpp_binary(self, backend: str) -> str:
        variable = f"RECORDER_WHISPER_CPP_{backend.upper()}"
        configured = os.environ.get(variable)
        if configured:
            path = Path(configured).expanduser()
            if path.is_file():
                return str(path)
        found = shutil.which("whisper-cli") or shutil.which("whisper-cpp")
        if found:
            return found
        raise RuntimeError(f"whisper.cpp binary is not configured ({variable})")

    def _transcribe_whisper_cpp(self, engine, audio_path, language, initial_prompt,
                                condition_on_previous):
        from core.whisper_cpp import WhisperCppTranscriber

        backend = engine.removeprefix("whisper.cpp-")
        runner = WhisperCppTranscriber(
            self._whisper_cpp_binary(backend), backend,
            whisper_cpp_model_path(self.model), beam_size=self.beam_size,
            threads=max(1, min(8, os.cpu_count() or 4)),
        )
        segments = runner.transcribe(
            audio_path, language=language, prompt=initial_prompt,
            condition_on_previous=condition_on_previous,
        )
        return segments, runner.detected_language or language or "en"

    def _fallback_to_faster(
        self, audio_path, language, initial_prompt, condition_on_previous,
        progress, temperature, warnings, cause, backend="MLX",
    ):
        if not self._faster_available():
            raise RuntimeError(
                f"{backend} transcription failed ({cause}) and faster-whisper is unavailable. "
                "Install the CPU dependencies or choose a supported backend."
            ) from cause
        if self.requested_model == "auto":
            self.model = self._resolved_model("faster-whisper")
        warnings.append(t("tr_warn_fallback", backend=backend, exc=cause))
        self._engine = "faster-whisper"
        if progress:
            progress(0.05, t("tr_backend_fallback", backend=backend, model=self.model))
        return self._transcribe_faster(
            audio_path, language, initial_prompt, condition_on_previous,
            progress, temperature,
        )

    @staticmethod
    def _mlx_available() -> bool:
        if not Transcriber._mlx_supported_host():
            return False
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
        self, audio_path, language, initial_prompt, condition_on_previous,
        progress, temperature=TEMPERATURE_LADDER,
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
            temperature=temperature,
            no_speech_threshold=_NO_SPEECH_THRESHOLD,
            logprob_threshold=_LOGPROB_THRESHOLD,
            compression_ratio_threshold=_COMPRESSION_RATIO_THRESHOLD,
        )
        segs = result.get("segments", [])
        lang = result.get("language", language or "en")
        return segs, lang

    def _transcribe_mlx_chunked(self, audio_path, initial_prompt, progress,
                                chunk_sec=90.0, language: Optional[str] = None,
                                temperature=TEMPERATURE_LADDER):
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
                condition_on_previous_text=False, temperature=temperature,
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
            cached_model = find_faster_whisper_model(self.model)
            model_ref = str(cached_model) if cached_model else self.model
            threads = max(1, min(8, os.cpu_count() or 4))
            logger.info("Loading faster-whisper %s (%s, CPU, %d threads)…",
                        self.model, self.compute_type, threads)
            try:
                self._faster_model = WhisperModel(
                    model_ref, device="cpu", compute_type=self.compute_type,
                    cpu_threads=threads, num_workers=1,
                    download_root=str(faster_whisper_cache_dir()),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load the CPU Whisper model '{self.model}'. "
                    "Check the network connection or pre-download it with "
                    f"'python download_models.py --engine cpu {self.model}'. "
                    f"Original error: {exc}"
                ) from exc
        return self._faster_model

    def _transcribe_faster(
        self, audio_path, language, initial_prompt, condition_on_previous,
        progress, temperature=TEMPERATURE_LADDER,
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
            temperature=temperature,
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
            elif lang != "en" and english_function_words_dominate(text):
                # Accented English whose detection landed on a language outside
                # the function-word table (it/pt/nl/…): the IELTS guarantee that
                # an English answer stays English still applies.
                lang = "en"
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


def _local_model_dir(model: str):
    """Return a local mlx model directory for `model` if present, else None.

    Looks in the install dir and the project tree under models/whisper-<name>-mlx
    (populated by download_models.py)."""
    from pathlib import Path as _P
    name = f"whisper-{model}-mlx"
    candidates = [
        _P.home() / "Library" / "Application Support" / "Recorder" / "models" / name,
        mlx_cache_dir() / name,
        _P(__file__).resolve().parent.parent / "models" / name,
    ]
    for d in candidates:
        if (d / "config.json").exists() and any(d.glob("weights.*")):
            return d
    return None
