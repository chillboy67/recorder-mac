"""
Module 2 — AudioEnhancer (P1-A)

Enhances audio quality before VAD/ASR using DeepFilterNet 3.
Includes SNR pre-detection to skip enhancement on already-clean audio,
and chunked processing for long recordings.

Runs between AudioLoader and VADProcessor in the pipeline.

Usage:
    enhancer = AudioEnhancer(config)
    result = enhancer.process(audio_path)
"""

from __future__ import annotations

import gc
import logging
import time
from pathlib import Path

import numpy as np
import torch

from modules import EnhancedAudioResult

logger = logging.getLogger(__name__)

# SNR threshold defaults
DEFAULT_SKIP_SNR_DB = 30.0
DEFAULT_CHUNK_S = 30
DEFAULT_OVERLAP_S = 0.5


class AudioEnhancer:
    """Speech enhancement using DeepFilterNet 3."""

    def __init__(self, config: dict):
        cfg = config.get("enhancement", {})
        self.mode = cfg.get("mode", "light")  # none | light | strong
        self.skip_snr = cfg.get("skip_if_snr_above", DEFAULT_SKIP_SNR_DB)
        self.device = cfg.get("device", "mps")
        self.chunk_duration_s = cfg.get("chunk_duration_s", DEFAULT_CHUNK_S)
        self.overlap_s = cfg.get("overlap_s", DEFAULT_OVERLAP_S)

        # Validate device
        if self.device == "mps" and not torch.backends.mps.is_available():
            logger.warning("MPS not available, falling back to CPU")
            self.device = "cpu"

        logger.info("AudioEnhancer: using device=%s", self.device)

        self._model = None
        self._df_state = None

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------

    def process(self, audio_path: str) -> EnhancedAudioResult:
        """
        Enhance audio quality.

        Returns EnhancedAudioResult. If SNR is already good enough
        (above skip_snr), returns skipped=True without processing.
        """
        t0 = time.time()
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Estimate SNR
        snr_before = self._estimate_snr(str(audio_path))
        logger.info("Estimated SNR: %.1f dB", snr_before)

        # Skip if already clean
        if snr_before > self.skip_snr:
            dt_ms = (time.time() - t0) * 1000
            logger.info(
                "SNR %.1f dB > %.0f dB threshold — skipping enhancement",
                snr_before, self.skip_snr,
            )
            return EnhancedAudioResult(
                output_path=str(audio_path),
                skipped=True,
                snr_db_before=snr_before,
                snr_db_after=None,
                processing_time_ms=dt_ms,
            )

        if self.mode == "none":
            dt_ms = (time.time() - t0) * 1000
            return EnhancedAudioResult(
                output_path=str(audio_path),
                skipped=True,
                snr_db_before=snr_before,
                snr_db_after=None,
                processing_time_ms=dt_ms,
            )

        # Load DeepFilterNet
        model, df_state = self._load_model()

        # Determine attenuation limit
        atten_lim = {"light": 30, "strong": 100}.get(self.mode, 30)

        # Enhance — chunked for long audio
        output_path = self._output_path(audio_path)

        try:
            self._enhance_chunked(
                str(audio_path), str(output_path), model, df_state, atten_lim,
            )
        except Exception as exc:
            raise RuntimeError(
                f"DeepFilterNet enhancement failed: {exc}"
            ) from exc
        finally:
            # Free GPU memory
            if self.device == "mps":
                torch.mps.empty_cache()
            gc.collect()

        # Post-enhancement SNR
        snr_after = self._estimate_snr(str(output_path))
        dt_ms = (time.time() - t0) * 1000

        logger.info(
            "AudioEnhancer: SNR %.1f dB → %.1f dB (mode=%s, %.0fms)",
            snr_before, snr_after, self.mode, dt_ms,
        )

        return EnhancedAudioResult(
            output_path=str(output_path),
            skipped=False,
            snr_db_before=snr_before,
            snr_db_after=snr_after,
            processing_time_ms=dt_ms,
        )

    # -----------------------------------------------------------
    # SNR Estimation
    # -----------------------------------------------------------

    @staticmethod
    def _estimate_snr(audio_path: str) -> float:
        """
        Estimate SNR using librosa.
        Top 10% loudest frames = signal, bottom 10% = noise.
        """
        try:
            import librosa
            y, sr = librosa.load(audio_path, sr=None)
            rms = librosa.feature.rms(y=y)[0]
            if len(rms) < 10:
                return 30.0  # too short to estimate, assume good
            rms_sorted = np.sort(rms)
            noise_level = np.mean(rms_sorted[: max(1, int(len(rms_sorted) * 0.1))]) + 1e-9
            signal_level = np.mean(rms_sorted[int(len(rms_sorted) * 0.9):]) + 1e-9
            return float(20 * np.log10(signal_level / noise_level))
        except Exception as exc:
            logger.warning("SNR estimation failed: %s, assuming low SNR", exc)
            return 0.0  # assume noisy → always enhance

    # -----------------------------------------------------------
    # DeepFilterNet Integration
    # -----------------------------------------------------------

    def _load_model(self):
        """Lazy-load DeepFilterNet 3 model."""
        if self._model is not None:
            return self._model, self._df_state

        logger.info("Loading DeepFilterNet 3 model...")
        t0 = time.time()

        try:
            from df.enhance import init_df
            model, df_state, _ = init_df()
        except ImportError:
            raise RuntimeError(
                "DeepFilterNet is required for audio enhancement. "
                "Install: pip install deepfilternet"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load DeepFilterNet model: {exc}"
            ) from exc

        self._model = model
        self._df_state = df_state
        dt = time.time() - t0
        logger.info("DeepFilterNet loaded in %.1fs", dt)
        return model, df_state

    def _enhance_chunked(
        self, input_path: str, output_path: str,
        model, df_state, atten_lim: int,
    ) -> None:
        """
        Process long audio in chunks to avoid OOM.
        Chunks are overlapped to prevent boundary artifacts.
        """
        from df.enhance import enhance, load_audio, save_audio

        # Load full audio
        audio, sr = load_audio(input_path, sr=16000)

        # If short enough, process directly
        total_duration = len(audio) / sr
        if total_duration <= self.chunk_duration_s:
            enhanced = enhance(model, df_state, audio, atten_lim_db=atten_lim)
            save_audio(output_path, enhanced, sr)
            return

        # Chunked processing
        chunk_samples = int(self.chunk_duration_s * sr)
        overlap_samples = int(self.overlap_s * sr)
        stride = chunk_samples - overlap_samples

        # Pad audio to align with chunks
        num_chunks = max(1, (len(audio) - overlap_samples + stride - 1) // stride)
        padded_len = (num_chunks - 1) * stride + chunk_samples
        if padded_len > len(audio):
            audio = np.pad(audio, (0, padded_len - len(audio)))

        enhanced = np.zeros_like(audio)
        weight = np.zeros_like(audio)

        # Hann window for cross-fade
        window = np.hanning(chunk_samples * 2)[chunk_samples:]

        for i in range(num_chunks):
            start = i * stride
            end = start + chunk_samples
            chunk = audio[start:end]

            # Enhance this chunk
            enhanced_chunk = enhance(
                model, df_state, chunk, atten_lim_db=atten_lim,
            )

            # Overlap-add with window
            enhanced[start:end] += enhanced_chunk * window
            weight[start:end] += window

        # Normalize by overlap weight
        weight = np.where(weight < 1e-6, 1.0, weight)
        enhanced = enhanced / weight

        # Trim padding
        enhanced = enhanced[: int(total_duration * sr)]

        save_audio(output_path, enhanced, sr)

    @staticmethod
    def _output_path(input_path: Path) -> Path:
        """Generate output path: {stem}_enhanced.wav."""
        return input_path.parent / f"{input_path.stem}_enhanced.wav"


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import sys
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("AudioEnhancer — Self Test")
    print("=" * 60)

    config = {
        "enhancement": {
            "mode": "light",
            "skip_if_snr_above": 30.0,
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
            "chunk_duration_s": 30,
            "overlap_s": 0.5,
        }
    }

    # Find a test audio file
    test_file = None
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "lecture_intel"
        wavs = sorted(temp_dir.glob("*_converted.wav"))
        if wavs:
            test_file = str(wavs[0])
        else:
            data_dir = Path(__file__).resolve().parents[2] / "data"
            for subdir in sorted(data_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                for ext in (".wav", ".webm", ".m4a"):
                    candidates = list(subdir.glob(f"*{ext}"))
                    if candidates:
                        test_file = str(candidates[0])
                        break
                if test_file:
                    break

    if not test_file:
        print("No test audio file found.")
        print("\nTesting SNR estimation only (no DeepFilterNet required):")
        print(f"  Model loaded:    {False}")
        print(f"  SNR estimation:  ready")
        print("\nPass an audio file path to test enhancement.")
        print("Example: python audio_enhancer.py /path/to/audio.wav")
        sys.exit(0)

    print(f"Testing with: {test_file}")

    # Test SNR estimation first (fast, no model needed)
    snr = AudioEnhancer._estimate_snr(test_file)
    print(f"Estimated SNR: {snr:.1f} dB")
    print(f"Skip threshold: {config['enhancement']['skip_if_snr_above']} dB")

    # Test full enhancement
    enhancer = AudioEnhancer(config)
    try:
        result = enhancer.process(test_file)
        print(f"\nResult:")
        print(f"  Skipped:    {result.skipped}")
        print(f"  SNR before: {result.snr_db_before:.1f} dB")
        if result.snr_db_after is not None:
            print(f"  SNR after:  {result.snr_db_after:.1f} dB")
        print(f"  Output:     {result.output_path}")
        print(f"  Time:       {result.processing_time_ms:.0f}ms")
    except Exception as exc:
        print(f"Enhancement failed: {exc}")
        import traceback
        traceback.print_exc()

    print("\n=== Test Complete ===")
