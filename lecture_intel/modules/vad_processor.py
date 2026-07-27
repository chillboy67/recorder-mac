"""
Module 3 — VADProcessor

Voice Activity Detection using Silero VAD v4.
Detects speech regions, merges short gaps, splits long segments,
and extracts per-segment audio files.

Output: list of Segment objects with time boundaries and audio paths.

Usage:
    vad = VADProcessor(config)
    segments = vad.process(audio_file)
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

from modules import AudioFile, Segment

logger = logging.getLogger(__name__)

# ============================================================
# Defaults
# ============================================================

DEFAULT_THRESHOLD = 0.4
DEFAULT_MIN_SPEECH_MS = 250
DEFAULT_MIN_SILENCE_MS = 500
DEFAULT_MAX_SEGMENT_SEC = 30
DEFAULT_SEGMENT_OVERLAP_SEC = 1.0


class VADProcessor:
    """Voice Activity Detection using Silero VAD."""

    def __init__(self, config: dict):
        """
        Args:
            config: The 'vad' section from config.yaml.
        """
        self.threshold = config.get("threshold", DEFAULT_THRESHOLD)
        self.min_speech_ms = config.get("min_speech_ms", DEFAULT_MIN_SPEECH_MS)
        self.min_silence_ms = config.get("min_silence_ms", DEFAULT_MIN_SILENCE_MS)
        self.max_segment_sec = config.get("max_segment_sec", DEFAULT_MAX_SEGMENT_SEC)
        self.segment_overlap_sec = config.get("segment_overlap_sec", DEFAULT_SEGMENT_OVERLAP_SEC)
        self._model: Optional[torch.nn.Module] = None
        self._sample_rate = 16000  # Silero VAD expects 16kHz

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------

    def process(self, audio_file: AudioFile) -> list[Segment]:
        """
        Run VAD on an AudioFile and return speech segments.

        If no speech is detected, returns the entire audio as one segment.
        """
        if not audio_file.path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file.path}")

        logger.info("Loading audio for VAD: %s", audio_file.path.name)
        waveform, sr = self._load_audio(audio_file.path)

        logger.info("Running Silero VAD (threshold=%.2f)...", self.threshold)
        speech_ts = self._get_speech_timestamps(waveform, sr)

        if not speech_ts:
            logger.warning(
                "No speech segments detected. Using entire audio as one segment."
            )
            speech_ts = [
                {"start": 0.0, "end": audio_file.duration_sec}
            ]

        logger.info("Detected %d raw speech region(s)", len(speech_ts))

        # Merge close gaps
        merged = self._merge_close_segments(speech_ts)
        if len(merged) < len(speech_ts):
            logger.info("Merged to %d segment(s) (gap < %.1fs)", len(merged),
                        self.min_silence_ms / 1000)

        # Split long segments
        split = self._split_long_segments(merged)
        if len(split) > len(merged):
            logger.info("Split to %d segment(s) (max %.0fs each)", len(split),
                        self.max_segment_sec)

        # Extract audio files
        segments = self._extract_audio(audio_file.path, split)

        total_speech = sum(s.end - s.start for s in segments)
        logger.info(
            "Final: %d segment(s), total speech: %.1fs, "
            "min: %.1fs, max: %.1fs",
            len(segments), total_speech,
            min(s.end - s.start for s in segments) if segments else 0,
            max(s.end - s.start for s in segments) if segments else 0,
        )

        return segments

    # -----------------------------------------------------------
    # VAD Model
    # -----------------------------------------------------------

    def _load_model(self) -> torch.nn.Module:
        """Lazy-load Silero VAD model."""
        if self._model is not None:
            return self._model

        logger.info("Loading Silero VAD model (snakers4/silero-vad)...")
        try:
            model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to download Silero VAD model. "
                "Check your network connection or manually download from "
                "https://github.com/snakers4/silero-vad\n"
                f"Original error: {exc}"
            ) from exc

        self._model = model
        return model

    # -----------------------------------------------------------
    # Audio loading
    # -----------------------------------------------------------

    def _load_audio(self, audio_path: Path) -> tuple[torch.Tensor, int]:
        """Load audio and resample to 16kHz if needed."""
        try:
            import soundfile as sf
            data, sr = sf.read(str(audio_path))
        except Exception:
            # Fallback to torchaudio
            try:
                waveform, sr = torchaudio.load(str(audio_path), backend="soundfile")
                data = waveform.squeeze(0).numpy()
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load audio: {audio_path}\nError: {exc}"
                ) from exc

        # Convert stereo to mono
        if data.ndim > 1 and data.shape[1] > 1:
            data = data.mean(axis=1)

        # Convert to torch tensor [1, samples]
        waveform = torch.from_numpy(data.copy()).float().unsqueeze(0)

        # Resample if needed
        if sr != self._sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self._sample_rate)
            waveform = resampler(waveform)

        return waveform, self._sample_rate

    # -----------------------------------------------------------
    # VAD Processing
    # -----------------------------------------------------------

    def _get_speech_timestamps(self, waveform: torch.Tensor, sr: int) -> list[dict]:
        """
        Run the Silero VAD model and return speech timestamps.
        Returns list of {"start": float_sec, "end": float_sec}.
        """
        model = self._load_model()
        model.eval()

        # Silero expects shape [batch, samples]
        audio = waveform.squeeze(0)

        try:
            speech_ts = self._run_vad(audio, sr)
        except Exception as exc:
            raise RuntimeError(
                f"VAD inference failed: {exc}"
            ) from exc

        return speech_ts

    def _run_vad(self, audio: torch.Tensor, sr: int) -> list[dict]:
        """
        Run VAD with get_speech_timestamps.
        Handles different Silero VAD API versions.
        """
        from torch.hub import load as hub_load

        # Silero VAD utility functions come from torch.hub
        # We use the model directly with its bundled utilities
        model = self._model
        try:
            # Silero VAD v4+ returns speech timestamps directly
            speech_timestamps = model.get_speech_timestamps(
                audio,
                model,
                threshold=self.threshold,
                sampling_rate=sr,
                min_speech_duration_ms=self.min_speech_ms,
                min_silence_duration_ms=self.min_silence_ms,
            )
            return [
                {"start": ts["start"] / sr, "end": ts["end"] / sr}
                for ts in speech_timestamps
            ]
        except AttributeError:
            pass

        # Fallback: use the VAD iterator approach
        try:
            _, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
            get_speech_ts = utils[0]
            speech_timestamps = get_speech_ts(
                audio,
                model,
                threshold=self.threshold,
                sampling_rate=sr,
                min_speech_duration_ms=self.min_speech_ms,
                min_silence_duration_ms=self.min_silence_ms,
            )
            return [
                {"start": ts["start"] / sr, "end": ts["end"] / sr}
                for ts in speech_timestamps
            ]
        except Exception:
            raise RuntimeError(
                "Silero VAD API has changed. Please update to v4+: "
                "pip install silero-vad"
            )

    # -----------------------------------------------------------
    # Segment merging
    # -----------------------------------------------------------

    def _merge_close_segments(self, segments: list[dict]) -> list[dict]:
        """
        Merge segments where the gap between end of one and start of next
        is less than min_silence_ms.
        """
        if not segments:
            return []

        min_gap_sec = self.min_silence_ms / 1000.0
        merged = [segments[0].copy()]

        for seg in segments[1:]:
            gap = seg["start"] - merged[-1]["end"]
            if gap < min_gap_sec:
                # Merge: extend the previous segment
                merged[-1]["end"] = seg["end"]
            else:
                merged.append(seg.copy())

        return merged

    # -----------------------------------------------------------
    # Long segment splitting
    # -----------------------------------------------------------

    def _split_long_segments(self, segments: list[dict]) -> list[dict]:
        """
        Split any segment longer than max_segment_sec into overlapping chunks.
        Overlap is controlled by segment_overlap_sec.
        """
        result = []
        overlap = self.segment_overlap_sec

        for seg in segments:
            duration = seg["end"] - seg["start"]
            if duration <= self.max_segment_sec:
                result.append(seg)
                continue

            # Split into chunks
            pos = seg["start"]
            while pos < seg["end"]:
                chunk_end = min(pos + self.max_segment_sec, seg["end"])
                chunk_duration = chunk_end - pos

                # Don't create a tiny final chunk — merge with previous
                if chunk_duration < 2.0 and result:
                    result[-1]["end"] = seg["end"]
                    break

                result.append({"start": pos, "end": chunk_end})
                pos = chunk_end - overlap

                # Prevent infinite loop on very short remaining
                if pos >= seg["end"] - 0.1:
                    break

        return result

    # -----------------------------------------------------------
    # Audio extraction
    # -----------------------------------------------------------

    def _extract_audio(self, audio_path: Path, segments: list[dict]) -> list[Segment]:
        """
        Extract each segment's audio into a separate WAV file using ffmpeg.
        Files are named: {original_stem}_seg_{index:04d}.wav
        """
        temp_dir = Path(tempfile.gettempdir()) / "lecture_intel" / "segments"
        temp_dir.mkdir(parents=True, exist_ok=True)
        stem = audio_path.stem

        results: list[Segment] = []

        for i, seg in enumerate(segments):
            seg_start = seg["start"]
            seg_duration = seg["end"] - seg["start"]
            out_path = temp_dir / f"{stem}_seg_{i:04d}.wav"

            cmd = [
                "ffmpeg",
                "-ss", str(seg_start),
                "-i", str(audio_path),
                "-t", str(seg_duration),
                "-c", "copy",
                "-y",
                "-loglevel", "error",
                str(out_path),
            ]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                logger.warning(
                    "ffmpeg extraction failed for segment %d (%.1f-%.1f): %s",
                    i, seg_start, seg["end"], proc.stderr.strip(),
                )
                # Fallback: re-encode instead of stream copy
                cmd[cmd.index("-c") + 1] = "pcm_s16le"
                retry = subprocess.run(cmd, capture_output=True, text=True)
                if retry.returncode != 0:
                    logger.warning("Segment %d extraction failed, skipping", i)
                    continue

            results.append(Segment(
                start=seg["start"],
                end=seg["end"],
                audio_path=out_path,
            ))

        return results


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("VADProcessor — Self Test")
    print("=" * 60)

    config = {
        "threshold": 0.4,
        "min_speech_ms": 250,
        "min_silence_ms": 500,
        "max_segment_sec": 30,
        "segment_overlap_sec": 1.0,
    }

    # Test with a converted WAV file
    test_file = None
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
    else:
        # Try to find a converted wav from a previous AudioLoader run
        temp_dir = Path(tempfile.gettempdir()) / "lecture_intel"
        wavs = sorted(temp_dir.glob("*_converted.wav"))
        if wavs:
            test_file = wavs[0]
        else:
            # Look for any existing wav in data/
            data_dir = Path(__file__).resolve().parents[2] / "data"
            for subdir in sorted(data_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                for ext in (".wav", ".webm", ".m4a"):
                    candidates = list(subdir.glob(f"*{ext}"))
                    if candidates:
                        test_file = candidates[0]
                        break
                if test_file:
                    break

    if not test_file or not test_file.exists():
        print("No test file found. First convert audio with audio_loader.py,")
        print("or pass a WAV file path as argument.")
        print("Example: python vad_processor.py /path/to/audio.wav")
        sys.exit(0)

    print(f"Testing with: {test_file}")

    # Create an AudioFile mock
    af = AudioFile(
        path=test_file,
        duration_sec=10.0,  # placeholder
        sample_rate=16000,
        channels=1,
    )

    vad = VADProcessor(config)
    try:
        segments = vad.process(af)
        print(f"\nFound {len(segments)} segment(s):")
        for seg in segments:
            dur = seg.end - seg.start
            print(f"  [{seg.start:7.2f}s - {seg.end:7.2f}s] ({dur:5.1f}s) "
                  f"→ {seg.audio_path.name}")
    except Exception as exc:
        print(f"✗ Error: {exc}")
        import traceback
        traceback.print_exc()

    print("\n=== Test Complete ===")
