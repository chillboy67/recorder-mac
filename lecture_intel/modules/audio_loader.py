"""
Module 1 — AudioLoader

Detects input audio format via magic bytes, converts to unified
16kHz mono PCM WAV using ffmpeg, and returns an AudioFile object.

Supports: mp3, m4a, wav, aac, flac, ogg, opus, webm

Usage:
    loader = AudioLoader(config)
    audio_file = loader.process(Path("lecture.mp3"))
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from modules import AudioFile

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_FORMAT = "wav"

MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    "mp3": [b"\xff\xfb", b"\xff\xf3", b"\xff\xf2", b"ID3"],
    "wav": [b"RIFF"],
    "flac": [b"fLaC"],
    "ogg": [b"OggS"],
    "m4a": [],   # ISOBMFF: check for 'ftyp' at offset 4
    "aac": [],   # same as m4a
    "webm": [b"\x1a\x45\xdf\xa3"],
}


class AudioLoader:
    """Load and normalize audio files to 16kHz mono WAV."""

    def __init__(self, config: dict):
        """
        Args:
            config: The 'audio' section from config.yaml.
        """
        self.target_sample_rate = config.get("target_sample_rate", TARGET_SAMPLE_RATE)
        self.target_channels = config.get("target_channels", TARGET_CHANNELS)
        self.target_format = config.get("target_format", TARGET_FORMAT)
        self.supported_formats = config.get(
            "supported_formats",
            ["mp3", "m4a", "wav", "aac", "flac", "ogg", "opus", "webm"],
        )
        self.max_file_size_gb = config.get("max_file_size_gb", 2)
        self._ffmpeg = self._find_ffmpeg()
        self._ffprobe = self._find_ffprobe()

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------

    def process(self, file_path: Path) -> AudioFile:
        """
        Load, validate, and convert an audio file.

        Returns an AudioFile pointing to a guaranteed 16kHz mono WAV.
        """
        file_path = file_path.resolve()
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        detected_fmt = self.detect_format(file_path)
        if detected_fmt is None:
            raise ValueError(
                f"Cannot detect audio format for {file_path.name}. "
                f"Supported: {', '.join(self.supported_formats)}"
            )
        if detected_fmt not in self.supported_formats:
            raise ValueError(
                f"Unsupported format '{detected_fmt}' for {file_path.name}. "
                f"Supported: {', '.join(self.supported_formats)}"
            )

        logger.info("Detected format: %s (%s)", detected_fmt, file_path.name)

        duration, sr, ch = self.get_audio_info(file_path)
        logger.info(
            "Audio info: duration=%.1fs, sample_rate=%dHz, channels=%d",
            duration, sr, ch,
        )

        if duration <= 0:
            raise ValueError(f"Audio file has zero duration: {file_path}")

        needs_conv = (
            sr != self.target_sample_rate
            or ch != self.target_channels
            or detected_fmt != self.target_format
        )

        if needs_conv:
            output_path = self._temp_wav_path(file_path)
            logger.info("Converting audio to %dHz mono WAV...", self.target_sample_rate)
            converted = self.convert(file_path, output_path)
            return AudioFile(
                path=converted,
                duration_sec=duration,
                sample_rate=self.target_sample_rate,
                channels=self.target_channels,
                original_path=file_path,
            )

        return AudioFile(
            path=file_path,
            duration_sec=duration,
            sample_rate=sr,
            channels=ch,
            original_path=file_path,
        )

    # -----------------------------------------------------------
    # Format detection
    # -----------------------------------------------------------

    def detect_format(self, file_path: Path) -> Optional[str]:
        """
        Detect audio format by reading magic bytes (not relying on extension).
        Returns lowercase format name or None.
        """
        try:
            with open(file_path, "rb") as fh:
                header = fh.read(16)
        except OSError as exc:
            raise RuntimeError(f"Cannot read file: {file_path}") from exc

        if len(header) < 4:
            return None

        # ISOBMFF (m4a / aac / mp4): 'ftyp' at offset 4
        if header[4:8] == b"ftyp":
            # Read brand to distinguish
            brand = header[8:12]
            if brand in (b"M4A ", b"M4A\x00"):
                return "m4a"
            if brand in (b"mp42", b"isom"):
                return "aac"
            return "m4a"  # default to m4a

        # Check known signatures
        for fmt, signatures in MAGIC_SIGNATURES.items():
            for sig in signatures:
                if header[: len(sig)] == sig:
                    return fmt

        return None

    # -----------------------------------------------------------
    # FFmpeg / FFprobe helpers
    # -----------------------------------------------------------

    def get_audio_info(self, file_path: Path) -> tuple[float, int, int]:
        """
        Extract duration, sample_rate, channels via ffprobe.
        Returns (duration_sec, sample_rate, channels).
        """
        cmd = [
            str(self._ffprobe),
            "-v", "error",
            "-show_entries", "format=duration:stream=duration,sample_rate,channels,codec_type",
            "-of", "json",
            str(file_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr.strip()}")

        try:
            data = json.loads(result.stdout)
            streams = data.get("streams", [])

            # Extract duration, sample_rate, channels from audio stream
            duration = 0.0
            sr = 0
            ch = 0

            for stream in streams:
                if stream.get("codec_type") == "audio":
                    stream_dur = stream.get("duration")
                    if stream_dur:
                        duration = float(stream_dur)
                    sr = int(stream.get("sample_rate", sr or 0))
                    ch = int(stream.get("channels", ch or 0))
                    break  # Found the audio stream

            # Fall back to first stream if no audio stream marked
            if not sr and streams:
                for stream in streams:
                    sr = int(stream.get("sample_rate", sr or 0))
                    ch = int(stream.get("channels", ch or 0))
                    stream_dur = stream.get("duration")
                    if stream_dur and duration == 0.0:
                        duration = float(stream_dur)
                    if sr and ch:
                        break

            # If stream duration is missing, use format duration
            if duration == 0.0:
                fmt = data.get("format", {})
                fmt_dur = fmt.get("duration")
                if fmt_dur:
                    duration = float(fmt_dur)

            return duration, sr, ch
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise RuntimeError(f"Failed to parse ffprobe output: {exc}") from exc

    def convert(self, input_path: Path, output_path: Path) -> Path:
        """
        Convert audio to target format using ffmpeg.
        For files > 2GB, uses streaming mode via subprocess.Popen.
        """
        file_size_gb = input_path.stat().st_size / (1024**3)
        is_large = file_size_gb >= self.max_file_size_gb

        cmd = [
            str(self._ffmpeg),
            "-i", str(input_path),
            "-ar", str(self.target_sample_rate),
            "-ac", str(self.target_channels),
            "-sample_fmt", "s16",
            "-y",  # overwrite
            "-loglevel", "error",
        ]

        if is_large:
            logger.info("Large file (%.1f GB), using streaming conversion", file_size_gb)

        cmd.append(str(output_path))

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg conversion failed: {result.stderr.strip()}"
            )
        if not output_path.exists():
            raise RuntimeError(
                f"ffmpeg conversion produced no output file: {output_path}"
            )

        return output_path

    # -----------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------

    def _find_ffmpeg(self) -> Path:
        """Locate ffmpeg binary."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            return Path(ffmpeg)
        # Check Homebrew default path
        brew_path = Path("/opt/homebrew/bin/ffmpeg")
        if brew_path.exists():
            return brew_path
        raise RuntimeError(
            "ffmpeg not found. Install it with: brew install ffmpeg"
        )

    def _find_ffprobe(self) -> Path:
        """Locate ffprobe binary."""
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            return Path(ffprobe)
        brew_path = Path("/opt/homebrew/bin/ffprobe")
        if brew_path.exists():
            return brew_path
        raise RuntimeError(
            "ffprobe not found. Install ffmpeg with: brew install ffmpeg"
        )

    @staticmethod
    def _temp_wav_path(original: Path) -> Path:
        """Generate a temporary WAV path."""
        temp_dir = Path(tempfile.gettempdir()) / "lecture_intel"
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir / f"{original.stem}_converted.wav"


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("AudioLoader — Self Test")
    print("=" * 60)

    config = {
        "target_sample_rate": 16000,
        "target_channels": 1,
        "target_format": "wav",
        "supported_formats": ["mp3", "m4a", "wav", "aac", "flac", "ogg", "opus", "webm"],
        "max_file_size_gb": 2,
    }

    loader = AudioLoader(config)

    # Test with files passed as arguments, or scan data/
    test_files = []
    if len(sys.argv) > 1:
        test_files = [Path(p) for p in sys.argv[1:]]
    else:
        data_dir = Path(__file__).resolve().parents[2] / "data"
        if data_dir.exists():
            for subdir in sorted(data_dir.iterdir()):
                if subdir.is_dir():
                    for f in subdir.iterdir():
                        if f.suffix.lower() in (".m4a", ".webm", ".mp3", ".wav", ".ogg"):
                            test_files.append(f)
                            break

    if not test_files:
        print("No test files found. Pass audio file paths as arguments.")
        print("Example: python audio_loader.py /path/to/audio.m4a")
        sys.exit(0)

    for tf in test_files[:3]:  # Test up to 3 files
        print(f"\n--- Testing: {tf.name} ---")
        try:
            af = loader.process(tf)
            print(f"  ✓ Path:       {af.path}")
            print(f"  ✓ Duration:   {af.duration_sec:.1f}s")
            print(f"  ✓ Sample rate: {af.sample_rate} Hz")
            print(f"  ✓ Channels:    {af.channels}")
            if af.original_path and af.original_path != af.path:
                print(f"  ✓ Original:   {af.original_path}")
        except Exception as exc:
            print(f"  ✗ Error: {exc}")

    print("\n=== Test Complete ===")
