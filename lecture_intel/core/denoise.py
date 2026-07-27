"""
Lightweight audio cleanup for the classroom mode.

DeepFilterNet (the original plan) is incompatible with current torchaudio and
drags in a heavy/fragile dependency. Instead we use ffmpeg's built-in filters,
which are always available (ffmpeg is already required) and robust:

  - highpass=f=80     : kill low-frequency room rumble / HVAC hum
  - afftdn            : FFT-based adaptive denoiser (steady background noise)
  - dynaudnorm        : gentle dynamic normalization so a distant lecturer's
                        voice comes up to a consistent level

This won't perform miracles on a terrible recording, but it reliably improves
far-field / echoey classroom audio before ASR without the risk of a broken
native dependency. It never alters the *content* — only the signal.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

# Conservative chain: remove rumble, denoise, normalize loudness.
_FILTER_CHAIN = (
    "highpass=f=80,"
    "afftdn=nr=12:nf=-25,"
    "dynaudnorm=f=200:g=5"
)


def denoise_audio(wav_path: Path) -> Path:
    """Return a cleaned 16k mono wav. Falls back to the input on any error."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found; skipping denoise")
        return wav_path

    out = Path(tempfile.mkstemp(suffix="_clean.wav", prefix="recorder_")[1])
    cmd = [
        ffmpeg, "-y", "-i", str(wav_path),
        "-af", _FILTER_CHAIN,
        "-ac", "1", "-ar", "16000",
        str(out),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out.exists() or out.stat().st_size == 0:
            logger.warning("denoise failed (%s); using original audio",
                           proc.stderr.strip()[:200])
            return wav_path
        logger.info("Classroom denoise applied: %s", out.name)
        return out
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("denoise error: %s; using original audio", exc)
        return wav_path
