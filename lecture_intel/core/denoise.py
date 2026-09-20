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
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Conservative chain: remove rumble, denoise, normalize loudness.
FILTER_CHAIN = (
    "highpass=f=80,"
    "afftdn=nr=12:nf=-25,"
    "dynaudnorm=f=200:g=5"
)


def measure_noise_floor(wav_path: Path) -> Optional[float]:
    """Predict the noise floor in dB via ffmpeg volumedetect (best effort).

    volumedetect only reports whole-file levels, so we use ``mean_volume`` as
    the proxy: on clean close-mic audio the silent stretches pull the mean
    toward digital black, while a far-field classroom recording carries a
    constant HVAC/fan floor that keeps the mean high. It is a heuristic gate,
    not a measurement — the value and the decision it drives are both recorded
    in meta.json so the choice stays auditable. Returns None when ffmpeg is
    missing or the parse fails (caller then defaults to denoising).
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    cmd = [ffmpeg, "-hide_banner", "-i", str(wav_path),
           "-af", "volumedetect", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        m = re.search(r"mean_volume:\s*(-?[\d.]+)\s*dB", proc.stderr)
        return float(m.group(1)) if m else None
    except Exception:
        return None


def denoise_audio(wav_path: Path) -> Path:
    """Return a cleaned 16k mono wav. Falls back to the input on any error."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("ffmpeg not found; skipping denoise")
        return wav_path

    out = Path(tempfile.mkstemp(suffix="_clean.wav", prefix="recorder_")[1])
    cmd = [
        ffmpeg, "-y", "-i", str(wav_path),
        "-af", FILTER_CHAIN,
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
