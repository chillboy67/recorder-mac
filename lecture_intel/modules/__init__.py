"""
Lecture Intelligence System - shared data types.

The dataclasses here (AudioFile / ASRSegment / ASRWord / ASRResult) are the
common vocabulary passed between `core/` and the GUI. `modules/audio_loader.py`
is the only implementation remaining alongside them; the other modules that used
to live in this package belonged to the retired 11-step pipeline and have been
deleted — `core/engine.py` is the sole orchestrator now.

All Result objects MUST include:
  - processing_time_ms: float
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class AudioFile:
    """Output of AudioLoader; a converted, normalized audio file."""
    path: Path
    duration_sec: float
    sample_rate: int
    channels: int
    original_path: Optional[Path] = None
    processing_time_ms: float = 0.0


@dataclass
class ASRWord:
    """A single word with timing and confidence from ASR."""
    word: str
    start: float
    end: float
    confidence: float


@dataclass
class ASRSegment:
    """A transcribed segment with word-level detail."""
    id: int
    start: float
    end: float
    text: str
    language: str              # Whisper code ("en"/"zh"/"ja"/"fr"…) or "mixed"
    confidence: float
    words: list[ASRWord] = field(default_factory=list)
    speaker_id: Optional[str] = None


@dataclass
class ASRResult:
    """Full ASR output containing segments and metadata."""
    segments: list[ASRSegment]
    full_text: str
    language: str
    model_used: str
    audio_duration_sec: float = 0.0
    warnings: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0
    # Fidelity audit trail: repeat-arbitration verdicts, dropped-duplicate
    # markers, etc. Every entry keeps the original text so annotations never
    # destroy what the speaker (or the model) actually produced.
    annotations: list[dict] = field(default_factory=list)
