"""
Lecture Intelligence System - Modules Package.

Each module exposes a single main class with:
  - __init__(self, config: dict)  — receives its section from config.yaml
  - process(self, input) -> output  — main processing method

All Result objects MUST include:
  - processing_time_ms: float

Modules are independent; pipeline.py is the sole orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ============================================================
# P0 Types (existing, updated with processing_time_ms)
# ============================================================

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
class Segment:
    """Output of VADProcessor; a single speech segment with time boundaries."""
    start: float
    end: float
    audio_path: Path


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


@dataclass
class CorrectedChunk:
    """A single corrected chunk from LLM post-processing."""
    original: str
    corrected: str
    chunk_index: int
    warning: Optional[str] = None


@dataclass
class LLMResult:
    """Full LLM correction output."""
    chunks: list[CorrectedChunk]
    full_corrected: str
    warnings: list[str] = field(default_factory=list)
    processing_time_ms: float = 0.0


@dataclass
class PipelineResult:
    """Complete pipeline output summary."""
    input_path: Path
    audio_file: AudioFile
    segment_count: int
    asr_result: ASRResult
    llm_result: Optional[LLMResult]
    output_files: dict[str, Path]
    elapsed_sec: float
    warnings: list[str] = field(default_factory=list)


# ============================================================
# P1-A: Audio Enhancer Types
# ============================================================

@dataclass
class EnhancedAudioResult:
    """Output of AudioEnhancer."""
    output_path: str
    skipped: bool
    snr_db_before: float
    snr_db_after: Optional[float]
    processing_time_ms: float = 0.0


# ============================================================
# P1-B: Course Classifier Types
# ============================================================

@dataclass
class ClassificationResult:
    """Output of CourseClassifier."""
    course: str               # e.g. "deep_learning", "general_lecture"
    confidence: float         # 0.0 ~ 1.0
    top3: list[tuple]         # [(course, score), ...]
    domain_hints: list[str]   # detected keywords
    method: str               # "keyword" | "embedding"
    processing_time_ms: float = 0.0


# ============================================================
# P1-C: Terminology Corrector Types
# ============================================================

@dataclass
class CorrectionRecord:
    """A single correction made by TerminologyCorrector."""
    original: str
    corrected: str
    level: int                # 1-4, matching correction level
    position: int             # character position in text


@dataclass
class CorrectionResult:
    """Output of TerminologyCorrector."""
    corrected_text: str
    corrections_made: list[CorrectionRecord]
    correction_count: int
    dictionaries_used: list[str]
    processing_time_ms: float = 0.0


# ============================================================
# P1-D: Transcript Merger Types
# ============================================================

@dataclass
class MergedSegment:
    """A merged transcript segment."""
    id: int
    start: float
    end: float
    text: str
    speaker_id: Optional[str]
    source_segment_ids: list[int] = field(default_factory=list)


@dataclass
class MergeResult:
    """Output of TranscriptMerger."""
    segments: list[MergedSegment]
    full_text: str
    merge_count: int
    processing_time_ms: float = 0.0


# ============================================================
# P2-A: Speaker Diarizer Types
# ============================================================

@dataclass
class SpeakerSegment:
    """A diarized speaker segment."""
    start: float
    end: float
    speaker_id: str         # raw: SPEAKER_00
    speaker_label: str      # final: Person 1


@dataclass
class DiarizationResult:
    """Output of SpeakerDiarizer."""
    segments: list[SpeakerSegment]
    speaker_count: int
    speaker_map: dict[str, str]    # {"SPEAKER_00": "Person 1", ...}
    processing_time_ms: float = 0.0


# ============================================================
# P2-B: Lecture Structurer Types
# ============================================================

@dataclass
class KeyPoint:
    """A highlighted key point in the lecture."""
    text: str
    point_type: str         # "emphasis" | "definition" | "repetition" | "contrast"
    confidence: float
    start_char: int
    end_char: int


@dataclass
class StructuredResult:
    """Output of LectureStructurer."""
    markdown: str
    key_points: list[KeyPoint]
    definitions_count: int
    emphasis_count: int
    section_breaks: int
    processing_time_ms: float = 0.0
