"""
Pipeline — Main Orchestrator

Connects all modules in sequence. Modules are independent;
pipeline.py is the sole integration point.

Data flow (P0 + P1 + P2):
  Path → AudioLoader → AudioEnhancer (P1-A) → SpeakerDiarizer (P2-A)
       → VADProcessor → ASREngine → TranscriptMerger (P1-D)
       → CourseClassifier (P1-B) → TerminologyCorrector (P1-C)
       → LLMCorrector (P0) → LectureStructurer (P2-B) → Exporter

Usage:
    pipeline = Pipeline(config_path)
    result = pipeline.process(input_path)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import yaml

from modules import (
    AudioFile, ASRResult, LLMResult, PipelineResult, Segment,
    ClassificationResult,
)
from modules.audio_loader import AudioLoader
from modules.audio_enhancer import AudioEnhancer
from modules.vad_processor import VADProcessor
from modules.speaker_diarizer import SpeakerDiarizer
from modules.asr_engine import ASREngine
from modules.transcript_merger import TranscriptMerger
from modules.course_classifier import CourseClassifier
from modules.terminology_corrector import TerminologyCorrector
from modules.llm_corrector import LLMCorrector
from modules.lecture_structurer import LectureStructurer
from modules.exporter import Exporter

logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline orchestrator for lecture processing."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize all modules with their respective config sections
        self.audio_loader = AudioLoader(self.config.get("audio", {}))
        self.audio_enhancer = AudioEnhancer(self.config.get("enhancement", {}))
        self.speaker_diarizer = SpeakerDiarizer(self.config.get("diarization", {}))
        self.vad_processor = VADProcessor(self.config.get("vad", {}))
        self.asr_engine = ASREngine(self.config.get("asr", {}))
        self.transcript_merger = TranscriptMerger(self.config.get("merger", {}))
        self.course_classifier = CourseClassifier(self.config.get("classifier", {}))
        self.terminology_corrector = TerminologyCorrector(self.config.get("correction", {}))
        self.llm_corrector = LLMCorrector(self.config.get("llm", {}))
        self.lecture_structurer = LectureStructurer(self.config.get("structuring", {}))
        self.exporter = Exporter(self.config.get("export", {}))

        self.pipeline_cfg = self.config.get("pipeline", {})
        self.output_dir = Path(self.pipeline_cfg.get("output_dir", "./output"))
        self.cleanup_temp = self.pipeline_cfg.get("cleanup_temp", True)

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------

    def process(
        self,
        input_path: Path,
        output_dir: Optional[Path] = None,
        course_type: str = "general",
        domain_keywords: Optional[list[str]] = None,
        enhance: bool = True,
        formats: Optional[list[str]] = None,
    ) -> PipelineResult:
        """Run the full lecture processing pipeline."""
        warnings: list[str] = []
        t_start = time.time()
        output_dir = output_dir or self.output_dir
        output_dir = Path(output_dir)
        domain_keywords = domain_keywords or []

        if formats:
            self.exporter.formats = formats

        working_audio = input_path
        classification: Optional[ClassificationResult] = None
        working_text: Optional[str] = None

        # === Step 1: Audio Loading (P0) ===
        audio_file = self._step_load_audio(input_path)
        working_audio = audio_file.path

        # === Step 2: Audio Enhancement (P1-A) ===
        if self.pipeline_cfg.get("enable_audio_enhancement", False):
            enhance_result = self.audio_enhancer.process(str(working_audio))
            if not enhance_result.skipped:
                working_audio = Path(enhance_result.output_path)
                logger.info(
                    "Audio enhanced: SNR %.1f → %.1f dB",
                    enhance_result.snr_db_before,
                    enhance_result.snr_db_after or 0,
                )

        # === Step 3: Speaker Diarization (P2-A) ===
        speaker_segments = None
        if self.pipeline_cfg.get("enable_speaker_diarization", False):
            diarize_result = self.speaker_diarizer.process(str(working_audio))
            speaker_segments = diarize_result.segments
            logger.info(
                "Diarization: %d speakers: %s",
                diarize_result.speaker_count,
                list(diarize_result.speaker_map.values()),
            )

        # === Step 4: VAD (P0) ===
        segments = self._step_vad(AudioFile(
            path=working_audio,
            duration_sec=audio_file.duration_sec,
            sample_rate=audio_file.sample_rate,
            channels=audio_file.channels,
        ))

        # === Step 5: ASR (P0) ===
        asr_result = self._step_asr(segments, audio_file.duration_sec)
        warnings.extend(asr_result.warnings)

        # Attach speaker info to ASR segments
        if speaker_segments:
            asr_result = self._attach_speakers(asr_result, speaker_segments)

        # === Step 6: Transcript Merging (P1-D) ===
        if self.pipeline_cfg.get("enable_transcript_merging", True):
            merge_result = self.transcript_merger.process(asr_result.segments)
            working_text = merge_result.full_text
            merged_segments = merge_result.segments
        else:
            working_text = asr_result.full_text
            merged_segments = asr_result.segments

        # === Step 7: Course Classification (P1-B) ===
        if self.pipeline_cfg.get("enable_course_classification", True):
            sample = working_text[:2000]
            classification = self.course_classifier.process(sample)

        # === Step 8: Terminology Correction (P1-C) ===
        if self.pipeline_cfg.get("enable_terminology_correction", True) and classification:
            course = classification.course if classification else "general_lecture"
            conf = classification.confidence if classification else 0.0
            corr_result = self.terminology_corrector.process(
                working_text, course=course, confidence=conf,
            )
            working_text = corr_result.corrected_text
            logger.info(
                "Terminology: %d corrections from %s",
                corr_result.correction_count,
                corr_result.dictionaries_used,
            )

        # === Step 9: LLM Correction (P0) ===
        llm_result: Optional[LLMResult] = None
        if enhance and self.pipeline_cfg.get("enable_llm_correction", True):
            course_hint = classification.course if classification else "general_lecture"
            llm_result = self.llm_corrector.process(
                asr_result, course_hint, domain_keywords,
            )
            if llm_result.warnings:
                warnings.extend(llm_result.warnings)
            if llm_result.full_corrected:
                working_text = llm_result.full_corrected

        # === Step 10: Lecture Structuring (P2-B) ===
        if self.pipeline_cfg.get("enable_lecture_structuring", False):
            course_hint = classification.course if classification else "general_lecture"
            structure_result = self.lecture_structurer.process(
                working_text, course=course_hint,
            )
            working_text = structure_result.markdown
            logger.info(
                "Structuring: %d key points, %d definitions",
                structure_result.emphasis_count,
                structure_result.definitions_count,
            )

        # === Step 11: Export (P0) ===
        output_files = self._step_export(
            asr_result, llm_result, working_text, merged_segments,
            output_dir, input_path.stem,
        )

        elapsed = time.time() - t_start
        result = PipelineResult(
            input_path=input_path,
            audio_file=audio_file,
            segment_count=len(segments),
            asr_result=asr_result,
            llm_result=llm_result,
            output_files=output_files,
            elapsed_sec=elapsed,
            warnings=warnings,
        )
        self._print_summary(result)
        return result

    def process_batch(
        self,
        input_dir: Path,
        output_dir: Optional[Path] = None,
        course_type: str = "general",
        domain_keywords: Optional[list[str]] = None,
        enhance: bool = True,
        formats: Optional[list[str]] = None,
    ) -> list[PipelineResult]:
        """Process all audio files in a directory."""
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {input_dir}")

        supported = {".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus", ".webm"}
        audio_files = sorted(
            f for f in input_dir.rglob("*")
            if f.suffix.lower() in supported and not f.name.startswith(".")
        )

        if not audio_files:
            logger.warning("No supported audio files found in %s", input_dir)
            return []

        logger.info("Batch processing %d file(s)", len(audio_files))
        results: list[PipelineResult] = []

        for i, audio_path in enumerate(audio_files):
            logger.info("=== File %d/%d: %s ===", i + 1, len(audio_files), audio_path.name)
            try:
                result = self.process(
                    input_path=audio_path, output_dir=output_dir,
                    course_type=course_type, domain_keywords=domain_keywords,
                    enhance=enhance, formats=formats,
                )
                results.append(result)
            except Exception as exc:
                logger.error("Failed to process %s: %s", audio_path, exc)

        succeeded = len(results)
        failed = len(audio_files) - succeeded
        logger.info("Batch complete: %d succeeded, %d failed", succeeded, failed)
        return results

    # -----------------------------------------------------------
    # Pipeline Steps
    # -----------------------------------------------------------

    def _step_load_audio(self, input_path: Path) -> AudioFile:
        logger.info("Step 1/11: Loading audio...")
        t0 = time.time()
        af = self.audio_loader.process(input_path)
        dt = time.time() - t0
        logger.info("✓ Audio: %s (%.1fs, %dHz) in %.1fs",
                     input_path.name, af.duration_sec, af.sample_rate, dt)
        return af

    def _step_vad(self, audio_file: AudioFile) -> list[Segment]:
        logger.info("Step 4/11: Detecting speech segments...")
        t0 = time.time()
        segments = self.vad_processor.process(audio_file)
        dt = time.time() - t0
        total_speech = sum(s.end - s.start for s in segments)
        logger.info("✓ VAD: %d segments, %.1fs speech in %.1fs",
                     len(segments), total_speech, dt)
        return segments

    def _step_asr(self, segments: list[Segment], duration: float) -> ASRResult:
        logger.info("Step 5/11: Transcribing...")
        t0 = time.time()
        result = self.asr_engine.process(segments)
        result.audio_duration_sec = duration
        dt = time.time() - t0
        logger.info("✓ ASR: language=%s, engine=%s, %d chars in %.1fs",
                     result.language, result.model_used,
                     len(result.full_text), dt)
        return result

    def _step_export(
        self, asr_result: ASRResult, llm_result: Optional[LLMResult],
        working_text: str, merged_segments, output_dir: Path, base_name: str,
    ) -> dict[str, Path]:
        logger.info("Step 11/11: Exporting...")
        t0 = time.time()

        # Build ASRSegment-compatible list for exporter
        from modules import ASRSegment as _ASRSegment
        export_segments = []
        for seg in merged_segments[:]:
            export_segments.append(_ASRSegment(
                id=seg.id if hasattr(seg, 'id') else 0,
                start=seg.start, end=seg.end,
                text=seg.text.strip() if hasattr(seg, 'text') else str(seg),
                language=asr_result.language, confidence=0.9,
            ))

        # Build ASRResult for exporter
        export_asr = ASRResult(
            segments=export_segments if export_segments else asr_result.segments,
            full_text=working_text,
            language=asr_result.language,
            model_used=asr_result.model_used,
            audio_duration_sec=asr_result.audio_duration_sec,
            warnings=asr_result.warnings,
        )

        outputs = self.exporter.process(export_asr, llm_result, output_dir, base_name)
        dt = time.time() - t0
        logger.info("✓ Export: %d formats in %.1fs", len(outputs), dt)
        return outputs

    def _attach_speakers(self, asr_result: ASRResult, speaker_segments) -> ASRResult:
        """Attach speaker labels to ASR segments based on time overlap."""
        for asr_seg in asr_result.segments:
            best_speaker = None
            best_overlap = 0.0
            for spk_seg in speaker_segments:
                overlap_start = max(asr_seg.start, spk_seg.start)
                overlap_end = min(asr_seg.end, spk_seg.end)
                overlap = max(0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = spk_seg.speaker_label
            if best_speaker:
                asr_seg.speaker_id = best_speaker
        return asr_result

    # -----------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------

    def _load_config(self) -> dict:
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)

    def _print_summary(self, result: PipelineResult) -> None:
        print()
        print("=" * 60)
        print("  Pipeline Complete")
        print("=" * 60)
        print(f"  Input:        {result.input_path.name}")
        print(f"  Duration:     {result.audio_file.duration_sec:.1f}s")
        print(f"  Segments:     {result.segment_count}")
        print(f"  Language:     {result.asr_result.language}")
        print(f"  ASR Engine:   {result.asr_result.model_used}")
        if result.llm_result:
            print(f"  LLM Corrected: yes ({len(result.llm_result.chunks)} chunks)")
        print(f"  Elapsed:      {result.elapsed_sec:.1f}s")
        if result.warnings:
            print(f"  Warnings:     {len(result.warnings)}")
        print(f"  Outputs:")
        for fmt, path in result.output_files.items():
            print(f"    {fmt:5s} → {path}")
        print("=" * 60)


# ============================================================
# Module-level run_pipeline (GUI entry point)
# ============================================================


def run_pipeline(
    input_path: str,
    output_path: str,
    config: dict,
    progress_callback: callable | None = None,
) -> dict:
    """
    Run the full lecture processing pipeline with optional progress reporting.

    Args:
        input_path: Path to the input audio file.
        output_path: Directory for output files (base name derived from input).
        config: Full configuration dict (same structure as config.yaml).
        progress_callback: If provided, called with dicts of shape:
            {"step": str, "message": str, "percent": int, "status": str}

    Returns:
        dict with keys: course, confidence, total_time_s, output_files, warnings.
    """
    import time as _time
    from pathlib import Path as _Path

    _t_start = _time.time()

    def _report(step: str, message: str, percent: int, status: str = "running"):
        if progress_callback:
            progress_callback({
                "step": step,
                "message": message,
                "percent": percent,
                "status": status,
            })

    input_p = _Path(input_path)
    output_p = _Path(output_path)

    # Build a Pipeline-compatible config — write to temp file then load
    import tempfile as _tempfile
    import yaml as _yaml

    with _tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as _tf:
        _yaml.dump(config, _tf)
        _config_path = _Path(_tf.name)

    try:
        pipeline = Pipeline(_config_path)

        # Apply additional runtime overrides from config that were already set
        pipeline_cfg = config.get("pipeline", {})

        # Determine which steps will actually run
        do_enhance = pipeline_cfg.get("enable_audio_enhancement", False)
        do_diarize = pipeline_cfg.get("enable_speaker_diarization", False)
        do_structure = pipeline_cfg.get("enable_lecture_structuring", False)

        _report("load", "Loading audio...", 5)
        audio_file = pipeline._step_load_audio(input_p)
        _report("load", "Audio loaded", 8, status="done")

        working_audio = audio_file.path

        # Enhancement
        if do_enhance:
            _report("enhance", "Enhancing audio...", 10)
            enhance_result = pipeline.audio_enhancer.process(str(working_audio))
            if not enhance_result.skipped:
                working_audio = _Path(enhance_result.output_path)
            _report("enhance", "Audio enhanced", 18, status="done")
        else:
            _report("enhance", "Audio enhancement skipped", 18, status="done")

        # Diarization
        speaker_segments = None
        if do_diarize:
            _report("diarize", "Detecting speakers...", 20)
            diarize_result = pipeline.speaker_diarizer.process(str(working_audio))
            speaker_segments = diarize_result.segments
            _report("diarize", f"Diarization: {diarize_result.speaker_count} speakers", 25, status="done")

        # VAD
        _report("vad", "Detecting speech segments...", 26)
        segments = pipeline._step_vad(AudioFile(
            path=working_audio,
            duration_sec=audio_file.duration_sec,
            sample_rate=audio_file.sample_rate,
            channels=audio_file.channels,
        ))
        _report("vad", "VAD complete", 30, status="done")

        # ASR
        _report("asr", "Transcribing (this takes a while)...", 32)
        asr_result = pipeline._step_asr(segments, audio_file.duration_sec)
        _report("asr", "Transcription complete", 65, status="done")

        warnings_list = list(asr_result.warnings)

        # Attach speakers
        if speaker_segments:
            asr_result = pipeline._attach_speakers(asr_result, speaker_segments)

        # Transcript Merging
        _report("merge", "Merging segments...", 67)
        if pipeline_cfg.get("enable_transcript_merging", True):
            merge_result = pipeline.transcript_merger.process(asr_result.segments)
            working_text = merge_result.full_text
            merged_segments = merge_result.segments
        else:
            working_text = asr_result.full_text
            merged_segments = asr_result.segments
        _report("merge", "Segments merged", 70, status="done")

        # Course Classification
        classification = None
        _report("classify", "Classifying course domain...", 72)
        if pipeline_cfg.get("enable_course_classification", True):
            sample = working_text[:2000]
            classification = pipeline.course_classifier.process(sample)
        _report("classify", "Course classified", 75, status="done")

        # Terminology Correction
        _report("correct", "Applying terminology corrections...", 76)
        if pipeline_cfg.get("enable_terminology_correction", True) and classification:
            course_name = classification.course if classification else "general_lecture"
            conf = classification.confidence if classification else 0.0
            corr_result = pipeline.terminology_corrector.process(
                working_text, course=course_name, confidence=conf,
            )
            working_text = corr_result.corrected_text
        _report("correct", "Terminology corrected", 80, status="done")

        # LLM Correction
        llm_result = None
        if pipeline_cfg.get("enable_llm_correction", True):
            _report("llm", "LLM correction (Phi-4)...", 82)
            course_hint = classification.course if classification else "general_lecture"
            llm_result = pipeline.llm_corrector.process(
                asr_result, course_hint, [],
            )
            if llm_result.warnings:
                warnings_list.extend(llm_result.warnings)
            if llm_result.full_corrected:
                working_text = llm_result.full_corrected
            _report("llm", "LLM correction complete", 92, status="done")
        else:
            _report("llm", "LLM correction skipped", 92, status="done")

        # Lecture Structuring
        if do_structure:
            _report("structure", "Structuring lecture...", 93)
            course_hint = classification.course if classification else "general_lecture"
            structure_result = pipeline.lecture_structurer.process(
                working_text, course=course_hint,
            )
            working_text = structure_result.markdown
            _report("structure", "Structuring complete", 96, status="done")
        else:
            _report("structure", "Structuring skipped", 96, status="done")

        # Export
        _report("export", "Exporting files...", 97)
        output_files = pipeline._step_export(
            asr_result, llm_result, working_text, merged_segments,
            output_p, input_p.stem,
        )
        _report("export", "Done", 100, status="done")

        elapsed = _time.time() - _t_start

        return {
            "course": classification.course if classification else "general_lecture",
            "confidence": classification.confidence if classification else 0.0,
            "total_time_s": elapsed,
            "output_files": {k: str(v) for k, v in output_files.items()},
            "warnings": warnings_list,
        }

    finally:
        # Clean up temp config
        if _config_path.exists():
            _config_path.unlink()


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("=" * 60)
    print("Pipeline — Self Test")
    print("=" * 60)

    config_path = Path(__file__).resolve().parent / "config.yaml"
    if not config_path.exists():
        print(f"Config not found at {config_path}")
        sys.exit(1)

    test_file = None
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
    else:
        data_dir = Path(__file__).resolve().parents[1] / "data"
        if data_dir.exists():
            for subdir in sorted(data_dir.iterdir()):
                if not subdir.is_dir():
                    continue
                for ext in (".webm", ".m4a", ".wav"):
                    candidates = list(subdir.glob(f"*{ext}"))
                    if candidates:
                        test_file = candidates[0]
                        break
                if test_file:
                    break

    if not test_file or not test_file.exists():
        print("No test audio found. Pass path as argument.")
        sys.exit(0)

    print(f"\nProcessing: {test_file}")
    pipeline = Pipeline(config_path)

    try:
        result = pipeline.process(
            input_path=test_file,
            course_type="general",
            enhance=False,
            formats=["txt", "md", "json"],
        )
    except Exception as exc:
        print(f"\nPipeline failed: {exc}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n=== Test Complete ===")
