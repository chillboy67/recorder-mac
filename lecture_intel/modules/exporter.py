"""
Module 11 — Exporter

Exports ASR and LLM-corrected transcripts to multiple formats:
txt, md, docx, srt, json.

Output files are written to the specified output directory.

Usage:
    exporter = Exporter(config)
    outputs = exporter.process(asr_result, llm_result, output_dir, base_name)
"""

from __future__ import annotations

import json
import logging
from datetime import timedelta
from pathlib import Path
from typing import Optional

from modules import ASRResult, LLMResult, ASRSegment

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

SRT_MAX_CHARS_PER_LINE = 42
SRT_MAX_LINES_PER_CUE = 2


class Exporter:
    """Export transcripts to various output formats."""

    def __init__(self, config: dict):
        """
        Args:
            config: The 'export' section from config.yaml.
        """
        self.formats: list[str] = config.get("formats", ["txt", "md", "srt", "json"])
        self.include_timestamps: bool = config.get("include_timestamps", True)
        self.speaker_labels: bool = config.get("speaker_labels", False)
        self.docx_enabled: bool = config.get("docx_enabled", False)

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------

    def process(
        self,
        asr_result: ASRResult,
        llm_result: Optional[LLMResult],
        output_dir: Path,
        base_name: str,
    ) -> dict[str, Path]:
        """
        Export transcript to all configured formats.

        Args:
            asr_result: ASR output with segments and full text.
            llm_result: Optional LLM correction result.
            output_dir: Directory to write output files.
            base_name: Base filename (without extension).

        Returns:
            Dict mapping format name to output file path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Determine which text to use
        text = llm_result.full_corrected if llm_result else asr_result.full_text
        segments = asr_result.segments
        warnings = list(asr_result.warnings)
        if llm_result:
            warnings.extend(llm_result.warnings)

        outputs: dict[str, Path] = {}

        for fmt in self.formats:
            try:
                if fmt == "txt":
                    path = self.export_txt(text, segments, output_dir, base_name)
                elif fmt == "md":
                    path = self.export_md(text, segments, output_dir, base_name,
                                          asr_result, warnings)
                elif fmt == "srt":
                    path = self.export_srt(segments, output_dir, base_name)
                elif fmt == "json":
                    path = self.export_json(asr_result, llm_result,
                                            output_dir, base_name)
                elif fmt == "docx":
                    if self.docx_enabled:
                        path = self.export_docx(text, segments, output_dir, base_name)
                    else:
                        logger.info("Skipping docx (disabled in config)")
                        continue
                else:
                    logger.warning("Unknown format: %s, skipping", fmt)
                    continue

                outputs[fmt] = path
                logger.info("Exported %s: %s", fmt, path.name)

            except Exception as exc:
                logger.error("Failed to export %s: %s", fmt, exc)

        return outputs

    # -----------------------------------------------------------
    # TXT
    # -----------------------------------------------------------

    def export_txt(
        self,
        text: str,
        segments: list[ASRSegment],
        output_dir: Path,
        base_name: str,
    ) -> Path:
        """Export as plain text, optionally with timestamp markers."""
        path = output_dir / f"{base_name}.txt"

        lines = []
        if self.include_timestamps and segments:
            for seg in segments:
                ts = self._format_timestamp_short(seg.start)
                lines.append(f"[{ts}] {seg.text}")
        else:
            lines.append(text)

        path.write_text("\n\n".join(lines), encoding="utf-8")
        return path

    # -----------------------------------------------------------
    # Markdown
    # -----------------------------------------------------------

    def export_md(
        self,
        text: str,
        segments: list[ASRSegment],
        output_dir: Path,
        base_name: str,
        asr_result: Optional[ASRResult] = None,
        warnings: Optional[list[str]] = None,
    ) -> Path:
        """Export as structured Markdown document."""
        path = output_dir / f"{base_name}.md"

        lines = []
        lines.append(f"# Lecture Transcript: {base_name}")
        lines.append("")

        # Metadata
        if asr_result:
            lines.append(f"**Language:** {asr_result.language}")
            lines.append(f"**ASR Engine:** {asr_result.model_used}")
            if asr_result.audio_duration_sec:
                lines.append(
                    f"**Duration:** "
                    f"{self._format_duration(asr_result.audio_duration_sec)}"
                )
            lines.append(f"**Segments:** {len(asr_result.segments)}")
            lines.append("")

        # Warnings
        warnings = warnings or []
        if warnings:
            lines.append("### ⚠ Processing Warnings")
            for w in warnings:
                lines.append(f"- {w}")
            lines.append("")

        # Transcript
        lines.append("---")
        lines.append("")
        lines.append("## Transcript")
        lines.append("")

        if self.include_timestamps and segments:
            for seg in segments:
                ts = self._format_timestamp_full(seg.start)
                lang_badge = f"`{seg.language}`" if seg.language != "en" else ""
                prefix = f"**[{ts}]** {lang_badge} " if lang_badge else f"**[{ts}]** "
                lines.append(f"{prefix}{seg.text}")
                lines.append("")
            # If a corrected text differs from original, append it
            original_full = " ".join(s.text for s in segments)
            if text and text.strip() != original_full.strip():
                lines.append("---")
                lines.append("")
                lines.append("## Corrected Transcript (LLM)")
                lines.append("")
                for para in text.split("\n"):
                    para = para.strip()
                    if para:
                        lines.append(para)
                        lines.append("")
        else:
            # Split text into paragraphs
            paragraphs = text.split("\n")
            for para in paragraphs:
                para = para.strip()
                if para:
                    lines.append(para)
                    lines.append("")

        # Footer
        lines.append("---")
        lines.append("")
        lines.append("*Generated by Lecture Intelligence System*")

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    # -----------------------------------------------------------
    # SRT (Subtitle)
    # -----------------------------------------------------------

    def export_srt(
        self,
        segments: list[ASRSegment],
        output_dir: Path,
        base_name: str,
    ) -> Path:
        """
        Generate SRT subtitle file from word-level timestamps.
        Falls back to segment-level timestamps if word data unavailable.
        """
        path = output_dir / f"{base_name}.srt"

        cues = self._build_srt_cues(segments)

        lines = []
        for i, cue in enumerate(cues, start=1):
            lines.append(str(i))
            start_ts = self._format_srt_time(cue["start"])
            end_ts = self._format_srt_time(cue["end"])
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(cue["text"])
            lines.append("")  # Blank line between cues

        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _build_srt_cues(self, segments: list[ASRSegment]) -> list[dict]:
        """
        Build SRT cue blocks from segment word timestamps.

        Each cue is roughly 1-2 lines, max 42 chars per line.
        """
        cues: list[dict] = []

        for seg in segments:
            if seg.words:
                # Use word-level timestamps to create natural sub-cues
                current_words = []
                current_start = seg.words[0].start
                current_chars = 0

                for w in seg.words:
                    word_len = len(w.word)
                    if (current_chars + word_len > SRT_MAX_CHARS_PER_LINE * SRT_MAX_LINES_PER_CUE
                            and current_words):
                        # Flush current cue
                        cues.append({
                            "start": current_start,
                            "end": current_words[-1].end,
                            "text": " ".join(wd.word for wd in current_words),
                        })
                        current_words = [w]
                        current_start = w.start
                        current_chars = word_len
                    else:
                        current_words.append(w)
                        current_chars += word_len + 1  # +1 for space

                # Flush remaining words
                if current_words:
                    cues.append({
                        "start": current_start,
                        "end": current_words[-1].end,
                        "text": " ".join(wd.word for wd in current_words),
                    })
            else:
                # Fall back to segment-level timing
                # Split long segments into multiple cues
                text = seg.text
                if len(text) <= SRT_MAX_CHARS_PER_LINE * SRT_MAX_LINES_PER_CUE:
                    cues.append({
                        "start": seg.start,
                        "end": seg.end,
                        "text": text,
                    })
                else:
                    # Evenly split the segment
                    words = text.split()
                    mid = len(words) // 2
                    mid_time = seg.start + (seg.end - seg.start) / 2
                    cues.append({
                        "start": seg.start,
                        "end": mid_time,
                        "text": " ".join(words[:mid]),
                    })
                    cues.append({
                        "start": mid_time,
                        "end": seg.end,
                        "text": " ".join(words[mid:]),
                    })

        return cues

    # -----------------------------------------------------------
    # JSON
    # -----------------------------------------------------------

    def export_json(
        self,
        asr_result: ASRResult,
        llm_result: Optional[LLMResult],
        output_dir: Path,
        base_name: str,
    ) -> Path:
        """Export complete structured data as JSON."""
        path = output_dir / f"{base_name}.json"

        data = {
            "metadata": {
                "language": asr_result.language,
                "model_used": asr_result.model_used,
                "audio_duration_sec": asr_result.audio_duration_sec,
                "segment_count": len(asr_result.segments),
            },
            "segments": [
                {
                    "id": seg.id,
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text,
                    "language": seg.language,
                    "confidence": seg.confidence,
                    "words": [
                        {
                            "word": w.word,
                            "start": w.start,
                            "end": w.end,
                            "confidence": w.confidence,
                        }
                        for w in seg.words
                    ],
                }
                for seg in asr_result.segments
            ],
            "full_text": asr_result.full_text,
            "corrected_text": llm_result.full_corrected if llm_result else None,
            "warnings": (
                asr_result.warnings + (llm_result.warnings if llm_result else [])
            ),
        }

        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path

    # -----------------------------------------------------------
    # DOCX
    # -----------------------------------------------------------

    def export_docx(
        self,
        text: str,
        segments: list[ASRSegment],
        output_dir: Path,
        base_name: str,
    ) -> Path:
        """Export as a formatted Word document (.docx)."""
        try:
            from docx import Document
            from docx.shared import Pt, Inches, RGBColor
        except ImportError:
            raise RuntimeError(
                "python-docx is required for docx export. "
                "Install: pip install python-docx"
            )

        path = output_dir / f"{base_name}.docx"
        doc = Document()

        # Title
        title = doc.add_heading(f"Lecture Transcript: {base_name}", level=1)

        # Transcript body
        if self.include_timestamps and segments:
            for seg in segments:
                # Timestamp in small gray text
                ts_para = doc.add_paragraph()
                ts_run = ts_para.add_run(f"[{self._format_timestamp_full(seg.start)}]")
                ts_run.font.size = Pt(8)
                ts_run.font.color.rgb = RGBColor(128, 128, 128)

                # Segment text
                text_para = doc.add_paragraph(seg.text)
                text_para.paragraph_format.space_after = Pt(6)
        else:
            for paragraph in text.split("\n"):
                para = paragraph.strip()
                if para:
                    doc.add_paragraph(para)

        doc.save(str(path))
        return path

    # -----------------------------------------------------------
    # Formatting utilities
    # -----------------------------------------------------------

    @staticmethod
    def _format_timestamp_short(seconds: float) -> str:
        """Format as MM:SS."""
        td = timedelta(seconds=int(seconds))
        total_secs = int(td.total_seconds())
        mins, secs = divmod(total_secs, 60)
        return f"{mins:02d}:{secs:02d}"

    @staticmethod
    def _format_timestamp_full(seconds: float) -> str:
        """Format as HH:MM:SS."""
        td = timedelta(seconds=int(seconds))
        total_secs = int(td.total_seconds())
        hours, remainder = divmod(total_secs, 3600)
        mins, secs = divmod(remainder, 60)
        return f"{hours:02d}:{mins:02d}:{secs:02d}"

    @staticmethod
    def _format_srt_time(seconds: float) -> str:
        """Format as HH:MM:SS,mmm (SRT format)."""
        td = timedelta(seconds=seconds)
        total_secs = int(td.total_seconds())
        hours, remainder = divmod(total_secs, 3600)
        mins, secs = divmod(remainder, 60)
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{mins:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration as 'Xh Ym Zs'."""
        td = timedelta(seconds=int(seconds))
        total = int(td.total_seconds())
        h, r = divmod(total, 3600)
        m, s = divmod(r, 60)
        if h > 0:
            return f"{h}h {m}m {s}s"
        elif m > 0:
            return f"{m}m {s}s"
        return f"{s}s"


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import sys
    import tempfile
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("Exporter — Self Test")
    print("=" * 60)

    from modules import ASRResult, ASRSegment, ASRWord, LLMResult, CorrectedChunk

    # Create sample data
    sample_segments = [
        ASRSegment(
            id=0, start=0.0, end=5.2,
            text="Today we will discuss the attention mechanism in neural networks.",
            language="en", confidence=0.92,
            words=[
                ASRWord("Today", 0.0, 0.3, 0.99),
                ASRWord("we", 0.3, 0.5, 0.98),
                ASRWord("will", 0.5, 0.7, 0.99),
                ASRWord("discuss", 0.7, 1.0, 0.95),
                ASRWord("the", 1.0, 1.1, 0.99),
                ASRWord("attention", 1.1, 1.8, 0.88),
                ASRWord("mechanism", 1.8, 2.5, 0.90),
                ASRWord("in", 2.5, 2.6, 0.99),
                ASRWord("neural", 2.6, 3.0, 0.93),
                ASRWord("networks", 3.0, 3.5, 0.94),
            ],
        ),
        ASRSegment(
            id=1, start=5.5, end=12.0,
            text="The seq2seq model revolutionized machine translation and natural language processing.",
            language="en", confidence=0.89,
            words=[
                ASRWord("The", 5.5, 5.6, 0.99),
                ASRWord("seq2seq", 5.6, 6.5, 0.75),
                ASRWord("model", 6.5, 6.9, 0.97),
                ASRWord("revolutionized", 6.9, 8.0, 0.85),
                ASRWord("machine", 8.0, 8.4, 0.96),
                ASRWord("translation", 8.4, 9.2, 0.91),
                ASRWord("and", 9.2, 9.4, 0.99),
                ASRWord("natural", 9.4, 9.8, 0.94),
                ASRWord("language", 9.8, 10.3, 0.95),
                ASRWord("processing", 10.3, 11.0, 0.92),
            ],
        ),
    ]

    asr_result = ASRResult(
        segments=sample_segments,
        full_text="Today we will discuss the attention mechanism in neural networks. "
                  "The seq2seq model revolutionized machine translation and natural "
                  "language processing.",
        language="en",
        model_used="faster-whisper",
        audio_duration_sec=12.0,
    )

    llm_corrected_chunks = [
        CorrectedChunk(
            original=asr_result.full_text,
            corrected="Today we will discuss the attention mechanism in neural networks. "
                      "The seq2seq model revolutionized machine translation and natural "
                      "language processing.",
            chunk_index=0,
        ),
    ]
    llm_result = LLMResult(
        chunks=llm_corrected_chunks,
        full_corrected=llm_corrected_chunks[0].corrected,
    )

    # Test export
    config = {
        "formats": ["txt", "md", "srt", "json"],
        "include_timestamps": True,
        "speaker_labels": False,
        "docx_enabled": False,
    }

    exporter = Exporter(config)
    output_dir = Path(tempfile.gettempdir()) / "lecture_intel" / "test_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    outputs = exporter.process(asr_result, llm_result, output_dir, "test_lecture")

    print(f"\nGenerated {len(outputs)} file(s):")
    for fmt, path in outputs.items():
        size = path.stat().st_size
        print(f"  {fmt:5s} → {path.name} ({size:,} bytes)")

    # Show content previews
    print("\n--- TXT preview ---")
    txt_path = output_dir / "test_lecture.txt"
    if txt_path.exists():
        print(txt_path.read_text()[:300])

    print("\n--- SRT preview ---")
    srt_path = output_dir / "test_lecture.srt"
    if srt_path.exists():
        print(srt_path.read_text()[:400])

    print("\n--- MD preview ---")
    md_path = output_dir / "test_lecture.md"
    if md_path.exists():
        print(md_path.read_text()[:400])

    print("\n=== Test Complete ===")
