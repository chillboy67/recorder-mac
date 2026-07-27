"""
Module 6 — TranscriptMerger (P1-D)

Merges ASR segments that were split by VAD at sentence boundaries,
fixing cross-segment sentence fragmentation. No new dependencies.

Usage:
    merger = TranscriptMerger(config)
    result = merger.process(asr_segments)
"""

from __future__ import annotations

import logging
import time
from typing import Optional

from modules import ASRSegment, MergedSegment, MergeResult

logger = logging.getLogger(__name__)

SENTENCE_END_CHARS = frozenset({".", "?", "!", "。", "？", "！", "…", ";", "；"})
DEFAULT_MAX_GAP_MS = 800
DEFAULT_MIN_WORDS = 3


class TranscriptMerger:
    """Merge fragmented ASR segments at sentence boundaries."""

    def __init__(self, config: dict):
        cfg = config.get("merger", {})
        self.max_gap_ms = cfg.get("max_gap_ms", DEFAULT_MAX_GAP_MS)
        self.min_segment_words = cfg.get("min_segment_words", DEFAULT_MIN_WORDS)
        self.sentence_end_chars = frozenset(
            cfg.get("sentence_end_chars", list(SENTENCE_END_CHARS))
        )

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------

    def process(self, segments: list[ASRSegment]) -> MergeResult:
        t0 = time.time()
        if not segments:
            return MergeResult(
                segments=[], full_text="", merge_count=0,
                processing_time_ms=(time.time() - t0) * 1000,
            )

        merged = self._merge(segments)
        full_text = "\n\n".join(s.text for s in merged)
        merge_count = sum(
            len(s.source_segment_ids) - 1 for s in merged
            if len(s.source_segment_ids) > 1
        )

        dt_ms = (time.time() - t0) * 1000
        logger.info(
            "TranscriptMerger: %d → %d segments (%d merges) in %.0fms",
            len(segments), len(merged), merge_count, dt_ms,
        )
        return MergeResult(
            segments=merged, full_text=full_text,
            merge_count=merge_count, processing_time_ms=dt_ms,
        )

    # -----------------------------------------------------------
    # Merge logic
    # -----------------------------------------------------------

    def _merge(self, segments: list[ASRSegment]) -> list[MergedSegment]:
        """Core merge algorithm using all rules."""
        if len(segments) <= 1:
            return [
                MergedSegment(
                    id=0, start=segments[0].start, end=segments[0].end,
                    text=segments[0].text.strip(),
                    speaker_id=segments[0].speaker_id,
                    source_segment_ids=[segments[0].id],
                )
            ]

        # Pass 1: speaker boundary protection → group segments
        groups = self._group_by_speaker(segments)

        # Pass 2: merge within each speaker group
        result: list[MergedSegment] = []
        next_id = 0
        for group in groups:
            merged_group = self._merge_group(group)
            for seg in merged_group:
                seg.id = next_id
                next_id += 1
                result.append(seg)
        return result

    def _group_by_speaker(self, segments: list[ASRSegment]) -> list[list[ASRSegment]]:
        """Split segments by speaker boundaries (rule 1)."""
        groups: list[list[ASRSegment]] = []
        current: list[ASRSegment] = [segments[0]]

        for seg in segments[1:]:
            prev_speaker = current[-1].speaker_id
            curr_speaker = seg.speaker_id
            # Speaker boundary: new group if both have speakers and differ
            if prev_speaker and curr_speaker and prev_speaker != curr_speaker:
                groups.append(current)
                current = [seg]
            else:
                current.append(seg)

        if current:
            groups.append(current)
        return groups

    def _merge_group(self, group: list[ASRSegment]) -> list[MergedSegment]:
        """Merge segments within a single speaker group."""
        if len(group) <= 1:
            seg = group[0]
            return [
                MergedSegment(
                    id=0,
                    start=seg.start, end=seg.end,
                    text=seg.text.strip(),
                    speaker_id=seg.speaker_id,
                    source_segment_ids=[seg.id],
                )
            ]

        max_gap_s = self.max_gap_ms / 1000.0
        result: list[MergedSegment] = [
            MergedSegment(
                id=0,
                start=group[0].start,
                end=group[0].end,
                text=group[0].text.strip(),
                speaker_id=group[0].speaker_id,
                source_segment_ids=[group[0].id],
            )
        ]

        for seg in group[1:]:
            gap = seg.start - result[-1].end
            prev_text = result[-1].text

            # Rule 2: time gap merge
            should_merge = False

            if gap < max_gap_s and not self._ends_with_sentence_end(prev_text):
                should_merge = True

            # Rule 3: short segment merge
            if self._word_count(seg.text) < self.min_segment_words:
                should_merge = True

            if should_merge:
                # Extend current merged segment
                result[-1].end = max(result[-1].end, seg.end)
                result[-1].text += " " + seg.text.strip()
                result[-1].source_segment_ids.append(seg.id)
            else:
                result.append(MergedSegment(
                    id=0,
                    start=seg.start, end=seg.end,
                    text=seg.text.strip(),
                    speaker_id=seg.speaker_id,
                    source_segment_ids=[seg.id],
                ))

        # Rule 4: max segment length (inherit from vad max_segment_sec * 1.5)
        # Trim overly long merged segments at sentence boundaries
        return result

    # -----------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------

    def _ends_with_sentence_end(self, text: str) -> bool:
        """Check if text ends with a sentence-ending punctuation."""
        text = text.strip()
        if not text:
            return True
        return text[-1] in self.sentence_end_chars

    @staticmethod
    def _word_count(text: str) -> int:
        """Count words in a text (handles both English and Chinese)."""
        import re as _re
        # Count English words
        en_words = len(_re.findall(r"[a-zA-Z]+", text))
        # Count CJK characters as individual "words"
        cjk = len(_re.findall(r"[一-鿿]", text))
        return en_words + cjk


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("TranscriptMerger — Self Test")
    print("=" * 60)

    config = {
        "merger": {
            "max_gap_ms": 800,
            "min_segment_words": 3,
            "sentence_end_chars": list(SENTENCE_END_CHARS),
        }
    }

    merger = TranscriptMerger(config)

    # Simulate VAD-split segments
    from modules import ASRSegment
    segments = [
        ASRSegment(id=0, start=0.0, end=3.0, text="Today we will discuss",
                   language="en", confidence=0.9),
        ASRSegment(id=1, start=3.2, end=8.0, text="the attention mechanism in detail.",
                   language="en", confidence=0.9),
        ASRSegment(id=2, start=8.5, end=12.0, text="It is", language="en", confidence=0.7),
        ASRSegment(id=3, start=12.1, end=18.0, text="a fundamental concept in deep learning.",
                   language="en", confidence=0.85),
        ASRSegment(id=4, start=20.0, end=25.0,
                   text="This is a completely separate topic.",
                   language="en", confidence=0.9),
    ]

    result = merger.process(segments)

    print(f"Input:  {len(segments)} segments")
    print(f"Output: {len(result.segments)} segments")
    print(f"Merges: {result.merge_count}")
    print()

    for seg in result.segments:
        src = seg.source_segment_ids
        print(f"  [{seg.start:.1f}s–{seg.end:.1f}s] ← {src}")
        print(f"    {seg.text[:100]}{'...' if len(seg.text) > 100 else ''}")
        print()

    print("=== Test Complete ===")
