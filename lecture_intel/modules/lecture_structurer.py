"""
Module 10 — LectureStructurer (P2-B)

Highlights key content in lecture transcripts without guessing titles.
Detects emphasis phrases, definitions, repeated concepts, contrasts,
topic breaks, and code blocks.

Usage:
    structurer = LectureStructurer(config)
    result = structurer.process(corrected_text, course="deep_learning")
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

from modules import KeyPoint, StructuredResult

logger = logging.getLogger(__name__)

# ============================================================
# Signal Patterns
# ============================================================

EMPHASIS_MARKERS: dict[str, dict[str, list[str]]] = {
    "high": {
        "zh": ["重点", "关键", "注意", "记住", "非常重要", "核心", "本质上",
               "最重要的是", "一定要", "需要强调", "划重点", "关键在于"],
        "en": ["important", "key point", "crucial", "essential", "critical",
               "remember", "note that", "pay attention", "the key is",
               "fundamentally", "at its core", "this is the core"],
    },
    "medium": {
        "zh": ["也就是说", "换句话说", "简单来说", "实际上",
               "这意味着", "所以说", "因此", "这说明"],
        "en": ["in other words", "essentially", "basically", "that is to say",
               "which means", "therefore", "so what this means is",
               "the point is", "what we're saying is"],
    },
}

CONTRAST_MARKERS: dict[str, list[str]] = {
    "zh": ["但是", "然而", "不过", "区别在于", "与此不同",
           "相比之下", "而不是", "不同于", "而"],
    "en": ["however", "but", "whereas", "unlike", "in contrast",
           "the difference is", "rather than", "as opposed to",
           "on the other hand", "while"],
}

DEFINITION_PATTERNS: list[str] = [
    # English
    r"(?P<term>[\w][\w\s\-]{2,30})\s+(?:is|are|refers?\s+to|is\s+defined\s+as|means?)\s+(?P<def>[^.!?。？！]{10,})",
    r"we\s+(?:call|define|refer\s+to)\s+(?P<term>[\w\s\-]{2,20})\s+as\s+(?P<def>[^.!?]{10,})",
    # Chinese
    r"(?P<term>[一-鿿\w]{2,10})(?:是指|是|指的是|定义为|称为|就是)\s*(?P<def>[^。？！]{5,})",
    r"所谓(?P<term>[一-鿿\w]{2,10})[，,]?(?:就是|是指|是)\s*(?P<def>[^。？！]{5,})",
]

SECTION_ANNOUNCE_PATTERNS: list[str] = [
    r"(?:now\s+)?let'?s\s+(?:talk\s+about|move\s+on\s+to|look\s+at|discuss)\s+(?P<topic>[\w\s\-]{3,40})",
    r"(?:today|next)\s+we(?:'re\s+going\s+to|\s+will)\s+(?:cover|discuss|look\s+at)\s+(?P<topic>[\w\s\-]{3,30})",
    r"接下来(?:我们|我)?(?:来|要)?(?:讲|谈|讨论|介绍)\s*(?P<topic>[一-鿿\w]{2,15})",
    r"下面(?:介绍|讲|来看)\s*(?P<topic>[一-鿿\w]{2,15})",
]

CODE_PATTERNS: list[str] = [
    r"^\s{4,}\S",                    # indented code
    r"\b(?:def|class|import|from)\s+\w+",  # Python
    r"\b(?:function|const|let|var)\s+\w+",  # JavaScript
    r"\b\w+\s*\([^)]*\)\s*[{:]",      # function call with brace
]


class LectureStructurer:
    """Highlight key content in lecture transcripts."""

    def __init__(self, config: dict):
        cfg = config.get("structuring", {})
        self.highlight_emphasis = cfg.get("highlight_emphasis", True)
        self.highlight_definitions = cfg.get("highlight_definitions", True)
        self.highlight_repetitions = cfg.get("highlight_repetitions", True)
        self.highlight_contrasts = cfg.get("highlight_contrasts", True)
        self.topic_break_threshold = cfg.get("topic_break_threshold", 0.42)
        self.min_section_words = cfg.get("min_section_words", 60)
        self.detect_code_blocks = cfg.get("detect_code_blocks", True)
        self.generate_titles = cfg.get("generate_titles_from_speech", True)

        self._embedding_model = None

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------

    def process(self, corrected_text: str, course: str) -> StructuredResult:
        t0 = time.time()
        key_points: list[KeyPoint] = []
        markdown = corrected_text

        if not corrected_text or not corrected_text.strip():
            dt_ms = (time.time() - t0) * 1000
            return StructuredResult(
                markdown="", key_points=[], definitions_count=0,
                emphasis_count=0, section_breaks=0, processing_time_ms=dt_ms,
            )

        sentences = self._split_sentences(corrected_text)

        # Signal 1: Emphasis markers
        emphasis_count = 0
        if self.highlight_emphasis:
            markdown, emphasis_kps = self._mark_emphasis(markdown, sentences)
            key_points.extend(emphasis_kps)
            emphasis_count = len(emphasis_kps)

        # Signal 2: Definitions
        definitions_count = 0
        if self.highlight_definitions:
            markdown, def_kps = self._mark_definitions(markdown)
            key_points.extend(def_kps)
            definitions_count = len(def_kps)

        # Signal 3: Repeated concepts
        if self.highlight_repetitions:
            markdown = self._mark_repeated_concepts(markdown, sentences)

        # Signal 4: Contrast markers
        if self.highlight_contrasts:
            markdown, contrast_kps = self._mark_contrasts(markdown, sentences)
            key_points.extend(contrast_kps)

        # Signal 5: Topic breaks (separators only)
        section_breaks = 0
        if self.topic_break_threshold > 0:
            markdown, section_breaks = self._insert_topic_breaks(
                markdown, sentences, course,
            )

        # Generate titles from speaker announcements
        if self.generate_titles:
            markdown = self._extract_section_titles(markdown)

        # Code block detection
        if self.detect_code_blocks:
            markdown = self._mark_code_blocks(markdown)

        dt_ms = (time.time() - t0) * 1000
        logger.info(
            "LectureStructurer: %d key points, %d definitions, "
            "%d emphasis, %d section breaks in %.0fms",
            len(key_points), definitions_count, emphasis_count,
            section_breaks, dt_ms,
        )

        return StructuredResult(
            markdown=markdown,
            key_points=key_points,
            definitions_count=definitions_count,
            emphasis_count=emphasis_count,
            section_breaks=section_breaks,
            processing_time_ms=dt_ms,
        )

    # -----------------------------------------------------------
    # Signal 1: Emphasis Detection
    # -----------------------------------------------------------

    def _mark_emphasis(
        self, text: str, sentences: list[str],
    ) -> tuple[str, list[KeyPoint]]:
        key_points: list[KeyPoint] = []

        for sent in sentences:
            sent_stripped = sent.strip()
            if not sent_stripped:
                continue

            is_high = self._contains_any(sent_stripped.lower(), EMPHASIS_MARKERS["high"])
            is_medium = self._contains_any(sent_stripped.lower(), EMPHASIS_MARKERS["medium"])

            if is_high:
                # Mark this sentence as a key point
                label = "🔑 **Key point**: "
                pos = text.find(sent_stripped)
                if pos >= 0:
                    text = text.replace(sent_stripped, label + sent_stripped, 1)
                    key_points.append(KeyPoint(
                        text=sent_stripped[:200],
                        point_type="emphasis",
                        confidence=0.9,
                        start_char=pos,
                        end_char=pos + len(sent_stripped),
                    ))
            elif is_medium:
                # Bold key terms (simplified: bold entire sentence as hint)
                pos = text.find(sent_stripped)
                if pos >= 0:
                    key_points.append(KeyPoint(
                        text=sent_stripped[:200],
                        point_type="emphasis",
                        confidence=0.6,
                        start_char=pos,
                        end_char=pos + len(sent_stripped),
                    ))

        return text, key_points

    # -----------------------------------------------------------
    # Signal 2: Definition Detection
    # -----------------------------------------------------------

    def _mark_definitions(self, text: str) -> tuple[str, list[KeyPoint]]:
        key_points: list[KeyPoint] = []

        for pattern in DEFINITION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                term = match.group("term").strip()
                definition = match.group("def").strip()
                original = match.group(0)

                # Format as bold term + definition
                replacement = f"**{term}**: {definition}"
                text = text.replace(original, replacement, 1)

                key_points.append(KeyPoint(
                    text=f"{term}: {definition[:100]}",
                    point_type="definition",
                    confidence=0.85,
                    start_char=match.start(),
                    end_char=match.end(),
                ))

        return text, key_points

    # -----------------------------------------------------------
    # Signal 3: Repeated Concepts
    # -----------------------------------------------------------

    def _mark_repeated_concepts(
        self, text: str, sentences: list[str],
    ) -> str:
        """Detect concepts repeated ≥3 times in a sliding window of 5 sentences."""
        if len(sentences) < 5:
            return text

        window_size = 5
        min_count = 3

        # Build word frequency per window
        from collections import Counter

        for i in range(len(sentences) - window_size + 1):
            window = sentences[i:i + window_size]
            window_text = " ".join(window).lower()

            # Extract potential terms (2+ word phrases, capitalized or technical)
            words = re.findall(r"[a-zA-Z][a-zA-Z\-]{3,}", window_text)
            word_counts = Counter(words)

            repeated = [w for w, c in word_counts.items() if c >= min_count]
            if repeated:
                # Insert core concept note before the window
                top_term = repeated[0]
                first_sent = window[0].strip()
                pos = text.find(first_sent)
                if pos >= 0:
                    concept_note = f"> **Core concept**: {top_term} (mentioned {word_counts[top_term]} times)\n\n"
                    # Only insert if not already present
                    if concept_note not in text[max(0, pos - 200):pos]:
                        text = text[:pos] + concept_note + text[pos:]

        return text

    # -----------------------------------------------------------
    # Signal 4: Contrast Detection
    # -----------------------------------------------------------

    def _mark_contrasts(
        self, text: str, sentences: list[str],
    ) -> tuple[str, list[KeyPoint]]:
        key_points: list[KeyPoint] = []

        for sent in sentences:
            sent_stripped = sent.strip()
            if not sent_stripped:
                continue

            if self._contains_any(sent_stripped.lower(), CONTRAST_MARKERS):
                label = "⚡ **Distinction**: "
                pos = text.find(sent_stripped)
                if pos >= 0 and not text[max(0, pos - 3):pos].endswith(label):
                    text = text.replace(sent_stripped, label + sent_stripped, 1)
                    key_points.append(KeyPoint(
                        text=sent_stripped[:200],
                        point_type="contrast",
                        confidence=0.8,
                        start_char=pos,
                        end_char=pos + len(sent_stripped),
                    ))

        return text, key_points

    # -----------------------------------------------------------
    # Signal 5: Topic Breaks
    # -----------------------------------------------------------

    def _insert_topic_breaks(
        self, text: str, sentences: list[str], course: str,
    ) -> tuple[str, int]:
        """
        Insert '---' separators at topic transitions.
        Uses embedding similarity when BGE-M3 is available,
        otherwise skipped.
        """
        if len(sentences) < 10:
            return text, 0

        try:
            model = self._get_embedding_model()
        except Exception:
            logger.debug("No embedding model available, skipping topic breaks")
            return text, 0

        # Group sentences into ~100-word windows
        windows = []
        current = []
        current_words = 0
        for s in sentences:
            wc = len(s.split())
            current.append(s)
            current_words += wc
            if current_words >= 100:
                windows.append(" ".join(current))
                current = []
                current_words = 0
        if current:
            windows.append(" ".join(current))

        if len(windows) < 2:
            return text, 0

        # Compute embeddings and similarities
        import numpy as np
        embeddings = []
        for w in windows:
            embeddings.append(model.encode(w))

        breaks = []
        low_count = 0
        for i in range(1, len(embeddings)):
            sim = float(np.dot(embeddings[i], embeddings[i - 1]) /
                        (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i - 1]) + 1e-9))
            if sim < self.topic_break_threshold:
                low_count += 1
            else:
                low_count = 0

            # Need 3 consecutive low-similarity windows
            if low_count >= 3:
                breaks.append(i - 2)
                low_count = 0

        # Insert separators (reverse order to preserve positions)
        for br_idx in reversed(breaks):
            window_text = windows[br_idx]
            first_sent = window_text.split(". ")[0].strip() + "."
            pos = text.find(first_sent)
            if pos >= 0:
                text = text[:pos] + "\n\n---\n\n" + text[pos:]

        return text, len(breaks)

    # -----------------------------------------------------------
    # Section Title Extraction
    # -----------------------------------------------------------

    def _extract_section_titles(self, text: str) -> str:
        """Extract section titles from speaker's own words."""
        for pattern in SECTION_ANNOUNCE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                topic = match.group("topic").strip()
                if len(topic) >= 3 and len(topic) <= 40:
                    original = match.group(0)
                    heading = f"## {topic}"
                    # Only replace if not already in a heading
                    if not text[max(0, match.start() - 5):match.start()].startswith("#"):
                        text = text.replace(original, f"\n\n{heading}\n\n{original}", 1)
        return text

    # -----------------------------------------------------------
    # Code Block Detection
    # -----------------------------------------------------------

    def _mark_code_blocks(self, text: str) -> str:
        """Wrap detected code snippets in markdown code fences."""
        lines = text.split("\n")
        in_code = False
        result = []

        for line in lines:
            is_code_line = any(
                re.search(p, line) for p in CODE_PATTERNS
            )
            if is_code_line and not in_code:
                result.append("```")
                in_code = True
            elif not is_code_line and in_code:
                result.append("```")
                in_code = False
            result.append(line)

        if in_code:
            result.append("```")

        return "\n".join(result)

    # -----------------------------------------------------------
    # Utilities
    # -----------------------------------------------------------

    @staticmethod
    def _contains_any(text_lower: str, markers: dict[str, list[str]]) -> bool:
        """Check if text contains any marker from zh or en lists."""
        for lang in ("zh", "en"):
            for marker in markers.get(lang, []):
                if marker.lower() in text_lower:
                    return True
        return False

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences."""
        parts = re.split(r"(?<=[。！？.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    def _get_embedding_model(self):
        """Lazy-load embedding model (reuse if already loaded elsewhere)."""
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from FlagEmbedding import FlagModel
            model = FlagModel("BAAI/bge-m3", use_fp16=True)
            self._embedding_model = model
            return model
        except ImportError:
            raise RuntimeError("FlagEmbedding required for topic break detection")


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("LectureStructurer — Self Test")
    print("=" * 60)

    config = {
        "structuring": {
            "highlight_emphasis": True,
            "highlight_definitions": True,
            "highlight_repetitions": True,
            "highlight_contrasts": True,
            "topic_break_threshold": 0.42,
            "min_section_words": 60,
            "detect_code_blocks": True,
            "generate_titles_from_speech": True,
        }
    }

    structurer = LectureStructurer(config)

    test_text = (
        "The attention mechanism is a key component in modern neural networks. "
        "This is important to understand. "
        "Self-attention is a mechanism where each token in a sequence attends to all "
        "other tokens in the same sequence. "
        "In other words, every token can look at every other token. "
        "However, cross-attention operates across two different sequences, "
        "unlike self-attention which operates within a single sequence. "
        "The key difference is that cross-attention connects the encoder and decoder. "
        "Now let's move on to multi-head attention. "
        "Multi-head attention means we use multiple attention heads in parallel. "
        "This is crucial for capturing different types of relationships. "
        "Remember that each head learns different projection matrices."
    )

    print(f"Input text ({len(test_text)} chars):")
    print(test_text[:200] + "...")
    print()

    result = structurer.process(test_text, course="deep_learning")

    print(f"Key points:   {len(result.key_points)}")
    print(f"Definitions:  {result.definitions_count}")
    print(f"Emphasis:     {result.emphasis_count}")
    print(f"Breaks:       {result.section_breaks}")
    print()
    print("Structured output:")
    print(result.markdown[:800])

    print("\n=== Test Complete ===")
