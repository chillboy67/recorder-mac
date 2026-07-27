"""
Module 8 — TerminologyCorrector (P1-C)

Corrects ASR terminology errors using domain-specific dictionaries.
Four levels of correction, applied in strict order:
  Level 1: exact phrase matching (longest-first)
  Level 2: exact word matching (case-insensitive)
  Level 3: regex pattern matching
  Level 4: fuzzy matching (Levenshtein <= threshold, rapidfuzz score >= 85)

Usage:
    corrector = TerminologyCorrector(config)
    result = corrector.process(text, course="deep_learning", confidence=0.85)
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from modules import CorrectionRecord, CorrectionResult

logger = logging.getLogger(__name__)

# Map course_id → dictionary filename
COURSE_TO_FILE_MAP: dict[str, str] = {
    "deep_learning":      "deep_learning_terms.json",
    "machine_learning":   "machine_learning_terms.json",
    "statistics":         "statistics_terms.json",
    "r_language":         "r_terms.json",
    "python_programming": "python_terms.json",
    "mathematics":        "mathematics_terms.json",
    "data_science":       "machine_learning_terms.json",  # reuse
}


class TerminologyCorrector:
    """Multi-level ASR terminology correction using domain dictionaries."""

    def __init__(self, config: dict):
        cfg = config.get("correction", {})
        # Resolve dicts_dir relative to this module's directory
        dicts_dir_str = cfg.get("dictionaries_dir", "./dictionaries")
        if not Path(dicts_dir_str).is_absolute():
            module_dir = Path(__file__).resolve().parent.parent
            self.dicts_dir = module_dir / dicts_dir_str
        else:
            self.dicts_dir = Path(dicts_dir_str)
        self.fuzzy_threshold = cfg.get("fuzzy_threshold", 2)
        self.min_word_length_for_fuzzy = cfg.get("min_word_length_for_fuzzy", 5)
        self.fuzzy_score_threshold = cfg.get("fuzzy_score_threshold", 85)
        self.enable_phonetic = cfg.get("enable_phonetic", False)

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------

    def process(
        self, text: str, course: str, confidence: float,
    ) -> CorrectionResult:
        """
        Apply all correction levels to the input text.

        If confidence < 0.70, only general_terms.json is loaded.
        """
        t0 = time.time()
        corrections: list[CorrectionRecord] = []

        if not text or not text.strip():
            return CorrectionResult(
                corrected_text=text, corrections_made=[],
                correction_count=0, dictionaries_used=[],
                processing_time_ms=(time.time() - t0) * 1000,
            )

        # Load dictionaries
        dicts = self._load_dictionaries(course, confidence)

        # Extract all phrase corrections, sorted by phrase length (longest first)
        exact_phrases: dict[str, str] = {}
        exact_words: dict[str, str] = {}
        regex_patterns: list[dict] = []

        for d in dicts:
            for k, v in d.get("exact_phrases", {}).items():
                if len(k) > len(exact_phrases.get(k, "")):
                    exact_phrases[k] = v
            for k, v in d.get("exact_words", {}).items():
                if k not in exact_words:
                    exact_words[k] = v
            regex_patterns.extend(d.get("regex_patterns", []))

        # Sort phrases by length (longest first for greedy matching)
        sorted_phrases = sorted(exact_phrases.items(), key=lambda x: len(x[0]), reverse=True)

        # --- Level 1: Exact phrase matching ---
        for phrase, replacement in sorted_phrases:
            pattern = re.compile(re.escape(phrase), re.IGNORECASE)
            for match in pattern.finditer(text):
                pos = match.start()
                corrections.append(CorrectionRecord(
                    original=match.group(),
                    corrected=replacement,
                    level=1,
                    position=pos,
                ))
            text = pattern.sub(replacement, text)

        # --- Level 2: Exact word matching ---
        for word, replacement in exact_words.items():
            # Match as whole word only (boundary-aware)
            pattern = re.compile(r"\b" + re.escape(word) + r"\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                pos = match.start()
                corrections.append(CorrectionRecord(
                    original=match.group(),
                    corrected=replacement,
                    level=2,
                    position=pos,
                ))
            text = pattern.sub(replacement, text)

        # --- Level 3: Regex pattern matching ---
        for rp in regex_patterns:
            try:
                pattern = re.compile(rp["pattern"], re.IGNORECASE)
                for match in pattern.finditer(text):
                    pos = match.start()
                    corrections.append(CorrectionRecord(
                        original=match.group(),
                        corrected=rp["replace"],
                        level=3,
                        position=pos,
                    ))
                text = pattern.sub(rp["replace"], text)
            except re.error as exc:
                logger.warning("Invalid regex pattern '%s': %s", rp["pattern"], exc)

        # --- Level 4: Fuzzy matching ---
        if self.fuzzy_threshold > 0:
            fuzzy_corrections = self._fuzzy_correct(
                text, exact_phrases, exact_words, corrections,
            )
            corrections.extend(fuzzy_corrections)

        # Deduplicate and sort by position
        seen_pos = set()
        unique_corrections = []
        for c in sorted(corrections, key=lambda x: x.position):
            key = (c.position, c.original, c.corrected)
            if key not in seen_pos:
                seen_pos.add(key)
                unique_corrections.append(c)

        dict_names = [d["meta"]["domain"] for d in dicts]
        dt_ms = (time.time() - t0) * 1000

        logger.info(
            "TerminologyCorrector: %d corrections from %s in %.0fms",
            len(unique_corrections), dict_names, dt_ms,
        )

        return CorrectionResult(
            corrected_text=text,
            corrections_made=unique_corrections,
            correction_count=len(unique_corrections),
            dictionaries_used=dict_names,
            processing_time_ms=dt_ms,
        )

    # -----------------------------------------------------------
    # Dictionary Loading
    # -----------------------------------------------------------

    def _load_dictionaries(self, course: str, confidence: float) -> list[dict]:
        """Load appropriate dictionaries based on course and confidence."""
        files_to_load = ["general_terms.json"]  # always loaded

        if confidence >= 0.70:
            domain_file = COURSE_TO_FILE_MAP.get(course)
            if domain_file:
                files_to_load.append(domain_file)

        dicts = []
        for fname in files_to_load:
            path = self.dicts_dir / fname
            if not path.exists():
                logger.warning("Dictionary not found: %s", path)
                continue
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    d = json.load(fh)
                    dicts.append(d)
                    logger.debug("Loaded dictionary: %s (%s)", fname, d["meta"]["domain"])
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Failed to load dictionary %s: %s", fname, exc)

        return dicts

    # -----------------------------------------------------------
    # Fuzzy Matching (Level 4)
    # -----------------------------------------------------------

    def _fuzzy_correct(
        self,
        text: str,
        exact_phrases: dict[str, str],
        exact_words: dict[str, str],
        existing_corrections: list[CorrectionRecord],
    ) -> list[CorrectionRecord]:
        """Apply fuzzy matching to individual words in the text."""
        try:
            from rapidfuzz import fuzz
        except ImportError:
            logger.warning("rapidfuzz not installed — fuzzy matching disabled")
            return []

        corrections: list[CorrectionRecord] = []

        # Build a flat list of known terms and their corrections
        known_terms: dict[str, str] = {}
        for phrase, replacement in exact_phrases.items():
            for word in phrase.split():
                if len(word) >= self.min_word_length_for_fuzzy:
                    known_terms[word.lower()] = word
        for word, replacement in exact_words.items():
            if len(word) >= self.min_word_length_for_fuzzy:
                known_terms[word.lower()] = replacement

        if not known_terms:
            return []

        # Split text into words, preserving positions
        word_positions: list[tuple[str, int, int]] = []  # (word, start, end)
        for match in re.finditer(r"[a-zA-Z]+", text):
            word_positions.append((match.group(), match.start(), match.end()))

        # Already-corrected positions (skip these)
        corrected_positions = {
            (c.position, c.position + len(c.original))
            for c in existing_corrections
        }

        for word, start, end in word_positions:
            if len(word) < self.min_word_length_for_fuzzy:
                continue
            if any(cs <= start and end <= ce for cs, ce in corrected_positions):
                continue

            word_lower = word.lower()

            # Find best fuzzy match
            best_score = 0
            best_match = None
            best_replacement = None

            for known, replacement in known_terms.items():
                score = fuzz.token_sort_ratio(word_lower, known)
                if score > best_score and score >= self.fuzzy_score_threshold:
                    best_score = score
                    best_match = known
                    best_replacement = replacement

            if best_match and best_replacement:
                # Preserve original capitalization style
                if word[0].isupper():
                    final = best_replacement[0].upper() + best_replacement[1:]
                else:
                    final = best_replacement

                corrections.append(CorrectionRecord(
                    original=word,
                    corrected=final,
                    level=4,
                    position=start,
                ))
                # Apply correction
                text = text[:start] + final + text[end:]

        return corrections


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("TerminologyCorrector — Self Test")
    print("=" * 60)

    config = {
        "correction": {
            "dictionaries_dir": "./dictionaries",
            "fuzzy_threshold": 2,
            "min_word_length_for_fuzzy": 5,
            "fuzzy_score_threshold": 85,
            "enable_phonetic": False,
        }
    }

    corrector = TerminologyCorrector(config)

    # Test text with common ASR errors
    test_text = (
        "Today we discuss a tension mechanism and seek to seek models. "
        "The back propagation algorithm uses gradient decent. "
        "We also cover gpt and bert architectures. "
        "The relu activation is common. "
        "We use batch norm and drop out for regularization."
    )

    print("\nTest 1: deep_learning (confidence=0.90)")
    print(f"Original:  {test_text}")
    result = corrector.process(test_text, course="deep_learning", confidence=0.90)
    print(f"Corrected: {result.corrected_text}")
    print(f"Corrections: {result.correction_count}")
    print(f"Dictionaries: {result.dictionaries_used}")
    for c in result.corrections_made:
        print(f"  L{c.level}: '{c.original}' → '{c.corrected}' at pos {c.position}")

    # Test with statistics terms
    test_text2 = (
        "The p value was less than 0.05 so we reject the null hypothesis. "
        "The central limit theorem states that the sample mean follows a normal distribution. "
        "We can use anova and mcmc for this analysis."
    )

    print("\nTest 2: statistics (confidence=0.85)")
    print(f"Original:  {test_text2}")
    result2 = corrector.process(test_text2, course="statistics", confidence=0.85)
    print(f"Corrected: {result2.corrected_text}")
    print(f"Corrections: {result2.correction_count}")

    # Test low confidence (general only)
    print("\nTest 3: low confidence (0.50) — general terms only")
    print(f"Original:  {test_text}")
    result3 = corrector.process(test_text, course="deep_learning", confidence=0.50)
    print(f"Corrected: {result3.corrected_text}")
    print(f"Dictionaries: {result3.dictionaries_used}")

    print("\n=== Test Complete ===")
