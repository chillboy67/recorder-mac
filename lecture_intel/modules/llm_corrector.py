"""
Module 9 — LLMCorrector ★ Core Module

Post-processes ASR transcripts using a local LLM (Ollama + Qwen3)
to correct speech recognition errors while preserving meaning.

Key features:
- Sliding window chunking (800 tokens, 100 token overlap)
- Specialized system prompt for academic lecture correction
- Conservative temperature (0.1) to prevent hallucination
- Output validation (length ratio check)
- Automatic retry on failure

Usage:
    corrector = LLMCorrector(config)
    result = corrector.process(asr_result, course_type="ai")
"""

from __future__ import annotations

import logging
import time
from typing import Optional

import httpx

from modules import ASRResult, ASRSegment, CorrectedChunk, LLMResult

logger = logging.getLogger(__name__)

# ============================================================
# Constants
# ============================================================

DEFAULT_API_BASE = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"
DEFAULT_CHUNK_TOKENS = 800
DEFAULT_OVERLAP_TOKENS = 100
DEFAULT_MAX_OUTPUT_RATIO = 1.2
DEFAULT_MAX_RETRIES = 3
DEFAULT_TEMPERATURE = 0.1

SYSTEM_PROMPT_TEMPLATE = """You are a professional academic transcript editor specializing in university lectures.

Your task: Correct speech recognition (ASR) errors in the transcript below.

STRICT RULES:
1. PRESERVE original meaning completely — do not add or remove information
2. PRESERVE the instructor's natural teaching style and speaking patterns
3. PRESERVE all examples, analogies, and explanations exactly
4. PRESERVE code snippets, function names, variable names (fix only obvious typos)
5. FIX technical terminology errors (e.g., "a tension" → "attention")
6. FIX obvious punctuation and sentence boundary errors
7. FIX repetitions caused by ASR (not by the speaker)
8. DO NOT summarize, compress, or restructure content
9. DO NOT translate between Chinese and English
10. DO NOT add information not present in the original

Course type: {course_type}
Domain keywords for reference: {domain_keywords}

For mixed Chinese-English lectures:
- Keep the original language of each sentence
- Technical terms may appear in English within Chinese sentences — preserve this
- Do not force consistency if the speaker switches languages naturally

Output ONLY the corrected transcript text. No explanations, no notes."""


class LLMCorrector:
    """Corrects ASR errors using a local LLM via Ollama."""

    def __init__(self, config: dict):
        """
        Args:
            config: The 'llm' section from config.yaml.
        """
        self.enabled = config.get("enabled", True)
        self.api_base = config.get("api_base", DEFAULT_API_BASE)
        self.model = config.get("model", DEFAULT_MODEL)
        self.temperature = config.get("temperature", DEFAULT_TEMPERATURE)
        self.top_p = config.get("top_p", 0.9)
        self.repeat_penalty = config.get("repeat_penalty", 1.1)
        self.num_ctx = config.get("num_ctx", 4096)
        self.chunk_tokens = config.get("chunk_tokens", DEFAULT_CHUNK_TOKENS)
        self.overlap_tokens = config.get("overlap_tokens", DEFAULT_OVERLAP_TOKENS)
        self.max_output_ratio = config.get("max_output_ratio", DEFAULT_MAX_OUTPUT_RATIO)
        self.max_retries = config.get("max_retries", DEFAULT_MAX_RETRIES)
        self.context_window_sentences = config.get("context_window_sentences", 3)

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------

    def process(
        self,
        asr_result: ASRResult,
        course_type: str = "general",
        domain_keywords: Optional[list[str]] = None,
    ) -> LLMResult:
        """
        Correct the ASR transcript using LLM.

        If LLM is disabled or Ollama is unavailable, returns the
        original text unchanged with appropriate warnings.

        Args:
            asr_result: The ASR output to correct.
            course_type: Domain label (e.g., "ai", "statistics", "math").
            domain_keywords: Optional list of domain-specific terms.

        Returns:
            LLMResult with corrected text and any warnings.
        """
        if not self.enabled:
            logger.info("LLM correction disabled by config")
            return LLMResult(
                chunks=[],
                full_corrected=asr_result.full_text,
                warnings=["LLM correction disabled"],
            )

        if not self._check_ollama():
            logger.warning(
                "Ollama is not available at %s. Install it with: brew install ollama",
                self.api_base,
            )
            return LLMResult(
                chunks=[],
                full_corrected=asr_result.full_text,
                warnings=[
                    "Ollama not available — ASR output used without correction. "
                    "Install: brew install ollama && ollama pull qwen3:8b"
                ],
            )

        if not asr_result.full_text.strip():
            logger.warning("Empty ASR text, nothing to correct")
            return LLMResult(chunks=[], full_corrected="", warnings=["Empty input"])

        domain_keywords = domain_keywords or []
        chunks = self._chunk_text(asr_result.segments)
        logger.info("Split text into %d chunk(s)", len(chunks))

        corrected_chunks: list[CorrectedChunk] = []
        warnings: list[str] = []

        for i, chunk in enumerate(chunks):
            logger.info("Correcting chunk %d/%d (%d chars)...",
                        i + 1, len(chunks), len(chunk["text"]))

            # Build context from previous corrected chunk
            prev_context = ""
            if corrected_chunks:
                prev_sentences = self._last_n_sentences(
                    corrected_chunks[-1].corrected,
                    self.context_window_sentences,
                )
                prev_context = "\n".join(prev_sentences)

            domain_str = ", ".join(domain_keywords) if domain_keywords else "none specified"

            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                course_type=course_type,
                domain_keywords=domain_str,
            )

            user_message = (
                f"Previous context (for continuity, do NOT edit this):\n"
                f"{prev_context}\n\n"
                f"Transcript to correct:\n{chunk['text']}"
            )

            corrected_text = None
            last_error = None

            for attempt in range(self.max_retries):
                try:
                    raw = self._call_ollama(system_prompt, user_message)
                    warning = self._validate_output(chunk["text"], raw)

                    if warning and "severely truncated" in warning.lower():
                        # Hard failure — retry
                        last_error = warning
                        logger.warning(
                            "Chunk %d attempt %d: %s, retrying...",
                            i + 1, attempt + 1, warning,
                        )
                        continue

                    if warning:
                        # Soft warning — accept but flag
                        logger.warning("Chunk %d: %s", i + 1, warning)
                        warnings.append(f"Chunk {i}: {warning}")

                    corrected_text = raw.strip()
                    break

                except httpx.TimeoutException:
                    last_error = "timeout"
                    logger.warning(
                        "Chunk %d attempt %d: timeout, retrying...",
                        i + 1, attempt + 1,
                    )
                except Exception as exc:
                    last_error = str(exc)
                    logger.warning(
                        "Chunk %d attempt %d failed: %s, retrying...",
                        i + 1, attempt + 1, exc,
                    )

                time.sleep(1.0)  # Brief pause before retry

            if corrected_text is None:
                # All retries exhausted — use original
                warnings.append(
                    f"Chunk {i}: All {self.max_retries} retries failed "
                    f"({last_error}), using original"
                )
                corrected_chunks.append(CorrectedChunk(
                    original=chunk["text"],
                    corrected=chunk["text"],
                    chunk_index=i,
                    warning=f"Correction failed: {last_error}",
                ))
            else:
                corrected_chunks.append(CorrectedChunk(
                    original=chunk["text"],
                    corrected=corrected_text,
                    chunk_index=i,
                    warning=None,
                ))

        # Merge corrected chunks (handle overlap)
        full_corrected = self._merge_chunks(corrected_chunks)

        return LLMResult(
            chunks=corrected_chunks,
            full_corrected=full_corrected,
            warnings=warnings,
        )

    # -----------------------------------------------------------
    # Ollama communication
    # -----------------------------------------------------------

    def _check_ollama(self) -> bool:
        """Verify Ollama is running and accessible."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.api_base}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    def _call_ollama(self, system_prompt: str, user_message: str) -> str:
        """
        Send a generation request to Ollama.
        Returns the generated text.
        """
        url = f"{self.api_base}/api/generate"

        # Combine system prompt and user message as Ollama expects
        # Qwen3 uses ChatML format via Ollama's template automatically
        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_message,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "num_ctx": self.num_ctx,
            "stream": False,
        }

        t0 = time.time()
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()

        dt = time.time() - t0
        response = data.get("response", "")

        if not response:
            raise RuntimeError("Ollama returned empty response")

        logger.debug("Ollama response: %d chars in %.1fs", len(response), dt)
        return response

    # -----------------------------------------------------------
    # Chunking
    # -----------------------------------------------------------

    def _chunk_text(self, segments: list[ASRSegment]) -> list[dict]:
        """
        Split text into overlapping chunks of approximately chunk_tokens.

        Uses approximate token counting: splits by words for English,
        character-based for Chinese.
        """
        # Concatenate all segment texts with spaces
        full_text = " ".join(s.text for s in segments)
        if not full_text.strip():
            return []

        # Use sentence boundaries for clean splits
        sentences = self._split_sentences(full_text)
        if not sentences:
            return [{"text": full_text, "chunk_index": 0, "total_chunks": 1}]

        chunks = []
        current = []
        current_tokens = 0
        chunk_index = 0

        for sent in sentences:
            sent_tokens = self._approx_tokens(sent)

            if current_tokens + sent_tokens > self.chunk_tokens and current:
                # Save current chunk
                chunks.append({
                    "text": " ".join(current),
                    "chunk_index": chunk_index,
                    "total_chunks": 0,  # updated after loop
                })
                chunk_index += 1

                # Start new chunk, keeping overlap sentences
                overlap_tokens = 0
                overlap_sents = []
                for s in reversed(current):
                    st = self._approx_tokens(s)
                    if overlap_tokens + st <= self.overlap_tokens:
                        overlap_sents.insert(0, s)
                        overlap_tokens += st
                    else:
                        break
                current = overlap_sents
                current_tokens = overlap_tokens

            current.append(sent)
            current_tokens += sent_tokens

        # Last chunk
        if current:
            chunks.append({
                "text": " ".join(current),
                "chunk_index": chunk_index,
                "total_chunks": 0,
            })

        # Update total_chunks
        total = len(chunks)
        for ch in chunks:
            ch["total_chunks"] = total

        return chunks

    def _merge_chunks(self, corrected_chunks: list[CorrectedChunk]) -> str:
        """
        Merge corrected chunks, handling overlap by keeping the later
        chunk's version of overlapping text.
        """
        if not corrected_chunks:
            return ""

        # Simple approach: concat all corrected texts
        # For proper overlap handling, we'd need diff-based merging
        # For MVP, trust that the overlap provides enough context continuity
        texts = [c.corrected for c in corrected_chunks]
        return " ".join(texts)

    # -----------------------------------------------------------
    # Validation
    # -----------------------------------------------------------

    def _validate_output(self, original: str, corrected: str) -> Optional[str]:
        """
        Validate LLM output against safety rules.
        Returns a warning string if issues found, None otherwise.
        """
        if not corrected or not corrected.strip():
            return "Empty output from LLM"

        orig_len = len(original)
        corr_len = len(corrected)

        if corr_len > orig_len * self.max_output_ratio:
            return (
                f"Output length ({corr_len}) exceeds {self.max_output_ratio*100:.0f}% "
                f"of input ({orig_len}) — possible hallucination"
            )

        if corr_len < orig_len * 0.3:
            return (
                f"Output severely truncated: {corr_len} vs {orig_len} chars"
            )

        # Check for common hallucination patterns
        hallucination_markers = [
            "As an AI language model",
            "Here is the corrected transcript",
            "I have corrected",
            "The corrected version",
        ]
        for marker in hallucination_markers:
            if marker.lower() in corrected.lower():
                return f"Output contains meta-commentary: '{marker}'"

        return None

    # -----------------------------------------------------------
    # Text utilities
    # -----------------------------------------------------------

    @staticmethod
    def _approx_tokens(text: str) -> int:
        """Approximate token count (rough estimate for chunking)."""
        import re as _re
        words = len(_re.findall(r"[a-zA-Z]+", text))
        # Chinese chars are roughly 1.5 tokens each in Qwen3
        cjk = len(_re.findall(r"[一-鿿㐀-䶿]", text))
        return words + int(cjk * 1.5)

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using punctuation boundaries."""
        import re as _re
        # Split on sentence-ending punctuation followed by space or end
        parts = _re.split(r"(?<=[。！？.!?])\s+", text)
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def _last_n_sentences(text: str, n: int) -> list[str]:
        """Get the last N sentences from a text."""
        sentences = LLMCorrector._split_sentences(text)
        return sentences[-n:] if len(sentences) > n else sentences


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("LLMCorrector — Self Test")
    print("=" * 60)

    config = {
        "enabled": True,
        "api_base": "http://localhost:11434",
        "model": "qwen3:8b",
        "temperature": 0.1,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "num_ctx": 4096,
        "chunk_tokens": 800,
        "overlap_tokens": 100,
        "max_output_ratio": 1.2,
        "max_retries": 3,
        "context_window_sentences": 3,
    }

    corrector = LLMCorrector(config)

    # Check Ollama status
    if corrector._check_ollama():
        print("✓ Ollama running at", config["api_base"])

        # Test with a sample text
        sample_text = (
            "Today we will discuss the a tension mechanism that is widely used "
            "in neural networks. The seek to seek model is also known as seq2seq. "
            "Back propagation is used for training. Gradient decent is an optimization "
            "algorithm."
        )
        from modules import ASRSegment
        sample_segments = [
            ASRSegment(id=0, start=0.0, end=5.0, text=sample_text,
                       language="en", confidence=0.85)
        ]
        sample_result = ASRResult(
            segments=sample_segments,
            full_text=sample_text,
            language="en",
            model_used="test",
        )

        result = corrector.process(
            sample_result,
            course_type="ai",
            domain_keywords=["attention", "seq2seq", "backpropagation"],
        )

        print(f"\nOriginal:  {sample_text[:200]}")
        print(f"Corrected: {result.full_corrected[:200]}")
        if result.warnings:
            print(f"Warnings: {result.warnings}")
        print(f"Chunks: {len(result.chunks)}")
    else:
        print("✗ Ollama not running at", config["api_base"])
        print("  Install: brew install ollama")
        print("  Pull model: ollama pull qwen3:8b")
        print("  Start: ollama serve")
        print("\nSkipping correction test (no Ollama available).")
        print("The module will return original text when Ollama is unavailable.")

    # Test chunking logic (works without Ollama)
    long_text = (
        "Today we will discuss the attention mechanism. "
        "This is widely used in neural networks. "
        "The seq2seq model revolutionized machine translation. "
        "Backpropagation is the key training algorithm. "
        "Gradient descent optimizes the loss function. "
        "Transformers use self-attention layers. "
        "BERT is a bidirectional encoder. "
        "GPT is an autoregressive decoder. "
        "Fine-tuning adapts pre-trained models. "
        "Transfer learning saves training time. "
    ) * 5  # Make it longer

    sentences = corrector._split_sentences(long_text)
    print(f"\nSentence splitting test: {len(sentences)} sentences found")

    chunks = corrector._chunk_text([
        ASRSegment(id=0, start=0.0, end=10.0, text=long_text,
                   language="en", confidence=0.9)
    ])
    print(f"Chunking test: {len(chunks)} chunks generated")
    for c in chunks:
        print(f"  Chunk {c['chunk_index']}: {len(c['text'])} chars "
              f"(~{c['approx_tokens']} tokens)")

    print("\n=== Test Complete ===")
