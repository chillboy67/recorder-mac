"""
Module 4 — SpeakerDiarizer (P2-A)

Identifies different speakers in audio recordings using pyannote-audio.
Labels speakers as "Person 1", "Person 2", etc. in order of first appearance.

Requires HuggingFace user agreement for pyannote models:
  1. Visit https://huggingface.co/pyannote/speaker-diarization-3.1
  2. Log in, accept the user agreement
  3. Run: huggingface-cli login
  4. First run downloads ~600MB model, then fully offline

Usage:
    diarizer = SpeakerDiarizer(config)
    result = diarizer.process(audio_path)
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from modules import DiarizationResult, SpeakerSegment

logger = logging.getLogger(__name__)

DEFAULT_CLUSTERING_THRESHOLD = 0.75
DEFAULT_MERGE_GAP_MS = 1500
DEFAULT_MIN_SEGMENT_S = 1.0
DEFAULT_EMBED_SIM_THRESHOLD = 0.82


class SpeakerDiarizer:
    """Speaker diarization using pyannote-audio with anti-fragmentation safeguards."""

    def __init__(self, config: dict):
        cfg = config.get("diarization", {})
        self.model_name = cfg.get("model", "pyannote/speaker-diarization-3.1")
        self.min_speakers = cfg.get("min_speakers", 1)
        self.max_speakers = cfg.get("max_speakers", 6)
        self.clustering_threshold = cfg.get("clustering_threshold", DEFAULT_CLUSTERING_THRESHOLD)
        self.merge_gap_ms = cfg.get("merge_gap_ms", DEFAULT_MERGE_GAP_MS)
        self.min_segment_duration_s = cfg.get("min_segment_duration_s", DEFAULT_MIN_SEGMENT_S)
        self.embedding_similarity_threshold = cfg.get(
            "embedding_similarity_threshold", DEFAULT_EMBED_SIM_THRESHOLD,
        )
        self.device = cfg.get("device", "mps")
        if self.device == "mps" and not torch.backends.mps.is_available():
            self.device = "cpu"

        self._pipeline = None
        self._embedding_model = None

    # -----------------------------------------------------------
    # Public API
    # -----------------------------------------------------------

    def process(self, audio_path: str) -> DiarizationResult:
        """
        Run speaker diarization on an audio file.

        Returns DiarizationResult with labeled speaker segments.
        """
        t0 = time.time()
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        logger.info("Running speaker diarization on %s...", audio_path.name)

        # Load pipeline
        pipeline = self._load_pipeline()

        # Run diarization
        try:
            diarization = pipeline(str(audio_path))
        except Exception as exc:
            raise RuntimeError(
                f"pyannote diarization failed: {exc}. "
                "Make sure you have accepted the HuggingFace user agreement at "
                "https://huggingface.co/pyannote/speaker-diarization-3.1"
            ) from exc

        # Convert to our data format
        raw_segments: list[SpeakerSegment] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            raw_segments.append(SpeakerSegment(
                start=turn.start,
                end=turn.end,
                speaker_id=speaker,
                speaker_label="",  # filled later
            ))

        if not raw_segments:
            dt_ms = (time.time() - t0) * 1000
            return DiarizationResult(
                segments=[], speaker_count=0, speaker_map={},
                processing_time_ms=dt_ms,
            )

        # Sort by start time
        raw_segments.sort(key=lambda s: s.start)

        # Layer 2: Short segment embedding verification
        verified = self._verify_short_segments(raw_segments, str(audio_path))

        # Layer 3: Sandwich merge (A → B → A pattern)
        merged = self._merge_sandwich_segments(verified)

        # Build speaker map: first appearance → Person N
        speaker_map = self._build_speaker_map(merged)

        # Apply labels
        for seg in merged:
            seg.speaker_label = speaker_map.get(seg.speaker_id, seg.speaker_id)

        unique_speakers = len(set(s.speaker_label for s in merged))
        dt_ms = (time.time() - t0) * 1000

        logger.info(
            "Diarization: %d segments, %d unique speakers in %.0fms",
            len(merged), unique_speakers, dt_ms,
        )

        return DiarizationResult(
            segments=merged,
            speaker_count=unique_speakers,
            speaker_map=speaker_map,
            processing_time_ms=dt_ms,
        )

    # -----------------------------------------------------------
    # Model Loading
    # -----------------------------------------------------------

    def _load_pipeline(self):
        """Lazy-load pyannote diarization pipeline."""
        if self._pipeline is not None:
            return self._pipeline

        try:
            from pyannote.audio import Pipeline
        except ImportError:
            raise RuntimeError(
                "pyannote.audio is required for speaker diarization. "
                "Install: pip install pyannote.audio"
            )

        hf_token = self._load_hf_token()
        if not hf_token:
            raise RuntimeError(
                "HuggingFace token required for pyannote models.\n"
                "1. Visit: https://huggingface.co/pyannote/speaker-diarization-3.1\n"
                "2. Accept the user agreement\n"
                "3. Run: huggingface-cli login\n"
                "4. Or set HF_TOKEN environment variable"
            )

        logger.info("Loading pyannote diarization pipeline...")
        t0 = time.time()

        try:
            pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=hf_token,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pyannote pipeline: {exc}"
            ) from exc

        # Configure clustering for less fragmentation
        pipeline.instantiate({
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 12,
                "threshold": self.clustering_threshold,
            },
            "segmentation": {
                "min_duration_off": 0.5,
            },
        })

        # Move to device
        if self.device != "cpu":
            try:
                pipeline.to(torch.device(self.device))
                logger.info("Pipeline moved to %s", self.device)
            except Exception:
                logger.warning("Could not move pipeline to %s, using CPU", self.device)

        self._pipeline = pipeline
        dt = time.time() - t0
        logger.info("pyannote pipeline loaded in %.1fs", dt)
        return pipeline

    @staticmethod
    def _load_hf_token() -> Optional[str]:
        """Load HuggingFace token from env or cache."""
        token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if token:
            return token

        # Try huggingface-cli cache
        cache_dir = Path.home() / ".cache" / "huggingface" / "token"
        if cache_dir.exists():
            return cache_dir.read_text().strip()

        # Try .huggingface directory
        hf_dir = Path.home() / ".huggingface" / "token"
        if hf_dir.exists():
            return hf_dir.read_text().strip()

        return None

    # -----------------------------------------------------------
    # Layer 2: Short Segment Embedding Verification
    # -----------------------------------------------------------

    def _verify_short_segments(
        self, segments: list[SpeakerSegment], audio_path: str,
    ) -> list[SpeakerSegment]:
        """
        For segments shorter than min_segment_duration_s, verify speaker
        identity using embedding similarity with neighboring segments.
        """
        result = segments.copy()

        for i, seg in enumerate(result):
            duration = seg.end - seg.start
            if duration >= self.min_segment_duration_s:
                continue

            # Find neighbors (segments within 10s before/after)
            neighbors = []
            for j, other in enumerate(result):
                if j == i:
                    continue
                if abs(other.start - seg.start) <= 10.0:
                    neighbors.append(other)

            if not neighbors:
                continue

            try:
                emb_model = self._get_embedding_model()
                seg_emb = self._extract_embedding(emb_model, audio_path, seg.start, duration)
                if seg_emb is None:
                    continue

                best_sim = 0.0
                best_speaker = None

                for neighbor in neighbors:
                    n_dur = max(neighbor.end - neighbor.start, 0.1)
                    n_emb = self._extract_embedding(
                        emb_model, audio_path, neighbor.start, n_dur,
                    )
                    if n_emb is None:
                        continue

                    sim = self._cosine_similarity(seg_emb, n_emb)
                    if sim > best_sim and sim > self.embedding_similarity_threshold:
                        best_sim = sim
                        best_speaker = neighbor.speaker_id

                if best_speaker:
                    logger.debug(
                        "Short segment (%.2fs): reassigned %s → %s (sim=%.3f)",
                        duration, seg.speaker_id, best_speaker, best_sim,
                    )
                    result[i].speaker_id = best_speaker

            except Exception as exc:
                logger.debug("Embedding verification skipped: %s", exc)

        return result

    def _get_embedding_model(self):
        """Lazy-load pyannote embedding model."""
        if self._embedding_model is not None:
            return self._embedding_model

        try:
            from pyannote.audio import Inference
            model = Inference("pyannote/embedding", window="whole")
            self._embedding_model = model
            return model
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model: {exc}"
            ) from exc

    @staticmethod
    def _extract_embedding(
        model, audio_path: str, start: float, duration: float,
    ) -> Optional[np.ndarray]:
        """Extract speaker embedding for a segment."""
        try:
            emb = model(
                {"uri": "seg", "audio": audio_path},
                start=start,
                duration=max(duration, 0.1),
            )
            return emb if isinstance(emb, np.ndarray) else np.array(emb)
        except Exception:
            return None

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    # -----------------------------------------------------------
    # Layer 3: Sandwich Merge
    # -----------------------------------------------------------

    def _merge_sandwich_segments(
        self, segments: list[SpeakerSegment],
    ) -> list[SpeakerSegment]:
        """
        Merge sandwich patterns: Person A → Person B (<1s) → Person A.
        The short Person B segment is likely a misrecognition of A.
        """
        if len(segments) < 3:
            return segments

        result = segments.copy()
        merge_gap_s = self.merge_gap_ms / 1000.0

        for i in range(1, len(result) - 1):
            prev = result[i - 1]
            curr = result[i]
            next_ = result[i + 1]

            total_gap = next_.start - prev.end
            curr_dur = curr.end - curr.start

            if (prev.speaker_id == next_.speaker_id
                    and curr_dur < self.min_segment_duration_s
                    and total_gap < merge_gap_s
                    and curr.speaker_id != prev.speaker_id):
                logger.debug(
                    "Sandwich merge: %s (%.2fs) reassigned → %s",
                    curr.speaker_id, curr_dur, prev.speaker_id,
                )
                result[i].speaker_id = prev.speaker_id

        return result

    # -----------------------------------------------------------
    # Speaker Naming
    # -----------------------------------------------------------

    @staticmethod
    def _build_speaker_map(segments: list[SpeakerSegment]) -> dict[str, str]:
        """Build speaker ID → label map by first appearance order."""
        speaker_map: dict[str, str] = {}
        counter = 1
        for seg in sorted(segments, key=lambda s: s.start):
            if seg.speaker_id not in speaker_map:
                speaker_map[seg.speaker_id] = f"Person {counter}"
                counter += 1
        return speaker_map


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    import sys
    import logging as _log
    _log.basicConfig(level=_log.INFO, format="%(levelname)s: %(message)s")

    print("=" * 60)
    print("SpeakerDiarizer — Self Test")
    print("=" * 60)

    config = {
        "diarization": {
            "model": "pyannote/speaker-diarization-3.1",
            "min_speakers": 1,
            "max_speakers": 6,
            "clustering_threshold": 0.75,
            "merge_gap_ms": 1500,
            "min_segment_duration_s": 1.0,
            "embedding_similarity_threshold": 0.82,
            "device": "mps" if torch.backends.mps.is_available() else "cpu",
        }
    }

    # Check HF token
    token = SpeakerDiarizer._load_hf_token()
    if token:
        print("✓ HuggingFace token found")
    else:
        print("✗ HuggingFace token not found")
        print("  Diarization requires accepting the pyannote user agreement:")
        print("  https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("  Then run: huggingface-cli login")

    # Check if pyannote is installed
    try:
        import pyannote.audio  # noqa
        print("✓ pyannote.audio installed")
    except ImportError:
        print("✗ pyannote.audio not installed")
        print("  Install: pip install pyannote.audio")

    # Find test audio
    test_file = None
    if len(sys.argv) > 1:
        test_file = sys.argv[1]
    else:
        import tempfile
        temp_dir = Path(tempfile.gettempdir()) / "lecture_intel"
        wavs = sorted(temp_dir.glob("*_converted.wav"))
        if wavs:
            test_file = str(wavs[0])

    if test_file and token:
        print(f"\nTesting with: {test_file}")
        diarizer = SpeakerDiarizer(config)
        try:
            result = diarizer.process(test_file)
            print(f"  Speakers: {result.speaker_count}")
            print(f"  Segments: {len(result.segments)}")
            print(f"  Map:      {result.speaker_map}")
            print(f"  Time:     {result.processing_time_ms:.0f}ms")
            for seg in result.segments[:5]:
                print(f"    [{seg.start:.1f}s–{seg.end:.1f}s] {seg.speaker_label}")
            if len(result.segments) > 5:
                print(f"    ... and {len(result.segments) - 5} more")
        except Exception as exc:
            print(f"  Failed: {exc}")
    else:
        print("\nSkipping full test (no audio file or no HF token)")

    print("\n=== Test Complete ===")
