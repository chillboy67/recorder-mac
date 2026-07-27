from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import subprocess
from app.config import settings
from app.models import PronunciationIssue, FluencyMetric
from app.services.text_utils import words


def analyze_pronunciation(audio_path: Path | None, transcript: str, work_dir: Path | None = None) -> tuple[list[PronunciationIssue], FluencyMetric, list[str]]:
    warnings: list[str] = []
    token_count = len(words(transcript))

    if audio_path is None:
        return [], FluencyMetric(note="No audio was provided; pronunciation feedback is skipped."), warnings

    if not settings.mfa_enabled:
        warnings.append("MFA is disabled; set MFA_ENABLED=true after installing Montreal Forced Aligner for phoneme-level diagnostics.")
        note = "Pronunciation feedback is limited until MFA alignment is enabled."
        return _heuristic_pronunciation(transcript), FluencyMetric(note=note), warnings

    if not work_dir or not shutil.which(settings.mfa_cli) or not settings.mfa_dictionary_path or not settings.mfa_acoustic_model:
        warnings.append("MFA is enabled, but the CLI, dictionary, or acoustic model is not configured; using pronunciation heuristics.")
        note = "MFA configuration is incomplete."
        return _heuristic_pronunciation(transcript), FluencyMetric(note=note, estimated_pause_count=max(0, token_count // 80)), warnings

    try:
        intervals = _run_mfa(audio_path, transcript, work_dir)
        return _issues_from_intervals(intervals), _fluency_from_intervals(intervals), warnings
    except Exception as exc:
        warnings.append(f"MFA alignment failed; using pronunciation heuristics. Details: {exc}")
        note = "MFA alignment failed; fallback diagnostics were used."
        return _heuristic_pronunciation(transcript), FluencyMetric(note=note, estimated_pause_count=max(0, token_count // 80)), warnings


def _heuristic_pronunciation(transcript: str) -> list[PronunciationIssue]:
    issues: list[PronunciationIssue] = []
    filler_words = {"um", "uh", "er", "erm"}
    for word in words(transcript):
        if word.lower() in filler_words:
            issues.append(
                PronunciationIssue(
                    word=word,
                    message="Frequent filler sounds can reduce fluency. Practice pausing silently instead.",
                    severity="low",
                )
            )
    if len(words(transcript)) < 20:
        issues.append(
            PronunciationIssue(
                word="overall",
                message="The answer is short; pronunciation scoring is less reliable without a longer sample.",
                severity="low",
            )
        )
    return issues


def _run_mfa(audio_path: Path, transcript: str, work_dir: Path) -> list[tuple[float, float, str]]:
    corpus_dir = work_dir / "mfa_corpus"
    output_dir = work_dir / "mfa_output"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "sample"
    _prepare_mfa_audio(audio_path, corpus_dir, stem)
    (corpus_dir / f"{stem}.lab").write_text(transcript, encoding="utf-8")
    command = [
        settings.mfa_cli,
        "align",
        str(corpus_dir),
        settings.mfa_dictionary_path,
        settings.mfa_acoustic_model,
        str(output_dir),
        "--clean",
        "--overwrite",
        "--single_speaker",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=_mfa_environment())
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip())
    textgrids = sorted(output_dir.rglob("*.TextGrid"))
    if not textgrids:
        raise RuntimeError("MFA did not produce a TextGrid file.")
    return _parse_word_intervals(textgrids[0])


def _prepare_mfa_audio(audio_path: Path, corpus_dir: Path, stem: str) -> Path:
    env = _mfa_environment()
    ffmpeg_cli = shutil.which("ffmpeg", path=env.get("PATH"))
    if not ffmpeg_cli:
        raise RuntimeError("ffmpeg is required to convert uploaded audio before MFA alignment.")

    wav_path = corpus_dir / f"{stem}.wav"
    command = [
        ffmpeg_cli,
        "-y",
        "-i",
        str(audio_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False, env=env)
    if completed.returncode != 0:
        raise RuntimeError(f"MFA audio conversion failed: {completed.stderr.strip() or completed.stdout.strip()}")
    return wav_path


def _mfa_environment() -> dict[str, str]:
    env = os.environ.copy()
    mfa_bin = str(Path(settings.mfa_cli).expanduser().resolve().parent)
    env["PATH"] = f"{mfa_bin}{os.pathsep}{env.get('PATH', '')}"
    return env


def _parse_word_intervals(textgrid_path: Path) -> list[tuple[float, float, str]]:
    text = textgrid_path.read_text(encoding="utf-8", errors="ignore")
    intervals: list[tuple[float, float, str]] = []
    pattern = re.compile(
        r"xmin = (?P<start>[0-9.]+)\s+xmax = (?P<end>[0-9.]+)\s+text = \"(?P<word>[^\"]*)\"",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        word = match.group("word").strip()
        if word:
            intervals.append((float(match.group("start")), float(match.group("end")), word))
    return intervals


def _issues_from_intervals(intervals: list[tuple[float, float, str]]) -> list[PronunciationIssue]:
    issues: list[PronunciationIssue] = []
    previous_end: float | None = None
    for start, end, word in intervals:
        duration = max(0.0, end - start)
        if previous_end is not None and start - previous_end > 0.9:
            issues.append(
                PronunciationIssue(
                    word=word,
                    start=previous_end,
                    end=start,
                    message="Long pause before this word may affect fluency and coherence.",
                    severity="medium",
                )
            )
        if len(word) <= 3 and duration > 0.75:
            issues.append(
                PronunciationIssue(
                    word=word,
                    start=start,
                    end=end,
                    message="This short word is unusually long in the alignment; check hesitation or vowel length.",
                    severity="low",
                )
            )
        if len(word) >= 7 and duration < 0.18:
            issues.append(
                PronunciationIssue(
                    word=word,
                    start=start,
                    end=end,
                    message="This longer word is very short in the alignment; check whether it was reduced or mispronounced.",
                    severity="medium",
                )
            )
        previous_end = end
    return issues


def _fluency_from_intervals(intervals: list[tuple[float, float, str]]) -> FluencyMetric:
    if not intervals:
        return FluencyMetric(note="MFA produced no word intervals.")
    duration_minutes = max((intervals[-1][1] - intervals[0][0]) / 60, 0.01)
    pause_count = sum(1 for previous, current in zip(intervals, intervals[1:]) if current[0] - previous[1] > 0.9)
    wpm = round(len(intervals) / duration_minutes, 1)
    return FluencyMetric(
        words_per_minute=wpm,
        estimated_pause_count=pause_count,
        note=f"MFA aligned {len(intervals)} words at approximately {wpm} WPM with {pause_count} long pauses.",
    )
