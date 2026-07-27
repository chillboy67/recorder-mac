from __future__ import annotations

import uuid
from pathlib import Path
from app.models import AnalysisResult, SpeakingPart
from app.services.grammar import check_grammar
from app.services.naturalness import suggest_naturalness, build_upgraded_answer
from app.services.pronunciation import analyze_pronunciation
from app.services.report import render_report
from app.services.text_utils import normalize_space
from app.services.transcription import transcribe_audio


async def analyze_text(
    text: str,
    template: str | None = None,
    job_id: str | None = None,
    assistant_notes: str | None = None,
    speaking_part: SpeakingPart = "part2",
) -> AnalysisResult:
    transcript = extract_student_speech(normalize_space(text))
    result = AnalysisResult(
        job_id=job_id or str(uuid.uuid4()),
        status="running",
        source="text",
        speaking_part=speaking_part,
        transcript=transcript,
        assistant_notes=normalize_space(assistant_notes or ""),
    )
    grammar, grammar_warnings = await check_grammar(transcript)
    result.grammar_issues = grammar
    result.warnings.extend(grammar_warnings)
    result.pronunciation_issues, result.fluency, pronunciation_warnings = analyze_pronunciation(None, transcript)
    result.warnings.extend(pronunciation_warnings)
    corrected = _apply_grammar_replacements(transcript, result.grammar_issues)
    result.natural_suggestions = suggest_naturalness(corrected)
    result.upgraded_answer = build_upgraded_answer(corrected)
    result.status = "complete"
    result.report_markdown = render_report(result, template)
    return result


async def analyze_audio(
    audio_path: Path,
    supplied_text: str | None,
    work_dir: Path,
    template: str | None = None,
    job_id: str | None = None,
    assistant_notes: str | None = None,
    speaking_part: SpeakingPart = "part2",
) -> AnalysisResult:
    result = AnalysisResult(
        job_id=job_id or str(uuid.uuid4()),
        status="running",
        source="audio",
        speaking_part=speaking_part,
        assistant_notes=normalize_space(assistant_notes or ""),
    )
    transcript = normalize_space(supplied_text or "")
    if not transcript:
        transcript, transcription_warnings = transcribe_audio(audio_path, work_dir)
        result.warnings.extend(transcription_warnings)
    transcript = extract_student_speech(transcript)
    result.transcript = transcript
    if not transcript:
        result.status = "failed"
        result.error = "No transcript is available. Install Whisper locally or provide a transcript with the audio."
        return result

    grammar, grammar_warnings = await check_grammar(transcript)
    result.grammar_issues = grammar
    result.warnings.extend(grammar_warnings)
    result.pronunciation_issues, result.fluency, pronunciation_warnings = analyze_pronunciation(audio_path, transcript, work_dir)
    result.warnings.extend(pronunciation_warnings)
    corrected = _apply_grammar_replacements(transcript, result.grammar_issues)
    result.natural_suggestions = suggest_naturalness(corrected)
    result.upgraded_answer = build_upgraded_answer(corrected)
    result.status = "complete"
    result.report_markdown = render_report(result, template)
    return result


def extract_student_speech(text: str) -> str:
    lines = [line.strip() for line in text.replace("\r", "\n").split("\n") if line.strip()]
    if len(lines) > 1:
        student_lines = []
        for line in lines:
            lowered = line.lower().lstrip("-:： ")
            if lowered.startswith(("student:", "student：", "学生:", "学生：", "s:")):
                student_lines.append(line.split(":", 1)[-1].split("：", 1)[-1].strip())
            elif lowered.startswith(("teacher:", "teacher：", "助教:", "助教：", "tutor:", "tutor：", "examiner:", "q:")):
                continue
            elif not _looks_like_non_student_turn(line):
                student_lines.append(line)
        if student_lines:
            return normalize_space(" ".join(student_lines))
    sentences = []
    for sentence in _rough_sentences(text):
        if not _looks_like_non_student_turn(sentence):
            sentences.append(sentence)
    return normalize_space(" ".join(sentences) or text)


def _rough_sentences(text: str) -> list[str]:
    import re

    return [part.strip() for part in re.split(r"(?<=[.!?？。])\s+", text) if part.strip()]


def _looks_like_non_student_turn(text: str) -> bool:
    lowered = text.lower().strip()
    question_starters = ("what ", "why ", "how ", "when ", "where ", "who ", "do you", "did you", "can you", "could you", "would you", "tell me", "describe ")
    correction_cues = ("you should", "you need to", "注意", "建议", "可以改", "应该", "不需要", "直接用", "纠正")
    return lowered.endswith(("?", "？")) or lowered.startswith(question_starters) or any(cue in lowered for cue in correction_cues)


def _apply_grammar_replacements(text: str, issues) -> str:
    corrected = text
    for issue in issues:
        if issue.replacement:
            corrected = _apply_known_replacement(corrected, issue.replacement)
            context = issue.context.strip()
            if issue.replacement.count(" ") and context and context in corrected:
                corrected = corrected.replace(context, context.replace(_find_problem_span(context, issue.replacement), issue.replacement), 1)
    return corrected


def _find_problem_span(context: str, replacement: str) -> str:
    replacement_words = replacement.split()
    if not replacement_words:
        return context
    first = replacement_words[0].lower()
    words = context.split()
    for index, word in enumerate(words):
        if word.lower().strip(".,!?") == first and index + len(replacement_words) <= len(words):
            return " ".join(words[index : index + len(replacement_words)])
    return context


def _apply_known_replacement(text: str, replacement: str) -> str:
    known = {
        "He has": ("He have", "he have"),
        "She has": ("She have", "she have"),
        "It has": ("It have", "it have"),
        "I have": ("I has", "i has"),
        "advice": ("advices",),
        "information": ("informations",),
        "discuss": ("discuss about",),
    }
    for problem in known.get(replacement, ()):
        text = text.replace(problem, replacement)
    return text
