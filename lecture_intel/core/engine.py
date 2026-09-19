"""
Engine — the single clean orchestrator.

Replaces the old 11-step `pipeline.py` (course classification, LLM rewriting,
lecture structuring …) with a focused, mode-driven flow:

    audio → [denoise] → transcribe(whole file) → [diarize / main-speaker]
          → [IELTS analysis] → export

The GUI and CLI both call `run(...)`. Progress is reported through a callback
so the UI stays responsive.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable, Optional

from modules import ASRResult
from modules.audio_loader import AudioLoader

from core import export as exporter
from core import llm as llm_mod
from core import provenance
from core.i18n import t
from core.languages import normalize_language, prefers_asian_model
from core.modes import Mode, get_mode
from core.transcriber import TEMPERATURE_LADDER, Transcriber

logger = logging.getLogger(__name__)

ProgressCB = Callable[[dict], None]


def run(
    input_path: str | Path,
    output_dir: str | Path,
    mode_key: str = "general",
    *,
    model: str = "large-v3",
    engine: Optional[str] = None,   # None → use the mode's preferred engine
    language: Optional[str] = None,  # None → mode default; "auto"/"" → detect per chunk
    initial_prompt: Optional[str] = None,
    formats: Optional[list[str]] = None,
    languagetool_url: str = "http://127.0.0.1:8010/v2/check",
    use_llm: bool = False,
    llm_model: str = llm_mod.DEFAULT_MODEL,            # head of the non-Chinese candidates
    chinese_model: str = llm_mod.DEFAULT_CHINESE_MODEL,  # head of the Chinese candidates
    progress: Optional[ProgressCB] = None,
) -> dict:
    """Run the full pipeline for one file. Returns a result summary dict."""
    mode = get_mode(mode_key)
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    base = input_path.stem
    t_start = time.time()
    warnings: list[str] = []

    def report(step, message, percent, status="running"):
        if progress:
            progress({"step": step, "message": message,
                      "percent": percent, "status": status})

    # 0) Archive the input for provenance -----------------------------------
    # The output dir becomes self-contained evidence: original.wav (capture-faithful,
    # never modified) + meta.json tracing every later transformation.
    provenance.archive_input(input_path, output_dir)

    # 1) Load + normalize to 16k mono wav -----------------------------------
    report("load", t("eng_load_start"), 4)
    loader = AudioLoader({})
    audio = loader.process(input_path)
    wav_path = audio.path
    provenance.append_meta(output_dir, {
        "name": "normalize",
        "params": {"sample_rate": 16000, "channels": 1, "format": "wav"},
        "input_sha256": provenance.sha256_file(input_path),
        "output_sha256": provenance.sha256_file(wav_path),
    })
    report("load", t("eng_load_done"), 8, "done")

    # 2) Denoise (classroom, gated) ------------------------------------------
    if mode.denoise:
        report("denoise", t("eng_denoise_start"), 12)
        from core import denoise as denoise_mod
        floor = denoise_mod.measure_noise_floor(wav_path)
        # None (measure failed) → default to cleaning, as before the gate existed.
        decided = floor is None or floor > mode.denoise_noise_floor_db
        cleaned = denoise_mod.denoise_audio(wav_path) if decided else wav_path
        provenance.append_meta(output_dir, {
            "name": "denoise",
            "input_sha256": provenance.sha256_file(wav_path),
            "params": {"filter": denoise_mod.FILTER_CHAIN,
                       "noise_floor_db": floor,
                       "threshold_db": mode.denoise_noise_floor_db,
                       "applied": cleaned != wav_path},
            "output_sha256": provenance.sha256_file(cleaned),
        })
        wav_path = cleaned
        report("denoise", t("eng_denoise_done"), 16, "done")

    # 3) Transcribe (whole file) --------------------------------------------
    report("asr", t("eng_asr_start"), 20)

    # A caller-supplied language overrides the mode preset; "auto"/"" normalize
    # to None, which re-enables per-chunk detection on mixed-language audio.
    # `is not None` (not `or`) so an explicit "auto" wins over a mode preset.
    lang = normalize_language(language) if language is not None else mode.language
    if lang is not None:
        logger.info("Language pinned to %s; per-chunk language detection disabled", lang)

    def asr_progress(frac, msg):
        report("asr", msg, 20 + int(frac * 55))

    transcriber = Transcriber(model=model, engine=engine or mode.engine)
    asr: ASRResult = transcriber.transcribe(
        wav_path,
        language=lang,
        initial_prompt=(initial_prompt if initial_prompt is not None else mode.initial_prompt),
        condition_on_previous=mode.condition_on_previous,
        chunked=mode.chunked_language,
        chunk_sec=mode.chunk_sec,
        duration_sec=audio.duration_sec,
        progress=asr_progress,
    )
    warnings.extend(asr.warnings)
    report("asr", t("eng_asr_done", count=len(asr.segments)), 76, "done")
    provenance.append_meta(output_dir, {
        "name": "transcribe",
        "input_sha256": provenance.sha256_file(wav_path),
        "params": {"model": model, "engine": engine or mode.engine,
                   "language": lang,
                   "temperature_ladder": list(TEMPERATURE_LADDER),
                   "condition_on_previous": mode.condition_on_previous},
        "output": {"segments": len(asr.segments), "language": asr.language},
    })

    labels: Optional[dict[int, str]] = None
    ielts_report = None
    classroom_report = None
    classroom_md: Optional[str] = None
    general_tidy_md: Optional[str] = None

    # Is the local LLM usable, and which one? Route by the detected language:
    # CJK audio → the Chinese candidates; everything else → the multilingual
    # ones. Neither role is a single hardcoded name: the caller's preferred
    # model heads the list, then each role degrades through whatever else the
    # user happens to have pulled, so installing "the wrong" family still works.
    llm_on = False
    if use_llm:
        asian = prefers_asian_model(asr.language)
        head = chinese_model if asian else llm_model
        own = llm_mod.CHINESE_CANDIDATES if asian else llm_mod.NON_CHINESE_CANDIDATES
        other = llm_mod.NON_CHINESE_CANDIDATES if asian else llm_mod.CHINESE_CANDIDATES
        resolved = llm_mod.resolve_first((head,) + own + other)
        if resolved:
            llm_model = resolved
            llm_on = True
            logger.info("LLM on: language=%s → model=%s", asr.language, llm_model)
        else:
            warnings.append(t("eng_llm_unavailable"))
            logger.warning("use_llm requested but no Ollama model available")

    # 4) Speaker handling ----------------------------------------------------
    if mode.diarize:
        report("diarize", t("eng_diarize_start"), 80)
        from core.diarize import diarize as run_diarize
        dia = run_diarize(asr, wav_path, expected_speakers=mode.expected_speakers)
        labels = dia.labels
        report("diarize", t("eng_diarize_done", count=dia.speaker_count), 85, "done")
    elif mode.keep_main_speaker_only:
        report("diarize", t("eng_main_start"), 80)
        from core.diarize import main_speaker_ids
        kept = main_speaker_ids(asr, wav_path)
        if kept is not None:
            before = len(asr.segments)
            removed = [s for s in asr.segments if s.id not in kept]
            asr.segments = [s for s in asr.segments if s.id in kept]
            asr.full_text = " ".join(s.text for s in asr.segments).strip()
            logger.info("Classroom: %d → %d segments after main-speaker filter",
                        before, len(asr.segments))
            # Dropped speech must not vanish silently: every removed interval
            # lands in meta.json with its text, so the edit is reviewable.
            provenance.append_meta(output_dir, {
                "name": "keep_main_speaker_only",
                "kept_segment_ids": sorted(kept),
                "removed": [{"segment_id": s.id, "start": round(s.start, 3),
                             "end": round(s.end, 3),
                             "speaker_id": s.speaker_id, "text": s.text}
                            for s in removed]})
        report("diarize", t("eng_main_done"), 85, "done")

    # 5) IELTS analysis ------------------------------------------------------
    if mode.analyze_ielts:
        report("analyze", t("eng_analyze_start"), 88)
        from core import ielts as ielts_mod
        dia_obj = dia if mode.diarize else None
        ielts_report = ielts_mod.analyze(
            asr,
            labels or {},
            candidate_seconds=getattr(dia_obj, "candidate_seconds", audio.duration_sec),
            examiner_seconds=getattr(dia_obj, "examiner_seconds", 0.0),
            other_seconds=getattr(dia_obj, "other_seconds", 0.0),
            languagetool_url=languagetool_url,
        )
        # LLM-enhanced examiner-style feedback on the candidate's English.
        if llm_on:
            report("analyze", t("eng_llm_feedback"), 91)
            fb = llm_mod.ielts_feedback(ielts_report.transcript_candidate, model=llm_model)
            if fb:
                ielts_report.markdown += (
                    "\n\n---\n\n# 🤖 AI 考官点评（本地大模型）\n\n" + fb + "\n")
        report("analyze", t("eng_analyze_done"), 94, "done")

    # 5b) Classroom: AI-corrected transcript (context-aware) + key-point summary
    if mode.summarize:
        from core import classroom as classroom_mod
        classroom_report = classroom_mod.summarize(asr)   # always have a fallback
        if llm_on:
            report("analyze", t("eng_correct_start"), 90)
            def _corr_prog(frac, msg):
                report("analyze", msg, 88 + int(frac * 4))
            corrected = llm_mod.correct_transcript(
                asr.full_text, context=_lecture_context(asr.language),
                language=asr.language, model=llm_model,
                progress=_corr_prog)
            report("analyze", t("eng_extract_start"), 93)
            summary_md = llm_mod.summarize_lecture(
                corrected or asr.full_text, language=asr.language, model=llm_model)
            parts = ["# 课堂重点总结（本地大模型）", ""]
            if summary_md:
                parts.append(summary_md)
            parts += ["", "## 校对后全文（AI 据上课内容判断）", "", corrected or asr.full_text]
            classroom_md = "\n".join(parts)
        if classroom_md is None:
            classroom_md = classroom_report.markdown   # heuristic fallback
        report("analyze", t("eng_extract_done"), 94, "done")

    # 5c) General: AI determines the true transcript (accuracy-focused correction)
    if (not mode.analyze_ielts and not mode.summarize) and llm_on:
        report("analyze", t("eng_general_correct"), 90)
        def _corr_prog(frac, msg):
            report("analyze", msg, 88 + int(frac * 6))
        general_tidy_md = llm_mod.correct_transcript(
            asr.full_text, language=asr.language, model=llm_model, progress=_corr_prog)
        report("analyze", t("eng_general_done"), 94, "done")

    # 6) Export --------------------------------------------------------------
    report("export", t("eng_export_start"), 96)
    fmts = formats or mode.formats
    if ielts_report:
        extra_md, extra_suffix = ielts_report.markdown, "ielts"
    elif classroom_md:
        extra_md, extra_suffix = classroom_md, "summary"
    elif general_tidy_md:
        extra_md, extra_suffix = ("# AI 校对版（本地大模型）\n\n" + general_tidy_md), "corrected"
    else:
        extra_md, extra_suffix = None, "report"
    outputs = exporter.export_all(
        asr, output_dir, base, fmts,
        labels=labels,
        extra_markdown=extra_md,
        extra_markdown_suffix=extra_suffix,
    )
    report("export", t("eng_export_done"), 100, "done")
    provenance.append_meta(output_dir, {
        "name": "export",
        "files": {k: {"file": v.name, "sha256": provenance.sha256_file(v)}
                  for k, v in outputs.items()},
    })

    elapsed = time.time() - t_start
    summary = {
        "mode": mode.key,
        "language": asr.language,
        "model": asr.model_used,
        "duration_sec": audio.duration_sec,
        "segment_count": len(asr.segments),
        "total_time_s": elapsed,
        "output_files": {k: str(v) for k, v in outputs.items()},
        "warnings": warnings,
        "ielts": _ielts_summary(ielts_report) if ielts_report else None,
        "classroom": ({"markdown": classroom_md,
                       "llm": bool(llm_on),
                       "emphasis_count": len(classroom_report.emphasis_points),
                       "definition_count": len(classroom_report.definitions)}
                      if classroom_report else None),
        "tidy_markdown": ("# AI 校对版（本地大模型）\n\n" + general_tidy_md) if general_tidy_md else None,
        "llm_used": bool(llm_on),
    }
    logger.info("Engine done in %.1fs (%s)", elapsed, mode.key)
    return summary


def _ielts_summary(r) -> dict:
    return {
        "wpm": r.words_per_minute,
        "filler_count": r.filler_count,
        "long_pause_count": r.long_pause_count,
        "pron_issue_count": len(r.pron_issues),
        "grammar_issue_count": len(r.grammar_issues),
        "naturalness_count": len(r.naturalness),
        "markdown": r.markdown,
    }


def _lecture_context(language: str) -> str:
    """Background hint for the correction prompt, in the transcript's language
    (code-switching transcripts keep the Chinese hint, matching their prompt)."""
    if not language or language in ("zh", "mixed"):
        return "一节课的课堂录音"
    return "a classroom lecture recording"
