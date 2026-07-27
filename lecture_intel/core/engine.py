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
from core.modes import Mode, get_mode
from core.transcriber import Transcriber

logger = logging.getLogger(__name__)

ProgressCB = Callable[[dict], None]


def run(
    input_path: str | Path,
    output_dir: str | Path,
    mode_key: str = "general",
    *,
    model: str = "large-v3",
    engine: Optional[str] = None,   # None → use the mode's preferred engine
    initial_prompt: Optional[str] = None,
    formats: Optional[list[str]] = None,
    languagetool_url: str = "http://127.0.0.1:8010/v2/check",
    use_llm: bool = False,
    llm_model: str = "llama3.1:8b",        # English model
    chinese_model: str = "qwen-zh:7b",     # Chinese model (uncensored Qwen2.5)
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

    # 1) Load + normalize to 16k mono wav -----------------------------------
    report("load", "加载音频…", 4)
    loader = AudioLoader({})
    audio = loader.process(input_path)
    wav_path = audio.path
    report("load", "音频已加载", 8, "done")

    # 2) Denoise (classroom) -------------------------------------------------
    if mode.denoise:
        report("denoise", "降噪处理…", 12)
        from core.denoise import denoise_audio
        wav_path = denoise_audio(wav_path)
        report("denoise", "降噪完成", 16, "done")

    # 3) Transcribe (whole file) --------------------------------------------
    report("asr", "转写中（首次会下载模型，请耐心等待）…", 20)

    def asr_progress(frac, msg):
        report("asr", msg, 20 + int(frac * 55))

    transcriber = Transcriber(model=model, engine=engine or mode.engine)
    asr: ASRResult = transcriber.transcribe(
        wav_path,
        language=mode.language,
        initial_prompt=(initial_prompt if initial_prompt is not None else mode.initial_prompt),
        condition_on_previous=mode.condition_on_previous,
        chunked=mode.chunked_language,
        chunk_sec=mode.chunk_sec,
        duration_sec=audio.duration_sec,
        progress=asr_progress,
    )
    warnings.extend(asr.warnings)
    report("asr", f"转写完成（{len(asr.segments)} 段）", 76, "done")

    labels: Optional[dict[int, str]] = None
    ielts_report = None
    classroom_report = None
    classroom_md: Optional[str] = None
    general_tidy_md: Optional[str] = None

    # Is the local LLM usable, and which one? Route by the detected language:
    # Chinese audio → Chinese model; English → English model. Fall back to
    # whichever is installed if the preferred one isn't.
    llm_on = False
    if use_llm:
        from core import llm as llm_mod
        prefer = chinese_model if asr.language in ("zh", "mixed") else llm_model
        alt = llm_model if prefer == chinese_model else chinese_model
        resolved = (llm_mod.resolve_model(prefer)
                    or llm_mod.resolve_model(alt))
        if resolved:
            llm_model = resolved
            llm_on = True
            logger.info("LLM on: language=%s → model=%s", asr.language, llm_model)
        else:
            warnings.append("已勾选本地大模型，但 Ollama 未运行或模型未安装，已回退离线处理。")
            logger.warning("use_llm requested but no Ollama model available")

    # 4) Speaker handling ----------------------------------------------------
    if mode.diarize:
        report("diarize", "区分说话人…", 80)
        from core.diarize import diarize as run_diarize
        dia = run_diarize(asr, wav_path, expected_speakers=mode.expected_speakers)
        labels = dia.labels
        report("diarize", f"识别到 {dia.speaker_count} 个说话人", 85, "done")
    elif mode.keep_main_speaker_only:
        report("diarize", "聚焦主讲人，排除旁人…", 80)
        from core.diarize import main_speaker_ids
        kept = main_speaker_ids(asr, wav_path)
        if kept is not None:
            before = len(asr.segments)
            asr.segments = [s for s in asr.segments if s.id in kept]
            asr.full_text = " ".join(s.text for s in asr.segments).strip()
            logger.info("Classroom: %d → %d segments after main-speaker filter",
                        before, len(asr.segments))
        report("diarize", "已聚焦主讲人", 85, "done")

    # 5) IELTS analysis ------------------------------------------------------
    if mode.analyze_ielts:
        report("analyze", "分析发音 / 语法 / 表达…", 88)
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
            report("analyze", "大模型点评中…", 91)
            fb = llm_mod.ielts_feedback(ielts_report.transcript_candidate, model=llm_model)
            if fb:
                ielts_report.markdown += (
                    "\n\n---\n\n# 🤖 AI 考官点评（本地大模型）\n\n" + fb + "\n")
        report("analyze", "分析完成", 94, "done")

    # 5b) Classroom: AI-corrected transcript (context-aware) + key-point summary
    if mode.summarize:
        from core import classroom as classroom_mod
        classroom_report = classroom_mod.summarize(asr)   # always have a fallback
        if llm_on:
            report("analyze", "AI 根据上课内容校对原文…", 90)
            def _corr_prog(frac, msg):
                report("analyze", msg, 88 + int(frac * 4))
            corrected = llm_mod.correct_transcript(
                asr.full_text, context="一节课的课堂录音", model=llm_model,
                progress=_corr_prog)
            report("analyze", "AI 提炼重点…", 93)
            summary_md = llm_mod.summarize_lecture(corrected or asr.full_text, model=llm_model)
            parts = ["# 课堂重点总结（本地大模型）", ""]
            if summary_md:
                parts.append(summary_md)
            parts += ["", "## 校对后全文（AI 据上课内容判断）", "", corrected or asr.full_text]
            classroom_md = "\n".join(parts)
        if classroom_md is None:
            classroom_md = classroom_report.markdown   # heuristic fallback
        report("analyze", "重点提取完成", 94, "done")

    # 5c) General: AI determines the true transcript (accuracy-focused correction)
    if (not mode.analyze_ielts and not mode.summarize) and llm_on:
        report("analyze", "AI 校对全文（提高准确性）…", 90)
        def _corr_prog(frac, msg):
            report("analyze", msg, 88 + int(frac * 6))
        general_tidy_md = llm_mod.correct_transcript(
            asr.full_text, model=llm_model, progress=_corr_prog)
        report("analyze", "整理完成", 94, "done")

    # 6) Export --------------------------------------------------------------
    report("export", "导出结果…", 96)
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
    report("export", "完成", 100, "done")

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
