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
from core import provenance
from core.i18n import t
from core.languages import normalize_language
from core.modes import Mode, get_mode
from core.transcriber import TEMPERATURE_LADDER, Transcriber

logger = logging.getLogger(__name__)

ProgressCB = Callable[[dict], None]


def run(
    input_path: str | Path,
    output_dir: str | Path,
    mode_key: str = "general",
    *,
    model: str = "auto",
    engine: Optional[str] = None,   # None → use the mode's preferred engine
    language: Optional[str] = None,  # None → mode default; "auto"/"" → detect per chunk
    initial_prompt: Optional[str] = None,
    formats: Optional[list[str]] = None,
    languagetool_url: str = "http://127.0.0.1:8010/v2/check",
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
    original = provenance.archive_input(input_path, output_dir)

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
    # Transcription is the long, real wait; it starts at 30% so the ring rests
    # there rather than near the bottom, and its progress spans 30 → 75%.
    report("asr", t("eng_asr_start"), 30)

    # A caller-supplied language overrides the mode preset; "auto"/"" normalize
    # to None, which re-enables per-chunk detection on mixed-language audio.
    # `is not None` (not `or`) so an explicit "auto" wins over a mode preset.
    lang = normalize_language(language) if language is not None else mode.language
    if lang is not None:
        logger.info("Language pinned to %s; per-chunk language detection disabled", lang)

    def asr_progress(frac, msg):
        report("asr", msg, 30 + int(frac * 45))

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
    actual_engine, _, actual_model = asr.model_used.partition(":")
    provenance.append_meta(output_dir, {
        "name": "transcribe",
        "input_sha256": provenance.sha256_file(wav_path),
        "params": {"requested_model": model,
                   "model": actual_model or asr.model_used,
                   "engine": actual_engine or (engine or mode.engine),
                   "language": lang,
                   "temperature_ladder": list(TEMPERATURE_LADDER),
                   "condition_on_previous": mode.condition_on_previous},
        "output": {"segments": len(asr.segments), "language": asr.language},
    })

    # 3b) Repeat arbitration: stutter vs ASR loop — annotate everywhere, fold
    # only confirmed loops in classroom mode (original always kept).
    from core.repeat_arbitration import (adjacent_duplicate_segments,
                                         apply_verdicts, arbitrate)
    # L3 re-hears repeat runs against the PRE-DENOISE original: denoise can
    # induce decode loops, and re-hearing the same denoised audio only
    # reproduces the artifact. Real stutter is in the original acoustics.
    verdicts = arbitrate(asr, wav_path, transcriber, oracle_path=original)
    apply_verdicts(asr, verdicts, mode.key)
    # Adjacent duplicate segments (a loop that spans segment boundaries):
    # classroom keeps the old drop-the-copy behaviour (recorded in meta);
    # general/IELTS only annotate — the text stays in the transcript.
    dups = adjacent_duplicate_segments(asr)
    dropped_segments: list[dict] = []
    if dups:
        if mode.key == "classroom":
            drop_ids = {d["segment_id"] for d in dups}
            asr.segments = [s for s in asr.segments if s.id not in drop_ids]
            for i, s in enumerate(asr.segments):
                s.id = i
            asr.full_text = " ".join(s.text for s in asr.segments).strip()
            dropped_segments = dups
            provenance.append_meta(output_dir, {
                "name": "drop_duplicate_segments", "dropped": dups})
        else:
            asr.annotations.extend(
                {"type": "adjacent_duplicate_segment", **d} for d in dups)
    # The fidelity audit trail lives in meta.json too, not only the json
    # export — general/IELTS modes don't export json by default.
    provenance.append_meta(output_dir, {
        "name": "annotations", "annotations": asr.annotations,
        "oracle_audio": str(original)})

    labels: Optional[dict[int, str]] = None
    ielts_report = None
    classroom_report = None
    classroom_md: Optional[str] = None

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
        report("analyze", t("eng_analyze_done"), 94, "done")

    # 5b) Classroom: key-point summary from the transcript
    if mode.summarize:
        from core import classroom as classroom_mod
        classroom_report = classroom_mod.summarize(asr)
        classroom_md = classroom_report.markdown
        report("analyze", t("eng_extract_done"), 94, "done")

    # 6) Export --------------------------------------------------------------
    report("export", t("eng_export_start"), 96)
    fmts = formats or mode.formats
    if ielts_report:
        extra_md, extra_suffix = ielts_report.markdown, "ielts"
    elif classroom_md:
        extra_md, extra_suffix = classroom_md, "summary"
    else:
        extra_md, extra_suffix = None, "report"
    outputs = exporter.export_all(
        asr, output_dir, base, fmts,
        labels=labels,
        extra_markdown=extra_md,
        extra_markdown_suffix=extra_suffix,
        dropped_segments=dropped_segments,
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
                       "emphasis_count": len(classroom_report.emphasis_points),
                       "definition_count": len(classroom_report.definitions)}
                      if classroom_report else None),
        # The fidelity audit trail used to stop at meta.json, which meant the
        # GUI could never show it (general/IELTS don't even export json by
        # default). Surfacing it in the summary is what puts it on screen.
        "annotations": asr.annotations,
        "fidelity": _fidelity_summary(asr, dropped_segments),
    }
    logger.info("Engine done in %.1fs (%s)", elapsed, mode.key)
    return summary


def _fidelity_summary(asr, dropped_segments: list[dict]) -> dict:
    """Counts + the ready-to-render lines for the fidelity audit trail.

    `lines` is produced by ``core.export.fidelity_lines`` — the same function
    that writes the .md/.txt tail, so the screen and the file always agree.
    The full record stays in ``asr.annotations`` and meta.json.
    """
    repeats = [a for a in asr.annotations
               if a.get("type") == "repeat_arbitration"]
    folded = [a for a in repeats if a.get("folded")]
    return {
        "asr_loop": sum(1 for a in repeats if a.get("verdict") == "asr_loop"),
        "real_speech": sum(1 for a in repeats
                           if a.get("verdict") == "real_speech"),
        "uncertain": sum(1 for a in repeats if a.get("verdict") == "uncertain"),
        "adjacent_duplicates": sum(
            1 for a in asr.annotations
            if a.get("type") == "adjacent_duplicate_segment"),
        "folded_count": len(folded) + len(dropped_segments),
        "dropped_count": len(dropped_segments),
        "total": len(asr.annotations) + len(dropped_segments),
        "lines": exporter.fidelity_lines(asr, dropped_segments),
    }


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
