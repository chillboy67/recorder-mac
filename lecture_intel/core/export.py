"""
Output writers: txt / md / srt / json, with optional speaker labels.

Self-contained (doesn't depend on the old Exporter's LLM coupling). Timestamps
are always available; speaker labels are emitted only when diarization ran.
"""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Optional

from modules import ASRResult, ASRSegment

from core.i18n import t

# Speaker label → i18n key; exported files follow the UI language.
_SPEAKER_KEYS = {
    "candidate": "exp_speaker_candidate",
    "examiner": "exp_speaker_examiner",
    "other": "exp_speaker_other",
    "main": "exp_speaker_main",
}


# ── Fidelity annotations ─────────────────────────────────────────────────
# The transcript body is never rewritten by this section. It is a report
# *about* the transcript: which spans the three-level arbitration flagged, what
# it decided, and on what evidence. Nothing here is part of the utterance.
_VERDICT_KEYS = {
    "asr_loop": "exp_verdict_asr_loop",
    "real_speech": "exp_verdict_real_speech",
    "uncertain": "exp_verdict_uncertain",
}


def _evidence_brief(ev: dict) -> str:
    """One short, checkable justification drawn from the arbitration evidence."""
    if not ev:
        return ""
    level = ev.get("level")
    if level == 1:
        return t("exp_ev_l1", span=ev.get("span_sec"), expected=ev.get("expected_sec"))
    if level == 2:
        return t("exp_ev_l2", bursts=ev.get("voiced_bursts"), k=ev.get("k"))
    if level == 3:
        where = t("exp_ev_original" if ev.get("oracle") == "original" else "exp_ev_pipeline")
        seeds = ev.get("seeds")
        if seeds is None:
            return t("exp_ev_l3_failed", where=where)
        return t("exp_ev_l3", where=where, seeds=seeds, k=ev.get("k"))
    return str(ev.get("reason") or "")


def fidelity_lines(asr, dropped_segments: Optional[list] = None) -> list[str]:
    """Every fidelity annotation as one readable line.

    Shared by the exported .md/.txt tail and the GUI tab so the file and the
    screen can never drift apart. Qt-free on purpose.
    """
    out: list[str] = []
    for a in asr.annotations:
        kind = a.get("type")
        ts = f"{_ts_short(a.get('start', 0))}–{_ts_short(a.get('end', 0))}"
        if kind == "repeat_arbitration":
            key = _VERDICT_KEYS.get(a.get("verdict"))
            verdict = t(key) if key else str(a.get("verdict", "?"))
            state = t("exp_state_folded" if a.get("folded") else "exp_state_kept")
            ev = _evidence_brief(a.get("evidence") or {})
            line = t("exp_repeat_line", ts=ts, verdict=verdict, state=state,
                     original=a.get("original", ""))
            out.append(line + t("exp_evidence", evidence=ev) if ev else line)
        elif kind == "adjacent_duplicate_segment":
            out.append(t("exp_dup_kept", ts=ts, text=a.get("text", "")))
    for d in (dropped_segments or []):
        ts = f"{_ts_short(d.get('start', 0))}–{_ts_short(d.get('end', 0))}"
        out.append(t("exp_dup_dropped", ts=ts, text=d.get("text", "")))
    return out


def export_all(
    asr: ASRResult,
    out_dir: Path,
    base: str,
    formats: list[str],
    labels: Optional[dict[int, str]] = None,
    extra_markdown: Optional[str] = None,
    extra_markdown_suffix: str = "report",
    dropped_segments: Optional[list] = None,
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        # probe writability (macOS TCC may block protected folders)
        probe = out_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except (PermissionError, OSError):
        from core.paths import data_root
        fallback = data_root() / base
        fallback.mkdir(parents=True, exist_ok=True)
        out_dir = fallback

    outputs: dict[str, Path] = {}

    for fmt in formats:
        try:
            if fmt == "txt":
                outputs["txt"] = _txt(asr, out_dir, base, labels, dropped_segments)
            elif fmt == "md":
                outputs["md"] = _md(asr, out_dir, base, labels, dropped_segments)
            elif fmt == "docx":
                outputs["docx"] = _docx(asr, out_dir, base, labels, extra_markdown)
            elif fmt == "doc":
                outputs["doc"] = _doc(asr, out_dir, base, labels, extra_markdown)
            elif fmt == "srt":
                outputs["srt"] = _srt(asr, out_dir, base, labels)
            elif fmt == "json":
                outputs["json"] = _json(asr, out_dir, base, labels)
        except Exception as exc:  # one bad writer shouldn't sink the rest
            import logging
            logging.getLogger(__name__).warning("export %s failed: %s", fmt, exc)

    if extra_markdown:
        p = out_dir / f"{base}.{extra_markdown_suffix}.md"
        p.write_text(extra_markdown, encoding="utf-8")
        outputs["report"] = p

    return outputs


def _label(labels, seg_id) -> str:
    if not labels:
        return ""
    raw = labels.get(seg_id)
    if not raw:
        return ""
    key = _SPEAKER_KEYS.get(raw)
    return t(key) if key else raw


def _txt(asr, out_dir, base, labels, dropped_segments=None) -> Path:
    lines = []
    for s in asr.segments:
        ts = _ts_short(s.start)
        spk = _label(labels, s.id)
        prefix = f"[{ts}] " + (f"{spk}: " if spk else "")
        lines.append(prefix + s.text)
    body = "\n".join(lines) if lines else asr.full_text
    fidelity = fidelity_lines(asr, dropped_segments)
    if fidelity:
        # Appended after the transcript, never interleaved into it.
        body += "\n\n" + "\n".join(
            [t("exp_fidelity_heading"), "", t("exp_fidelity_note"), ""] + fidelity)
    p = out_dir / f"{base}.txt"
    p.write_text(body, encoding="utf-8")
    return p


def _md(asr, out_dir, base, labels, dropped_segments=None) -> Path:
    L = [f"# {base}", "", t("exp_md_language", language=asr.language),
         t("exp_md_engine", engine=asr.model_used),
         t("exp_md_duration", seconds=f"{asr.audio_duration_sec:.0f}"),
         "", f"## {t('exp_md_transcript')}", ""]
    for s in asr.segments:
        ts = _ts_short(s.start)
        spk = _label(labels, s.id)
        if spk:
            L.append(f"**{spk}** `[{ts}]` {s.text}")
        else:
            L.append(f"`[{ts}]` {s.text}")
        L.append("")
    fidelity = fidelity_lines(asr, dropped_segments)
    if fidelity:
        L.extend(["", f"## {t('exp_fidelity_heading')}", "", f"> {t('exp_fidelity_note')}", ""])
        L.extend(f"- {line}" for line in fidelity)
        L.append("")
    p = out_dir / f"{base}.md"
    p.write_text("\n".join(L), encoding="utf-8")
    return p


def _srt(asr, out_dir, base, labels) -> Path:
    blocks = []
    for i, s in enumerate(asr.segments, 1):
        spk = _label(labels, s.id)
        text = (f"{spk}: " if spk else "") + s.text
        blocks.append(f"{i}\n{_ts_srt(s.start)} --> {_ts_srt(s.end)}\n{text}\n")
    p = out_dir / f"{base}.srt"
    p.write_text("\n".join(blocks), encoding="utf-8")
    return p


def _json(asr, out_dir, base, labels) -> Path:
    data = {
        "language": asr.language,
        "model": asr.model_used,
        "duration_sec": asr.audio_duration_sec,
        "full_text": asr.full_text,
        "segments": [
            {
                "id": s.id,
                "start": round(s.start, 3),
                "end": round(s.end, 3),
                "speaker": labels.get(s.id) if labels else None,
                "text": s.text,
                "avg_logprob": round(s.confidence, 3),
                "words": [
                    {"word": w.word, "start": round(w.start, 3),
                     "end": round(w.end, 3), "confidence": round(w.confidence, 3)}
                    for w in s.words
                ],
            }
            for s in asr.segments
        ],
        "warnings": asr.warnings,
        "annotations": asr.annotations,
    }
    p = out_dir / f"{base}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ----------------------------------------------------------------------
# Word documents
# ----------------------------------------------------------------------

def _build_docx_document(asr, base, labels, report_md):
    """Return a python-docx Document. For IELTS (report_md given) the doc IS the
    report (which already contains the verbatim transcript); otherwise it's the
    timestamped transcript."""
    from docx import Document

    doc = Document()
    if report_md:
        _md_into_docx(doc, report_md)
    else:
        doc.add_heading(base, level=0)
        meta = doc.add_paragraph()
        meta.add_run(t("exp_docx_meta", language=asr.language, engine=asr.model_used,
                       seconds=f"{asr.audio_duration_sec:.0f}")).italic = True
        for s in asr.segments:
            ts = _ts_short(s.start)
            spk = _label(labels, s.id)
            p = doc.add_paragraph()
            p.add_run(f"[{ts}] ").bold = True
            if spk:
                p.add_run(f"{spk}: ").bold = True
            p.add_run(s.text)
    return doc


def _md_into_docx(doc, md: str) -> None:
    """Render the lightweight markdown our reports use into a docx Document."""
    for raw in md.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        if line.startswith("### "):
            doc.add_heading(line[4:], level=3)
        elif line.startswith("## "):
            doc.add_heading(line[3:], level=2)
        elif line.startswith("# "):
            doc.add_heading(line[2:], level=1)
        elif line.startswith("> "):
            p = doc.add_paragraph(line[2:])
            p.style = doc.styles["Intense Quote"] if "Intense Quote" in [s.name for s in doc.styles] else p.style
        elif line.lstrip().startswith("- "):
            doc.add_paragraph(line.lstrip()[2:], style="List Bullet")
        else:
            doc.add_paragraph(line)


def _docx(asr, out_dir, base, labels, report_md) -> Path:
    doc = _build_docx_document(asr, base, labels, report_md)
    p = out_dir / f"{base}.docx"
    doc.save(str(p))
    return p


def _doc(asr, out_dir, base, labels, report_md) -> Path:
    """Legacy .doc via macOS `textutil`, converting from a generated .docx."""
    import shutil
    import subprocess
    import tempfile

    # Build a docx first (reuse formatting), then convert.
    doc = _build_docx_document(asr, base, labels, report_md)
    with tempfile.NamedTemporaryFile(suffix=".docx", prefix="recorder_",
                                     delete=False) as tmp:
        tmp_docx = Path(tmp.name)
    try:
        doc.save(str(tmp_docx))

        out = out_dir / f"{base}.doc"
        textutil = shutil.which("textutil")
        if not textutil:
            raise RuntimeError(t("export_textutil_missing"))
        proc = subprocess.run(
            [textutil, "-convert", "doc", str(tmp_docx), "-output", str(out)],
            capture_output=True, text=True, check=False,
        )
        if proc.returncode != 0 or not out.exists():
            raise RuntimeError(
                t("export_textutil_failed", err=proc.stderr.strip()[:160]))
        return out
    finally:
        tmp_docx.unlink(missing_ok=True)


def _ts_short(sec: float) -> str:
    td = timedelta(seconds=int(sec))
    m, s = divmod(int(td.total_seconds()), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def _ts_srt(sec: float) -> str:
    ms = int((sec - int(sec)) * 1000)
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
