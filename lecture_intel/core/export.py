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

_SPEAKER_ZH = {
    "candidate": "考生",
    "examiner": "教官",
    "other": "其他",
    "main": "主讲",
}


def export_all(
    asr: ASRResult,
    out_dir: Path,
    base: str,
    formats: list[str],
    labels: Optional[dict[int, str]] = None,
    extra_markdown: Optional[str] = None,
    extra_markdown_suffix: str = "report",
) -> dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    for fmt in formats:
        if fmt == "txt":
            outputs["txt"] = _txt(asr, out_dir, base, labels)
        elif fmt == "md":
            outputs["md"] = _md(asr, out_dir, base, labels)
        elif fmt == "srt":
            outputs["srt"] = _srt(asr, out_dir, base, labels)
        elif fmt == "json":
            outputs["json"] = _json(asr, out_dir, base, labels)

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
    return _SPEAKER_ZH.get(raw, raw)


def _txt(asr, out_dir, base, labels) -> Path:
    lines = []
    for s in asr.segments:
        ts = _ts_short(s.start)
        spk = _label(labels, s.id)
        prefix = f"[{ts}] " + (f"{spk}: " if spk else "")
        lines.append(prefix + s.text)
    p = out_dir / f"{base}.txt"
    p.write_text("\n".join(lines) if lines else asr.full_text, encoding="utf-8")
    return p


def _md(asr, out_dir, base, labels) -> Path:
    L = [f"# {base}", "", f"- 语言：{asr.language}", f"- 引擎：{asr.model_used}",
         f"- 时长：{asr.audio_duration_sec:.0f}s", "", "## 转写", ""]
    for s in asr.segments:
        ts = _ts_short(s.start)
        spk = _label(labels, s.id)
        if spk:
            L.append(f"**{spk}** `[{ts}]` {s.text}")
        else:
            L.append(f"`[{ts}]` {s.text}")
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
    }
    p = out_dir / f"{base}.json"
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


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
