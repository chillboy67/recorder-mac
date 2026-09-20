"""
Filler retention eval — how many spoken disfluency markers survive general-mode
transcription, broken down by language × category.

Why this exists: Whisper upstream actively drops some filler words while
decoding (faster-whisper#901, whisper.cpp#965, both closed unfixed). Before
claiming "faithful to the original" we measure what actually survives on this
app's own pipeline, on this user's own voice. See eval/README.md for the
decision gate the numbers feed.

Usage (with the app's own venv, mlx/faster-whisper installed):

    .venv/bin/python eval/measure_filler_recall.py
    .venv/bin/python eval/measure_filler_recall.py --model large-v3-turbo
    .venv/bin/python eval/measure_filler_recall.py --only zh_fill_en_1,en_rep_no

Audio lives in ``eval/audio/<id>.wav`` (wav/m4a/mp3/flac all work — the engine
normalizes). Entries without audio are skipped, not failed. Results are written
to ``eval/report.md``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ROOT = EVAL_DIR.parent          # lecture_intel/
sys.path.insert(0, str(ROOT))

AUDIO_EXTS = [".wav", ".m4a", ".mp3", ".flac", ".ogg", ".aac"]


def normalize(text: str) -> str:
    """Comparison form: case-folded, all punctuation and whitespace stripped.

    Stripping punctuation means half-word markers (``should—`` → ``should``)
    match however the dash is rendered; it also means a half-word can false-
    positive when the completed word survives (``pheno`` ⊂ ``phenomenal``).
    That fuzziness only affects the ``half`` category and is flagged in the
    per-entry detail for human review — the gate decision rests on filler and
    repeat categories, which are exact.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def find_audio(audio_dir: Path, entry_id: str) -> Path | None:
    for ext in AUDIO_EXTS:
        p = audio_dir / f"{entry_id}{ext}"
        if p.exists():
            return p
    return None


def run_entry(audio: Path, out_dir: Path, model: str) -> str:
    """Transcribe one clip through the real general-mode pipeline; return text."""
    from core.engine import run
    # General mode's default formats (txt/md/docx) exclude json; force the json
    # export so the eval can read back full_text.
    run(audio, out_dir, mode_key="general", model=model, formats=["json"])
    data = json.loads((out_dir / f"{audio.stem}.json").read_text(encoding="utf-8"))
    return data.get("full_text", "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="large-v3",
                    help="Whisper model name as shown in the app (default large-v3)")
    ap.add_argument("--only", default="",
                    help="comma-separated entry ids to measure (default: all with audio)")
    ap.add_argument("--out", default=str(EVAL_DIR / "report.md"))
    ap.add_argument("--output-root", default=str(EVAL_DIR / "output"),
                    help="per-entry engine output dirs (contains original.wav + exports)")
    args = ap.parse_args()

    entries = json.loads((EVAL_DIR / "filler_set.json").read_text(encoding="utf-8"))
    only = {s.strip() for s in args.only.split(",") if s.strip()} if args.only else None
    audio_dir = EVAL_DIR / "audio"
    audio_dir.mkdir(exist_ok=True)

    rows = []           # per entry: id/lang/category/markers/retained/text
    skipped, errors = [], []
    for e in entries:
        if only and e["id"] not in only:
            continue
        audio = find_audio(audio_dir, e["id"])
        if audio is None:
            skipped.append(e["id"])
            continue
        out_dir = Path(args.output_root) / e["id"]
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            text = run_entry(audio, out_dir, args.model)
        except Exception as exc:
            errors.append((e["id"], f"{type(exc).__name__}: {exc}"))
            continue
        norm = normalize(text)
        retained = [m for m in e["markers"] if normalize(m) in norm]
        rows.append({
            "id": e["id"], "lang": e["lang"], "category": e["category"],
            "markers": e["markers"], "retained": retained, "text": text,
        })
        n = len(e["markers"])
        print(f"{e['id']:<18} {len(retained)}/{n} retained")

    # ── aggregate ──────────────────────────────────────────────────────
    by_lc = defaultdict(lambda: [0, 0])     # (lang, category) → [kept, total]
    for r in rows:
        kept = len(r["retained"])
        for key in ((r["lang"], r["category"]), (r["lang"], "*")):
            by_lc[key][0] += kept
            by_lc[key][1] += len(r["markers"])

    def pct(pair):
        return f"{100.0 * pair[0] / pair[1]:.0f}%" if pair[1] else "—"

    L = []
    L.append("# Filler retention report / 口头填充保留率评测\n")
    L.append(f"- model: `{args.model}`  ·  run: {datetime.now():%Y-%m-%d %H:%M}  ·  "
             f"entries measured: {len(rows)}, skipped (no audio): {len(skipped)}, "
             f"errors: {len(errors)}")
    L.append("- matching: normalized substring (case/punct/space-insensitive); "
             "`half` category can false-positive when the completed word survives\n")
    L.append("## Recall by language × category / 分语言×类别召回\n")
    L.append("| lang | category | retained / markers | recall |")
    L.append("|------|----------|--------------------|--------|")
    for (lang, cat) in sorted(by_lc):
        if cat == "*":
            continue
        kept, total = by_lc[(lang, cat)]
        L.append(f"| {lang} | {cat} | {kept}/{total} | {pct((kept, total))} |")
    L.append("| lang | **all** | **retained/markers** | **recall** |")
    for lang in sorted({k[0] for k in by_lc}):
        kept, total = by_lc[(lang, "*")]
        L.append(f"| {lang} | **overall** | **{kept}/{total}** | **{pct((kept, total))}** |")
    L.append("\n## Per-entry detail / 逐条明细\n")
    for r in rows:
        missing = [m for m in r["markers"] if m not in r["retained"]]
        flag = "" if not missing else f"  MISSING: {missing}"
        L.append(f"- **{r['id']}** ({r['lang']}/{r['category']})"
                 f" {len(r['retained'])}/{len(r['markers'])}{flag}")
        L.append(f"  - transcript: {r['text']!r}")
    if skipped:
        L.append("\n## Skipped (no audio) / 未录跳过\n")
        L.append(", ".join(skipped))
    if errors:
        L.append("\n## Errors / 出错\n")
        for eid, msg in errors:
            L.append(f"- {eid}: {msg}")
    Path(args.out).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"\nreport → {args.out}")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
