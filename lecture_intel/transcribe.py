#!/usr/bin/env python3
"""
Recorder CLI — headless transcription.

    python transcribe.py audio.m4a                 # general mode
    python transcribe.py lecture.mp3 -m classroom  # classroom (denoise + main speaker)
    python transcribe.py ielts.webm -m ielts       # IELTS analysis report
    python transcribe.py a.wav --model large-v3-turbo -f txt -f srt -o ./out
    python transcribe.py talk.m4a -l ja                           # pin the language
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> int:
    ap = argparse.ArgumentParser(description="Recorder — local speech-to-text")
    ap.add_argument("input", help="audio file")
    ap.add_argument("-m", "--mode", default="general",
                    choices=["general", "classroom", "ielts"])
    ap.add_argument("-o", "--output", default=None, help="output dir (default: <input>_output)")
    ap.add_argument("--model", default="large-v3",
                    help="large-v3 | large-v3-turbo | medium | small")
    ap.add_argument("--engine", default="auto",
                    choices=["auto", "mlx-whisper", "faster-whisper"])
    ap.add_argument("-l", "--language", default=None, metavar="CODE",
                    help="pin the spoken language (zh en ja ko fr de es…). "
                         "Default: auto-detect, which also enables per-chunk "
                         "code-switching; pinning trades that away")
    ap.add_argument("-f", "--format", action="append", dest="formats",
                    help="txt|md|srt|json (repeatable)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    inp = Path(args.input)
    if not inp.exists():
        print(f"✗ not found: {inp}")
        return 1
    out = Path(args.output) if args.output else inp.parent / f"{inp.stem}_output"

    from core.engine import run

    def show(info):
        bar = "█" * (info["percent"] // 5)
        print(f"\r[{bar:<20}] {info['percent']:3d}%  {info['message']:<40}", end="", flush=True)

    summary = run(
        input_path=inp, output_dir=out, mode_key=args.mode,
        model=args.model, engine=args.engine, language=args.language,
        formats=args.formats, progress=show,
    )
    print()
    print("=" * 60)
    print(f"  mode={summary['mode']}  lang={summary['language']}  "
          f"engine={summary['model']}")
    print(f"  segments={summary['segment_count']}  "
          f"time={summary['total_time_s']:.1f}s")
    for fmt, path in summary["output_files"].items():
        print(f"    {fmt:6s} → {path}")
    if summary.get("ielts"):
        i = summary["ielts"]
        print(f"  IELTS: {i['wpm']:.0f} WPM · 发音疑点 {i['pron_issue_count']} · "
              f"语法 {i['grammar_issue_count']} · 表达 {i['naturalness_count']}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
