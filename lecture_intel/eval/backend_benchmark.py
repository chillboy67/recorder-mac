#!/usr/bin/env python3
"""Compare Recorder ASR backends on the same local audio fixture.

Example (run from lecture_intel/):
    .venv/bin/python eval/backend_benchmark.py sample.wav \
        --engines faster-whisper,whisper.cpp-vulkan \
        --model small --warmups 1 --repeats 3 --reference reference.txt

A requested GPU run counts as a GPU result only if Recorder's JSON result reports
that exact backend. CPU fallbacks are retained as failed rows for transparency.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SUPPORTED_ENGINES = (
    "faster-whisper", "whisper.cpp-vulkan", "whisper.cpp-openvino",
    "mlx-whisper", "auto",
)


def normalize_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[\u3400-\u9fff]|[\u3040-\u30ff]|[\uac00-\ud7af]+|[^\W_]+", text, flags=re.UNICODE)


def error_rate(reference: str, hypothesis: str) -> float:
    """Whitespace/token WER; for CJK this is a character/token proxy, not WER."""
    ref, hyp = normalize_words(reference), normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, left in enumerate(ref, 1):
        current = [i]
        for j, right in enumerate(hyp, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (left != right),
            ))
        previous = current
    return previous[-1] / len(ref)


def _monitor_memory(child: subprocess.Popen[Any]) -> int | None:
    """Poll peak process-tree RSS while output is redirected to a file."""
    try:
        from importlib import import_module
        psutil = import_module("psutil")
        process = psutil.Process(child.pid)
    except Exception:
        process = None

    peak = 0
    while child.poll() is None:
        if process is not None:
            try:
                rss = process.memory_info().rss
                for descendant in process.children(recursive=True):
                    try:
                        rss += descendant.memory_info().rss
                    except Exception:
                        pass
                peak = max(peak, rss)
            except Exception:
                pass
        time.sleep(0.03)
    return peak or None


def run_once(
    audio: Path,
    output_dir: Path,
    engine: str,
    model: str,
    reference: str | None = None,
) -> dict[str, Any]:
    audio = audio.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable, str(ROOT / "transcribe.py"), str(audio),
        "--mode", "general", "--engine", engine, "--model", model,
        "--format", "json", "--output", str(output_dir), "--verbose",
    ]
    started = time.perf_counter()
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as console_file:
        child = subprocess.Popen(
            command, cwd=ROOT, stdout=console_file, stderr=subprocess.STDOUT,
            text=True,
        )
        peak_rss = _monitor_memory(child)
        elapsed = time.perf_counter() - started
        console_file.seek(0)
        console = console_file.read()
    output_file = output_dir / f"{audio.stem}.json"
    safe_console = console.replace(str(audio), audio.name)
    row: dict[str, Any] = {
        "requested_engine": engine,
        "actual_engine": None,
        "model": model,
        "elapsed_sec": round(elapsed, 3),
        "audio_duration_sec": None,
        "rtf": None,
        "peak_rss_mb": round(peak_rss / (1024 * 1024), 1) if peak_rss else None,
        "wer": None,
        "returncode": child.returncode,
        "status": "error",
        "error": None,
    }
    if child.returncode != 0:
        row["error"] = safe_console[-4000:] or f"CLI exited with {child.returncode}"
        return row
    try:
        result = json.loads(output_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        row["error"] = f"Could not read CLI JSON result: {exc}; {safe_console[-1000:]}"
        return row

    actual_engine, _, actual_model = result.get("model", "").partition(":")
    row["actual_engine"] = actual_engine or None
    row["model"] = actual_model or model
    duration = result.get("duration_sec")
    if isinstance(duration, (int, float)) and duration > 0:
        row["audio_duration_sec"] = round(float(duration), 3)
        row["rtf"] = round(elapsed / float(duration), 4)
    if reference is not None:
        row["wer"] = round(error_rate(reference, result.get("full_text", "")), 4)
    expected = engine if engine != "auto" else actual_engine
    if actual_engine != expected:
        row["status"] = "backend-mismatch"
        row["error"] = (
            f"Requested {engine}, but Recorder reports {actual_engine or 'unknown'}; "
            "this run does not count as the requested backend."
        )
    else:
        row["status"] = "ok"
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "ok":
            groups.setdefault(row["requested_engine"], []).append(row)
    summary = {}
    for engine, group in groups.items():
        elapsed_values = [r["elapsed_sec"] for r in group]
        rtf_values = [r["rtf"] for r in group if r["rtf"] is not None]
        summary[engine] = {
            "successful_runs": len(group),
            "median_elapsed_sec": round(statistics.median(elapsed_values), 3),
            "elapsed_range_sec": [round(min(elapsed_values), 3),
                                  round(max(elapsed_values), 3)],
            "median_rtf": round(statistics.median(rtf_values), 4)
                if rtf_values else None,
            "rtf_range": [round(min(rtf_values), 4), round(max(rtf_values), 4)]
                if rtf_values else None,
            "median_peak_rss_mb": round(statistics.median(
                r["peak_rss_mb"] for r in group if r["peak_rss_mb"] is not None), 1)
                if any(r["peak_rss_mb"] is not None for r in group) else None,
            "median_wer": round(statistics.median(
                r["wer"] for r in group if r["wer"] is not None), 4)
                if any(r["wer"] is not None for r in group) else None,
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="same local audio fixture for all backends")
    parser.add_argument("--engines", default="faster-whisper,whisper.cpp-vulkan",
                        help=f"comma-separated: {', '.join(SUPPORTED_ENGINES)}")
    parser.add_argument("--model", default="small")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference", type=Path,
                        help="UTF-8 reference transcript; computes token WER proxy")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "eval" / "backend_benchmark.json")
    parser.add_argument("--runs-dir", type=Path,
                        default=ROOT / "eval" / "backend_benchmark_runs")
    args = parser.parse_args(argv)
    if not args.audio.is_file():
        parser.error(f"audio file does not exist: {args.audio}")
    if args.warmups < 0 or args.repeats < 1:
        parser.error("warmups must be >= 0 and repeats must be >= 1")
    engines = [item.strip() for item in args.engines.split(",") if item.strip()]
    unknown = sorted(set(engines) - set(SUPPORTED_ENGINES))
    if not engines or unknown:
        parser.error(f"invalid/empty engine list; unknown: {unknown}")
    reference = args.reference.read_text(encoding="utf-8") if args.reference else None
    args.output = args.output.resolve()
    args.runs_dir = args.runs_dir.resolve()
    args.audio = args.audio.resolve()

    rows = []
    for engine in engines:
        for index in range(args.warmups + args.repeats):
            warmup = index < args.warmups
            run_dir = args.runs_dir / engine.replace("/", "-") / f"run-{index + 1:02d}"
            row = run_once(args.audio, run_dir, engine, args.model, reference)
            row["warmup"] = warmup
            rows.append(row)
            print(
                f"{engine}: actual={row['actual_engine'] or 'unknown'} "
                f"status={row['status']} elapsed={row['elapsed_sec']:.2f}s "
                f"rtf={row['rtf']} peak-rss={row['peak_rss_mb']}MB"
            )
            if row["error"]:
                print(row["error"])

    measured = [row for row in rows if not row["warmup"]]
    payload = {
        "audio": args.audio.name,
        "model": args.model,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "measurement_scope": "end-to-end process startup, model load, audio pipeline, and export",
        "memory_sampling": "psutil process-tree RSS polling; null if psutil is unavailable",
        "wer_note": "English token WER proxy; CJK uses characters and is not word WER.",
        "summary": summarize(measured),
        "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"\nJSON report → {args.output}")
    failed_gpu = any(
        row["status"] != "ok" and row["requested_engine"].startswith("whisper.cpp-")
        for row in measured
    )
    return 1 if failed_gpu or any(row["status"] == "error" for row in measured) else 0


if __name__ == "__main__":
    raise SystemExit(main())
