# Recorder (app code)

This directory holds the actual **Recorder** app — a local, offline speech-to-text
tool with three modes: 通用 / 课堂 / 雅思. (The folder name `lecture_intel` is
historical.)

See the top-level [../README.md](../README.md) for the product overview.

## Run

Install the platform-appropriate dependencies with `python -m pip install -r requirements.txt`.
`mlx-whisper` is installed only on Apple Silicon; `faster-whisper` provides the
CPU inference path on macOS Intel, Windows, and Linux. Start the GUI or CLI with
the Python executable from your virtual environment:

```bash
python app.py
python transcribe.py audio.m4a -m ielts
```

`make_app.sh` remains a macOS-only launcher installer. Windows/Linux packaging
and system-audio loopback recording are not provided yet. On Windows/Linux,
optional whisper.cpp Vulkan acceleration supports compatible Intel/AMD GPUs;
Intel GPU acceleration is also available through the separately configured
OpenVINO backend. Apple Silicon uses MLX/Metal when available; Intel Macs use
the CPU path. See [docs/GPU_BACKENDS.md](docs/GPU_BACKENDS.md) for build,
configuration, and real-hardware acceptance steps.

## Architecture

```
app.py            PySide6 GUI entry (double-click target)
transcribe.py     CLI entry
make_app.sh        builds dist/Recorder.app (lightweight launcher → venv)

core/             ← the engine (mode-driven)

  modes.py        general / classroom / ielts presets
  transcriber.py  Whisper (MLX / whisper.cpp Vulkan or OpenVINO / CPU fallback)
  denoise.py      ffmpeg cleanup for classroom mode
  diarize.py      token-free speaker separation (Resemblyzer + clustering)
  ielts.py        pronunciation (confidence-based) / grammar / phrasing / report
  export.py       txt / md / srt / json with speaker labels
  engine.py       the single orchestrator: run(input, output, mode)

gui/              PySide6 widgets (input / settings / progress / results)
modules/          shared dataclasses + AudioLoader (both still in use)
tests/            core-logic regression tests (no model needed)
```

The legacy 11-step `pipeline.py` and its course-classifier / LLM-correction /
lecture-structuring modules have been deleted — `core/engine.py` replaced them.

## Models (offline / China-friendly)

The default model is device-aware: Apple Silicon MLX uses `large-v3`; the CPU
profile uses `small`. You can still select any listed model manually. MLX and
faster-whisper use different model formats and separate platform cache folders.
Set `RECORDER_MODEL_CACHE_DIR` to move both caches to another writable volume.

```bash
python download_models.py                     # MLX + CPU fallback on Apple Silicon; CPU elsewhere
python download_models.py --engine cpu small  # pre-download CPU weights
python download_models.py --engine mlx --hf   # use huggingface.co for MLX files
python download_models.py --engine all small  # download MLX + CPU formats
python download_models.py --engine cpp-vulkan small   # whisper.cpp Vulkan model
python download_models.py --engine cpp-openvino small # whisper.cpp OpenVINO model
```

CPU weights are stored in the operating system's user cache (`%LOCALAPPDATA%` on
Windows, `~/Library/Caches` on macOS, or `$XDG_CACHE_HOME` / `~/.cache` on Linux).
Pre-download the selected CPU model before going offline. MLX and CPU weights are
not interchangeable.

The downloader uses the **hf-mirror.com** endpoint by default; pass `--hf` to
use `huggingface.co` directly. CPU weights are downloaded through the
`faster-whisper`/Hugging Face cache API. If a model was not pre-downloaded, the
app attempts to fetch it on first use; for fully offline runs, download the
selected model before disconnecting.

## Local LLM enhancement (optional, via Ollama)

Tick "本地大模型增强" after installing [Ollama](https://ollama.com). Everything
stays on-device; **weights are never part of this repo**.

What it adds (falls back to offline heuristics if Ollama is off):
- **通用**: light punctuation / recognition-error tidy (no paraphrase)
- **课堂**: lecture-context correction + key-point summary
- **雅思**: examiner-style critique; candidate transcript stays verbatim

Defaults: Chinese → Qwen, everything else → Mistral; the exact model you
install is your choice. With Ollama absent, or nothing matching pulled,
enhancement is skipped automatically.

**Install, hardware tiers, China mirrors, pull commands:**  
→ **[docs/LLM_MODELS.md](docs/LLM_MODELS.md)**

## Notes on accuracy

- Default model is `large-v3` (~3GB). For faster runs choose
  `large-v3-turbo` in the UI or `--model large-v3-turbo` on the CLI.
- Apple Silicon prefers `mlx-whisper`; Windows/Linux can use an explicitly
  configured whisper.cpp Vulkan/OpenVINO backend and fall back to
  `faster-whisper` CPU. Intel Macs currently use CPU. Backend setup and hardware
  validation details: [docs/GPU_BACKENDS.md](docs/GPU_BACKENDS.md).
- We feed the **whole file** to Whisper rather than pre-chunking — this is the
  single biggest accuracy improvement over the old pipeline.
- Nothing is ever paraphrased. Errors in speech are preserved verbatim.
- The optional AI enhancement only writes a **separate** corrected companion
  (punctuation, obvious recognition typos); the verbatim transcript files are
  never overwritten, and the speaker's own errors are flagged, not fixed.
- **Code-switching (中英混合):** plain mlx does one global language pass and
  translates the minority language away. IELTS mode keeps the GPU but splits the
  audio at silences and detects language **per chunk** (`chunked_language`), so
  Chinese coach feedback stays Chinese and English answers stay English — on the
  GPU. General/classroom use the fast single-pass (monolingual). faster-whisper
  (CPU) remains the fallback if an accelerated backend is unavailable or its
  runtime logs do not confirm GPU execution.
- **Speaker separation** is frame-level voice embeddings + clustering, with a
  language fallback: when two same-gender voices are acoustically too close to
  split, turns that are not in the candidate's English are attributed to the
  coach (教官) and the rest to the student (考生). A Chinese, Japanese, Korean,
  Russian or Thai coach is caught by script — verified on real recordings. A
  French/German/Spanish coach, whose letters match English, would need the
  per-chunk language attached to each segment, and that is **not reliable yet**:
  it only helps when a chunk holds one language, and the acoustic path's role
  score ignores segment language. The student's English is what the
  pronunciation/grammar analysis runs on.

## Test

```bash
.venv/bin/python3 -m pytest tests/ -q          # from this dir
# or from repo root:  python -m pytest
```
