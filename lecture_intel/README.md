# Recorder (app code)

This directory holds the actual **Recorder** app — a local, offline speech-to-text
tool with three modes: 通用 / 课堂 / 雅思. (The folder name `lecture_intel` is
historical.)

See the top-level [../README.md](../README.md) for the product overview.

## Run

```bash
uv venv                                  # if .venv doesn't exist
uv pip install -r requirements.txt

# GUI
.venv/bin/python3 app.py

# Double-clickable app
./make_app.sh && open dist/

# CLI
.venv/bin/python3 transcribe.py audio.m4a -m ielts
```

## Architecture

```
app.py            PySide6 GUI entry (double-click target)
transcribe.py     CLI entry
make_app.sh        builds dist/Recorder.app (lightweight launcher → venv)

core/             ← the new, focused engine
  modes.py        general / classroom / ielts presets
  transcriber.py  whole-file Whisper (mlx-whisper → faster-whisper fallback)
  denoise.py      ffmpeg cleanup for classroom mode
  diarize.py      token-free speaker separation (Resemblyzer + clustering)
  ielts.py        pronunciation (confidence-based) / grammar / phrasing / report
  export.py       txt / md / srt / json with speaker labels
  engine.py       the single orchestrator: run(input, output, mode)

gui/              PySide6 widgets (input / settings / progress / results)
modules/          shared dataclasses + AudioLoader (reused);
                  the old 11-step pipeline.py lives here but is no longer used.
tests/            core-logic regression tests (no model needed)
```

The old `pipeline.py` + course-classifier / LLM-correction / lecture-structuring
modules are superseded by `core/engine.py` and kept only for reference.

## Models (offline / China-friendly)

Pre-download all models into `~/Library/Application Support/Recorder/models/`
so the app runs fully offline and never waits on HuggingFace:

```bash
.venv/bin/python3 download_models.py          # all 4 models, via hf-mirror.com
.venv/bin/python3 download_models.py small     # just one
.venv/bin/python3 download_models.py --hf      # use huggingface.co instead
```

The transcriber loads a local model folder when present (no network call at
runtime). The downloader uses plain curl through the **hf-mirror.com** mirror —
this avoids the `huggingface_hub` xet/LFS transfer stalling that happens on
mainland-China networks. If a model isn't downloaded, the app still falls back
to fetching it from HuggingFace on first use.

## Local LLM enhancement (optional, via Ollama)

Tick "本地大模型增强" in the app to enable, after a one-time setup. Everything
stays on-device. **Model weights are never part of this repo** — each machine
runs `ollama pull` locally.

### Current code defaults (language auto-route)

| Audio language | Setting key | Default tag |
|----------------|-------------|-------------|
| Chinese / mixed | `chinese_model` | `qwen-zh:7b` |
| Otherwise (e.g. English) | `llm_model` | `llama3.1:8b` |

If the preferred model is missing, the other installed one is used; if neither
is available, offline heuristics run.

### Recommended dual strategy (upgrade path)

| Role | Family | 16GB Apple Silicon default | 24GB+ |
|------|--------|----------------------------|--------|
| Asian (zh/ja/ko, Chinese reports) | **Qwen** | `qwen2.5:7b` | `qwen2.5:14b` |
| European (en/fr/es/de, …) | **Mistral** | `mistral` (7B) | `mistral-nemo` |

```bash
brew install ollama && brew services start ollama

# 16GB-friendly pair (official registry)
ollama pull qwen2.5:7b
ollama pull mistral

# Optional: alias so current code names still resolve
# ollama cp qwen2.5:7b qwen-zh:7b
# ollama cp mistral llama3.1:8b   # only if you want Mistral under the old key
```

**Full guide** (hardware tiers 8GB–64GB, Whisper co-existence, China mirrors,
legacy ModelScope `qwen-zh` / Llama install, what not to run on 16GB):

→ **[docs/LLM_MODELS.md](docs/LLM_MODELS.md)**

What the LLM adds per mode (falls back to offline heuristics if Ollama is off):
- **通用**: light punctuation / recognition-error tidy (no paraphrase).
- **课堂**: lecture-context correction + key-point summary.
- **雅思**: examiner-style critique on top of confidence-based pronunciation
  flags. Candidate transcript stays verbatim.

All via local `/api/chat`. `resolve_model` tolerates ModelScope-style names.

## Notes on accuracy

- Default model is `large-v3` (~3GB). For faster runs choose
  `large-v3-turbo` in the UI or `--model large-v3-turbo` on the CLI.
- On Apple Silicon the `mlx-whisper` engine is used automatically; on other
  machines it falls back to `faster-whisper` (CPU).
- We feed the **whole file** to Whisper rather than pre-chunking — this is the
  single biggest accuracy improvement over the old pipeline.
- Nothing is ever paraphrased. Errors in speech are preserved verbatim.
- **Code-switching (中英混合):** plain mlx does one global language pass and
  translates the minority language away. IELTS mode keeps the GPU but splits the
  audio at silences and detects language **per chunk** (`chunked_language`), so
  Chinese coach feedback stays Chinese and English answers stay English — on the
  GPU. General/classroom use the fast single-pass (monolingual). faster-whisper
  (CPU) remains the automatic fallback if mlx is unavailable.
- **Speaker separation** is frame-level voice embeddings + clustering, with a
  language fallback: when two same-gender voices are acoustically too close to
  split, Chinese segments are attributed to the coach (教官) and English to the
  student (考生). The student's English is what the pronunciation/grammar
  analysis runs on.

## Test

```bash
.venv/bin/python3 -m pytest tests/ -q          # from this dir
# or from repo root:  python -m pytest
```
