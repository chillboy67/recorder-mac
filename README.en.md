# Recorder — Local, Offline Speech-to-Text (macOS)

[中文](README.md) | English

A double-click-to-launch, fully local audio-to-text desktop app for MacBook.
Everything runs offline — no cloud API calls, audio and text never leave the machine.

Built on **transcription accuracy** as the foundation, with three purpose-built modes on top:

| Mode | Scenario | Output |
|------|----------|--------|
| **General transcription** | Meetings, interviews, voice memos, any audio file | A faithful, word-for-word transcript (automatic Chinese/English detection) |
| **Classroom recording** | Open rooms, echo, a teacher far from the mic, side chatter | Noise reduction + transcript of the main speaker only + automatic key-point summary |
| **IELTS speaking coach** | One-on-one coach/student practice sessions (Chinese/English code-switching) | Coach/student separation + flagged pronunciation/grammar/phrasing issues + feedback report |

> The main codebase lives in [`lecture_intel/`](lecture_intel/) (a legacy directory name — the app itself is called Recorder).

---

## Why another transcription app

Most commercial recording-to-text apps share one default behavior: **they quietly fix what you said wrong**.
Grammar gets corrected, sentences get smoothed out, accents get flattened — great if your goal is
"capture what was decided in the meeting," terrible if your goal is "find out how I actually spoke,"
because the diagnostic value lives exactly in those errors.

Recorder's first principle:

> **No mode ever rewrites the speaker's original words. Mistakes are kept exactly as spoken.**

Anything worth flagging (uncertain pronunciation, grammar issues, non-native phrasing) is **annotated**,
never edited into the transcript body. The optional local LLM enhancement only touches transcription
errors caused by unclear audio — it never polishes or rewrites.

---

## Core features

- **Fully offline.** The Whisper model runs on-device; once downloaded, no network is needed. No accounts, no uploads.
- **Apple Silicon GPU acceleration.** Defaults to `mlx-whisper` (Metal), automatically falling back to
  `faster-whisper` (CPU) when unavailable.
- **Chinese/English code-switching without bias.** Chinese segments stay Chinese, English segments stay English —
  never silently "translated" into a single language (see "Engineering notes" below for how).
- **Two recording sources**: microphone, **system audio capture** (online classes/web video, works even with
  headphones on), plus a mixed-source mode.
- **Per-word confidence scores** are always on — this is what powers the IELTS mode's "possible pronunciation issue" flags.
- **Stable on long recordings.** A 2-hour recording is processed in silence-bounded chunks on a 16GB machine,
  with bounded memory and genuine progress reporting.
- **Crash-safe.** Transcription runs in an isolated subprocess — even if the model process gets killed by the
  system, the UI stays up and the recorded audio is never lost.
- **Export** to txt / md / doc / docx, with timestamps and speaker labels.
- **Light/dark theme**, follows the system setting.
- **Optional local LLM (Ollama)** for tidy-up / classroom summary / IELTS notes.  
  Model picks and install: [lecture_intel/docs/LLM_MODELS.md](lecture_intel/docs/LLM_MODELS.md).

---

## Mode details

### General transcription
Load → transcribe the whole file → export. Automatic Chinese/English detection, word-for-word fidelity.

### Classroom recording
ffmpeg preprocessing (high-pass filter for low-frequency rumble + adaptive noise reduction + loudness
normalization) → transcription → speaker clustering that **keeps only the speaker with the longest talk time**
(side chatter is dropped) → automatic key-point extraction (emphasis phrasing, definition sentences, frequent
terms, longest-explained segments) generating a `*.summary.md` file.

Classroom mode also ships a domain dictionary (machine learning / deep learning / statistics / math / Python / R)
that only corrects term spelling, never sentence structure.

### IELTS speaking coach
Transcription → **token-free two-speaker separation** (voiceprint embeddings + clustering, automatically
determines coach vs. student, swappable with one click in the UI) → analysis → feedback report.

Analysis covers four categories, all annotation-only, never rewriting:

1. **Possible pronunciation issues**: words where Whisper's per-word confidence is abnormally low — where the
   acoustic model "wasn't sure" — which usually correlates with unclear, mispronounced, or accented words.
   Multiple noise-reduction passes (skipping segment-start artifacts, filtering common words, requiring word
   length ≥ 4) reduce false positives.
2. **Grammar issues**: tense, subject-verb agreement, articles, etc. Uses a local LanguageTool service when
   available, otherwise falls back to built-in rules.
3. **Phrasing issues**: translated-from-Chinese English, unnatural collocations.
4. **Coach's corrections**: the correct phrasing the coach gave in the moment, listed for reference.

Finally, everything is rolled up into a report scored against the four IELTS speaking dimensions (fluency &
coherence / vocabulary / grammar / pronunciation), with the full transcript embedded in the report.

---

## Quick start

Requires macOS (Apple Silicon recommended) and [ffmpeg](https://ffmpeg.org) (`brew install ffmpeg`).

```bash
cd lecture_intel

# 1) Environment
uv venv
uv pip install -r requirements.txt

# 2) Build the double-click app (installs to /Applications/Recorder.app)
./make_app.sh

# Or just run the GUI directly
.venv/bin/python3 app.py
```

### Pre-download models (recommended)

```bash
cd lecture_intel
.venv/bin/python3 download_models.py     # downloads to the local model directory
```

Three model tiers are selectable in the UI: **most accurate** `large-v3` / **balanced** `large-v3-turbo` /
**fastest** `small`. Once downloaded, the app runs fully offline at runtime.

### Command line (no UI)

```bash
.venv/bin/python3 transcribe.py recording.m4a                  # general
.venv/bin/python3 transcribe.py class.mp3  -m classroom        # classroom
.venv/bin/python3 transcribe.py ielts.webm -m ielts            # IELTS feedback
.venv/bin/python3 transcribe.py a.wav --model large-v3-turbo -f txt -f docx
.venv/bin/python3 transcribe.py talk.m4a -l ja                 # pin the language (default: auto-detect)
```

Supports m4a / mp3 / wav / webm / flac / aac / ogg / opus.

### Optional: local LLM enhancement

With [Ollama](https://ollama.com), enable “AI enhancement” in the UI to fix recognition typos, generate classroom summaries, and add IELTS-style notes. When off, offline rules apply. Models run on-device — do not commit weights to Git.

Recommended by language family: **Asian (zh/ja/ko/…) → Qwen**, **European (en/fr/es/de/…) → Mistral**. Sizing by RAM, install steps, and co-existence with Whisper:

**[lecture_intel/docs/LLM_MODELS.md](lecture_intel/docs/LLM_MODELS.md)**

---

## Architecture

```
lecture_intel/
├── app.py                GUI entry point (the double-click target)
├── transcribe.py         CLI entry point
├── make_app.sh           Builds and installs /Applications/Recorder.app
├── download_models.py    Whisper model pre-download (direct mirror)
├── docs/LLM_MODELS.md    Hardware tiers × Ollama (Qwen/Mistral) guide
│
├── core/                 The engine
│   ├── engine.py         Single orchestration entry point: run(input, output, mode)
│   ├── modes.py          Parameter presets for general / classroom / ielts
│   ├── transcriber.py    Whisper wrapper (mlx → faster-whisper fallback, chunking, per-word confidence)
│   ├── denoise.py        ffmpeg-based noise reduction (classroom)
│   ├── diarize.py        Token-free speaker separation (voiceprint embeddings + clustering + temporal smoothing)
│   ├── ielts.py          Pronunciation / grammar / phrasing analysis and report generation
│   ├── classroom.py      Key-point extraction and summarization
│   ├── llm.py            Optional local LLM enhancement (Ollama)
│   ├── sysaudio.py       System audio capture driver
│   ├── runner.py         Subprocess executor (crash isolation)
│   ├── paths.py          User data directory (single source of truth)
│   └── export.py         txt / md / doc / docx export
│
├── gui/                  PySide6 UI (home / record / process / results + theming)
├── native/               SystemAudioRecorder.swift (ScreenCaptureKit-based system audio capture)
├── dictionaries/         Classroom term dictionary
└── tests/                Unit tests that run without needing the model (python -m pytest)
```

Data flow:

```
Audio →[classroom: noise reduction]→ silence-bounded chunked transcription (GPU) →[speaker separation / keep main speaker]
      →[IELTS analysis | classroom summary]→ export
```

---

## Engineering notes

Decisions that actually mattered in turning "it runs" into "it's accurate and stable":

**Feed the whole file to Whisper instead of pre-slicing into 30s chunks.**
An early version used VAD to cut segments before transcribing each one — words got dropped at the cut points,
duplicated at overlaps, and context broke across boundaries. Whisper's own 30s window with cross-window context
is most accurate when fed a full segment. Very long audio is only split into large chunks at **silence**, never
mid-sentence.

**Chunking + per-chunk language detection solves Chinese/English code-switching.**
A single global language decision "translates away" the minority language (English mode swallows Chinese,
Chinese mode swallows English). By detecting language independently per silence-bounded chunk, both languages
stay faithful while everything still runs on the GPU.

**Chunking also solves memory and progress reporting as a side effect.**
A 2-hour recording no longer builds one giant tensor (no OOM on a 16GB machine), and the progress bar advances
per chunk instead of stalling at some percentage. Repetition loops from transcription hallucination are handled
by dedicated suppression logic.

**Transcription runs in a subprocess, not a thread.**
The model's compilation recursion is deep enough to blow a worker thread's small stack, and if a thread is still
running when the app quits, it aborts outright. Switching to a `multiprocessing` subprocess with the main thread
polling a results queue means a subprocess crash is just a subprocess crash — the UI keeps running, recorded
audio stays safe, and quitting just terminates the subprocess.

**Speaker separation avoids anything requiring an auth token.**
Voiceprint embeddings + hierarchical clustering are enough for the two-speaker case. Same-gender voices are often
acoustically hard to separate, so there's a **language-based fallback**: when acoustic separation is clearly
unbalanced and a turn is written in a script foreign to the candidate's language, roles are assigned by language
(an IELTS candidate always answers in English, so whoever else is speaking is the coach — Chinese, Japanese,
Korean, Russian, Thai alike) — this is what makes coach/student separation actually reliable. Known gap: a
Latin-script coach (fr/de/es) shares the candidate's script and needs per-segment language detection to tell apart.

**System audio capture writes its own WAV file.**
ScreenCaptureKit hands back non-interleaved Float32 audio that neither AVAudioFile nor its converters will accept,
so the Swift side manually de-interleaves it into interleaved Int16 and writes the WAV header by hand. Requires a
one-time "Screen Recording" permission grant; the UI gives clear guidance when it hasn't been granted yet.

**Packaging avoids py2app / PyInstaller.**
Freezing torch and mlx into a bundle makes it huge and fragile. Instead, a lightweight `.app` launcher points at
an installed runtime copy, so dependencies can be upgraded normally. Because macOS TCC blocks a Finder-launched
app from reading `~/Documents`, the install script places the runnable copy and virtual environment under
`~/Library/Application Support/Recorder` and launches from there — true double-click-to-open.

---

## Development history

The project ran from May to July 2026, roughly in five phases:

| Time | Phase |
|------|-------|
| 2026-05-07 – 05-17 | IELTS speaking-scorer prototype: grammar / naturalness / pronunciation rules + a FastAPI backend (`backend/`) |
| 2026-05-30 – 06-01 | Pivoted to classroom recordings, built an 11-step pipeline (`modules/` + `pipeline.py`) |
| 2026-06-13 – 06-14 | Rebuilt from scratch: the mode-driven `core/` engine, full-file transcription replacing self-sliced VAD segments, subprocess crash isolation |
| 2026-06-18 – 06-20 | System audio capture, light/dark theming, unified user data directory |
| 2026-07-05 – 07-07 | Redesigned UI into four screens, three-way choice of recording source |

The project wasn't under git for most of its life — it was only imported into version control on 2026-07-28.
Because of that, every commit's date is the import date; commits were organized into 23 commits matching the
phases above, with each commit's body noting the actual original development period it corresponds to.

## Multilingual (in progress)

Transcription today is **zh/en-first**; optional LLM sizing already follows **Qwen (Asian) / Mistral (European)**.
Full multilingual (e.g. ja/ko/EU coach sessions) lands in three steps:

1. **A shared `language` setting** + language helpers — not a standalone "detector app", but one setting every later
   stage shares (role separation, reports, model routing);
2. Diarization without hard-coded Chinese assumptions;
3. True multi-language ASR labels; then analysis and optional coach-side translation.

Progress: the language helper layer now lives in `core/languages.py` and labels by Unicode script
(kana → Japanese, Hangul → Korean, Han → Chinese; scripts shared by several languages — Latin, Cyrillic,
Arabic — defer to Whisper's own detection). Step 3's transcription labels are done: Japanese is no longer
reported as Chinese, Korean is no longer reported as English, and Latin-script languages no longer collapse
to English. The zh/en code-switching thresholds are unchanged. Steps 1 and 2 are complete: the
setting reaches the engine, the CLI (`transcribe.py -l ja`) and a 语言 picker in the UI, and the
coach/student split no longer assumes a Chinese-speaking coach — Japanese, Korean, Russian and
Thai coaches are attributed by script. Known gap: a Latin-script coach (fr/de/es) shares the
candidate's script and needs per-segment language detection. What remains is step 3's analysis
and optional translation.

See the [LLM guide](lecture_intel/docs/LLM_MODELS.md) for routing notes.

## Privacy

- The model and all processing run on-device — **no network requests at all** (aside from the initial model download).
- Recordings and transcripts are written to a local data directory (see `core/paths.py`); the repository ships no audio.
- The optional LLM enhancement runs through local Ollama and also never touches the network.

---

## Development

```bash
python -m pytest          # run from the repo root; tests don't need the model
```

After changing code, re-run `lecture_intel/make_app.sh` to sync the installed app.

Full requirements and design trade-offs are documented in [REQUIREMENTS.md](REQUIREMENTS.md).

## Directory notes

- `lecture_intel/` — the app itself (the only actively maintained code).
- `aura_gui/` — a standalone design draft for the UI redesign, kept for reference.
- `lecture_intel/gui_old/`, `pipeline.py`, and the lecture-classification/structured modules in `modules/` —
  earlier implementations superseded by the `core/` engine, kept as archive (`modules/audio_loader.py` and its
  dataclasses are still in use).
- `backend/` + `frontend/` — the original FastAPI-based IELTS scoring prototype; its analysis approach
  (pronunciation confidence, grammar rules, report templates) has been folded into `lecture_intel/core/ielts.py`.
  Kept for archival purposes only, safe to ignore.

## License

[MIT](LICENSE)
