# GPU backends for Windows and Linux

Recorder keeps Apple Silicon on MLX/Metal and Intel Macs on the `faster-whisper`
CPU path. On Windows/Linux, the optional `whisper.cpp` backends provide:

- **Vulkan**: broad vendor support, including Intel and AMD integrated GPUs (and
  compatible discrete GPUs). Device support depends on the installed Vulkan
  runtime/driver.
- **OpenVINO GPU**: Intel GPU path, with the encoder offloaded to OpenVINO; this
  is not an all-operations GPU execution path. A compatible Intel GPU driver and
  OpenVINO runtime are required.

Both are optional external `whisper-cli` builds. They do not add a Python GPU
runtime dependency. If requested backend execution cannot be confirmed in its
runtime logs, Recorder rejects the result as GPU and falls back to
`faster-whisper` CPU when available. The result's engine field and warning report
the actual backend used.

## Build whisper.cpp

Use an upstream checkout and build on the target operating system. Build/install
instructions and supported compiler/toolchain versions can change; follow the
current [whisper.cpp README](https://github.com/ggml-org/whisper.cpp) as well.

### Vulkan (Windows/Linux)

Install a Vulkan SDK or development headers/loader and a Vulkan-capable driver,
then build with Vulkan enabled:

```sh
cmake -B build -DGGML_VULKAN=1
cmake --build build --config Release --target whisper-cli -j
```

On Windows, the executable is commonly under `build/bin/Release/whisper-cli.exe`;
on Linux it is commonly `build/bin/whisper-cli`. Exact output paths vary by
whisper.cpp version and generator. Verify it starts and lists the intended GPU
before configuring Recorder.

### OpenVINO (Intel GPU, Windows/Linux)

Install a compatible OpenVINO runtime and Intel GPU driver, then configure and
build whisper.cpp with its OpenVINO support:

```sh
cmake -B build -DWHISPER_OPENVINO=1
cmake --build build --config Release --target whisper-cli -j
```

Check the current upstream build documentation for any additional OpenVINO CMake
options/runtime setup needed by the checked-out revision. The Recorder adapter
passes `-oved GPU` and only accepts output when the process log reports both
`device = GPU` and `OpenVINO model loaded`.

## Install model files

Run from `lecture_intel/`. The default cache is platform-specific; set
`RECORDER_MODEL_CACHE_DIR` to relocate it. Vulkan downloads the ggml model from
`ggerganov/whisper.cpp`. OpenVINO downloads the matching archive from
`Intel/whisper.cpp-openvino-models` and extracts the ggml model plus encoder IR.

```sh
python download_models.py --engine cpp-vulkan small
python download_models.py --engine cpp-openvino small
```

OpenVINO archive availability depends on the selected model: check the model
repository for a matching `ggml-{model}-models.zip`. The repository currently
describes each archive as containing `ggml-{model}.bin`,
`ggml-{model}-encoder-openvino.xml`, and
`ggml-{model}-encoder-openvino.bin`. Downloads require network access once; run
these commands before using Recorder offline.

## Configure and force a backend

Set the backend-specific environment variable to the built executable. Use an
absolute path if the executable is not on `PATH`:

```sh
# Linux/macOS shell syntax; use the equivalent environment settings in
# PowerShell on Windows.
export RECORDER_WHISPER_CPP_VULKAN=/path/to/whisper-cli
python transcribe.py recording.wav --engine whisper.cpp-vulkan --model small

export RECORDER_WHISPER_CPP_OPENVINO=/path/to/whisper-cli
python transcribe.py recording.wav --engine whisper.cpp-openvino --model small
```

PowerShell example:

```powershell
$env:RECORDER_WHISPER_CPP_VULKAN = 'C:\path\to\whisper-cli.exe'
python transcribe.py recording.wav --engine whisper.cpp-vulkan --model small
```

To allow automatic selection, configure the binary environment variable and
pre-download that backend's selected model. On non-Apple-Silicon hosts auto mode
tries a ready/configured Vulkan backend, then OpenVINO, then CPU. Apple Silicon
continues to prefer MLX. Use `--engine faster-whisper` to explicitly measure the
CPU baseline. A forced unavailable backend reports an error instead of silently
pretending it ran; a runtime GPU confirmation failure can fall back to CPU and
will be reported in the result.

## Acceptance test on real hardware

Mocks verify argument construction, JSON parsing, fallback, and log checks; they
do **not** prove hardware acceleration. For each machine/backend, record:

1. OS/version, CPU and exact GPU/iGPU, driver/runtime versions, whisper.cpp
   revision/build flags, model and audio fixture.
2. A forced-backend run with `--verbose`; verify the printed `engine=` is the
   requested backend and the log contains the expected Vulkan device or
   OpenVINO GPU confirmation. A CPU fallback is a failed GPU acceptance run.
3. CPU baseline with the same fixture/model and `--engine faster-whisper`.
   Compare wall time, real-time factor (`transcription seconds / audio seconds`),
   peak memory, and transcription quality/WER against a reviewed reference.
4. Repeat each run at least three times after warm-up, report median and range,
   and keep model, thread count, and power mode constant. Test a short fixture
   and a representative long recording.
5. Test automatic selection after explicitly configuring both backends, and
   verify the actual engine in the result; verify the graceful CPU fallback with
   a backend that is absent or fails runtime GPU confirmation.

Recorder includes a repeatable harness for these comparisons. From
`lecture_intel/`, pass the same fixture and optional human-reviewed reference to
all runs:

```sh
python eval/backend_benchmark.py path/to/sample.wav \
  --engines faster-whisper,whisper.cpp-vulkan \
  --model small --warmups 1 --repeats 3 --reference path/to/reference.txt
```

For OpenVINO use `--engines faster-whisper,whisper.cpp-openvino`. The harness
writes `eval/backend_benchmark.json` by default with each run, actual backend,
wall time, RTF, and median plus min/max ranges. GPU backend mismatches (including CPU
fallback) are explicitly marked and make the command fail. Peak RSS is sampled
using optional `psutil`; install it in the app environment if memory data is
needed (`python -m pip install psutil`). The reference score is a token WER
proxy; CJK uses character-level units, so inspect the transcript and don't treat
that value as a human-quality judgment. Keep fixture, model and power conditions
identical across backend runs.

Vulkan must be tested separately on at least one Intel iGPU and one AMD iGPU
system to claim coverage. OpenVINO must be tested separately on Intel GPU
hardware. Windows and Linux should each get at least one run before claiming
cross-platform validation. Current development-host tests cannot substitute for
these device tests.

## Known limitation

whisper.cpp emits token timestamps/probabilities rather than Recorder's exact
word-level contract. The adapter groups whitespace-delimited tokens
approximately; CJK output remains token-level, not lexical-word segmentation.
This can affect confidence-driven IELTS analysis and should be reviewed during
quality acceptance, not just speed benchmarking.
