from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.whisper_cpp import WhisperCppError, WhisperCppTranscriber  # noqa: E402


@pytest.fixture
def audio(tmp_path):
    path = tmp_path / "input.wav"
    path.write_bytes(b"wav")
    return path


def payload():
    return {
        "result": {"language": "en"},
        "transcription": [{
            "offsets": {"from": 1200, "to": 2300},
            "text": " Hello world!",
            "tokens": [
                {"text": " Hello", "p": 0.8, "offsets": {"from": 1200, "to": 1700}},
                {"text": " world", "p": 0.5, "offsets": {"from": 1700, "to": 2100}},
                {"text": "!", "p": 0.9, "offsets": {"from": 2100, "to": 2300}},
            ],
        }],
    }


def fake_runner(logs: str):
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        prefix = Path(command[command.index("-of") + 1])
        prefix.with_suffix(".json").write_text(json.dumps(payload()), encoding="utf-8")
        return type("Completed", (), {"returncode": 0, "stdout": logs, "stderr": ""})()

    return run, calls


def test_vulkan_flags_json_normalization_and_logprob(audio):
    runner, calls = fake_runner("ggml_vulkan: found GPU device 0: Intel Graphics")
    transcriber = WhisperCppTranscriber(
        "whisper-cli", "vulkan", "model.bin", runner=runner,
        beam_size=7, threads=3,
    )

    result = transcriber.transcribe(audio, prompt="lecture context", language="en")

    command, kwargs = calls[0]
    assert command[:6] == ["whisper-cli", "-m", "model.bin", "-f", str(audio), "-ojf"]
    assert command[command.index("-of") + 1].endswith("result")
    assert command[command.index("-l") + 1] == "en"
    assert command[command.index("-bs") + 1] == "7"
    assert command[command.index("-t") + 1] == "3"
    assert command[command.index("--prompt") + 1] == "lecture context"
    assert "--no-context" in command
    assert "-oved" not in command
    assert kwargs == {"capture_output": True, "text": True, "check": False}
    assert result == [{
        "start": 1.2, "end": 2.3, "text": "Hello world!",
        "avg_logprob": pytest.approx((__import__("math").log(0.8) + __import__("math").log(0.5)) / 2),
        "words": [
            {"word": "Hello", "start": 1.2, "end": 1.7, "probability": 0.8},
            {"word": "world!", "start": 1.7, "end": 2.3, "probability": 0.5},
        ],
    }]


def test_openvino_gpu_flags_and_log_verification(audio):
    runner, calls = fake_runner("OpenVINO model loaded; device = GPU")
    transcriber = WhisperCppTranscriber("cli", "openvino", "model.bin", runner=runner)
    transcriber.transcribe(audio, condition_on_previous=True)
    command = calls[0][0]
    assert command[command.index("-l") + 1] == "auto"
    assert command[command.index("-oved") + 1] == "GPU"
    assert "--no-context" not in command


def test_gpu_claim_rejected_if_selected_backend_not_confirmed(audio):
    for backend, logs in [
        ("vulkan", "ggml_vulkan: found CPU device 0"),
        ("openvino", "OpenVINO model loaded; device = CPU"),
        ("openvino", "device = GPU"),
    ]:
        runner, _ = fake_runner(logs)
        transcriber = WhisperCppTranscriber("cli", backend, "model.bin", runner=runner)
        with pytest.raises(WhisperCppError, match="refusing to report GPU"):
            transcriber.transcribe(audio)


def test_nonzero_exit_includes_stderr(audio):
    def run(command, **kwargs):
        return type("Completed", (), {"returncode": 2, "stdout": "", "stderr": "bad model"})()

    transcriber = WhisperCppTranscriber("cli", "vulkan", "model.bin", runner=run)
    with pytest.raises(WhisperCppError, match="bad model"):
        transcriber.transcribe(audio)


def test_invalid_json_shape_and_missing_timestamps_fail(audio):
    def run(command, **kwargs):
        Path(command[command.index("-of") + 1] + ".json").write_text(
            json.dumps({"transcription": [{"text": "missing offsets", "tokens": []}]}),
            encoding="utf-8",
        )
        return type("Completed", (), {
            "returncode": 0, "stdout": "ggml_vulkan: found GPU device 0", "stderr": ""
        })()

    transcriber = WhisperCppTranscriber("cli", "vulkan", "model.bin", runner=run)
    with pytest.raises(WhisperCppError, match="offsets"):
        transcriber.transcribe(audio)


def test_cjk_uses_token_level_units_and_skips_special_tokens():
    words = WhisperCppTranscriber._tokens_to_words([
        {"text": "[_BEG_]", "p": 1.0, "offsets": {"from": 0, "to": 0}},
        {"text": " 今", "p": 0.8, "offsets": {"from": 0, "to": 100}},
        {"text": "天", "p": 0.7, "offsets": {"from": 100, "to": 200}},
    ], "transcription[0]", language="zh")
    assert [word["word"] for word in words] == ["今", "天"]


def test_nonexistent_audio_is_reported():
    transcriber = WhisperCppTranscriber("cli", "vulkan", "model.bin")
    with pytest.raises(WhisperCppError, match="Audio file does not exist"):
        transcriber.transcribe("not-a-real-audio.wav")
