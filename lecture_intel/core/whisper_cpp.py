"""Thin, verifiable adapter for the whisper.cpp command-line interface.

The adapter deliberately returns the raw segment dictionaries consumed by
``Transcriber`` rather than changing the application's transcription flow.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


class WhisperCppError(RuntimeError):
    """Raised when whisper.cpp fails, emits invalid output, or misses its GPU."""


# `--print-progress` writes lines like "cb_progress: progress =  35%" to stderr
# (older builds name the callback whisper_print_progress_callback).
_PROGRESS_RE = re.compile(r"progress\s*=\s*(\d+)\s*%")


def run_streaming(command: Sequence[str], *,
                  on_stderr_line: Optional[Callable[[str], None]] = None,
                  **_ignored: Any) -> subprocess.CompletedProcess:
    """``subprocess.run(capture_output=True, text=True)``, except each stderr
    line reaches ``on_stderr_line`` while the process is still running."""
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace")
    stdout: list[str] = []
    reader = threading.Thread(target=lambda: stdout.append(proc.stdout.read()),
                              daemon=True)
    reader.start()
    stderr: list[str] = []
    for line in proc.stderr:
        stderr.append(line)
        if on_stderr_line:
            on_stderr_line(line)
    reader.join()
    return subprocess.CompletedProcess(command, proc.wait(), "".join(stdout),
                                       "".join(stderr))


class WhisperCppTranscriber:
    """Run whisper.cpp with a requested Vulkan or OpenVINO GPU backend.

    ``binary``, ``backend`` and ``model`` are injectable to make invocation
    behavior independently testable. The subprocess runner can also be
    replaced with ``runner`` (normally ``run_streaming``).
    """

    def __init__(
        self,
        binary: str | Path,
        backend: str,
        model: str | Path,
        *,
        beam_size: int = 5,
        threads: int = 4,
        runner: Callable[..., Any] = run_streaming,
    ) -> None:
        if backend not in {"vulkan", "openvino"}:
            raise ValueError("backend must be 'vulkan' or 'openvino'")
        self.binary = str(binary)
        self.backend = backend
        self.model = str(model)
        self.beam_size = int(beam_size)
        self.threads = int(threads)
        self.runner = runner
        self.detected_language: Optional[str] = None

    def transcribe(
        self,
        audio: str | Path,
        *,
        language: Optional[str] = None,
        prompt: str = "",
        condition_on_previous: bool = False,
        progress: Optional[Callable[[float], None]] = None,
    ) -> list[dict[str, Any]]:
        """Transcribe audio and return ``start/end/text/avg_logprob/words`` dicts.

        ``condition_on_previous=False`` passes ``-mc 0`` (no text carried between
        windows); whisper-cli has no ``--no-context`` flag and would print its
        help instead of transcribing.
        Word boundaries are approximated from whitespace-prefixed token text;
        for CJK, where tokens often do not correspond to orthographic words,
        each token/group is only a token-level approximation, not true word
        segmentation. Token probabilities are used as word probabilities.
        ``progress`` receives the fraction done (0–1) as whisper.cpp reports it.
        """
        audio_path = Path(audio)
        if not audio_path.is_file():
            raise WhisperCppError(f"Audio file does not exist: {audio_path}")
        selected_language = language or "auto"
        with tempfile.TemporaryDirectory(prefix="whisper-cpp-") as tmpdir:
            prefix = Path(tmpdir) / "result"
            command = [
                self.binary, "-m", self.model, "-f", str(audio_path),
                "-ojf", "-of", str(prefix), "-l", selected_language,
                "-bs", str(self.beam_size), "-t", str(self.threads),
            ]
            if prompt:
                command.extend(["--prompt", prompt])
            if not condition_on_previous:
                command.extend(["-mc", "0"])
            if self.backend == "openvino":
                command.extend(["-oved", "GPU"])
            run_kwargs: dict[str, Any] = {"capture_output": True, "text": True,
                                          "check": False}
            if progress is not None:
                command.append("--print-progress")
                run_kwargs["on_stderr_line"] = lambda line: self._report_progress(
                    line, progress)
            try:
                completed = self.runner(command, **run_kwargs)
            except OSError as exc:
                raise WhisperCppError(f"Could not execute whisper.cpp: {exc}") from exc
            stdout = completed.stdout or ""
            stderr = completed.stderr or ""
            logs = f"{stdout}\n{stderr}"
            if completed.returncode:
                raise WhisperCppError(
                    f"whisper.cpp exited with status {completed.returncode}; "
                    f"stderr: {stderr.strip() or '(empty)'}"
                )
            self._verify_backend(logs)
            json_path = prefix.with_suffix(".json")
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except FileNotFoundError as exc:
                raise WhisperCppError(
                    f"whisper.cpp did not create JSON output: {json_path.name}"
                ) from exc
            except (OSError, json.JSONDecodeError) as exc:
                raise WhisperCppError(f"Invalid whisper.cpp JSON output: {exc}") from exc
            result = payload.get("result")
            self.detected_language = (result.get("language")
                                      if isinstance(result, dict) else None)
            return self._normalize(payload, language=self.detected_language)

    @staticmethod
    def _report_progress(line: str, progress: Callable[[float], None]) -> None:
        match = _PROGRESS_RE.search(line)
        if match:
            progress(min(100, int(match.group(1))) / 100)

    def _verify_backend(self, logs: str) -> None:
        lower = logs.lower()
        if self.backend == "vulkan":
            if "ggml_vulkan" not in lower or "found gpu device" not in lower:
                raise WhisperCppError(
                    "Vulkan was requested, but logs do not confirm a "
                    "ggml_vulkan device/GPU; refusing to report GPU execution"
                )
        elif "device = gpu" not in lower or "openvino model loaded" not in lower:
            raise WhisperCppError(
                "OpenVINO GPU was requested, but logs do not contain both "
                "'device = GPU' and 'OpenVINO model loaded'; refusing to "
                "report GPU execution"
            )

    @classmethod
    def _normalize(cls, payload: Any, *, language: Optional[str] = None) -> list[dict[str, Any]]:
        if not isinstance(payload, dict):
            raise WhisperCppError("whisper.cpp JSON top level must be an object")
        transcription = payload.get("transcription")
        if not isinstance(transcription, list):
            raise WhisperCppError("whisper.cpp JSON is missing transcription segments")
        result: list[dict[str, Any]] = []
        for index, segment in enumerate(transcription):
            where = f"transcription[{index}]"
            if not isinstance(segment, dict):
                raise WhisperCppError(f"{where} must be an object")
            offsets = cls._required_object(segment, "offsets", where)
            start = cls._milliseconds(offsets, "from", f"{where}.offsets")
            end = cls._milliseconds(offsets, "to", f"{where}.offsets")
            text = segment.get("text")
            if not isinstance(text, str):
                raise WhisperCppError(f"{where}.text must be a string")
            tokens = segment.get("tokens")
            if not isinstance(tokens, list):
                raise WhisperCppError(f"{where}.tokens must be an array")
            words = cls._tokens_to_words(tokens, where, language=language)
            probabilities = [w["probability"] for w in words]
            avg_logprob = (
                sum(math.log(max(p, 1e-12)) for p in probabilities) / len(probabilities)
                if probabilities else 0.0
            )
            result.append({
                "start": start,
                "end": end,
                "text": text.strip(),
                "avg_logprob": avg_logprob,
                "words": words,
            })
        return result

    @staticmethod
    def _required_object(value: dict[str, Any], key: str, where: str) -> dict[str, Any]:
        item = value.get(key)
        if not isinstance(item, dict):
            raise WhisperCppError(f"{where}.{key} must be an object")
        return item

    @staticmethod
    def _milliseconds(offsets: dict[str, Any], key: str, where: str) -> float:
        value = offsets.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise WhisperCppError(f"{where}.{key} must be a millisecond number")
        return float(value) / 1000.0

    @classmethod
    def _tokens_to_words(
        cls, tokens: Sequence[Any], where: str, *, language: Optional[str] = None
    ) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        cjk_tokens = {"zh", "ja", "ko", "th", "lo", "my"}
        for index, token in enumerate(tokens):
            token_where = f"{where}.tokens[{index}]"
            if not isinstance(token, dict):
                raise WhisperCppError(f"{token_where} must be an object")
            text = token.get("text")
            if not isinstance(text, str):
                raise WhisperCppError(f"{token_where}.text must be a string")
            if re.fullmatch(r"\[_[^]]+_\]", text.strip()):
                continue
            offsets = cls._required_object(token, "offsets", token_where)
            start = cls._milliseconds(offsets, "from", f"{token_where}.offsets")
            end = cls._milliseconds(offsets, "to", f"{token_where}.offsets")
            probability = token.get("p")
            if isinstance(probability, bool) or not isinstance(probability, (int, float)):
                raise WhisperCppError(f"{token_where}.p must be a number")
            probability = min(1.0, max(0.0, float(probability)))
            if language in cjk_tokens or not groups or text[:1].isspace():
                groups.append({"word": text.strip(), "start": start, "end": end,
                               "probability": probability})
            else:
                group = groups[-1]
                group["word"] += text
                group["end"] = end
                group["probability"] = min(group["probability"], probability)
        return [group for group in groups if group["word"]]


# Short alias for callers that prefer the backend name as the public class.
WhisperCpp = WhisperCppTranscriber
